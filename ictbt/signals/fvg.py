"""Fair value gap detection.

A fair value gap is a three-candle pattern where the first and third candles
do not overlap, leaving a price band that the middle candle traversed without
trading through on either side:

- **Bullish**: candle 1's high is strictly below candle 3's low. The band
  between them is the gap, and price is above it.
- **Bearish**: candle 1's low is strictly above candle 3's high. The band
  between them is the gap, and price is below it.

This is the most mechanically defined ICT concept — no free parameters in the
detection itself — which is why it is implemented first. The judgement calls
are all downstream of detection, and each is named below.

**When the gap becomes knowable.** The pattern is only visible once candle 3
has closed. Every row this module emits is therefore stamped at candle 3, and
`tradeable_from` points at candle 4. Stamping a gap at candle 2 — where it
visually sits on a chart — would let a backtest act on information that did
not exist yet, which inflates results in a way that is very hard to see once
it is buried in a P&L curve.

**Session scoping.** Detection never spans a session boundary. Yesterday's
15:45 candle and today's 09:45 candle are not a three-candle pattern; the
band between them is the overnight gap, which every equity has most nights
and which has nothing to do with intraday order flow. Left unscoped this
alone would manufacture roughly one "signal" per day.

**Fill.** "Not yet filled" needs a definition, and the two obvious ones give
different answers. `fill_mode="touch"` (the default) counts the gap as filled
the moment price re-enters the band at all; `fill_mode="full"` requires price
to traverse the whole band. Touch is the stricter, less flattering choice —
it invalidates gaps sooner, so fewer remain "active" to trade. It is chosen
for that reason and it is a choice, not a derivation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..calendar import rth_only, session_date
from ..schema import validate_bars

#: Columns of the frame returned by `find_fvgs`.
FVG_COLUMNS = (
    "direction",
    "top",
    "bottom",
    "size",
    "size_pct",
    "c1_at",
    "c2_at",
    "tradeable_from",
    "filled_at",
    "filled",
    "bars_to_fill",
)

_FILL_MODES = ("touch", "full")


def find_fvgs(
    df: pd.DataFrame,
    *,
    rth: bool = True,
    fill_mode: str = "touch",
    validate: bool = True,
) -> pd.DataFrame:
    """Locate every fair value gap in `df`.

    Returns one row per gap, indexed by `formed_at` — the timestamp of
    candle 3, the first bar at which the gap is knowable.

    Args:
        df: canonical bar frame.
        rth: restrict to regular trading hours before detecting. After-hours
            bars are sparse and jumpy; the gaps between them are illiquidity,
            not order flow.
        fill_mode: "touch" or "full". See module docstring.
        validate: check the input against the bar contract first.
    """
    if fill_mode not in _FILL_MODES:
        raise ValueError(f"fill_mode must be one of {_FILL_MODES}, got {fill_mode!r}")
    if validate:
        validate_bars(df, name="fvg input")

    frame = rth_only(df) if rth else df
    if len(frame) < 3:
        return _empty_fvgs()

    rows = [
        _scan_session(session, fill_mode)
        for _, session in frame.groupby(session_date(frame.index).to_numpy())
    ]
    rows = [r for r in rows if len(r)]
    if not rows:
        return _empty_fvgs()

    out = pd.concat(rows).sort_index()
    out.index.name = "formed_at"
    return out


def _scan_session(session: pd.DataFrame, fill_mode: str) -> pd.DataFrame:
    """Find and resolve every gap inside one trading session."""
    if len(session) < 3:
        return _empty_fvgs()

    high = session["high"].to_numpy()
    low = session["low"].to_numpy()
    index = session.index

    # Candle 1 at i, candle 2 at i+1, candle 3 at i+2. Strict inequality:
    # touching extremes leave no band, so they are not a gap.
    bullish = high[:-2] < low[2:]
    bearish = low[:-2] > high[2:]

    first = np.flatnonzero(bullish | bearish)
    if not len(first):
        return _empty_fvgs()

    third = first + 2
    is_bull = bullish[first]

    # Bullish: band runs from candle 1's high up to candle 3's low.
    # Bearish: band runs from candle 3's high up to candle 1's low.
    bottom = np.where(is_bull, high[first], high[third])
    top = np.where(is_bull, low[third], low[first])

    formed_at = index[third]
    # Candle 4. The last gap in a session may have no candle 4, in which case
    # it is never tradeable within that session and is marked NaT.
    next_pos = third + 1
    tradeable = np.where(next_pos < len(index), next_pos, -1)
    tradeable_from = pd.DatetimeIndex(
        [index[p] if p >= 0 else pd.NaT for p in tradeable], tz=index.tz
    )

    filled_at, bars_to_fill = _resolve_fills(
        high, low, third, top, bottom, is_bull, index, fill_mode
    )

    mid = (top + bottom) / 2.0
    size = top - bottom

    out = pd.DataFrame(
        {
            "direction": np.where(is_bull, "bullish", "bearish"),
            "top": top,
            "bottom": bottom,
            "size": size,
            "size_pct": np.divide(size, mid, out=np.zeros_like(size), where=mid != 0)
            * 100.0,
            "c1_at": index[first],
            "c2_at": index[first + 1],
            "tradeable_from": tradeable_from,
            "filled_at": filled_at,
            "filled": ~pd.isna(filled_at),
            "bars_to_fill": bars_to_fill,
        },
        index=formed_at,
    )
    out.index.name = "formed_at"
    return out


def _resolve_fills(
    high: np.ndarray,
    low: np.ndarray,
    third: np.ndarray,
    top: np.ndarray,
    bottom: np.ndarray,
    is_bull: np.ndarray,
    index: pd.DatetimeIndex,
    fill_mode: str,
) -> tuple[pd.DatetimeIndex, np.ndarray]:
    """First bar after candle 3 at which each gap is filled.

    The search starts at candle 4, never at candle 3 — candle 3 is what
    *created* the band, so treating its own extreme as a fill would close
    every gap at birth.

    Scoped to the session. A day-trading strategy does not carry a gap
    overnight, and bounding the scan keeps this linear per session rather
    than quadratic over the whole history.
    """
    filled_at: list[pd.Timestamp] = []
    bars_to_fill: list[float] = []

    for k in range(len(third)):
        start = third[k] + 1
        if start >= len(index):
            filled_at.append(pd.NaT)
            bars_to_fill.append(np.nan)
            continue

        if is_bull[k]:
            # Price sits above a bullish gap; a fill is a move back down.
            level = top[k] if fill_mode == "touch" else bottom[k]
            hit = np.flatnonzero(low[start:] <= level)
        else:
            # Price sits below a bearish gap; a fill is a move back up.
            level = bottom[k] if fill_mode == "touch" else top[k]
            hit = np.flatnonzero(high[start:] >= level)

        if len(hit):
            pos = start + hit[0]
            filled_at.append(index[pos])
            bars_to_fill.append(float(pos - third[k]))
        else:
            filled_at.append(pd.NaT)
            bars_to_fill.append(np.nan)

    return (
        pd.DatetimeIndex(filled_at, tz=index.tz),
        np.asarray(bars_to_fill, dtype="float64"),
    )


def fvg_signal(
    df: pd.DataFrame,
    *,
    direction: str | None = None,
    rth: bool = True,
    fill_mode: str = "touch",
) -> pd.Series:
    """Boolean series aligned to `df`: is a gap actionable on this bar?

    True on the bar at which a gap first becomes tradeable — candle 4, not
    candle 3. Aligning to candle 3 would mark the bar whose close revealed the
    pattern, and a backtest entering there would be entering on information it
    could not have had until that bar was over.
    """
    # Argument validation precedes any work: a bad argument is a bug whether
    # or not the data happens to contain a gap.
    if direction is not None and direction not in ("bullish", "bearish"):
        raise ValueError(
            f"direction must be 'bullish' or 'bearish', got {direction!r}"
        )

    gaps = find_fvgs(df, rth=rth, fill_mode=fill_mode)
    signal = pd.Series(False, index=df.index, name="fvg_signal")
    if gaps.empty:
        return signal

    if direction is not None:
        gaps = gaps[gaps["direction"] == direction]

    marks = gaps["tradeable_from"].dropna()
    signal.loc[signal.index.isin(marks)] = True
    return signal


def _empty_fvgs() -> pd.DataFrame:
    from ..schema import NY

    return pd.DataFrame(
        {
            "direction": pd.Series(dtype="object"),
            "top": pd.Series(dtype="float64"),
            "bottom": pd.Series(dtype="float64"),
            "size": pd.Series(dtype="float64"),
            "size_pct": pd.Series(dtype="float64"),
            "c1_at": pd.Series(dtype=f"datetime64[ns, {NY}]"),
            "c2_at": pd.Series(dtype=f"datetime64[ns, {NY}]"),
            "tradeable_from": pd.Series(dtype=f"datetime64[ns, {NY}]"),
            "filled_at": pd.Series(dtype=f"datetime64[ns, {NY}]"),
            "filled": pd.Series(dtype="bool"),
            "bars_to_fill": pd.Series(dtype="float64"),
        },
        index=pd.DatetimeIndex([], tz=NY, name="formed_at"),
    )
