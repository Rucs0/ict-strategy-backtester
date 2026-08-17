"""Liquidity sweep detection.

A sweep is a bar whose wick penetrates a prior swing point and whose close
returns inside it. The reading is that resting stop orders beyond the swing
were triggered and price was rejected — liquidity taken, direction unchanged.

- **Bullish sweep**: the bar's low breaks a prior swing *low*, and it closes
  back above that low. Sell-side liquidity taken.
- **Bearish sweep**: the bar's high breaks a prior swing *high*, and it closes
  back below it. Buy-side liquidity taken.

**Sweep versus break.** A bar that penetrates a swing and closes *beyond* it is
not a sweep — it is a break, and it means something close to the opposite. ICT
material moves between the two loosely; here they are separate outcomes, and a
broken level is retired rather than left available to be "swept" by a later
bar. Conflating them would let the same level generate a bullish sweep after
price had already broken down through it, which is a signal generated out of a
level that no longer existed.

**The lookahead guard, again.** A sweep can only reference a swing that was
*confirmed* by the time the sweep bar closed, not merely one that exists in
hindsight. Since a swing at bar `i` is confirmed at `i + n`, most of a
session's swings are unusable for much of that session. Using the swing bar's
timestamp instead would be a lookahead bias worth roughly `n` bars — small
enough to look plausible, large enough to invent an edge.

A bar cannot both confirm a swing and directly sweep it — confirming a swing
high requires the confirming bar's high to sit *below* it, and sweeping it
requires the opposite. That much is mutually exclusive. It does **not** make
the confirmation lag harmless, which is what an earlier version of this
docstring wrongly claimed.

The lag matters through queue order. Each bar is resolved against the nearest
live level only, so a level admitted too early sits in front and shields the
older levels behind it. Making swings live at their own bar loses 68 of 3,059
sweeps on ten years of SPY at n=2. Worked example from 2023-07-26: a swing low
at 437.35 is genuinely swept at 14:15, but the 14:00 bar prints a deeper low
of 437.18 that is not confirmed until 14:30. Admit that low early and it
occupies the front of the queue; the 14:15 bar's low of 437.33 fails to
penetrate 437.18, returns nothing, and the real sweep of 437.35 never fires.
The error direction is not even conservative — it moves results both ways.

**Parameters that are choices, not derivations.** `n` is inherited from swing
detection and has no justification anywhere. `min_penetration` defaults to 0,
meaning any penetration counts; a positive value demands the wick clear the
level by some margin, which is the kind of knob that quietly turns a null
result positive. Both must be swept in Phase 3.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..calendar import rth_only, session_date
from ..schema import validate_bars
from .swings import DEFAULT_N, find_swings

SWEEP_COLUMNS = (
    "direction",
    "level",
    "swing_at",
    "penetration",
    "penetration_pct",
    "bars_since_swing",
)


def find_sweeps(
    df: pd.DataFrame,
    *,
    n: int = DEFAULT_N,
    scope: str = "session",
    rth: bool = True,
    min_penetration: float = 0.0,
    validate: bool = True,
) -> pd.DataFrame:
    """Locate liquidity sweeps of prior confirmed swing points.

    Returns one row per sweep, indexed by the timestamp of the sweeping bar —
    which is also the bar at which the sweep is knowable, since the condition
    depends on that bar's own close.

    Args:
        df: canonical bar frame.
        n: bars either side for swing detection. Free parameter.
        scope: "session" or "continuous", passed to swing detection.
        rth: restrict to regular trading hours.
        min_penetration: how far past the level the wick must reach, in price
            units. 0 means any penetration counts. A knob — sweep it.
        validate: check the input against the bar contract.
    """
    if min_penetration < 0:
        raise ValueError(f"min_penetration must be >= 0, got {min_penetration}")
    if validate:
        validate_bars(df, name="sweep input")

    frame = rth_only(df) if rth else df
    if frame.empty:
        return _empty_sweeps()

    swings = find_swings(df, n=n, scope=scope, rth=rth, validate=False)
    if swings.empty:
        return _empty_sweeps()

    if scope == "continuous":
        pieces = [_scan(frame, swings, min_penetration)]
    else:
        pieces = []
        dates = session_date(frame.index).to_numpy()
        swing_dates = session_date(swings.index).to_numpy()
        for date, session in frame.groupby(dates):
            pieces.append(
                _scan(session, swings.loc[swing_dates == date], min_penetration)
            )

    pieces = [p for p in pieces if len(p)]
    if not pieces:
        return _empty_sweeps()

    out = pd.concat(pieces).sort_index()
    out.index.name = "swept_at"
    return out


def _scan(
    frame: pd.DataFrame, swings: pd.DataFrame, min_penetration: float
) -> pd.DataFrame:
    """Walk one stretch of bars, tracking which swing levels are still live."""
    if frame.empty or swings.empty:
        return _empty_sweeps()

    highs = frame["high"].to_numpy(dtype="float64")
    lows = frame["low"].to_numpy(dtype="float64")
    closes = frame["close"].to_numpy(dtype="float64")
    index = frame.index

    # Swings become usable at confirmation, not at the swing bar.
    confirmed = swings.dropna(subset=["confirmed_at"])
    by_confirmation: dict[pd.Timestamp, list[tuple]] = {}
    for swing_at, row in confirmed.iterrows():
        by_confirmation.setdefault(row["confirmed_at"], []).append(
            (row["kind"], float(row["price"]), swing_at)
        )

    # Most recent first, so a sweep resolves against the nearest live level.
    live_highs: list[tuple[float, pd.Timestamp]] = []
    live_lows: list[tuple[float, pd.Timestamp]] = []

    records: list[dict] = []

    for pos, timestamp in enumerate(index):
        high, low, close = highs[pos], lows[pos], closes[pos]

        # Resolve against levels that were already live when this bar opened.
        hit = _resolve(
            live_highs, "bearish", high, close, min_penetration, timestamp, index
        )
        if hit is not None:
            records.append(hit)
        hit = _resolve(
            live_lows, "bullish", low, close, min_penetration, timestamp, index
        )
        if hit is not None:
            records.append(hit)

        # Only now admit swings confirmed by this bar's close, so a level can
        # never be swept by the same bar that made it usable.
        for kind, price, swing_at in by_confirmation.get(timestamp, ()):
            if kind == "high":
                live_highs.insert(0, (price, swing_at))
            else:
                live_lows.insert(0, (price, swing_at))

    if not records:
        return _empty_sweeps()

    out = pd.DataFrame.from_records(records).set_index("swept_at")
    out.index.name = "swept_at"
    return out


def _resolve(
    live: list[tuple[float, pd.Timestamp]],
    direction: str,
    extreme: float,
    close: float,
    min_penetration: float,
    timestamp: pd.Timestamp,
    index: pd.DatetimeIndex,
) -> dict | None:
    """Test the nearest live level, retiring it if swept or broken.

    Returns a record if this bar swept the level, None otherwise. Levels the
    bar closed beyond are retired without a record: that is a break, and the
    level has stopped being a level.
    """
    if not live:
        return None

    bearish = direction == "bearish"
    level, swing_at = live[0]

    penetrated = (
        extreme > level + min_penetration
        if bearish
        else extreme < level - min_penetration
    )
    if not penetrated:
        return None

    live.pop(0)

    closed_back_inside = close < level if bearish else close > level
    if not closed_back_inside:
        return None  # a break, not a sweep

    penetration = (extreme - level) if bearish else (level - extreme)
    return {
        "swept_at": timestamp,
        "direction": direction,
        "level": level,
        "swing_at": swing_at,
        "penetration": penetration,
        "penetration_pct": penetration / level * 100.0 if level else np.nan,
        "bars_since_swing": float(
            index.get_loc(timestamp) - index.get_loc(swing_at)
        ),
    }


def sweep_signal(
    df: pd.DataFrame,
    *,
    n: int = DEFAULT_N,
    direction: str | None = None,
    scope: str = "session",
    rth: bool = True,
    min_penetration: float = 0.0,
) -> pd.Series:
    """Boolean series aligned to `df`: did this bar sweep a prior swing?

    Marked on the sweeping bar, which is correct here — unlike a swing, a
    sweep is fully determined by the bar's own high, low and close.
    """
    # Argument validation precedes any work: a bad argument is a bug whether
    # or not the data happens to contain a sweep.
    if direction is not None and direction not in ("bullish", "bearish"):
        raise ValueError(
            f"direction must be 'bullish' or 'bearish', got {direction!r}"
        )

    sweeps = find_sweeps(
        df, n=n, scope=scope, rth=rth, min_penetration=min_penetration
    )
    signal = pd.Series(False, index=df.index, name="sweep_signal")
    if sweeps.empty:
        return signal

    if direction is not None:
        sweeps = sweeps[sweeps["direction"] == direction]

    signal.loc[signal.index.isin(sweeps.index)] = True
    return signal


def _empty_sweeps() -> pd.DataFrame:
    from ..schema import NY

    return pd.DataFrame(
        {
            "direction": pd.Series(dtype="object"),
            "level": pd.Series(dtype="float64"),
            "swing_at": pd.Series(dtype=f"datetime64[ns, {NY}]"),
            "penetration": pd.Series(dtype="float64"),
            "penetration_pct": pd.Series(dtype="float64"),
            "bars_since_swing": pd.Series(dtype="float64"),
        },
        index=pd.DatetimeIndex([], tz=NY, name="swept_at"),
    )
