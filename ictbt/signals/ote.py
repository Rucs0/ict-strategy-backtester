"""Optimal Trade Entry: the Fibonacci retracement band.

After a leg from one swing to the next, the OTE is the band where price is
expected to be re-entered on a retracement — conventionally between 62% and
79% back into the leg.

**The ratios are conventional and have no established mechanism.** 0.618 and
0.786 are quoted because they derive from the golden ratio, and the golden
ratio is invoked because it appears in some natural growth processes. No
causal account connects that to order flow, and no ratio in the family has
been shown to outperform an arbitrary one at the same depth. They are exposed
as parameters here and defaulted to the quoted values, which is not an
endorsement of them.

A band is also trivially easier to hit the wider it is. 62%-79% is a 17-point
window on a leg; a comparison against an arbitrary band of the same width at a
different depth is the minimum needed before a hit rate means anything, in the
same way the fair value gap fill rate needed one.

**Leg construction is a choice.** A leg is formed between consecutive
confirmed swings of opposite kind. When two swings of the same kind occur in
sequence, the more extreme one is kept as the anchor — a higher high replaces
a lower one. Other readings are possible and would produce different legs.

**A leg is not usable until both of its swings are confirmed**, so the band
becomes active at the later swing's `confirmed_at`, never at the swing bar.

**The intrabar ordering limitation, stated rather than hidden.** With OHLC
bars there is no way to know whether a bar's high or its low came first. A bar
that both reaches into the OTE band and blows through the leg's origin is
therefore ambiguous: the entry may have filled before invalidation, or not at
all. This module marks such a bar invalidated and records no entry, which is
the conservative reading. It is still an assumption, and on a 15-minute bar it
covers a lot of unobserved price action. Anything sensitive to this needs
finer bars to resolve, not a cleverer rule.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..calendar import rth_only, session_date
from ..schema import validate_bars
from .swings import DEFAULT_N, find_swings

#: Conventional OTE band, as a fraction retraced back into the leg.
#: Chosen, not derived — see the module docstring.
DEFAULT_LOW_RATIO = 0.62
DEFAULT_HIGH_RATIO = 0.79

#: Columns of the returned frame. `leg_end_at` is the index, not a column.
OTE_COLUMNS = (
    "direction",
    "leg_low",
    "leg_high",
    "leg_start_at",
    "band_low",
    "band_high",
    "active_from",
    "entered_at",
    "invalidated_at",
    "bars_to_entry",
)


def find_ote_zones(
    df: pd.DataFrame,
    *,
    n: int = DEFAULT_N,
    low_ratio: float = DEFAULT_LOW_RATIO,
    high_ratio: float = DEFAULT_HIGH_RATIO,
    scope: str = "session",
    rth: bool = True,
    validate: bool = True,
) -> pd.DataFrame:
    """Build OTE bands from swing legs and resolve whether price entered them.

    Returns one row per leg, indexed by `leg_end_at`.
    """
    if not 0.0 < low_ratio < high_ratio < 1.0:
        raise ValueError(
            f"need 0 < low_ratio < high_ratio < 1, got {low_ratio} and {high_ratio}"
        )
    if validate:
        validate_bars(df, name="ote input")

    frame = rth_only(df) if rth else df
    if frame.empty:
        return _empty_ote()

    swings = find_swings(df, n=n, scope=scope, rth=rth, validate=False)
    if swings.empty:
        return _empty_ote()

    if scope == "continuous":
        pieces = [_scan(frame, swings, low_ratio, high_ratio)]
    else:
        pieces = []
        dates = session_date(frame.index).to_numpy()
        swing_dates = session_date(swings.index).to_numpy()
        for date, session in frame.groupby(dates):
            pieces.append(
                _scan(session, swings.loc[swing_dates == date],
                      low_ratio, high_ratio)
            )

    pieces = [p for p in pieces if len(p)]
    if not pieces:
        return _empty_ote()

    out = pd.concat(pieces).sort_index()
    out.index.name = "leg_end_at"
    return out


def build_legs(swings: pd.DataFrame) -> list[dict]:
    """Pair consecutive confirmed swings of opposite kind into legs.

    Same-kind runs collapse to their most extreme member, so a sequence of
    rising highs anchors on the highest.
    """
    confirmed = swings.dropna(subset=["confirmed_at"]).sort_index()
    legs: list[dict] = []
    anchor: tuple[str, float, pd.Timestamp, pd.Timestamp] | None = None

    for swing_at, row in confirmed.iterrows():
        kind = row["kind"]
        price = float(row["price"])
        confirmed_at = row["confirmed_at"]

        if anchor is None:
            anchor = (kind, price, swing_at, confirmed_at)
            continue

        a_kind, a_price, a_at, a_confirmed = anchor
        if kind == a_kind:
            more_extreme = price > a_price if kind == "high" else price < a_price
            if more_extreme:
                anchor = (kind, price, swing_at, confirmed_at)
            continue

        direction = "bullish" if kind == "high" else "bearish"
        leg_low = a_price if direction == "bullish" else price
        leg_high = price if direction == "bullish" else a_price
        if leg_high <= leg_low:
            anchor = (kind, price, swing_at, confirmed_at)
            continue

        legs.append(
            {
                "direction": direction,
                "leg_low": leg_low,
                "leg_high": leg_high,
                "leg_start_at": a_at,
                "leg_end_at": swing_at,
                # Both swings must be confirmed for the leg to exist.
                "active_from": max(a_confirmed, confirmed_at),
            }
        )
        anchor = (kind, price, swing_at, confirmed_at)

    return legs


def _scan(
    frame: pd.DataFrame, swings: pd.DataFrame, low_ratio: float, high_ratio: float
) -> pd.DataFrame:
    """Resolve each leg's band against subsequent price."""
    legs = build_legs(swings)
    if not legs:
        return _empty_ote()

    highs = frame["high"].to_numpy(dtype="float64")
    lows = frame["low"].to_numpy(dtype="float64")
    index = frame.index

    records: list[dict] = []
    for leg in legs:
        span = leg["leg_high"] - leg["leg_low"]
        if leg["direction"] == "bullish":
            # Retracement measured down from the leg high.
            band_high = leg["leg_high"] - low_ratio * span
            band_low = leg["leg_high"] - high_ratio * span
        else:
            # Retracement measured up from the leg low.
            band_low = leg["leg_low"] + low_ratio * span
            band_high = leg["leg_low"] + high_ratio * span

        start = index.searchsorted(leg["active_from"], side="right")
        entered_at, invalidated_at, bars_to_entry = _resolve(
            highs, lows, index, start, leg, band_low, band_high
        )

        records.append(
            {
                **leg,
                "band_low": band_low,
                "band_high": band_high,
                "entered_at": entered_at,
                "invalidated_at": invalidated_at,
                "bars_to_entry": bars_to_entry,
            }
        )

    out = pd.DataFrame.from_records(records).set_index("leg_end_at")
    out.index.name = "leg_end_at"
    return out[list(OTE_COLUMNS)]


def _resolve(
    highs: np.ndarray,
    lows: np.ndarray,
    index: pd.DatetimeIndex,
    start: int,
    leg: dict,
    band_low: float,
    band_high: float,
) -> tuple[pd.Timestamp, pd.Timestamp, float]:
    """Walk forward from activation until the band is entered or the leg dies."""
    bullish = leg["direction"] == "bullish"

    for pos in range(start, len(index)):
        # Invalidation is checked first: a bar that did both is treated as
        # invalidated, because intrabar order is unobservable. See the module
        # docstring.
        invalidated = (
            lows[pos] < leg["leg_low"] if bullish else highs[pos] > leg["leg_high"]
        )
        if invalidated:
            return pd.NaT, index[pos], np.nan

        entered = (
            lows[pos] <= band_high if bullish else highs[pos] >= band_low
        )
        if entered:
            end_pos = index.get_indexer([leg["leg_end_at"]])[0]
            return index[pos], pd.NaT, float(pos - end_pos)

    return pd.NaT, pd.NaT, np.nan


def ote_signal(
    df: pd.DataFrame,
    *,
    n: int = DEFAULT_N,
    direction: str | None = None,
    low_ratio: float = DEFAULT_LOW_RATIO,
    high_ratio: float = DEFAULT_HIGH_RATIO,
    scope: str = "session",
    rth: bool = True,
) -> pd.Series:
    """Boolean series aligned to `df`: did price enter an OTE band here?"""
    if direction is not None and direction not in ("bullish", "bearish"):
        raise ValueError(
            f"direction must be 'bullish' or 'bearish', got {direction!r}"
        )

    zones = find_ote_zones(
        df, n=n, low_ratio=low_ratio, high_ratio=high_ratio, scope=scope, rth=rth
    )
    signal = pd.Series(False, index=df.index, name="ote_signal")
    if zones.empty:
        return signal

    if direction is not None:
        zones = zones[zones["direction"] == direction]
    marks = zones["entered_at"].dropna()
    signal.loc[signal.index.isin(marks)] = True
    return signal


def _empty_ote() -> pd.DataFrame:
    from ..schema import NY

    stamp = f"datetime64[ns, {NY}]"
    return pd.DataFrame(
        {
            "direction": pd.Series(dtype="object"),
            "leg_low": pd.Series(dtype="float64"),
            "leg_high": pd.Series(dtype="float64"),
            "leg_start_at": pd.Series(dtype=stamp),
            "band_low": pd.Series(dtype="float64"),
            "band_high": pd.Series(dtype="float64"),
            "active_from": pd.Series(dtype=stamp),
            "entered_at": pd.Series(dtype=stamp),
            "invalidated_at": pd.Series(dtype=stamp),
            "bars_to_entry": pd.Series(dtype="float64"),
        },
        index=pd.DatetimeIndex([], tz=NY, name="leg_end_at"),
    )
