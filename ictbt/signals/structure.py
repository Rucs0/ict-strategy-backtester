"""Market structure: breaks of structure and market structure shifts.

A **break** is a bar that *closes* beyond a prior confirmed swing point. It is
the complement of a sweep: a sweep wicks through a level and closes back
inside, a break closes past it. The same bar cannot be both, and every
penetration of a level is exactly one of the two.

Two kinds of break are distinguished:

- **BOS** (break of structure) — a break in the same direction as the
  prevailing structure. Continuation.
- **MSS** (market structure shift) — a break *against* the prevailing
  structure. The one ICT treats as significant, on the reading that structure
  has changed hands.

**Defining "prevailing structure" is where the ambiguity lives.** ICT material
reads structure holistically from a chart — swing sequences, displacement,
context — and no mechanical rule reproduces that. The rule used here is the
narrowest one that needs no second definition: *structure is the direction of
the most recent break*. A close above the nearest live swing high sets
structure bullish; a close below the nearest live swing low sets it bearish.
A break agreeing with the current state is a BOS, a break contradicting it is
an MSS.

The consequences are worth stating plainly rather than hiding:

- The first break of a session has no prior state to contradict, so it is
  classified `bos` with `prior_structure` of None. It is not evidence of
  anything; it is where the state machine starts.
- The rule is self-referential. It never consults higher-high/higher-low
  sequences, trend filters, or any timeframe above the one being scanned.
  Someone reading structure off a chart would disagree with it sometimes.
- Structure resets each session by default. For a day-trading test that is
  the conservative choice; carrying it overnight is available via `scope`.

**Close, not wick.** Requiring a close beyond the level is a choice. A
wick-based rule would classify most sweeps as breaks and delete the
distinction the previous module exists to draw.

**Levels the close has passed are retired.** If a bar closes above several
live swing highs, all of them stop being levels — price has been accepted
above them. This differs from `ictbt.signals.sweeps`, which retires only the
nearest level per bar, because a sweep is a statement about one specific
level being defended while a break is a statement about where price now
trades.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..calendar import rth_only, session_date
from ..schema import validate_bars
from .swings import DEFAULT_N, find_swings

STRUCTURE_COLUMNS = (
    "direction",
    "event",
    "level",
    "swing_at",
    "prior_structure",
    "close",
    "bars_since_swing",
)


def find_structure_events(
    df: pd.DataFrame,
    *,
    n: int = DEFAULT_N,
    scope: str = "session",
    rth: bool = True,
    validate: bool = True,
) -> pd.DataFrame:
    """Locate breaks of structure and market structure shifts.

    Returns one row per break, indexed by the timestamp of the breaking bar.
    The event is knowable at that bar's close, since the condition depends
    only on that bar's close and on swings confirmed earlier.
    """
    if validate:
        validate_bars(df, name="structure input")

    frame = rth_only(df) if rth else df
    if frame.empty:
        return _empty_structure()

    swings = find_swings(df, n=n, scope=scope, rth=rth, validate=False)
    if swings.empty:
        return _empty_structure()

    if scope == "continuous":
        pieces = [_scan(frame, swings)]
    else:
        pieces = []
        dates = session_date(frame.index).to_numpy()
        swing_dates = session_date(swings.index).to_numpy()
        for date, session in frame.groupby(dates):
            pieces.append(_scan(session, swings.loc[swing_dates == date]))

    pieces = [p for p in pieces if len(p)]
    if not pieces:
        return _empty_structure()

    out = pd.concat(pieces).sort_index()
    out.index.name = "broken_at"
    return out


def _scan(frame: pd.DataFrame, swings: pd.DataFrame) -> pd.DataFrame:
    """Walk one stretch of bars, running the structure state machine."""
    if frame.empty or swings.empty:
        return _empty_structure()

    closes = frame["close"].to_numpy(dtype="float64")
    index = frame.index

    confirmed = swings.dropna(subset=["confirmed_at"])
    by_confirmation: dict[pd.Timestamp, list[tuple]] = {}
    for swing_at, row in confirmed.iterrows():
        by_confirmation.setdefault(row["confirmed_at"], []).append(
            (row["kind"], float(row["price"]), swing_at)
        )

    live_highs: list[tuple[float, pd.Timestamp]] = []
    live_lows: list[tuple[float, pd.Timestamp]] = []
    structure: str | None = None
    records: list[dict] = []

    for pos, timestamp in enumerate(index):
        close = closes[pos]

        broken_high = [lvl for lvl in live_highs if close > lvl[0]]
        broken_low = [lvl for lvl in live_lows if close < lvl[0]]

        if broken_high:
            level, swing_at = broken_high[0]  # nearest, i.e. most recent
            event = "mss" if structure == "bearish" else "bos"
            records.append(
                _record(timestamp, "bullish", event, level, swing_at,
                        structure, close, index)
            )
            structure = "bullish"
            live_highs = [lvl for lvl in live_highs if lvl not in broken_high]

        if broken_low:
            level, swing_at = broken_low[0]
            event = "mss" if structure == "bullish" else "bos"
            records.append(
                _record(timestamp, "bearish", event, level, swing_at,
                        structure, close, index)
            )
            structure = "bearish"
            live_lows = [lvl for lvl in live_lows if lvl not in broken_low]

        # Admitted only after this bar is resolved, so a bar can never break
        # a level that its own close made available. See ictbt.signals.sweeps
        # for why admitting early is not harmless.
        for kind, price, swing_at in by_confirmation.get(timestamp, ()):
            if kind == "high":
                live_highs.insert(0, (price, swing_at))
            else:
                live_lows.insert(0, (price, swing_at))

    if not records:
        return _empty_structure()

    out = pd.DataFrame.from_records(records).set_index("broken_at")
    out.index.name = "broken_at"
    return out


def _record(timestamp, direction, event, level, swing_at, prior, close, index):
    return {
        "broken_at": timestamp,
        "direction": direction,
        "event": event,
        "level": level,
        "swing_at": swing_at,
        "prior_structure": prior,
        "close": close,
        "bars_since_swing": float(
            index.get_loc(timestamp) - index.get_loc(swing_at)
        ),
    }


def mss_signal(
    df: pd.DataFrame,
    *,
    n: int = DEFAULT_N,
    direction: str | None = None,
    scope: str = "session",
    rth: bool = True,
) -> pd.Series:
    """Boolean series aligned to `df`: did this bar produce an MSS?

    Excludes BOS events. Use `find_structure_events` for both.
    """
    if direction is not None and direction not in ("bullish", "bearish"):
        raise ValueError(
            f"direction must be 'bullish' or 'bearish', got {direction!r}"
        )

    events = find_structure_events(df, n=n, scope=scope, rth=rth)
    signal = pd.Series(False, index=df.index, name="mss_signal")
    if events.empty:
        return signal

    events = events[events["event"] == "mss"]
    if direction is not None:
        events = events[events["direction"] == direction]
    signal.loc[signal.index.isin(events.index)] = True
    return signal


def _empty_structure() -> pd.DataFrame:
    from ..schema import NY

    return pd.DataFrame(
        {
            "direction": pd.Series(dtype="object"),
            "event": pd.Series(dtype="object"),
            "level": pd.Series(dtype="float64"),
            "swing_at": pd.Series(dtype=f"datetime64[ns, {NY}]"),
            "prior_structure": pd.Series(dtype="object"),
            "close": pd.Series(dtype="float64"),
            "bars_since_swing": pd.Series(dtype="float64"),
        },
        index=pd.DatetimeIndex([], tz=NY, name="broken_at"),
    )
