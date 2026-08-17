"""Trading sessions and New York wall-clock time.

Scope note: this module knows about *sessions* (which bars belong to which
trading day, and when that day actually opened and closed). It deliberately
does not know about killzones — those are a signal primitive and belong to
Phase 2.

**Why this uses a real exchange calendar.** An earlier version treated regular
hours as a fixed 09:30-16:00 wall-clock window and derived sessions from
whatever dates had bars. Checked against ten years of real SPY data, that was
wrong on every early-close day: NYSE shuts at 13:00 on roughly three days a
year, but Alpaca keeps returning bars until 16:00 because SPY continues to
trade after hours on other venues. A fixed window silently relabels three
hours of thin after-hours prints as regular session data.

That matters here more than it would elsewhere. After-hours bars are sparse
and jumpy, with wide price gaps between consecutive prints — which is exactly
the shape of a fair value gap. Left uncorrected, roughly 25 sessions in the
sample would manufacture FVG signals out of nothing but illiquidity.

So session bounds come from `pandas_market_calendars`, which carries the
actual NYSE holiday and early-close schedule. The tradeoff accepted here is a
dependency, and the risk that the calendar and the vendor's data disagree —
`ictbt.quality` reports that disagreement rather than hiding it.
"""

from __future__ import annotations

import datetime as dt
import functools

import pandas as pd
import pandas_market_calendars as mcal

from .schema import NY

#: Nominal regular trading hours. Real sessions come from the exchange
#: calendar; these remain as the default expectation and for building
#: fixtures.
RTH_OPEN = dt.time(9, 30)
RTH_CLOSE = dt.time(16, 0)

#: The standard NYSE early close.
EARLY_CLOSE = dt.time(13, 0)

#: Range the exchange schedule is materialized over. Wide enough to cover all
#: available Alpaca history (2016 onward) plus headroom.
_SCHEDULE_START = "2015-01-01"
_SCHEDULE_END = "2030-12-31"

_EXCHANGE = "NYSE"


@functools.lru_cache(maxsize=4)
def exchange_schedule(exchange: str = _EXCHANGE) -> pd.DataFrame:
    """Real open/close times per trading date, in New York time.

    Indexed by `datetime.date`, with `market_open` and `market_close` columns.
    Memoized — building it costs about a second and never changes at runtime.
    """
    calendar = mcal.get_calendar(exchange)
    sched = calendar.schedule(start_date=_SCHEDULE_START, end_date=_SCHEDULE_END)
    out = pd.DataFrame(
        {
            "market_open": sched["market_open"].dt.tz_convert(NY),
            "market_close": sched["market_close"].dt.tz_convert(NY),
        }
    )
    out.index = pd.Index([ts.date() for ts in sched.index], name="session_date")
    return out


def session_bounds(date: dt.date, exchange: str = _EXCHANGE) -> tuple[pd.Timestamp, pd.Timestamp] | None:
    """Actual open and close for `date`, or None if the market was closed."""
    sched = exchange_schedule(exchange)
    if date not in sched.index:
        return None
    row = sched.loc[date]
    return row["market_open"], row["market_close"]


def is_early_close(date: dt.date, exchange: str = _EXCHANGE) -> bool:
    """Did the exchange close before 16:00 on this date?"""
    bounds = session_bounds(date, exchange)
    return bounds is not None and bounds[1].time() < RTH_CLOSE


def is_rth(index: pd.DatetimeIndex, exchange: str = _EXCHANGE) -> pd.Series:
    """Boolean mask: does this bar fall inside the real regular session?

    Uses each date's actual close, so bars after a 13:00 early close are
    excluded rather than counted as regular hours. Bars on dates the exchange
    was shut — weekends, holidays — are excluded entirely.

    Bars are left-labelled (see `ictbt.schema`), so the test is
    ``open <= t < close``. The 15:59 bar is the last regular-hours minute bar
    on a full day; a bar labelled 16:00 opens at the close and is excluded.
    """
    _require_ny(index)
    if len(index) == 0:
        return pd.Series([], index=index, dtype=bool, name="is_rth")

    sched = exchange_schedule(exchange)
    dates = pd.Series(index.date, index=range(len(index)))
    opens = dates.map(sched["market_open"])
    closes = dates.map(sched["market_close"])

    known = opens.notna().to_numpy()
    bar_ns = index.asi8
    open_ns = pd.DatetimeIndex(opens).asi8
    close_ns = pd.DatetimeIndex(closes).asi8

    mask = known & (bar_ns >= open_ns) & (bar_ns < close_ns)
    return pd.Series(mask, index=index, name="is_rth")


def session_date(index: pd.DatetimeIndex) -> pd.Series:
    """The trading date each bar belongs to.

    For US equities the session does not cross midnight, so this is just the
    New York calendar date. It exists as a named function anyway because that
    stops being true the moment futures are added — an equity index future
    session opens at 18:00 the previous evening — and every caller that says
    ``.index.date`` inline would need finding and fixing.
    """
    _require_ny(index)
    return pd.Series(index.date, index=index, name="session_date")


def rth_only(df: pd.DataFrame, exchange: str = _EXCHANGE) -> pd.DataFrame:
    """Drop everything outside the real regular session."""
    return df.loc[is_rth(df.index, exchange).to_numpy()]


def sessions(df: pd.DataFrame, *, rth: bool = True) -> pd.DataFrame:
    """Summarize `df` one row per trading session.

    Returns a frame indexed by session date with the bar count, the first and
    last bar timestamp, the exchange's scheduled close, and whether that close
    was early. This is the input to the completeness checks in
    `ictbt.quality`.
    """
    frame = rth_only(df) if rth else df
    if frame.empty:
        return pd.DataFrame(
            {
                "bars": pd.Series(dtype="int64"),
                "first_bar": pd.Series(dtype=f"datetime64[ns, {NY}]"),
                "last_bar": pd.Series(dtype=f"datetime64[ns, {NY}]"),
                "scheduled_close": pd.Series(dtype="object"),
                "early_close": pd.Series(dtype="bool"),
            },
            index=pd.Index([], name="session_date"),
        )

    grouper = session_date(frame.index).to_numpy()
    grouped = frame.groupby(grouper, sort=True)
    out = pd.DataFrame(
        {
            "bars": grouped.size(),
            "first_bar": grouped.apply(lambda g: g.index[0]),
            "last_bar": grouped.apply(lambda g: g.index[-1]),
        }
    )
    out.index.name = "session_date"

    sched = exchange_schedule()
    closes = pd.Series(out.index, index=out.index).map(sched["market_close"])
    out["scheduled_close"] = [
        ts.time() if pd.notna(ts) else None for ts in closes
    ]
    out["early_close"] = [
        bool(t is not None and t < RTH_CLOSE) for t in out["scheduled_close"]
    ]
    return out


def expected_rth_bars(
    minutes_per_bar: int,
    *,
    close: dt.time = RTH_CLOSE,
    open_: dt.time = RTH_OPEN,
) -> int:
    """How many bars a session running `open_` to `close` should contain.

    A full session is 09:30-16:00, i.e. 390 minutes, so 390 one-minute bars or
    26 fifteen-minute bars.
    """
    if minutes_per_bar <= 0:
        raise ValueError(f"minutes_per_bar must be positive, got {minutes_per_bar}")

    open_minutes = open_.hour * 60 + open_.minute
    close_minutes = close.hour * 60 + close.minute
    span = close_minutes - open_minutes
    if span <= 0:
        raise ValueError(f"close {close} is not after open {open_}")
    if span % minutes_per_bar:
        raise ValueError(
            f"a {span}-minute session does not divide evenly into "
            f"{minutes_per_bar}-minute bars"
        )
    return span // minutes_per_bar


def expected_bars_for_session(
    date: dt.date, minutes_per_bar: int, exchange: str = _EXCHANGE
) -> int:
    """Expected bar count for one specific date, honouring early closes.

    Returns 0 for dates the exchange was closed.
    """
    bounds = session_bounds(date, exchange)
    if bounds is None:
        return 0
    open_ts, close_ts = bounds
    span = int((close_ts - open_ts).total_seconds() // 60)
    return span // minutes_per_bar


def _require_ny(index: pd.DatetimeIndex) -> None:
    if not isinstance(index, pd.DatetimeIndex):
        raise TypeError(f"expected DatetimeIndex, got {type(index).__name__}")
    if index.tz is None or str(index.tz) != "America/New_York":
        raise ValueError(
            f"expected an America/New_York index, got tz={index.tz!r}. "
            "Session logic is wall-clock logic; converting late is how "
            "daylight-saving bugs get in."
        )
