"""Data integrity checks: missing bars, short sessions, suspicious prices.

**The forward-fill trap.** The instinct when a minute bar is missing is to
reindex onto a complete grid and forward-fill. Do not do that here. A
forward-filled bar has ``open == high == low == close``, which is a zero-range
candle, and a zero-range candle fabricates exactly the patterns this project
is trying to measure:

- A fair value gap is candle 1's high below candle 3's low. Insert a
  synthetic flat candle and you have manufactured a gap that no trade ever
  printed.
- A liquidity sweep needs a wick through a swing point. Synthetic bars have
  no wicks, so they suppress real sweeps while inventing clean structure.

So the rule is: gaps are reported, never filled. If a session is too damaged
to use, drop the whole session — an honest missing day beats an invented one.

Note also that a missing minute bar is not automatically an error. Alpaca
emits no bar for a minute in which nothing traded, which is routine in
illiquid names and in the quiet middle of the day. Density therefore says
something about the instrument as well as the feed, and the thresholds below
are review triggers, not verdicts.

**Post-close bars.** Alpaca returns bars past the exchange's close on
early-close days, because the instrument keeps trading after hours elsewhere.
`ictbt.calendar` excludes them from regular hours; this module counts them, so
that a disagreement between the exchange calendar and the vendor's data is
visible rather than silently resolved.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

import pandas as pd

from .calendar import (
    RTH_CLOSE,
    RTH_OPEN,
    exchange_schedule,
    expected_bars_for_session,
    rth_only,
    session_bounds,
    session_date,
    sessions,
)
from .schema import NY

#: A session with fewer than this fraction of its expected bars is flagged.
#: Chosen, not derived — 0.9 is a judgement call about what looks like a
#: damaged session rather than a quiet one. Worth a sensitivity check if
#: results ever turn on which sessions were excluded.
DEFAULT_COMPLETENESS_THRESHOLD = 0.90


@dataclass
class QualityReport:
    """What the integrity checks found. Truthy means something needs a look."""

    symbol: str
    timeframe_minutes: int
    total_bars: int
    session_count: int
    sessions_flagged: pd.DataFrame = field(default_factory=pd.DataFrame)
    duplicate_timestamps: int = 0
    zero_range_bars: int = 0
    zero_volume_bars: int = 0
    early_close_sessions: int = 0
    post_close_bars_excluded: int = 0

    @property
    def has_issues(self) -> bool:
        return bool(
            len(self.sessions_flagged)
            or self.duplicate_timestamps
            or self.zero_range_bars
        )

    def summary(self) -> str:
        lines = [
            f"{self.symbol} @ {self.timeframe_minutes}m — "
            f"{self.total_bars:,} bars across {self.session_count:,} sessions",
        ]
        if self.early_close_sessions:
            lines.append(
                f"  {self.early_close_sessions:,} early-close sessions; "
                f"{self.post_close_bars_excluded:,} after-hours bars excluded "
                "from regular hours"
            )
        if not self.has_issues:
            lines.append("  no issues found")
            return "\n".join(lines)

        if self.duplicate_timestamps:
            lines.append(f"  {self.duplicate_timestamps:,} duplicate timestamps")
        if self.zero_range_bars:
            lines.append(
                f"  {self.zero_range_bars:,} zero-range bars "
                "(high == low; real for illiquid minutes, but also what a "
                "forward-fill leaves behind)"
            )
        if self.zero_volume_bars:
            lines.append(f"  {self.zero_volume_bars:,} zero-volume bars")
        if len(self.sessions_flagged):
            lines.append(f"  {len(self.sessions_flagged):,} incomplete sessions:")
            for date, row in self.sessions_flagged.head(10).iterrows():
                lines.append(
                    f"    {date}  {int(row['bars']):>4}/{int(row['expected']):<4} bars "
                    f"({row['completeness']:.0%})  {row['likely_cause']}"
                )
            if len(self.sessions_flagged) > 10:
                lines.append(f"    ... and {len(self.sessions_flagged) - 10:,} more")
        return "\n".join(lines)


def check(
    df: pd.DataFrame,
    *,
    symbol: str,
    timeframe_minutes: int,
    threshold: float = DEFAULT_COMPLETENESS_THRESHOLD,
) -> QualityReport:
    """Run every integrity check over `df` and return a report.

    Reports; does not modify. Deciding what to exclude is the caller's call,
    because that decision changes results and should be visible in the
    strategy code rather than buried in a loader.
    """
    rth = rth_only(df)
    sess = sessions(df, rth=True)

    flagged = _flag_incomplete_sessions(sess, timeframe_minutes, threshold)

    zero_range = int((rth["high"] == rth["low"]).sum()) if len(rth) else 0
    zero_volume = int((rth["volume"] == 0).sum()) if len(rth) else 0

    return QualityReport(
        symbol=symbol,
        timeframe_minutes=timeframe_minutes,
        total_bars=len(df),
        session_count=len(sess),
        sessions_flagged=flagged,
        duplicate_timestamps=int(df.index.duplicated().sum()),
        zero_range_bars=zero_range,
        zero_volume_bars=zero_volume,
        early_close_sessions=int(sess["early_close"].sum()) if len(sess) else 0,
        post_close_bars_excluded=count_post_close_bars(df),
    )


def count_post_close_bars(df: pd.DataFrame) -> int:
    """Bars inside the nominal 09:30-16:00 window but after the real close.

    These are the after-hours prints on early-close days that a fixed
    wall-clock window would have mistaken for regular session data.
    """
    if df.empty:
        return 0
    sched = exchange_schedule()
    dates = pd.Series(df.index.date, index=range(len(df)))
    closes = dates.map(sched["market_close"])

    known = closes.notna().to_numpy()
    bar_ns = df.index.asi8
    close_ns = pd.DatetimeIndex(closes).asi8

    times = df.index.time
    in_nominal_window = (times >= RTH_OPEN) & (times < RTH_CLOSE)
    return int((known & in_nominal_window & (bar_ns >= close_ns)).sum())


def _flag_incomplete_sessions(
    sess: pd.DataFrame,
    timeframe_minutes: int,
    threshold: float,
) -> pd.DataFrame:
    """Sessions that fall short of their own scheduled length.

    Expectation is per-date, so a genuine early close is measured against a
    half day and is not flagged for being short.
    """
    columns = ["bars", "expected", "completeness", "likely_cause"]
    if sess.empty:
        return pd.DataFrame(columns=columns)

    out = sess.copy()
    out["expected"] = [
        expected_bars_for_session(date, timeframe_minutes) for date in out.index
    ]
    out = out[out["expected"] > 0]
    out["completeness"] = out["bars"] / out["expected"]
    out = out[out["completeness"] < threshold]
    if out.empty:
        return pd.DataFrame(columns=columns)

    out["likely_cause"] = [
        _guess_cause(int(row["bars"]), int(row["expected"]))
        for _, row in out.iterrows()
    ]
    return out[columns]


def _guess_cause(bars: int, expected: int) -> str:
    """A hint for the human reading the report — not a classification.

    Early closes are no longer guessed at; the exchange calendar supplies the
    real close, so anything still short here is genuinely missing data.
    """
    if bars < expected * 0.25:
        return "severely incomplete — consider excluding this session"
    return "incomplete — cause unknown, review before use"


def missing_bars(
    df: pd.DataFrame,
    *,
    timeframe_minutes: int,
    session: dt.date | None = None,
) -> pd.DatetimeIndex:
    """Timestamps absent from the regular-hours grid.

    Returned for inspection only. Nothing in this package fills them; see the
    module docstring for why.
    """
    rth = rth_only(df)
    if session is not None:
        mask = session_date(rth.index).to_numpy() == session
        rth = rth.loc[mask]
    if rth.empty:
        return pd.DatetimeIndex([], tz=NY, name="timestamp")

    missing: list[pd.DatetimeIndex] = []
    for date, group in rth.groupby(session_date(rth.index).to_numpy()):
        grid = _session_grid(date, timeframe_minutes, group.index[-1])
        missing.append(grid.difference(group.index))

    if not missing:
        return pd.DatetimeIndex([], tz=NY, name="timestamp")
    out = missing[0].append(missing[1:]) if len(missing) > 1 else missing[0]
    return pd.DatetimeIndex(out.sort_values(), name="timestamp")


def _session_grid(
    date: dt.date, timeframe_minutes: int, last_bar: pd.Timestamp
) -> pd.DatetimeIndex:
    """The complete bar grid for one session.

    Bounds come from the exchange calendar, so an early close produces a
    half-length grid rather than three hours of phantom missing bars. The grid
    is also capped at the session's own last bar, so a session that is
    genuinely truncated does not report every remaining slot as missing.
    """
    bounds = session_bounds(date)
    if bounds is None:
        return pd.DatetimeIndex([], tz=NY, name="timestamp")

    open_ts, close_ts = bounds
    end = min(last_bar, close_ts - pd.Timedelta(minutes=timeframe_minutes))
    if end < open_ts:
        return pd.DatetimeIndex([], tz=NY, name="timestamp")
    return pd.date_range(
        open_ts, end, freq=f"{timeframe_minutes}min", tz=NY, name="timestamp"
    )
