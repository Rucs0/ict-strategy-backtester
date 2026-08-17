"""Gap handling.

The central assertion in this file is a negative one: gaps are reported and
never filled. See `ictbt.quality` for why a forward-filled bar fabricates the
exact patterns this project measures.
"""

from __future__ import annotations

import datetime as dt

import pandas as pd

from ictbt import quality
from ictbt.calendar import rth_only, sessions
from ictbt.schema import NY

from .conftest import make_bars, make_session


class TestMissingBars:
    def test_complete_session_reports_no_gaps(self, full_session):
        missing = quality.missing_bars(full_session, timeframe_minutes=15)
        assert len(missing) == 0

    def test_a_hole_in_the_middle_is_reported(self):
        session = make_session("2026-03-09", minutes=15)
        with_hole = session.drop(session.index[5:8])

        missing = quality.missing_bars(with_hole, timeframe_minutes=15)

        assert [t.strftime("%H:%M") for t in missing] == ["10:45", "11:00", "11:15"]

    def test_gaps_are_reported_not_filled(self):
        """The returned frame is untouched — no synthetic bars appear."""
        session = make_session("2026-03-09", minutes=15)
        with_hole = session.drop(session.index[5:8])

        quality.missing_bars(with_hole, timeframe_minutes=15)

        assert len(with_hole) == 23
        assert "10:45" not in [t.strftime("%H:%M") for t in with_hole.index]

    def test_a_short_session_does_not_report_phantom_trailing_gaps(self):
        """An early close should not read as three missing hours.

        The grid stops at the session's own last bar, so a genuine half day
        reports zero missing bars rather than every bar between 13:00 and the
        regular close.
        """
        half = make_session("2026-11-27", minutes=15, bars=14)  # 09:30-12:45
        missing = quality.missing_bars(half, timeframe_minutes=15)
        assert len(missing) == 0

    def test_gaps_are_scoped_per_session(self):
        """The overnight span between two sessions is not a gap."""
        two_days = pd.concat(
            [
                make_session("2026-03-09", minutes=15),
                make_session("2026-03-10", minutes=15),
            ]
        )
        missing = quality.missing_bars(two_days, timeframe_minutes=15)
        assert len(missing) == 0

    def test_missing_bars_can_be_scoped_to_one_session(self):
        two_days = pd.concat(
            [
                make_session("2026-03-09", minutes=15).drop(
                    make_session("2026-03-09", minutes=15).index[3:4]
                ),
                make_session("2026-03-10", minutes=15),
            ]
        )
        missing = quality.missing_bars(
            two_days, timeframe_minutes=15, session=dt.date(2026, 3, 10)
        )
        assert len(missing) == 0


class TestQualityReport:
    def test_clean_data_reports_no_issues(self, full_session):
        report = quality.check(full_session, symbol="SPY", timeframe_minutes=15)
        assert not report.has_issues
        assert report.total_bars == 26
        assert report.session_count == 1

    def test_incomplete_session_is_flagged(self):
        stub = make_session("2026-03-09", minutes=15, bars=10)
        report = quality.check(stub, symbol="SPY", timeframe_minutes=15)

        assert report.has_issues
        assert len(report.sessions_flagged) == 1
        row = report.sessions_flagged.iloc[0]
        assert row["bars"] == 10
        assert row["expected"] == 26

    def test_a_complete_early_close_session_is_not_flagged(self):
        """2026-11-27 closes at 13:00, so 14 bars is a *complete* session.

        The earlier version of this check measured every session against a
        full 26 bars and flagged all ~25 early closes in the sample as
        damaged. The expectation now comes from the exchange calendar.
        """
        half = make_session("2026-11-27", minutes=15, bars=14)
        report = quality.check(half, symbol="SPY", timeframe_minutes=15)

        assert len(report.sessions_flagged) == 0
        assert report.early_close_sessions == 1

    def test_a_badly_truncated_session_is_not_called_a_half_day(self):
        stub = make_session("2026-03-09", minutes=15, bars=3)
        report = quality.check(stub, symbol="SPY", timeframe_minutes=15)

        row = report.sessions_flagged.iloc[0]
        assert "half day" not in row["likely_cause"]
        assert "excluding" in row["likely_cause"]

    def test_zero_range_bars_are_counted(self):
        """Flat candles are legitimate in illiquid minutes, but they are also
        the signature of a forward-fill, so they get surfaced either way."""
        bars = make_bars(
            [
                ("09:30", 100.0, 100.0, 100.0, 100.0),
                ("09:45", 100.0, 101.0, 99.0, 100.5),
            ]
        )
        report = quality.check(bars, symbol="SPY", timeframe_minutes=15)
        assert report.zero_range_bars == 1

    def test_summary_renders_without_error(self, full_session):
        report = quality.check(full_session, symbol="SPY", timeframe_minutes=15)
        assert "SPY" in report.summary()
        assert "no issues" in report.summary()


class TestPostCloseBars:
    """The bug found against ten years of real SPY data.

    Alpaca keeps returning bars after an early close, because the instrument
    goes on trading after hours on other venues. A fixed 09:30-16:00 window
    counted those thin, jumpy prints as regular session data — and the price
    gaps between them look exactly like fair value gaps.
    """

    def test_bars_after_an_early_close_are_excluded_from_rth(self):
        full_window = make_session("2026-11-27", minutes=15, bars=26)  # to 15:45
        kept = rth_only(full_window)

        assert len(kept) == 14
        assert kept.index[-1].strftime("%H:%M") == "12:45"

    def test_those_bars_are_counted_rather_than_silently_dropped(self):
        full_window = make_session("2026-11-27", minutes=15, bars=26)
        report = quality.check(full_window, symbol="SPY", timeframe_minutes=15)

        assert report.post_close_bars_excluded == 12
        assert report.early_close_sessions == 1

    def test_a_normal_session_has_no_post_close_bars(self, full_session):
        report = quality.check(full_session, symbol="SPY", timeframe_minutes=15)
        assert report.post_close_bars_excluded == 0
        assert report.early_close_sessions == 0

    def test_post_close_bars_do_not_create_phantom_gaps(self):
        """Trailing after-hours bars must not make the session look complete.

        Before the fix, an early-close day showed 26 bars and zero missing —
        which is how ~25 polluted sessions passed every check.
        """
        full_window = make_session("2026-11-27", minutes=15, bars=26)
        missing = quality.missing_bars(full_window, timeframe_minutes=15)
        assert len(missing) == 0
        assert len(rth_only(full_window)) == 14
