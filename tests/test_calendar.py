"""Session boundaries.

The boundary cases here are all off-by-one risks created by left-labelled
bars. They look pedantic and they are exactly the kind of thing that shifts a
backtest by one bar without changing anything visibly.
"""

from __future__ import annotations

import datetime as dt

import pytest

from ictbt.calendar import (
    EARLY_CLOSE,
    expected_bars_for_session,
    expected_rth_bars,
    is_early_close,
    is_rth,
    rth_only,
    session_bounds,
    sessions,
)

from .conftest import make_bars, make_session


class TestRegularHoursBoundaries:
    def test_open_bar_is_included(self):
        bars = make_bars([("09:30", 100.0, 101.0, 99.0, 100.0)])
        assert is_rth(bars.index).tolist() == [True]

    def test_bar_before_the_open_is_excluded(self):
        bars = make_bars([("09:29", 100.0, 101.0, 99.0, 100.0)])
        assert is_rth(bars.index).tolist() == [False]

    def test_1559_is_the_last_included_minute_bar(self):
        """Left labelling: the 15:59 bar covers 15:59:00-15:59:59."""
        bars = make_bars([("15:59", 100.0, 101.0, 99.0, 100.0)])
        assert is_rth(bars.index).tolist() == [True]

    def test_the_1600_bar_is_excluded(self):
        """A bar labelled 16:00 opens at the close, so it is after hours."""
        bars = make_bars([("16:00", 100.0, 101.0, 99.0, 100.0)])
        assert is_rth(bars.index).tolist() == [False]

    def test_premarket_and_afterhours_are_dropped(self):
        bars = make_bars(
            [
                ("04:00", 100.0, 101.0, 99.0, 100.0),
                ("09:30", 100.0, 101.0, 99.0, 100.0),
                ("15:45", 100.0, 101.0, 99.0, 100.0),
                ("18:30", 100.0, 101.0, 99.0, 100.0),
            ]
        )
        kept = rth_only(bars)
        assert [t.strftime("%H:%M") for t in kept.index] == ["09:30", "15:45"]


class TestExpectedBarCounts:
    @pytest.mark.parametrize(
        "minutes,expected",
        [(1, 390), (5, 78), (15, 26), (30, 13), (65, 6)],
    )
    def test_full_session_counts(self, minutes, expected):
        """390 minutes of regular hours, divided by the bar size."""
        assert expected_rth_bars(minutes) == expected

    def test_half_day_count(self):
        """09:30-13:00 is 210 minutes, so 14 fifteen-minute bars."""
        assert expected_rth_bars(15, close=EARLY_CLOSE) == 14

    def test_timeframe_that_does_not_divide_the_session_is_rejected(self):
        """390 is not divisible by 7; silently truncating would hide a bug."""
        with pytest.raises(ValueError, match="does not divide evenly"):
            expected_rth_bars(7)

    def test_non_positive_timeframe_is_rejected(self):
        with pytest.raises(ValueError, match="must be positive"):
            expected_rth_bars(0)


class TestSessions:
    def test_full_session_is_26_bars_from_0930_to_1545(self, full_session):
        summary = sessions(full_session)
        assert len(summary) == 1
        row = summary.iloc[0]
        assert row["bars"] == 26
        assert row["first_bar"].strftime("%H:%M") == "09:30"
        assert row["last_bar"].strftime("%H:%M") == "15:45"

    def test_multiple_sessions_are_grouped_by_date(self):
        import pandas as pd

        two_days = pd.concat(
            [
                make_session("2026-03-09", minutes=15),
                make_session("2026-03-10", minutes=15),
            ]
        )
        summary = sessions(two_days)
        assert list(summary.index) == [dt.date(2026, 3, 9), dt.date(2026, 3, 10)]
        assert summary["bars"].tolist() == [26, 26]

    def test_extended_hours_bars_do_not_create_phantom_sessions(self):
        """A weekend or holiday with only after-hours prints is not a session."""
        import pandas as pd

        mixed = pd.concat(
            [
                make_session("2026-03-09", minutes=15),
                make_bars([("18:00", 100.0, 100.5, 99.5, 100.0)], date="2026-03-10"),
            ]
        )
        summary = sessions(mixed, rth=True)
        assert list(summary.index) == [dt.date(2026, 3, 9)]

    def test_empty_input_returns_empty_summary(self):
        from ictbt.schema import empty_bars

        summary = sessions(empty_bars())
        assert summary.empty
        assert "bars" in summary.columns
        assert "early_close" in summary.columns


class TestExchangeCalendar:
    """Session bounds come from the real NYSE schedule, not a fixed window."""

    def test_a_normal_day_closes_at_1600(self):
        bounds = session_bounds(dt.date(2026, 3, 9))
        assert bounds is not None
        assert bounds[0].strftime("%H:%M") == "09:30"
        assert bounds[1].strftime("%H:%M") == "16:00"

    @pytest.mark.parametrize(
        "date",
        [
            dt.date(2024, 7, 3),  # day before Independence Day
            dt.date(2024, 11, 29),  # day after Thanksgiving
            dt.date(2024, 12, 24),  # Christmas Eve
            dt.date(2026, 11, 27),  # day after Thanksgiving 2026
        ],
    )
    def test_known_early_closes_are_recognized(self, date):
        assert is_early_close(date)
        assert session_bounds(date)[1].strftime("%H:%M") == "13:00"

    def test_a_holiday_is_not_a_session(self):
        assert session_bounds(dt.date(2024, 12, 25)) is None

    def test_a_weekend_is_not_a_session(self):
        assert session_bounds(dt.date(2026, 3, 7)) is None

    def test_bars_on_a_holiday_are_never_regular_hours(self):
        """After-hours prints on a closed date must not become a session."""
        bars = make_bars(
            [("10:00", 100.0, 101.0, 99.0, 100.0)], date="2024-12-25"
        )
        assert is_rth(bars.index).tolist() == [False]

    def test_expected_bars_shrinks_on_an_early_close(self):
        assert expected_bars_for_session(dt.date(2026, 3, 9), 15) == 26
        assert expected_bars_for_session(dt.date(2026, 11, 27), 15) == 14

    def test_expected_bars_is_zero_when_the_market_was_shut(self):
        assert expected_bars_for_session(dt.date(2024, 12, 25), 15) == 0

    def test_the_1300_bar_is_excluded_on_an_early_close_day(self):
        """13:00 is the close that day, so a 13:00 bar is after hours."""
        bars = make_bars(
            [
                ("12:45", 100.0, 101.0, 99.0, 100.0),
                ("13:00", 100.0, 101.0, 99.0, 100.0),
            ],
            date="2026-11-27",
        )
        assert is_rth(bars.index).tolist() == [True, False]
