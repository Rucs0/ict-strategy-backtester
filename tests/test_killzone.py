"""Killzone windows.

The boundary and daylight-saving cases matter most. A killzone is a
wall-clock claim, so it must land on the same local times year-round, and it
must agree with the left-labelled bar convention at both edges.
"""

from __future__ import annotations

import datetime as dt

import pandas as pd
import pytest

from ictbt.signals import (
    KILLZONES,
    Killzone,
    describe_killzones,
    filter_to_killzone,
    get_killzone,
    in_killzone,
    killzone_mask,
)

from .conftest import make_bars

SUMMER = "2026-03-09"  # after spring-forward
WINTER = "2026-11-02"  # after fall-back


class TestWindowBoundaries:
    def test_start_is_inclusive(self):
        bars = make_bars([("10:00", 100.0, 101.0, 99.0, 100.0)])
        assert in_killzone(bars.index, "silver_bullet_am").tolist() == [True]

    def test_end_is_exclusive(self):
        """A window ending 11:00 excludes the 11:00 bar, which opens as it closes."""
        bars = make_bars([("11:00", 100.0, 101.0, 99.0, 100.0)])
        assert in_killzone(bars.index, "silver_bullet_am").tolist() == [False]

    def test_last_bar_inside_the_window_is_included(self):
        bars = make_bars([("10:45", 100.0, 101.0, 99.0, 100.0)])
        assert in_killzone(bars.index, "silver_bullet_am").tolist() == [True]

    def test_bar_before_the_window_is_excluded(self):
        bars = make_bars([("09:45", 100.0, 101.0, 99.0, 100.0)])
        assert in_killzone(bars.index, "silver_bullet_am").tolist() == [False]

    def test_full_boundary_sequence(self):
        bars = make_bars(
            [
                ("09:45", 100.0, 101.0, 99.0, 100.0),
                ("10:00", 100.0, 101.0, 99.0, 100.0),
                ("10:45", 100.0, 101.0, 99.0, 100.0),
                ("11:00", 100.0, 101.0, 99.0, 100.0),
            ]
        )
        assert in_killzone(bars.index, "silver_bullet_am").tolist() == [
            False,
            True,
            True,
            False,
        ]


class TestDaylightSaving:
    def test_the_window_lands_on_the_same_wall_clock_in_both_seasons(self):
        """A killzone is a local-time claim; the UTC offset is irrelevant."""
        for date in (SUMMER, WINTER):
            bars = make_bars(
                [
                    ("09:45", 100.0, 101.0, 99.0, 100.0),
                    ("10:00", 100.0, 101.0, 99.0, 100.0),
                ],
                date=date,
            )
            assert in_killzone(bars.index, "silver_bullet_am").tolist() == [
                False,
                True,
            ], date

    def test_the_underlying_utc_hour_really_does_differ(self):
        """Guards the test above from passing for the wrong reason."""
        summer = make_bars([("10:00", 100.0, 101.0, 99.0, 100.0)], date=SUMMER)
        winter = make_bars([("10:00", 100.0, 101.0, 99.0, 100.0)], date=WINTER)

        assert summer.index[0].tz_convert("UTC").hour == 14
        assert winter.index[0].tz_convert("UTC").hour == 15


class TestMidnightWrap:
    def test_asian_window_wraps_midnight(self):
        kz = get_killzone("asian")
        assert kz.wraps_midnight

    def test_bars_on_both_sides_of_midnight_are_included(self):
        bars = make_bars(
            [
                ("19:45", 100.0, 101.0, 99.0, 100.0),
                ("20:00", 100.0, 101.0, 99.0, 100.0),
                ("23:45", 100.0, 101.0, 99.0, 100.0),
            ]
        )
        assert in_killzone(bars.index, "asian").tolist() == [False, True, True]

    def test_wrapping_duration_is_computed_correctly(self):
        assert get_killzone("asian").duration_minutes() == 240


class TestRegistry:
    def test_unknown_name_raises_with_the_known_names(self):
        with pytest.raises(KeyError, match="unknown killzone"):
            get_killzone("london_lunch")

    def test_a_custom_window_needs_no_registration(self):
        """Windows are data, so a sensitivity sweep does not touch logic."""
        custom = Killzone("custom", dt.time(9, 45), dt.time(10, 15))
        bars = make_bars(
            [
                ("09:30", 100.0, 101.0, 99.0, 100.0),
                ("09:45", 100.0, 101.0, 99.0, 100.0),
                ("10:15", 100.0, 101.0, 99.0, 100.0),
            ]
        )
        assert in_killzone(bars.index, custom).tolist() == [False, True, False]

    def test_windows_outside_equity_hours_are_flagged(self):
        """Asian and London hours cannot be tested on RTH equity data at all."""
        assert not KILLZONES["asian"].overlaps_rth
        assert not KILLZONES["london"].overlaps_rth
        assert KILLZONES["silver_bullet_am"].overlaps_rth
        assert KILLZONES["ny_pm"].overlaps_rth

    def test_ny_am_partially_overlaps_rth(self):
        """07:00-10:00 starts before the open; only 09:30-10:00 is testable."""
        assert KILLZONES["ny_am"].overlaps_rth

    def test_describe_renders_the_registry(self):
        table = describe_killzones()
        assert "silver_bullet_am" in table.index
        assert set(table.columns) == {"start", "end", "minutes", "overlaps_rth"}


class TestComposition:
    def test_mask_unions_several_windows(self):
        bars = make_bars(
            [
                ("10:30", 100.0, 101.0, 99.0, 100.0),  # silver bullet
                ("12:30", 100.0, 101.0, 99.0, 100.0),  # neither
                ("14:00", 100.0, 101.0, 99.0, 100.0),  # ny_pm
            ]
        )
        mask = killzone_mask(bars.index, ["silver_bullet_am", "ny_pm"])
        assert mask.tolist() == [True, False, True]

    def test_empty_list_selects_nothing(self):
        bars = make_bars([("10:30", 100.0, 101.0, 99.0, 100.0)])
        assert not killzone_mask(bars.index, []).any()

    def test_filter_returns_only_matching_bars(self):
        bars = make_bars(
            [
                ("09:45", 100.0, 101.0, 99.0, 100.0),
                ("10:30", 100.0, 101.0, 99.0, 100.0),
            ]
        )
        kept = filter_to_killzone(bars, "silver_bullet_am")
        assert [t.strftime("%H:%M") for t in kept.index] == ["10:30"]

    def test_a_naive_index_is_rejected(self):
        bars = make_bars([("10:30", 100.0, 101.0, 99.0, 100.0)])
        bars.index = bars.index.tz_localize(None)
        with pytest.raises(ValueError, match="America/New_York"):
            in_killzone(bars.index, "silver_bullet_am")
