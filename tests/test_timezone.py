"""Timezone correctness — the failure mode that silently invalidates everything.

ICT concepts are defined in New York wall-clock time. The UTC offset for New
York changes twice a year, so any code that stores a fixed offset, or that
converts to local time late, will drift by an hour for part of the year. An
hour is enough to move a bar out of one killzone and into another.
"""

from __future__ import annotations

import datetime as dt

import pandas as pd
import pytest

from ictbt.calendar import is_rth, session_date
from ictbt.schema import NY, SchemaError, to_canonical, validate_bars

from .conftest import make_bars

# US daylight saving in 2026: begins Sunday 8 March, ends Sunday 1 November.
SUMMER_SESSION = "2026-03-09"  # Monday after the spring-forward
WINTER_SESSION = "2026-11-02"  # Monday after the fall-back


class TestDaylightSaving:
    def test_ny_open_maps_to_different_utc_hours_across_dst(self):
        """13:30 UTC in summer, 14:30 UTC in winter — both are 09:30 in NY.

        This is the whole reason the index is stored in New York time. If it
        were stored as UTC and filtered on UTC hour, one of these two sessions
        would have its killzones off by an hour.
        """
        summer_open = pd.Timestamp(f"{SUMMER_SESSION} 09:30", tz=NY)
        winter_open = pd.Timestamp(f"{WINTER_SESSION} 09:30", tz=NY)

        assert summer_open.utcoffset() == dt.timedelta(hours=-4)
        assert winter_open.utcoffset() == dt.timedelta(hours=-5)

        assert summer_open.tz_convert("UTC").hour == 13
        assert winter_open.tz_convert("UTC").hour == 14

    def test_utc_bars_convert_to_correct_ny_wall_clock_in_both_seasons(self):
        """A UTC-sourced bar lands on the right New York minute year-round."""
        utc_index = pd.DatetimeIndex(
            [
                pd.Timestamp(f"{SUMMER_SESSION} 13:30", tz="UTC"),
                pd.Timestamp(f"{WINTER_SESSION} 14:30", tz="UTC"),
            ],
            name="timestamp",
        )
        frame = pd.DataFrame(
            {
                "open": [100.0, 100.0],
                "high": [101.0, 101.0],
                "low": [99.0, 99.0],
                "close": [100.5, 100.5],
                "volume": [1000.0, 1000.0],
                "trade_count": [10.0, 10.0],
                "vwap": [100.0, 100.0],
            },
            index=utc_index,
        )

        canonical = to_canonical(frame)

        assert [t.strftime("%H:%M") for t in canonical.index] == ["09:30", "09:30"]
        assert is_rth(canonical.index).all()

    def test_rth_mask_is_stable_across_the_dst_boundary(self):
        """The same wall-clock time is in or out of RTH regardless of season."""
        for date in (SUMMER_SESSION, WINTER_SESSION):
            bars = make_bars(
                [
                    ("09:29", 100.0, 100.0, 100.0, 100.0),
                    ("09:30", 100.0, 100.0, 100.0, 100.0),
                    ("15:59", 100.0, 100.0, 100.0, 100.0),
                    ("16:00", 100.0, 100.0, 100.0, 100.0),
                ],
                date=date,
            )
            assert is_rth(bars.index).tolist() == [False, True, True, False], date


class TestSchemaRejectsAmbiguousTime:
    def test_naive_index_is_rejected(self):
        bars = make_bars()
        bars.index = bars.index.tz_localize(None)
        with pytest.raises(SchemaError, match="timezone-naive"):
            validate_bars(bars)

    def test_utc_index_is_rejected_even_though_it_is_aware(self):
        """Aware-but-wrong-zone is the subtler bug, so it fails too."""
        bars = make_bars().tz_convert("UTC")
        with pytest.raises(SchemaError, match="expected 'America/New_York'"):
            validate_bars(bars)

    def test_canonicalizing_a_naive_frame_refuses_to_guess(self):
        bars = make_bars()
        bars.index = bars.index.tz_localize(None)
        with pytest.raises(SchemaError, match="not inferable"):
            to_canonical(bars)


class TestSessionDate:
    def test_session_date_is_the_new_york_calendar_date(self):
        """A 09:30 NY bar belongs to that NY date, not the UTC date.

        Not currently a distinction for equities, but the 20:00 NY bar of an
        overnight session is already the next UTC day, so this guards the
        futures extension.
        """
        bars = make_bars([("09:30", 100.0, 101.0, 99.0, 100.0)], date=SUMMER_SESSION)
        assert session_date(bars.index).tolist() == [dt.date(2026, 3, 9)]

    def test_late_bar_stays_on_its_new_york_date(self):
        bars = make_bars([("20:00", 100.0, 101.0, 99.0, 100.0)], date=SUMMER_SESSION)
        assert bars.index[0].tz_convert("UTC").date() == dt.date(2026, 3, 10)
        assert session_date(bars.index).tolist() == [dt.date(2026, 3, 9)]
