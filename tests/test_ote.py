"""OTE bands.

Band arithmetic is checked against numbers that can be done by hand: a leg
from 100 to 200 has a span of 100, so the 62%-79% band sits at 121 to 138 on
a bullish leg and 162 to 179 on a bearish one.
"""

from __future__ import annotations

import pandas as pd
import pytest

from ictbt.signals.ote import (
    DEFAULT_HIGH_RATIO,
    DEFAULT_LOW_RATIO,
    build_legs,
    find_ote_zones,
    ote_signal,
)
from ictbt.signals.swings import find_swings

from .conftest import make_bars

TIMES = [
    "09:30", "09:45", "10:00", "10:15", "10:30", "10:45", "11:00",
    "11:15", "11:30", "11:45", "12:00", "12:15", "12:30", "12:45",
]


def frame(rows: list[tuple[float, float, float, float]]) -> pd.DataFrame:
    return make_bars(
        [(t, o, h, low, c) for t, (o, h, low, c) in zip(TIMES, rows)],
        date="2026-03-09",
    )


# Swing low 100 at 10:00 (confirmed 10:30), swing high 200 at 11:00
# (confirmed 11:30). Bullish leg 100 -> 200, span 100.
BULLISH_LEG = [
    (150.0, 160.0, 140.0, 155.0),   # 09:30
    (155.0, 158.0, 130.0, 135.0),   # 09:45
    (135.0, 140.0, 100.0, 138.0),   # 10:00  swing low 100
    (138.0, 145.0, 120.0, 142.0),   # 10:15
    (142.0, 150.0, 125.0, 148.0),   # 10:30  confirms low
    (148.0, 170.0, 145.0, 165.0),   # 10:45
    (165.0, 200.0, 160.0, 195.0),   # 11:00  swing high 200
    (188.0, 190.0, 180.0, 185.0),   # 11:15  (high 190 < 200)
    (185.0, 188.0, 175.0, 180.0),   # 11:30  confirms high
]


class TestBandArithmetic:
    def test_bullish_band_is_measured_down_from_the_leg_high(self):
        """Leg 100 -> 200. 62% back is 138, 79% back is 121."""
        bars = frame(BULLISH_LEG + [(180.0, 182.0, 130.0, 140.0)])
        zones = find_ote_zones(bars, n=2, rth=False)

        assert len(zones) == 1
        zone = zones.iloc[0]
        assert zone["direction"] == "bullish"
        assert zone["leg_low"] == 100.0
        assert zone["leg_high"] == 200.0
        assert zone["band_high"] == pytest.approx(138.0)
        assert zone["band_low"] == pytest.approx(121.0)

    def test_custom_ratios_move_the_band(self):
        bars = frame(BULLISH_LEG + [(180.0, 182.0, 130.0, 140.0)])
        zone = find_ote_zones(
            bars, n=2, rth=False, low_ratio=0.5, high_ratio=0.9
        ).iloc[0]

        assert zone["band_high"] == pytest.approx(150.0)
        assert zone["band_low"] == pytest.approx(110.0)

    def test_the_conventional_ratios_are_the_defaults(self):
        """Pinned so a silent change is caught. Conventional, not derived."""
        assert DEFAULT_LOW_RATIO == 0.62
        assert DEFAULT_HIGH_RATIO == 0.79

    @pytest.mark.parametrize(
        "low,high",
        [(0.79, 0.62), (0.0, 0.79), (0.62, 1.0), (-0.1, 0.5), (0.5, 0.5)],
    )
    def test_invalid_ratio_pairs_are_rejected(self, low, high):
        bars = frame(BULLISH_LEG)
        with pytest.raises(ValueError, match="low_ratio"):
            find_ote_zones(bars, n=2, rth=False, low_ratio=low, high_ratio=high)


class TestEntry:
    def test_price_reaching_into_the_band_is_an_entry(self):
        """A bar dipping to 130 enters the 121-138 band."""
        bars = frame(BULLISH_LEG + [(180.0, 182.0, 130.0, 140.0)])
        zone = find_ote_zones(bars, n=2, rth=False).iloc[0]

        assert zone["entered_at"].strftime("%H:%M") == "11:45"
        assert pd.isna(zone["invalidated_at"])

    def test_a_shallow_retracement_is_not_an_entry(self):
        """A bar dipping only to 145 stays above the band's 138 top."""
        bars = frame(BULLISH_LEG + [(180.0, 182.0, 145.0, 150.0)])
        zone = find_ote_zones(bars, n=2, rth=False).iloc[0]

        assert pd.isna(zone["entered_at"])

    def test_the_band_is_not_live_before_both_swings_confirm(self):
        """At n=3 the high is unconfirmed when the retracement arrives."""
        bars = frame(BULLISH_LEG + [(180.0, 182.0, 130.0, 140.0)])

        assert find_ote_zones(bars, n=2, rth=False)["entered_at"].notna().all()
        at_3 = find_ote_zones(bars, n=3, rth=False)
        assert at_3.empty or at_3["entered_at"].isna().all()

    def test_signal_marks_the_entering_bar(self):
        bars = frame(BULLISH_LEG + [(180.0, 182.0, 130.0, 140.0)])
        signal = ote_signal(bars, n=2, rth=False)

        assert [t.strftime("%H:%M") for t in signal[signal].index] == ["11:45"]


class TestInvalidation:
    def test_breaching_the_leg_origin_kills_the_leg(self):
        """Price below 100 means the leg is gone, not retraced."""
        bars = frame(BULLISH_LEG + [(180.0, 182.0, 95.0, 99.0)])
        zone = find_ote_zones(bars, n=2, rth=False).iloc[0]

        assert pd.isna(zone["entered_at"])
        assert zone["invalidated_at"].strftime("%H:%M") == "11:45"

    def test_a_bar_doing_both_is_treated_as_invalidated(self):
        """The intrabar ambiguity, resolved conservatively and on purpose.

        This bar's low of 95 is below the leg origin and also passes through
        the 121-138 band. OHLC data cannot say which came first, so no entry
        is recorded. A finer timeframe is the only real fix.
        """
        bars = frame(BULLISH_LEG + [(180.0, 182.0, 95.0, 130.0)])
        zone = find_ote_zones(bars, n=2, rth=False).iloc[0]

        assert pd.isna(zone["entered_at"])
        assert pd.notna(zone["invalidated_at"])

    def test_an_entry_before_invalidation_still_counts(self):
        bars = frame(
            BULLISH_LEG
            + [
                (180.0, 182.0, 130.0, 140.0),  # 11:45 entry
                (140.0, 142.0, 95.0, 99.0),    # 12:00 later breakdown
            ]
        )
        zone = find_ote_zones(bars, n=2, rth=False).iloc[0]

        assert zone["entered_at"].strftime("%H:%M") == "11:45"
        assert pd.isna(zone["invalidated_at"])


class TestLegConstruction:
    def test_consecutive_opposite_swings_form_a_leg(self):
        bars = frame(BULLISH_LEG + [(180.0, 182.0, 130.0, 140.0)])
        swings = find_swings(bars, n=2, scope="session", rth=False)
        legs = build_legs(swings)

        assert len(legs) == 1
        assert legs[0]["direction"] == "bullish"
        assert legs[0]["leg_low"] == 100.0
        assert legs[0]["leg_high"] == 200.0

    def test_a_same_kind_run_collapses_to_the_extreme(self):
        """Two swing highs in a row anchor on the higher one."""
        swings = pd.DataFrame(
            {
                "kind": ["high", "high", "low"],
                "price": [110.0, 120.0, 90.0],
                "confirmed_at": pd.DatetimeIndex(
                    ["2026-03-09 10:00", "2026-03-09 10:30", "2026-03-09 11:00"],
                    tz="America/New_York",
                ),
                "bars_to_confirm": [2.0, 2.0, 2.0],
            },
            index=pd.DatetimeIndex(
                ["2026-03-09 09:30", "2026-03-09 10:00", "2026-03-09 10:30"],
                tz="America/New_York",
                name="swing_at",
            ),
        )
        legs = build_legs(swings)

        assert len(legs) == 1
        assert legs[0]["direction"] == "bearish"
        assert legs[0]["leg_high"] == 120.0
        assert legs[0]["leg_low"] == 90.0

    def test_a_leg_activates_at_the_later_confirmation(self):
        bars = frame(BULLISH_LEG + [(180.0, 182.0, 130.0, 140.0)])
        zone = find_ote_zones(bars, n=2, rth=False).iloc[0]

        assert zone["active_from"].strftime("%H:%M") == "11:30"
        # `leg_end_at` is the index: the band goes live after the leg's
        # closing swing, never at it.
        assert zone["active_from"] > zone.name

    def test_unconfirmed_swings_are_excluded(self):
        swings = pd.DataFrame(
            {
                "kind": ["low", "high"],
                "price": [100.0, 200.0],
                "confirmed_at": pd.DatetimeIndex(
                    ["2026-03-09 10:00", pd.NaT], tz="America/New_York"
                ),
                "bars_to_confirm": [2.0, 2.0],
            },
            index=pd.DatetimeIndex(
                ["2026-03-09 09:30", "2026-03-09 10:00"],
                tz="America/New_York",
                name="swing_at",
            ),
        )
        assert build_legs(swings) == []


class TestOutput:
    def test_invalid_direction_is_rejected(self):
        with pytest.raises(ValueError, match="direction"):
            ote_signal(frame(BULLISH_LEG), direction="up", rth=False)

    def test_signal_is_aligned_to_the_input_index(self):
        bars = frame(BULLISH_LEG + [(180.0, 182.0, 130.0, 140.0)])
        assert ote_signal(bars, n=2, rth=False).index.equals(bars.index)

    def test_legs_do_not_cross_session_boundaries(self):
        monday = frame(BULLISH_LEG)
        tuesday = make_bars(
            [("09:30", 180.0, 182.0, 130.0, 140.0)], date="2026-03-10"
        )
        zones = find_ote_zones(pd.concat([monday, tuesday]), n=2, rth=False)
        assert zones.empty or zones["entered_at"].isna().all()

    def test_empty_input_returns_an_empty_frame(self):
        from ictbt.schema import empty_bars

        zones = find_ote_zones(empty_bars())
        assert zones.empty
        assert "band_low" in zones.columns
