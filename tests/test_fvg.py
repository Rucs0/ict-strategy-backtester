"""Fair value gap detection, against hand-built three-candle patterns.

Every fixture here is small enough to verify by eye. The gap in each is
stated in the test name or a comment so the assertion can be checked against
the numbers without re-deriving the pattern.
"""

from __future__ import annotations

import pandas as pd
import pytest

from ictbt.signals import find_fvgs, fvg_signal

from .conftest import make_bars, make_session


def bars(rows, date="2026-03-09"):
    return make_bars(rows, date=date)


# A clean bullish gap: candle 1 tops at 100, candle 3 bottoms at 102.
# The band is 100 -> 102.
BULLISH = [
    ("09:30", 99.0, 100.0, 98.0, 99.5),
    ("09:45", 100.0, 103.0, 99.8, 102.5),
    ("10:00", 102.5, 104.0, 102.0, 103.5),
]

# A clean bearish gap: candle 1 bottoms at 100, candle 3 tops at 98.
# The band is 98 -> 100.
BEARISH = [
    ("09:30", 101.0, 102.0, 100.0, 100.5),
    ("09:45", 100.0, 100.2, 97.0, 97.5),
    ("10:00", 97.5, 98.0, 96.0, 96.5),
]


class TestDetection:
    def test_bullish_gap_is_found_with_correct_bounds(self):
        gaps = find_fvgs(bars(BULLISH), rth=False)

        assert len(gaps) == 1
        gap = gaps.iloc[0]
        assert gap["direction"] == "bullish"
        assert gap["bottom"] == 100.0
        assert gap["top"] == 102.0
        assert gap["size"] == pytest.approx(2.0)

    def test_bearish_gap_is_found_with_correct_bounds(self):
        gaps = find_fvgs(bars(BEARISH), rth=False)

        assert len(gaps) == 1
        gap = gaps.iloc[0]
        assert gap["direction"] == "bearish"
        assert gap["bottom"] == 98.0
        assert gap["top"] == 100.0
        assert gap["size"] == pytest.approx(2.0)

    def test_overlapping_candles_are_not_a_gap(self):
        overlapping = [
            ("09:30", 99.0, 101.0, 98.0, 100.0),
            ("09:45", 100.0, 103.0, 99.0, 102.0),
            ("10:00", 102.0, 104.0, 100.5, 103.0),  # low 100.5 < c1 high 101
        ]
        assert find_fvgs(bars(overlapping), rth=False).empty

    def test_exactly_touching_extremes_are_not_a_gap(self):
        """c1.high == c3.low leaves a band of zero width, so there is no gap."""
        touching = [
            ("09:30", 99.0, 100.0, 98.0, 99.5),
            ("09:45", 100.0, 103.0, 99.8, 102.5),
            ("10:00", 102.5, 104.0, 100.0, 103.5),  # low exactly 100.0
        ]
        assert find_fvgs(bars(touching), rth=False).empty

    def test_fewer_than_three_candles_yields_nothing(self):
        assert find_fvgs(bars(BULLISH[:2]), rth=False).empty

    def test_candle_two_is_not_required_to_be_large(self):
        """The definition constrains candles 1 and 3 only.

        Order blocks need an "impulsive move" and have no mechanical
        definition; the FVG deliberately does not, which is why it is
        testable without inventing a threshold.
        """
        gaps = find_fvgs(bars(BULLISH), rth=False)
        assert len(gaps) == 1


class TestLookaheadBias:
    """The gap is knowable only after candle 3 closes."""

    def test_gap_is_stamped_at_candle_three(self):
        gaps = find_fvgs(bars(BULLISH), rth=False)
        assert gaps.index[0].strftime("%H:%M") == "10:00"

    def test_component_candles_are_recorded(self):
        gap = find_fvgs(bars(BULLISH), rth=False).iloc[0]
        assert gap["c1_at"].strftime("%H:%M") == "09:30"
        assert gap["c2_at"].strftime("%H:%M") == "09:45"

    def test_tradeable_from_is_candle_four(self):
        rows = BULLISH + [("10:15", 103.5, 105.0, 103.0, 104.0)]
        gap = find_fvgs(bars(rows), rth=False).iloc[0]
        assert gap["tradeable_from"].strftime("%H:%M") == "10:15"

    def test_signal_fires_on_candle_four_not_candle_three(self):
        rows = BULLISH + [("10:15", 103.5, 105.0, 103.0, 104.0)]
        signal = fvg_signal(bars(rows), rth=False)

        fired = [t.strftime("%H:%M") for t in signal[signal].index]
        assert fired == ["10:15"]

    def test_a_gap_on_the_last_bar_is_never_tradeable(self):
        """No candle 4 means no bar at which the gap could have been acted on."""
        gap = find_fvgs(bars(BULLISH), rth=False).iloc[0]
        assert pd.isna(gap["tradeable_from"])

        signal = fvg_signal(bars(BULLISH), rth=False)
        assert not signal.any()


class TestSessionScoping:
    """Overnight gaps are not fair value gaps."""

    def test_pattern_does_not_span_a_session_boundary(self):
        """Two bars on Monday, one on Tuesday, with a large overnight gap.

        Unscoped, these three bars satisfy the arithmetic. They are not a
        three-candle pattern — the band between them is the overnight gap
        that most equities have most nights.
        """
        monday = make_bars(
            [
                ("15:30", 99.0, 100.0, 98.0, 99.5),
                ("15:45", 99.5, 100.0, 99.0, 99.8),
            ],
            date="2026-03-09",
        )
        tuesday = make_bars(
            [("09:30", 105.0, 106.0, 104.0, 105.5)], date="2026-03-10"
        )

        assert find_fvgs(pd.concat([monday, tuesday]), rth=False).empty

    def test_gaps_in_separate_sessions_are_both_found(self):
        both = pd.concat(
            [bars(BULLISH, date="2026-03-09"), bars(BULLISH, date="2026-03-10")]
        )
        gaps = find_fvgs(both, rth=False)

        assert len(gaps) == 2
        assert [t.date().isoformat() for t in gaps.index] == [
            "2026-03-09",
            "2026-03-10",
        ]

    def test_after_hours_bars_are_excluded_by_default(self):
        """Sparse premarket prints gap constantly; that is illiquidity."""
        premarket = make_bars(
            [
                ("04:00", 99.0, 100.0, 98.0, 99.5),
                ("04:15", 100.0, 103.0, 99.8, 102.5),
                ("04:30", 102.5, 104.0, 102.0, 103.5),
            ],
            date="2026-03-09",
        )
        assert find_fvgs(premarket, rth=True).empty
        assert len(find_fvgs(premarket, rth=False)) == 1


class TestFill:
    def test_an_untouched_gap_is_unfilled(self):
        rows = BULLISH + [
            ("10:15", 103.5, 105.0, 103.0, 104.0),
            ("10:30", 104.0, 106.0, 103.5, 105.0),
        ]
        gap = find_fvgs(bars(rows), rth=False).iloc[0]

        assert not gap["filled"]
        assert pd.isna(gap["filled_at"])

    def test_touch_mode_fills_on_first_re_entry(self):
        """Band is 100 -> 102; a later low of 101.5 enters it."""
        rows = BULLISH + [
            ("10:15", 103.5, 104.0, 103.0, 103.5),
            ("10:30", 103.5, 104.0, 101.5, 102.0),
        ]
        gap = find_fvgs(bars(rows), rth=False, fill_mode="touch").iloc[0]

        assert gap["filled"]
        assert gap["filled_at"].strftime("%H:%M") == "10:30"
        assert gap["bars_to_fill"] == 2

    def test_full_mode_requires_traversing_the_whole_band(self):
        """A low of 101.5 touches the band but does not cross it to 100."""
        rows = BULLISH + [
            ("10:15", 103.5, 104.0, 103.0, 103.5),
            ("10:30", 103.5, 104.0, 101.5, 102.0),
        ]
        gap = find_fvgs(bars(rows), rth=False, fill_mode="full").iloc[0]
        assert not gap["filled"]

    def test_full_mode_fills_when_price_crosses_the_band(self):
        rows = BULLISH + [
            ("10:15", 103.5, 104.0, 103.0, 103.5),
            ("10:30", 103.5, 104.0, 99.5, 100.0),
        ]
        gap = find_fvgs(bars(rows), rth=False, fill_mode="full").iloc[0]

        assert gap["filled"]
        assert gap["filled_at"].strftime("%H:%M") == "10:30"

    def test_candle_three_cannot_fill_its_own_gap(self):
        """Candle 3's low defines the band's top; it must not count as a fill.

        Searching from candle 3 rather than candle 4 would mark every gap
        filled at birth, which silently produces zero tradeable signals.
        """
        gap = find_fvgs(bars(BULLISH), rth=False).iloc[0]
        assert not gap["filled"]

    def test_bearish_gap_fills_on_a_move_up(self):
        """Band is 98 -> 100; a later high of 98.5 enters it."""
        rows = BEARISH + [
            ("10:15", 96.5, 97.0, 96.0, 96.5),
            ("10:30", 96.5, 98.5, 96.0, 98.0),
        ]
        gap = find_fvgs(bars(rows), rth=False).iloc[0]

        assert gap["filled"]
        assert gap["filled_at"].strftime("%H:%M") == "10:30"

    def test_fill_search_does_not_cross_into_the_next_session(self):
        monday = bars(BULLISH, date="2026-03-09")
        tuesday = make_bars(
            [("09:30", 99.0, 99.5, 95.0, 96.0)], date="2026-03-10"
        )
        gaps = find_fvgs(pd.concat([monday, tuesday]), rth=False)

        assert not gaps.iloc[0]["filled"]


class TestInputHandling:
    def test_invalid_fill_mode_is_rejected(self):
        with pytest.raises(ValueError, match="fill_mode"):
            find_fvgs(bars(BULLISH), fill_mode="partial")

    def test_invalid_direction_is_rejected(self):
        with pytest.raises(ValueError, match="direction"):
            fvg_signal(bars(BULLISH), direction="up")

    def test_direction_filter_selects_one_side(self):
        rows = BULLISH + [("10:15", 103.5, 105.0, 103.0, 104.0)]
        assert fvg_signal(bars(rows), direction="bullish", rth=False).any()
        assert not fvg_signal(bars(rows), direction="bearish", rth=False).any()

    def test_signal_is_aligned_to_the_input_index(self):
        """Including bars dropped by the RTH filter, so it can be assigned back."""
        session = make_session("2026-03-09", minutes=15)
        signal = fvg_signal(session)
        assert signal.index.equals(session.index)

    def test_empty_input_returns_an_empty_frame(self):
        from ictbt.schema import empty_bars

        gaps = find_fvgs(empty_bars())
        assert gaps.empty
        assert "direction" in gaps.columns
