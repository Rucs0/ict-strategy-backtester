"""Liquidity sweep detection.

Each fixture builds a swing first, waits for it to be confirmed, then presents
a candidate sweeping bar. The swing and the sweep are separated in the
fixtures so that the confirmation-lag requirement is exercised rather than
accidentally satisfied.
"""

from __future__ import annotations

import pandas as pd
import pytest

from ictbt.signals.sweeps import find_sweeps, sweep_signal

from .conftest import make_bars

TIMES = [
    "09:30", "09:45", "10:00", "10:15", "10:30", "10:45",
    "11:00", "11:15", "11:30", "11:45", "12:00", "12:15",
]


def frame(rows: list[tuple[float, float, float, float]]) -> pd.DataFrame:
    """Bars from (open, high, low, close) tuples on consecutive 15m slots."""
    return make_bars(
        [(t, o, h, low, c) for t, (o, h, low, c) in zip(TIMES, rows)],
        date="2026-03-09",
    )


# A swing high of 110 at 10:00, confirmed at 10:30 with n=2.
# Bars either side stay below it.
SWING_HIGH = [
    (100.0, 102.0, 99.0, 101.0),   # 09:30
    (101.0, 104.0, 100.0, 103.0),  # 09:45
    (103.0, 110.0, 102.0, 104.0),  # 10:00  <- swing high 110
    (104.0, 106.0, 103.0, 105.0),  # 10:15
    (105.0, 107.0, 104.0, 106.0),  # 10:30  <- confirmation
]

# A swing low of 90 at 10:00, confirmed at 10:30 with n=2.
SWING_LOW = [
    (100.0, 101.0, 98.0, 99.0),    # 09:30
    (99.0, 100.0, 96.0, 97.0),     # 09:45
    (97.0, 98.0, 90.0, 96.0),      # 10:00  <- swing low 90
    (96.0, 97.0, 94.0, 95.0),      # 10:15
    (95.0, 96.0, 93.0, 94.0),      # 10:30  <- confirmation
]


class TestBearishSweep:
    def test_wick_through_a_swing_high_closing_back_below_is_a_sweep(self):
        """Bar at 10:45 wicks to 112 (above 110) and closes at 108."""
        bars = frame(SWING_HIGH + [(106.0, 112.0, 105.0, 108.0)])
        sweeps = find_sweeps(bars, n=2, rth=False)

        assert len(sweeps) == 1
        sweep = sweeps.iloc[0]
        assert sweep["direction"] == "bearish"
        assert sweep["level"] == 110.0
        assert sweeps.index[0].strftime("%H:%M") == "10:45"
        assert sweep["penetration"] == pytest.approx(2.0)

    def test_closing_above_the_level_is_a_break_not_a_sweep(self):
        """Same penetration, but the close stays above 110."""
        bars = frame(SWING_HIGH + [(106.0, 112.0, 105.0, 111.0)])
        assert find_sweeps(bars, n=2, rth=False).empty

    def test_not_reaching_the_level_is_nothing(self):
        bars = frame(SWING_HIGH + [(106.0, 109.0, 105.0, 107.0)])
        assert find_sweeps(bars, n=2, rth=False).empty

    def test_a_broken_level_is_retired(self):
        """Once price closes above the swing high, that level is gone.

        Without retirement the same level would remain available and a later
        bar could 'sweep' a high that price had already accepted trading
        above.
        """
        bars = frame(
            SWING_HIGH
            + [
                (106.0, 112.0, 105.0, 111.0),  # 10:45 break, closes above
                (111.0, 113.0, 108.5, 109.0),  # 11:00 would sweep 110
            ]
        )
        assert find_sweeps(bars, n=2, rth=False).empty


class TestBullishSweep:
    def test_wick_through_a_swing_low_closing_back_above_is_a_sweep(self):
        """Bar at 10:45 wicks to 88 (below 90) and closes at 92."""
        bars = frame(SWING_LOW + [(94.0, 95.0, 88.0, 92.0)])
        sweeps = find_sweeps(bars, n=2, rth=False)

        assert len(sweeps) == 1
        sweep = sweeps.iloc[0]
        assert sweep["direction"] == "bullish"
        assert sweep["level"] == 90.0
        assert sweep["penetration"] == pytest.approx(2.0)

    def test_closing_below_the_level_is_a_break(self):
        bars = frame(SWING_LOW + [(94.0, 95.0, 88.0, 89.0)])
        assert find_sweeps(bars, n=2, rth=False).empty


class TestConfirmationRequirement:
    """A sweep may only reference a swing that was already confirmed."""

    def test_a_swing_cannot_be_swept_before_it_is_confirmed(self):
        """The bar right after the swing wicks above it, but with n=2 the
        swing is not confirmed until two bars later — and in fact a bar that
        exceeds the swing high disqualifies it as a swing at all."""
        bars = frame(
            [
                (100.0, 102.0, 99.0, 101.0),
                (101.0, 104.0, 100.0, 103.0),
                (103.0, 110.0, 102.0, 104.0),  # candidate swing high 110
                (104.0, 112.0, 103.0, 105.0),  # wicks above, one bar later
                (105.0, 107.0, 104.0, 106.0),
            ]
        )
        assert find_sweeps(bars, n=2, rth=False).empty

    def test_the_confirming_bar_cannot_itself_sweep_the_level(self):
        """Mutually exclusive by construction, and pinned here anyway.

        Confirming a swing high requires the confirming bar to stay below it;
        sweeping it requires exceeding it. A sweep recorded on a confirmation
        bar would mean the ordering had gone wrong.
        """
        bars = frame(SWING_HIGH + [(106.0, 112.0, 105.0, 108.0)])
        sweeps = find_sweeps(bars, n=2, rth=False)
        assert sweeps.index[0].strftime("%H:%M") == "10:45"

    def test_a_pending_swing_must_not_shield_an_older_level(self):
        """Regression: the case a naive reading of the lag argument misses.

        A bar cannot sweep the swing it confirms, which makes the lag look
        harmless. It is not — each bar resolves against the *nearest* live
        level only, so a level admitted before confirmation sits in front and
        hides the ones behind it.

        Here 100.0 and 101.0 are confirmed swing lows. The 11:30 bar sweeps
        101.0 and prints a new low of 99.0 that is not confirmed until 12:00.
        The 11:45 bar dips to 99.5, which sweeps 100.0. Admit 99.0 early and
        it takes the front of the queue, 99.5 fails to penetrate it, and the
        genuine sweep of 100.0 is lost.

        Reduced from SPY 2023-07-26, where this pattern costs 68 of 3,059
        sweeps across the sample.
        """
        bars = frame(
            [
                (106.0, 110.0, 105.0, 106.0),   # 09:30
                (105.0, 110.0, 104.0, 105.0),   # 09:45
                (104.0, 110.0, 100.0, 104.0),   # 10:00  swing low 100.0
                (104.0, 110.0, 103.0, 104.0),   # 10:15
                (103.0, 110.0, 102.0, 103.0),   # 10:30  confirms 100.0
                (102.0, 110.0, 101.0, 102.0),   # 10:45  swing low 101.0
                (103.0, 110.0, 102.5, 103.0),   # 11:00
                (103.0, 110.0, 102.2, 103.0),   # 11:15  confirms 101.0
                (102.0, 110.0, 99.0, 102.0),    # 11:30  sweeps 101.0
                (101.0, 110.0, 99.5, 101.0),    # 11:45  sweeps 100.0
                (100.0, 110.0, 99.6, 100.0),    # 12:00  confirms 99.0
            ]
        )
        sweeps = find_sweeps(bars, n=2, rth=False)

        assert [t.strftime("%H:%M") for t in sweeps.index] == ["11:30", "11:45"]
        assert list(sweeps["level"]) == [101.0, 100.0]

    def test_larger_n_delays_when_a_sweep_becomes_possible(self):
        """At n=3 the swing is not yet confirmed when the wick arrives."""
        bars = frame(SWING_HIGH + [(106.0, 112.0, 105.0, 108.0)])

        assert len(find_sweeps(bars, n=2, rth=False)) == 1
        assert find_sweeps(bars, n=3, rth=False).empty


class TestMinPenetration:
    def test_zero_threshold_accepts_any_penetration(self):
        bars = frame(SWING_HIGH + [(106.0, 110.5, 105.0, 108.0)])
        assert len(find_sweeps(bars, n=2, rth=False, min_penetration=0.0)) == 1

    def test_a_threshold_rejects_a_shallow_wick(self):
        """A knob that can quietly turn a null result positive; pinned."""
        bars = frame(SWING_HIGH + [(106.0, 110.5, 105.0, 108.0)])
        assert find_sweeps(bars, n=2, rth=False, min_penetration=1.0).empty

    def test_a_threshold_still_accepts_a_deep_wick(self):
        bars = frame(SWING_HIGH + [(106.0, 112.0, 105.0, 108.0)])
        assert len(find_sweeps(bars, n=2, rth=False, min_penetration=1.0)) == 1

    def test_negative_threshold_is_rejected(self):
        with pytest.raises(ValueError, match="min_penetration"):
            find_sweeps(frame(SWING_HIGH), n=2, rth=False, min_penetration=-1.0)


class TestBookkeeping:
    def test_a_level_is_swept_at_most_once(self):
        bars = frame(
            SWING_HIGH
            + [
                (106.0, 112.0, 105.0, 108.0),  # 10:45 sweep
                (108.0, 113.0, 107.0, 109.0),  # 11:00 would sweep again
            ]
        )
        assert len(find_sweeps(bars, n=2, rth=False)) == 1

    def test_the_swept_swing_is_identified(self):
        bars = frame(SWING_HIGH + [(106.0, 112.0, 105.0, 108.0)])
        sweep = find_sweeps(bars, n=2, rth=False).iloc[0]

        assert sweep["swing_at"].strftime("%H:%M") == "10:00"
        assert sweep["bars_since_swing"] == 3

    def test_sweeps_do_not_cross_session_boundaries(self):
        monday = frame(SWING_HIGH)
        tuesday = make_bars(
            [("09:30", 106.0, 112.0, 105.0, 108.0)], date="2026-03-10"
        )
        assert find_sweeps(pd.concat([monday, tuesday]), n=2, rth=False).empty


class TestOutput:
    def test_signal_marks_the_sweeping_bar(self):
        bars = frame(SWING_HIGH + [(106.0, 112.0, 105.0, 108.0)])
        signal = sweep_signal(bars, n=2, rth=False)

        assert [t.strftime("%H:%M") for t in signal[signal].index] == ["10:45"]

    def test_direction_filter_selects_one_side(self):
        bars = frame(SWING_HIGH + [(106.0, 112.0, 105.0, 108.0)])
        assert sweep_signal(bars, n=2, direction="bearish", rth=False).any()
        assert not sweep_signal(bars, n=2, direction="bullish", rth=False).any()

    def test_invalid_direction_is_rejected(self):
        with pytest.raises(ValueError, match="direction"):
            sweep_signal(frame(SWING_HIGH), direction="up", rth=False)

    def test_signal_is_aligned_to_the_input_index(self):
        bars = frame(SWING_HIGH + [(106.0, 112.0, 105.0, 108.0)])
        assert sweep_signal(bars, n=2, rth=False).index.equals(bars.index)

    def test_empty_input_returns_an_empty_frame(self):
        from ictbt.schema import empty_bars

        sweeps = find_sweeps(empty_bars())
        assert sweeps.empty
        assert "direction" in sweeps.columns
