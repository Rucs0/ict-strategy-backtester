"""Swing point detection against hand-built fractals.

Fixtures give the high (or low) of each bar explicitly, so the expected swing
can be read off the numbers without re-deriving the pattern. The opposite
extreme is pinned far away and identical on every bar, so only one side can
produce swings and ties suppress the other.
"""

from __future__ import annotations

import pandas as pd
import pytest

from ictbt.signals.swings import DEFAULT_N, find_swings, swing_signal

from .conftest import make_bars

TIMES = [
    "09:30", "09:45", "10:00", "10:15", "10:30", "10:45",
    "11:00", "11:15", "11:30", "11:45", "12:00",
]


def peak(highs: list[float], date: str = "2026-03-09") -> pd.DataFrame:
    """Bars with the given highs; every low is identical and far below."""
    floor = min(highs) - 5.0
    return make_bars(
        [(t, h - 0.5, h, floor, h - 0.2) for t, h in zip(TIMES, highs)], date=date
    )


def trough(lows: list[float], date: str = "2026-03-09") -> pd.DataFrame:
    """Bars with the given lows; every high is identical and far above."""
    ceiling = max(lows) + 5.0
    return make_bars(
        [(t, low + 0.5, ceiling, low, low + 0.2) for t, low in zip(TIMES, lows)],
        date=date,
    )


class TestSwingHighs:
    def test_a_clean_peak_is_a_swing_high(self):
        """Highs 100,101,105,101,100 — the 105 beats two bars either side."""
        swings = find_swings(peak([100, 101, 105, 101, 100]), n=2, rth=False)

        assert len(swings) == 1
        assert swings.iloc[0]["kind"] == "high"
        assert swings.iloc[0]["price"] == 105.0
        assert swings.index[0].strftime("%H:%M") == "10:00"

    def test_a_clean_trough_is_a_swing_low(self):
        """Lows 99,98,95,98,99 — the 95 undercuts two bars either side."""
        swings = find_swings(trough([99, 98, 95, 98, 99]), n=2, rth=False)

        assert len(swings) == 1
        assert swings.iloc[0]["kind"] == "low"
        assert swings.iloc[0]["price"] == 95.0

    def test_a_bar_beaten_on_one_side_is_not_a_swing(self):
        """The 105 loses to the 106 immediately to its right."""
        assert find_swings(peak([100, 101, 105, 106, 100]), n=2, rth=False).empty

    def test_a_tie_is_not_a_swing(self):
        """Equal highs are resting liquidity, not a fractal.

        Counting both bars would double-count the very cluster a liquidity
        sweep is meant to target.
        """
        assert find_swings(peak([100, 101, 105, 105, 100]), n=2, rth=False).empty

    def test_edge_bars_cannot_be_swings(self):
        """The first and last n bars lack the neighbours to be evaluated."""
        swings = find_swings(peak([109, 101, 100, 101, 109]), n=2, rth=False)
        assert swings.empty


class TestParameterN:
    def test_n_controls_how_many_neighbours_must_be_beaten(self):
        """Highs 100,102,106,103,107,103,100.

        The 106 beats its immediate neighbours, so it is a swing at n=1. It
        does not beat the 107 two bars to its right, so it is not one at n=2.
        The 107 survives both.
        """
        bars = peak([100, 102, 106, 103, 107, 103, 100])

        at_1 = find_swings(bars, n=1, rth=False)
        at_2 = find_swings(bars, n=2, rth=False)

        assert sorted(at_1.loc[at_1["kind"] == "high", "price"]) == [106.0, 107.0]
        assert sorted(at_2.loc[at_2["kind"] == "high", "price"]) == [107.0]

    def test_larger_n_is_strictly_more_selective(self):
        """Anything surviving n=3 must also survive n=2."""
        bars = peak([100, 103, 102, 104, 109, 104, 102, 103, 100])
        wide = set(find_swings(bars, n=3, rth=False).index)
        narrow = set(find_swings(bars, n=2, rth=False).index)
        assert wide <= narrow

    def test_n_below_one_is_rejected(self):
        with pytest.raises(ValueError, match="at least 1"):
            find_swings(peak([100, 101, 105, 101, 100]), n=0, rth=False)

    def test_default_n_is_two_and_is_an_arbitrary_choice(self):
        """Pinned so a silent change to the default is caught.

        No ICT source specifies n. Anything built on swings inherits it, so a
        result that holds at 2 and dies at 3 is noise.
        """
        assert DEFAULT_N == 2

    def test_too_few_bars_yields_nothing(self):
        assert find_swings(peak([100, 101, 105]), n=2, rth=False).empty


class TestConfirmationLag:
    """A swing is not knowable until n further bars have closed."""

    def test_confirmed_at_is_n_bars_after_the_swing(self):
        swings = find_swings(peak([100, 101, 105, 101, 100]), n=2, rth=False)
        row = swings.iloc[0]

        assert swings.index[0].strftime("%H:%M") == "10:00"
        assert row["confirmed_at"].strftime("%H:%M") == "10:30"
        assert row["bars_to_confirm"] == 2

    def test_confirmation_lag_scales_with_n(self):
        bars = peak([100, 101, 102, 109, 102, 101, 100])
        row = find_swings(bars, n=3, rth=False).iloc[0]

        swing_pos = bars.index.get_loc(row.name)
        confirm_pos = bars.index.get_loc(row["confirmed_at"])
        assert confirm_pos - swing_pos == 3

    def test_signal_fires_on_confirmation_not_on_the_swing_bar(self):
        """Acting on the swing bar itself would be trading a shape that had
        not finished forming — the chart only shows it there in hindsight."""
        signal = swing_signal(peak([100, 101, 105, 101, 100]), n=2, rth=False)
        fired = [t.strftime("%H:%M") for t in signal[signal].index]

        assert fired == ["10:30"]
        assert "10:00" not in fired


class TestScope:
    def test_session_scope_restarts_each_day(self):
        two = pd.concat(
            [
                peak([100, 101, 105, 101, 100], date="2026-03-09"),
                peak([100, 101, 107, 101, 100], date="2026-03-10"),
            ]
        )
        swings = find_swings(two, n=2, scope="session", rth=False)

        assert len(swings) == 2
        assert sorted(swings["price"]) == [105.0, 107.0]

    def test_continuous_scope_allows_confirmation_across_sessions(self):
        """A swing near a session end can only be confirmed by the next day."""
        monday = peak([100, 101, 109], date="2026-03-09")
        tuesday = peak([101, 100], date="2026-03-10")
        joined = pd.concat([monday, tuesday])

        session_scoped = find_swings(joined, n=2, scope="session", rth=False)
        continuous = find_swings(joined, n=2, scope="continuous", rth=False)

        assert session_scoped.empty
        assert len(continuous) == 1
        assert continuous.iloc[0]["price"] == 109.0

    def test_invalid_scope_is_rejected(self):
        with pytest.raises(ValueError, match="scope"):
            find_swings(peak([100, 101, 105, 101, 100]), scope="weekly", rth=False)


class TestOutput:
    def test_invalid_kind_is_rejected(self):
        with pytest.raises(ValueError, match="kind"):
            swing_signal(peak([100, 101, 105, 101, 100]), kind="top", rth=False)

    def test_kind_filter_selects_one_side(self):
        bars = peak([100, 101, 105, 101, 100])
        assert swing_signal(bars, kind="high", n=2, rth=False).any()
        assert not swing_signal(bars, kind="low", n=2, rth=False).any()

    def test_signal_is_aligned_to_the_input_index(self):
        bars = peak([100, 101, 105, 101, 100])
        assert swing_signal(bars, n=2, rth=False).index.equals(bars.index)

    def test_empty_input_returns_an_empty_frame(self):
        from ictbt.schema import empty_bars

        swings = find_swings(empty_bars())
        assert swings.empty
        assert "kind" in swings.columns
