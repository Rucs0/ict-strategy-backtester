"""Market structure events.

The state machine matters as much as the break detection here, so the tests
pin the BOS/MSS classification and its dependence on what came before, not
just whether a level was broken.
"""

from __future__ import annotations

import pandas as pd
import pytest

from ictbt.signals.structure import find_structure_events, mss_signal
from ictbt.signals.sweeps import find_sweeps

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


# Swing high 110 at 10:00, confirmed 10:30. Lows held flat so only highs swing.
SWING_HIGH = [
    (100.0, 102.0, 95.0, 101.0),   # 09:30
    (101.0, 104.0, 95.0, 103.0),   # 09:45
    (103.0, 110.0, 95.0, 104.0),   # 10:00  <- swing high 110
    (104.0, 106.0, 95.0, 105.0),   # 10:15
    (105.0, 107.0, 95.0, 106.0),   # 10:30  <- confirmation
]


class TestBreakDetection:
    def test_a_close_above_a_swing_high_is_a_break(self):
        bars = frame(SWING_HIGH + [(106.0, 112.0, 95.0, 111.0)])
        events = find_structure_events(bars, n=2, rth=False)

        assert len(events) == 1
        event = events.iloc[0]
        assert event["direction"] == "bullish"
        assert event["level"] == 110.0
        assert events.index[0].strftime("%H:%M") == "10:45"

    def test_a_wick_through_without_a_close_beyond_is_not_a_break(self):
        """That is a sweep, and it belongs to the other module."""
        bars = frame(SWING_HIGH + [(106.0, 112.0, 95.0, 108.0)])
        assert find_structure_events(bars, n=2, rth=False).empty

    def test_break_and_sweep_are_mutually_exclusive(self):
        """Every penetration is exactly one of the two, never both."""
        swept = frame(SWING_HIGH + [(106.0, 112.0, 95.0, 108.0)])
        broken = frame(SWING_HIGH + [(106.0, 112.0, 95.0, 111.0)])

        assert len(find_sweeps(swept, n=2, rth=False)) == 1
        assert find_structure_events(swept, n=2, rth=False).empty

        assert find_sweeps(broken, n=2, rth=False).empty
        assert len(find_structure_events(broken, n=2, rth=False)) == 1

    def test_a_close_exactly_at_the_level_is_not_a_break(self):
        bars = frame(SWING_HIGH + [(106.0, 112.0, 95.0, 110.0)])
        assert find_structure_events(bars, n=2, rth=False).empty

    def test_an_unconfirmed_swing_cannot_be_broken(self):
        """At n=3 the swing is not yet confirmed when the break bar arrives."""
        bars = frame(SWING_HIGH + [(106.0, 112.0, 95.0, 111.0)])
        assert len(find_structure_events(bars, n=2, rth=False)) == 1
        assert find_structure_events(bars, n=3, rth=False).empty


class TestClassification:
    def test_the_first_break_of_a_session_is_a_bos_with_no_prior_state(self):
        """Where the state machine starts, not evidence of anything."""
        bars = frame(SWING_HIGH + [(106.0, 112.0, 95.0, 111.0)])
        event = find_structure_events(bars, n=2, rth=False).iloc[0]

        assert event["event"] == "bos"
        assert event["prior_structure"] is None

    def test_a_break_against_the_prevailing_direction_is_an_mss(self):
        """Bullish break sets structure bullish; a later bearish break is
        then a shift rather than a continuation."""
        bars = frame(
            [
                (100.0, 102.0, 98.0, 101.0),   # 09:30
                (101.0, 104.0, 97.0, 103.0),   # 09:45
                (103.0, 110.0, 96.0, 104.0),   # 10:00  swing high 110
                (104.0, 106.0, 99.0, 105.0),   # 10:15
                (105.0, 107.0, 100.0, 106.0),  # 10:30  confirms high 110
                (106.0, 112.0, 106.0, 111.0),  # 10:45  bullish break -> bos
                (111.0, 113.0, 107.0, 112.0),  # 11:00
                (112.0, 114.0, 105.0, 113.0),  # 11:15  swing low 105
                (113.0, 115.0, 108.0, 114.0),  # 11:30
                (114.0, 116.0, 109.0, 115.0),  # 11:45  confirms low 105
                (101.0, 102.0, 95.0, 100.0),   # 12:00  bearish break -> mss
            ]
        )
        events = find_structure_events(bars, n=2, rth=False)

        kinds = list(zip(events["direction"], events["event"]))
        assert ("bullish", "bos") in kinds
        assert ("bearish", "mss") in kinds

        shift = events[events["event"] == "mss"].iloc[0]
        assert shift["prior_structure"] == "bullish"
        assert shift["direction"] == "bearish"

    def test_the_mirror_case_a_bullish_mss_after_bearish_structure(self):
        """Both directions need pinning separately.

        A mutation that classified every bullish break as a BOS passed the
        whole suite, because every MSS test here ran in the bearish
        direction.
        """
        bars = frame(
            [
                (106.0, 110.0, 105.0, 107.0),  # 09:30
                (107.0, 109.0, 103.0, 105.0),  # 09:45
                (105.0, 108.0, 100.0, 104.0),  # 10:00  swing low 100
                (104.0, 107.0, 102.0, 105.0),  # 10:15
                (105.0, 106.0, 104.0, 105.0),  # 10:30  confirms low 100
                (104.0, 105.0, 95.0, 96.0),    # 10:45  bearish break -> bos
                (96.0, 106.0, 96.0, 105.0),    # 11:00
                (105.0, 112.0, 97.0, 106.0),   # 11:15  swing high 112
                (106.0, 108.0, 98.0, 107.0),   # 11:30
                (107.0, 109.0, 99.0, 108.0),   # 11:45  confirms high 112
                (113.0, 118.0, 100.0, 117.0),  # 12:00  bullish break -> mss
            ]
        )
        events = find_structure_events(bars, n=2, rth=False)

        kinds = list(zip(events["direction"], events["event"]))
        assert ("bearish", "bos") in kinds
        assert ("bullish", "mss") in kinds

        shift = events[events["event"] == "mss"].iloc[0]
        assert shift["direction"] == "bullish"
        assert shift["prior_structure"] == "bearish"
        assert shift["level"] == 112.0

    def test_a_second_break_in_the_same_direction_is_a_bos(self):
        bars = frame(
            [
                (100.0, 102.0, 95.0, 101.0),
                (101.0, 104.0, 95.0, 103.0),
                (103.0, 110.0, 95.0, 104.0),   # swing high 110
                (104.0, 106.0, 95.0, 105.0),
                (105.0, 107.0, 95.0, 106.0),   # confirms
                (106.0, 112.0, 95.0, 111.0),   # bullish break -> bos
                (111.0, 113.0, 95.0, 112.0),
                (112.0, 118.0, 95.0, 113.0),   # swing high 118
                (113.0, 115.0, 95.0, 114.0),
                (114.0, 116.0, 95.0, 115.0),   # confirms 118
                (115.0, 120.0, 95.0, 119.0),   # bullish break -> bos again
            ]
        )
        events = find_structure_events(bars, n=2, rth=False)
        bullish = events[events["direction"] == "bullish"]

        assert len(bullish) == 2
        assert list(bullish["event"]) == ["bos", "bos"]


class TestLevelRetirement:
    def test_a_broken_level_does_not_break_again(self):
        bars = frame(
            SWING_HIGH
            + [
                (106.0, 112.0, 95.0, 111.0),  # break
                (111.0, 114.0, 95.0, 113.0),  # would break 110 again
            ]
        )
        assert len(find_structure_events(bars, n=2, rth=False)) == 1

    def test_a_close_past_several_levels_retires_all_of_them(self):
        """Price accepted above a level means that level has stopped being
        one, regardless of how recently it formed."""
        bars = frame(
            [
                (100.0, 102.0, 95.0, 101.0),
                (101.0, 104.0, 95.0, 103.0),
                (103.0, 118.0, 95.0, 104.0),   # swing high 118
                (104.0, 106.0, 95.0, 105.0),
                (105.0, 107.0, 95.0, 106.0),   # confirms 118
                (106.0, 112.0, 95.0, 107.0),   # swing high 112
                (107.0, 109.0, 95.0, 108.0),
                (108.0, 110.0, 95.0, 109.0),   # confirms 112
                (109.0, 122.0, 95.0, 121.0),   # closes above both
                (121.0, 123.0, 95.0, 122.0),
            ]
        )
        events = find_structure_events(bars, n=2, rth=False)

        # One event is emitted, for the nearest level, and both are retired.
        assert len(events) == 1
        assert events.iloc[0]["level"] == 112.0

    def test_events_do_not_cross_session_boundaries(self):
        monday = frame(SWING_HIGH)
        tuesday = make_bars(
            [("09:30", 106.0, 112.0, 95.0, 111.0)], date="2026-03-10"
        )
        assert find_structure_events(
            pd.concat([monday, tuesday]), n=2, rth=False
        ).empty

    def test_structure_state_resets_each_session(self):
        """A conservative default for a day-trading test."""
        day = [
            (100.0, 102.0, 95.0, 101.0),
            (101.0, 104.0, 95.0, 103.0),
            (103.0, 110.0, 95.0, 104.0),
            (104.0, 106.0, 95.0, 105.0),
            (105.0, 107.0, 95.0, 106.0),
            (106.0, 112.0, 95.0, 111.0),
        ]
        two = pd.concat(
            [
                make_bars(
                    [(t, *r) for t, r in zip(TIMES, day)], date="2026-03-09"
                ),
                make_bars(
                    [(t, *r) for t, r in zip(TIMES, day)], date="2026-03-10"
                ),
            ]
        )
        events = find_structure_events(two, n=2, rth=False)

        assert len(events) == 2
        assert list(events["event"]) == ["bos", "bos"]
        assert list(events["prior_structure"]) == [None, None]


class TestOutput:
    def test_mss_signal_excludes_bos(self):
        bars = frame(SWING_HIGH + [(106.0, 112.0, 95.0, 111.0)])

        assert not find_structure_events(bars, n=2, rth=False).empty
        assert not mss_signal(bars, n=2, rth=False).any()

    def test_invalid_direction_is_rejected(self):
        with pytest.raises(ValueError, match="direction"):
            mss_signal(frame(SWING_HIGH), direction="up", rth=False)

    def test_signal_is_aligned_to_the_input_index(self):
        bars = frame(SWING_HIGH + [(106.0, 112.0, 95.0, 111.0)])
        assert mss_signal(bars, n=2, rth=False).index.equals(bars.index)

    def test_the_broken_swing_is_identified(self):
        bars = frame(SWING_HIGH + [(106.0, 112.0, 95.0, 111.0)])
        event = find_structure_events(bars, n=2, rth=False).iloc[0]

        assert event["swing_at"].strftime("%H:%M") == "10:00"
        assert event["bars_since_swing"] == 3

    def test_empty_input_returns_an_empty_frame(self):
        from ictbt.schema import empty_bars

        events = find_structure_events(empty_bars())
        assert events.empty
        assert "event" in events.columns
