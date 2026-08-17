"""The null models must preserve what they claim to preserve.

A shuffle that quietly changed the return distribution would make the
comparison meaningless in the flattering direction — real data would look
special because the baseline was broken, not because the pattern is real.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ictbt.nullmodel import (
    matched_touch_rate,
    session_positions,
    shuffle_session,
    shuffled_bars,
)
from ictbt.schema import validate_bars
from ictbt.signals import find_fvgs

from .conftest import make_session


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(0)


def noisy_session(date: str, rng: np.random.Generator, bars: int = 26) -> pd.DataFrame:
    """A session with genuinely varied bar shapes, for distribution checks."""
    steps = rng.normal(0, 0.002, bars)
    price = 100 * np.exp(np.cumsum(steps))
    opens = price
    closes = price * np.exp(rng.normal(0, 0.001, bars))
    highs = np.maximum(opens, closes) * (1 + np.abs(rng.normal(0, 0.001, bars)))
    lows = np.minimum(opens, closes) * (1 - np.abs(rng.normal(0, 0.001, bars)))

    index = pd.date_range(
        pd.Timestamp(f"{date} 09:30", tz="America/New_York"),
        periods=bars,
        freq="15min",
        tz="America/New_York",
        name="timestamp",
    )
    return pd.DataFrame(
        {
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": np.full(bars, 1000.0),
            "trade_count": np.full(bars, 10.0),
            "vwap": (highs + lows) / 2,
        },
        index=index,
    )


class TestShuffleValidity:
    def test_shuffled_output_is_a_valid_bar_frame(self, rng):
        shuffled = shuffle_session(noisy_session("2026-03-09", rng), rng)
        validate_bars(shuffled)

    def test_bar_count_is_preserved(self, rng):
        session = noisy_session("2026-03-09", rng)
        assert len(shuffle_session(session, rng)) == len(session)

    def test_index_is_unchanged(self, rng):
        session = noisy_session("2026-03-09", rng)
        assert shuffle_session(session, rng).index.equals(session.index)

    def test_a_single_bar_session_is_returned_untouched(self, rng):
        one = make_session("2026-03-09", minutes=15, bars=1)
        pd.testing.assert_frame_equal(shuffle_session(one, rng), one)


class TestShufflePreservesDistribution:
    """The shuffle must destroy order and nothing else."""

    def test_bar_shapes_are_preserved_as_a_multiset(self, rng):
        session = noisy_session("2026-03-09", rng)
        shuffled = shuffle_session(session, rng)

        def shapes(df):
            o = df["open"].to_numpy()
            return np.sort(
                np.round(
                    np.stack(
                        [
                            df["high"].to_numpy() / o,
                            df["low"].to_numpy() / o,
                            df["close"].to_numpy() / o,
                        ]
                    ).sum(axis=0),
                    9,
                )
            )

        np.testing.assert_allclose(shapes(session), shapes(shuffled), rtol=1e-9)

    def test_bar_return_multiset_is_preserved(self, rng):
        session = noisy_session("2026-03-09", rng)
        shuffled = shuffle_session(session, rng)

        def returns(df):
            return np.sort(
                np.round(df["close"].to_numpy() / df["open"].to_numpy(), 9)
            )

        np.testing.assert_allclose(returns(session), returns(shuffled), rtol=1e-9)

    def test_order_actually_changes(self, rng):
        session = noisy_session("2026-03-09", rng)
        shuffled = shuffle_session(session, rng)
        assert not np.allclose(
            session["close"].to_numpy(), shuffled["close"].to_numpy()
        )

    def test_volume_travels_with_its_bar(self, rng):
        session = noisy_session("2026-03-09", rng)
        session["volume"] = np.arange(len(session), dtype="float64")
        shuffled = shuffle_session(session, rng)

        assert sorted(shuffled["volume"]) == sorted(session["volume"])


class TestShuffledBars:
    def test_sessions_stay_separate(self, rng):
        two = pd.concat(
            [noisy_session("2026-03-09", rng), noisy_session("2026-03-10", rng)]
        )
        shuffled = shuffled_bars(two, rng, rth=False)

        assert len(shuffled) == len(two)
        assert shuffled.index.equals(two.index)
        validate_bars(shuffled)

    def test_shuffled_data_still_produces_detectable_gaps(self, rng):
        """Sanity check on the comparison itself.

        If shuffling produced zero gaps the baseline would be vacuous rather
        than informative, so this asserts the null model is actually capable
        of generating the pattern.
        """
        sessions = pd.concat(
            [noisy_session(f"2026-03-{d:02d}", rng) for d in (9, 10, 11, 12, 13)]
        )
        shuffled = shuffled_bars(sessions, rng, rth=False)
        assert len(find_fvgs(shuffled, rth=False)) > 0

    def test_empty_input_is_handled(self, rng):
        from ictbt.schema import empty_bars

        assert shuffled_bars(empty_bars(), rng).empty


class TestSessionPositions:
    def test_positions_restart_each_session(self):
        two = pd.concat(
            [
                make_session("2026-03-09", minutes=15, bars=3),
                make_session("2026-03-10", minutes=15, bars=3),
            ]
        )
        positions = session_positions(two, two.index)
        assert list(positions) == [0, 1, 2, 0, 1, 2]


class TestMatchedTouchRate:
    def test_returns_a_probability(self, rng):
        sessions = pd.concat(
            [noisy_session(f"2026-03-{d:02d}", rng) for d in (9, 10, 11, 12, 13)]
        )
        gaps = find_fvgs(sessions, rth=False)
        rate = matched_touch_rate(sessions, gaps, rng, rth=False)

        assert 0.0 <= rate <= 1.0

    def test_no_gaps_yields_nan(self, rng):
        from ictbt.schema import empty_bars

        sessions = noisy_session("2026-03-09", rng)
        assert np.isnan(
            matched_touch_rate(sessions, find_fvgs(empty_bars()), rng, rth=False)
        )
