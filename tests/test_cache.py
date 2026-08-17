"""Cache round-trips, coverage bookkeeping, and provenance guards.

No network. `normalize_barset` is tested against a frame shaped like
alpaca-py's response rather than against a live call.
"""

from __future__ import annotations

import pandas as pd
import pytest

from ictbt.cache import Coverage, ParquetCache
from ictbt.fetch import normalize_barset
from ictbt.schema import NY, validate_bars

from .conftest import make_session


def ny(ts: str) -> pd.Timestamp:
    return pd.Timestamp(ts, tz=NY)


@pytest.fixture
def cache(tmp_cache_root) -> ParquetCache:
    return ParquetCache(tmp_cache_root)


class TestRoundTrip:
    def test_stored_bars_come_back_unchanged(self, cache):
        session = make_session("2026-03-09", minutes=15)
        cache.store(
            session,
            symbol="SPY",
            timeframe="15m",
            feed="sip",
            adjustment="all",
            requested=Coverage(ny("2026-03-09"), ny("2026-03-10")),
        )

        loaded = cache.load("SPY", "15m")

        assert len(loaded) == len(session)
        pd.testing.assert_frame_equal(loaded, session, check_freq=False)

    def test_timezone_survives_the_parquet_round_trip(self, cache):
        """Parquet stores UTC internally; the loader must restore New York."""
        session = make_session("2026-03-09", minutes=15)
        cache.store(
            session,
            symbol="SPY",
            timeframe="15m",
            feed="sip",
            adjustment="all",
            requested=Coverage(ny("2026-03-09"), ny("2026-03-10")),
        )

        loaded = cache.load("SPY", "15m")

        assert str(loaded.index.tz) == "America/New_York"
        assert loaded.index[0].strftime("%H:%M") == "09:30"
        validate_bars(loaded)

    def test_loading_an_absent_series_returns_empty_not_an_error(self, cache):
        loaded = cache.load("NOPE", "15m")
        assert loaded.empty
        validate_bars(loaded)

    def test_merging_two_sessions_keeps_both(self, cache):
        for date in ("2026-03-09", "2026-03-10"):
            cache.store(
                make_session(date, minutes=15),
                symbol="SPY",
                timeframe="15m",
                feed="sip",
                adjustment="all",
                requested=Coverage(ny(date), ny(f"{date} 23:59")),
            )

        loaded = cache.load("SPY", "15m")
        assert len(loaded) == 52
        assert loaded.index.is_monotonic_increasing


class TestCoverage:
    def test_an_empty_cache_is_entirely_missing(self, cache):
        gaps = cache.missing_ranges("SPY", "15m", ny("2026-03-01"), ny("2026-03-31"))
        assert len(gaps) == 1
        assert gaps[0].start == ny("2026-03-01")

    def test_a_fully_covered_range_reports_no_gaps(self, cache):
        cache.store(
            make_session("2026-03-09", minutes=15),
            symbol="SPY",
            timeframe="15m",
            feed="sip",
            adjustment="all",
            requested=Coverage(ny("2026-03-01"), ny("2026-03-31")),
        )
        gaps = cache.missing_ranges("SPY", "15m", ny("2026-03-05"), ny("2026-03-15"))
        assert gaps == []

    def test_extending_forward_only_requests_the_new_tail(self, cache):
        """Re-running a backtest with a later end date must not re-download."""
        cache.store(
            make_session("2026-03-09", minutes=15),
            symbol="SPY",
            timeframe="15m",
            feed="sip",
            adjustment="all",
            requested=Coverage(ny("2026-03-01"), ny("2026-03-15")),
        )

        gaps = cache.missing_ranges("SPY", "15m", ny("2026-03-01"), ny("2026-03-31"))

        assert len(gaps) == 1
        assert gaps[0].start == ny("2026-03-15")
        assert gaps[0].end == ny("2026-03-31")

    def test_a_covered_hole_in_the_middle_is_not_re_requested(self, cache):
        """An empty stretch that was already asked about stays empty.

        This is the case a naive min/max-of-index cache gets wrong: a holiday
        week has no bars, so index bounds suggest it was never fetched.
        """
        session = make_session("2026-03-09", minutes=15)
        for block in (
            Coverage(ny("2026-03-01"), ny("2026-03-10")),
            Coverage(ny("2026-03-10"), ny("2026-03-20")),
        ):
            cache.store(
                session,
                symbol="SPY",
                timeframe="15m",
                feed="sip",
                adjustment="all",
                requested=block,
            )

        gaps = cache.missing_ranges("SPY", "15m", ny("2026-03-02"), ny("2026-03-19"))
        assert gaps == []

    def test_naive_coverage_bounds_are_rejected(self, cache):
        with pytest.raises(ValueError, match="timezone-aware"):
            cache.store(
                make_session("2026-03-09", minutes=15),
                symbol="SPY",
                timeframe="15m",
                feed="sip",
                adjustment="all",
                requested=Coverage(
                    pd.Timestamp("2026-03-09"), pd.Timestamp("2026-03-10")
                ),
            )


class TestProvenanceGuards:
    """Feed and adjustment changes must not be silently merged."""

    def _seed(self, cache, *, feed="sip", adjustment="all"):
        cache.store(
            make_session("2026-03-09", minutes=15),
            symbol="SPY",
            timeframe="15m",
            feed=feed,
            adjustment=adjustment,
            requested=Coverage(ny("2026-03-09"), ny("2026-03-10")),
        )

    def test_mixing_feeds_is_refused(self, cache):
        """IEX and SIP bars for the same minute are different bars."""
        self._seed(cache, feed="sip")
        with pytest.raises(ValueError, match="cached from feed"):
            self._seed(cache, feed="iex")

    def test_mixing_adjustments_is_refused(self, cache):
        """Raw and split-adjusted prices differ by a factor across a split."""
        self._seed(cache, adjustment="all")
        with pytest.raises(ValueError, match="cached with adjustment"):
            self._seed(cache, adjustment="raw")

    def test_provenance_is_recorded(self, cache):
        self._seed(cache)
        meta = cache.load_meta("SPY", "15m")
        assert meta.feed == "sip"
        assert meta.adjustment == "all"


class TestNormalizeBarset:
    """alpaca-py returns a (symbol, timestamp) MultiIndex in UTC."""

    def _alpaca_shaped(self) -> pd.DataFrame:
        index = pd.MultiIndex.from_tuples(
            [
                ("SPY", pd.Timestamp("2026-03-09 13:30", tz="UTC")),
                ("SPY", pd.Timestamp("2026-03-09 13:45", tz="UTC")),
            ],
            names=["symbol", "timestamp"],
        )
        return pd.DataFrame(
            {
                "open": [100.0, 100.5],
                "high": [101.0, 101.5],
                "low": [99.5, 100.0],
                "close": [100.5, 101.0],
                "volume": [1000.0, 1200.0],
                "trade_count": [10.0, 12.0],
                "vwap": [100.2, 100.8],
            },
            index=index,
        )

    def test_multiindex_is_flattened_and_converted_to_new_york(self):
        out = normalize_barset(self._alpaca_shaped(), "SPY")

        validate_bars(out)
        assert [t.strftime("%H:%M") for t in out.index] == ["09:30", "09:45"]
        assert str(out.index.tz) == "America/New_York"

    def test_an_empty_response_becomes_an_empty_canonical_frame(self):
        out = normalize_barset(pd.DataFrame(), "SPY")
        assert out.empty
        validate_bars(out)

    def test_a_response_for_another_symbol_yields_nothing(self):
        out = normalize_barset(self._alpaca_shaped(), "QQQ")
        assert out.empty
