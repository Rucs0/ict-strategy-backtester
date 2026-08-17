"""The bar contract: what a valid frame is, and what gets rejected."""

from __future__ import annotations

import pandas as pd
import pytest

from ictbt.schema import (
    OHLCV_COLUMNS,
    SchemaError,
    empty_bars,
    to_canonical,
    validate_bars,
)

from .conftest import make_bars


class TestValidBars:
    def test_a_hand_built_frame_validates(self):
        validate_bars(make_bars())

    def test_empty_bars_satisfies_the_contract(self):
        validate_bars(empty_bars())


class TestStructuralRejections:
    def test_unsorted_index_is_rejected(self):
        bars = make_bars(
            [
                ("09:45", 100.0, 101.0, 99.0, 100.0),
                ("09:30", 100.0, 101.0, 99.0, 100.0),
            ]
        )
        with pytest.raises(SchemaError, match="sorted"):
            validate_bars(bars)

    def test_duplicate_timestamps_are_rejected(self):
        bars = make_bars(
            [
                ("09:30", 100.0, 101.0, 99.0, 100.0),
                ("09:30", 100.0, 101.0, 99.0, 100.0),
            ]
        )
        with pytest.raises(SchemaError, match="duplicate"):
            validate_bars(bars)

    def test_missing_column_is_rejected(self):
        bars = make_bars().drop(columns=["vwap"])
        with pytest.raises(SchemaError, match="missing required columns"):
            validate_bars(bars)

    def test_unnamed_index_is_rejected(self):
        bars = make_bars()
        bars.index.name = None
        with pytest.raises(SchemaError, match="named 'timestamp'"):
            validate_bars(bars)

    def test_null_price_is_rejected(self):
        bars = make_bars()
        bars.loc[bars.index[0], "close"] = float("nan")
        with pytest.raises(SchemaError, match="null values"):
            validate_bars(bars)


class TestOhlcConsistency:
    """These catch vendor bad ticks, and more often our own join bugs."""

    def test_high_below_low_is_rejected(self):
        bars = make_bars([("09:30", 100.0, 98.0, 99.0, 100.0)])
        with pytest.raises(SchemaError, match="high < low"):
            validate_bars(bars)

    def test_high_below_the_body_is_rejected(self):
        bars = make_bars([("09:30", 100.0, 100.5, 99.0, 101.0)])
        with pytest.raises(SchemaError, match=r"high < max\(open, close\)"):
            validate_bars(bars)

    def test_low_above_the_body_is_rejected(self):
        bars = make_bars([("09:30", 100.0, 101.0, 100.5, 100.2)])
        with pytest.raises(SchemaError, match=r"low > min\(open, close\)"):
            validate_bars(bars)

    def test_negative_volume_is_rejected(self):
        bars = make_bars()
        bars.loc[bars.index[0], "volume"] = -1.0
        with pytest.raises(SchemaError, match="negative volume"):
            validate_bars(bars)

    def test_a_legitimate_flat_bar_is_accepted(self):
        """open == high == low == close is degenerate but real."""
        validate_bars(make_bars([("09:30", 100.0, 100.0, 100.0, 100.0)]))


class TestToCanonical:
    def test_columns_are_reordered(self):
        bars = make_bars()[["vwap", "close", "open", "high", "low", "volume", "trade_count"]]
        assert tuple(to_canonical(bars).columns) == OHLCV_COLUMNS

    def test_duplicates_resolve_to_the_later_record(self):
        """A restated bar supersedes the original."""
        bars = make_bars(
            [
                ("09:30", 100.0, 101.0, 99.0, 100.0),
                ("09:30", 100.0, 101.0, 99.0, 100.9),
            ]
        )
        out = to_canonical(bars)
        assert len(out) == 1
        assert out["close"].iloc[0] == 100.9

    def test_rows_are_sorted(self):
        bars = make_bars(
            [
                ("09:45", 100.0, 101.0, 99.0, 100.0),
                ("09:30", 100.0, 101.0, 99.0, 100.0),
            ]
        )
        out = to_canonical(bars)
        assert out.index.is_monotonic_increasing
        validate_bars(out)
