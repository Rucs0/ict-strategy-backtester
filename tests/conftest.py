"""Fixture builders for hand-constructed bar data.

Bars are built in code rather than loaded from CSV so that the fixtures are
readable in the test that uses them, and so that nothing depends on file
encoding or line endings.
"""

from __future__ import annotations

import datetime as dt

import pandas as pd
import pytest

from ictbt.schema import NY, OHLCV_COLUMNS


def make_bars(
    rows: list[tuple[str, float, float, float, float]] | None = None,
    *,
    date: str = "2026-03-09",
    tz=NY,
) -> pd.DataFrame:
    """Build a canonical bar frame from ``(time, open, high, low, close)`` rows.

    Times are New York wall clock, e.g. ``("09:30", 100, 101, 99.5, 100.5)``.
    """
    rows = rows or [("09:30", 100.0, 101.0, 99.0, 100.5)]
    index = pd.DatetimeIndex(
        [pd.Timestamp(f"{date} {t}", tz=tz) for t, *_ in rows], name="timestamp"
    )
    data = {
        "open": [r[1] for r in rows],
        "high": [r[2] for r in rows],
        "low": [r[3] for r in rows],
        "close": [r[4] for r in rows],
        "volume": [1000.0] * len(rows),
        "trade_count": [10.0] * len(rows),
        "vwap": [(r[2] + r[3]) / 2 for r in rows],
    }
    return pd.DataFrame(data, index=index)[list(OHLCV_COLUMNS)]


def make_session(
    date: str,
    *,
    minutes: int = 15,
    bars: int | None = None,
    start_time: str = "09:30",
    price: float = 100.0,
) -> pd.DataFrame:
    """Build a contiguous run of bars for one session.

    `bars=None` means a full regular-hours session at the given timeframe.
    """
    if bars is None:
        bars = 390 // minutes
    first = pd.Timestamp(f"{date} {start_time}", tz=NY)
    index = pd.date_range(
        first, periods=bars, freq=f"{minutes}min", tz=NY, name="timestamp"
    )
    opens = [price + i * 0.1 for i in range(bars)]
    return pd.DataFrame(
        {
            "open": opens,
            "high": [o + 0.5 for o in opens],
            "low": [o - 0.5 for o in opens],
            "close": [o + 0.2 for o in opens],
            "volume": [1000.0] * bars,
            "trade_count": [10.0] * bars,
            "vwap": [o for o in opens],
        },
        index=index,
    )[list(OHLCV_COLUMNS)]


@pytest.fixture
def full_session() -> pd.DataFrame:
    """A complete 15-minute regular-hours session: 26 bars, 09:30 to 15:45."""
    return make_session("2026-03-09", minutes=15)


@pytest.fixture
def tmp_cache_root(tmp_path) -> "pathlib.Path":  # noqa: F821
    return tmp_path / "bars"
