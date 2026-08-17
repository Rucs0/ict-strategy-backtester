"""Fetching bars from Alpaca, with the cache in front of it.

Free-tier constraints that shape this module:

- **SIP is available historically, but not recently.** The consolidated tape
  is queryable on the free plan only for data at least 15 minutes old.
  Requests whose `end` is inside that window fail. `SIP_DELAY` clamps `end`
  back with a safety margin rather than letting the call error out.
- **History starts in 2016.** Earlier dates return nothing rather than an
  error, which is the kind of silence that looks like a bad symbol.
- **200 requests/minute.** Generous, but a multi-year minute-bar backfill is
  paginated internally by alpaca-py, so a wide date range is one call here
  and many over the wire.

Nothing in this module trades. It is read-only market data.
"""

from __future__ import annotations

import datetime as dt
import os
from pathlib import Path

import pandas as pd
from alpaca.data.enums import Adjustment, DataFeed
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

from .cache import Coverage, ParquetCache
from .schema import NY, empty_bars, to_canonical, validate_bars

#: Free-tier SIP embargo, plus a minute of slack for clock skew.
SIP_DELAY = dt.timedelta(minutes=16)

#: Alpaca equity history begins here. Asking earlier returns empty, not an error.
HISTORY_START = pd.Timestamp("2016-01-01", tz=NY)

#: Split- and dividend-adjusted. Raw prices contain discontinuities at
#: corporate actions that every gap-detection primitive would read as a
#: genuine fair value gap.
DEFAULT_ADJUSTMENT = Adjustment.ALL

#: Consolidated tape. IEX alone is a few percent of volume, which distorts
#: both the volume series and the extremes that liquidity sweeps depend on.
DEFAULT_FEED = DataFeed.SIP


class FetchError(RuntimeError):
    """A fetch could not be performed as requested."""


def timeframe_from_minutes(minutes: int) -> TimeFrame:
    """Map a minute count onto Alpaca's TimeFrame."""
    if minutes <= 0:
        raise ValueError(f"minutes must be positive, got {minutes}")
    if minutes % 60 == 0 and minutes < 1440:
        return TimeFrame(minutes // 60, TimeFrameUnit.Hour)
    # Member is `Minute`; its *value* is the wire string "Min".
    return TimeFrame(minutes, TimeFrameUnit.Minute)


def timeframe_label(minutes: int) -> str:
    """Cache filename for a timeframe, e.g. 15 -> '15m'."""
    return f"{minutes}m"


class AlpacaBarSource:
    """Cache-first reader for Alpaca stock bars."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        secret_key: str | None = None,
        cache: ParquetCache | None = None,
        feed: DataFeed = DEFAULT_FEED,
        adjustment: Adjustment = DEFAULT_ADJUSTMENT,
    ) -> None:
        self._api_key = api_key or os.environ.get("ALPACA_API_KEY")
        self._secret_key = secret_key or os.environ.get("ALPACA_SECRET_KEY")
        self.cache = cache or ParquetCache()
        self.feed = feed
        self.adjustment = adjustment
        self._client: StockHistoricalDataClient | None = None

    # -- client --------------------------------------------------------

    @property
    def client(self) -> StockHistoricalDataClient:
        """Built lazily so cache-only reads work with no credentials."""
        if self._client is None:
            if not self._api_key or not self._secret_key:
                raise FetchError(
                    "ALPACA_API_KEY / ALPACA_SECRET_KEY are not set. Copy "
                    ".env.example to .env and fill them in, or call with "
                    "allow_fetch=False to read only what is already cached."
                )
            self._client = StockHistoricalDataClient(
                api_key=self._api_key, secret_key=self._secret_key
            )
        return self._client

    # -- public API ----------------------------------------------------

    def get_bars(
        self,
        symbol: str,
        *,
        minutes: int,
        start: pd.Timestamp | str,
        end: pd.Timestamp | str,
        allow_fetch: bool = True,
    ) -> pd.DataFrame:
        """Bars for `symbol` over [start, end], fetching only what is missing.

        Returns a validated canonical frame sliced to the requested window.
        """
        start = _to_ny(start)
        end = _to_ny(end)
        if end <= start:
            raise ValueError(f"end {end} must be after start {start}")

        label = timeframe_label(minutes)
        start = max(start, HISTORY_START)
        end = _clamp_to_sip_embargo(end)
        if end <= start:
            raise FetchError(
                f"requested window collapses to nothing after clamping: "
                f"Alpaca history starts {HISTORY_START.date()} and free-tier "
                f"SIP data is embargoed for {SIP_DELAY}."
            )

        if allow_fetch:
            for gap in self.cache.missing_ranges(symbol, label, start, end):
                fetched = self._fetch(symbol, minutes, gap.start, gap.end)
                self.cache.store(
                    fetched,
                    symbol=symbol,
                    timeframe=label,
                    feed=self.feed.value,
                    adjustment=self.adjustment.value,
                    requested=gap,
                )

        cached = self.cache.load(symbol, label)
        window = cached.loc[(cached.index >= start) & (cached.index <= end)]
        validate_bars(window, name=f"{symbol}/{label}")
        return window

    # -- internals -----------------------------------------------------

    def _fetch(
        self, symbol: str, minutes: int, start: pd.Timestamp, end: pd.Timestamp
    ) -> pd.DataFrame:
        request = StockBarsRequest(
            symbol_or_symbols=symbol.upper(),
            timeframe=timeframe_from_minutes(minutes),
            start=start.to_pydatetime(),
            end=end.to_pydatetime(),
            feed=self.feed,
            adjustment=self.adjustment,
        )
        barset = self.client.get_stock_bars(request)
        return normalize_barset(barset.df, symbol)


def normalize_barset(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Turn alpaca-py's `BarSet.df` into a canonical bar frame.

    alpaca-py returns a (symbol, timestamp) MultiIndex even for a single
    symbol, with UTC timestamps. Split out as a free function so it can be
    tested against a captured response without any network access.
    """
    if df is None or df.empty:
        return empty_bars()

    out = df
    if isinstance(out.index, pd.MultiIndex):
        names = list(out.index.names)
        if "symbol" in names:
            key = symbol.upper()
            if key not in out.index.get_level_values("symbol"):
                return empty_bars()
            out = out.xs(key, level="symbol")
        else:
            out = out.droplevel(0)

    out = out.rename(columns={"trade_count": "trade_count", "vwap": "vwap"})
    out.index = pd.DatetimeIndex(out.index)
    if out.index.tz is None:
        # alpaca-py returns UTC; localize explicitly rather than assume local.
        out.index = out.index.tz_localize("UTC")

    canonical = to_canonical(out)
    validate_bars(canonical, name=f"fetched:{symbol}")
    return canonical


def load_credentials(env_path: Path | str = ".env") -> None:
    """Load ALPACA_* keys from a .env file into the environment."""
    from dotenv import load_dotenv

    load_dotenv(env_path)


def _clamp_to_sip_embargo(end: pd.Timestamp) -> pd.Timestamp:
    """Pull `end` back out of the free-tier SIP embargo window if needed."""
    latest = pd.Timestamp.now(tz=NY) - SIP_DELAY
    return min(end, latest)


def _to_ny(ts: pd.Timestamp | str) -> pd.Timestamp:
    out = pd.Timestamp(ts)
    if out.tz is None:
        out = out.tz_localize(NY)
    return out.tz_convert(NY)
