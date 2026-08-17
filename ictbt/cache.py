"""Local Parquet cache for bar data.

Fetching is rate-limited and slow; a backtest sweep re-reads the same bars
hundreds of times. Everything lands on disk once and is read from there after.

The cache tracks *what was requested*, not just what came back, in a sidecar
JSON file. This matters because an empty result is ambiguous: no bars between
two dates could mean the market was closed, or it could mean we never asked.
Without a coverage record the cache cannot tell the difference and will either
re-fetch forever or silently serve a hole as if it were real.

Layout::

    data/bars/{symbol}/{timeframe}.parquet   bars
    data/bars/{symbol}/{timeframe}.json      coverage + provenance
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from .schema import NY, to_canonical, validate_bars

DEFAULT_CACHE_ROOT = Path("data") / "bars"


@dataclass(frozen=True)
class Coverage:
    """One closed interval of time we have actually asked the vendor about."""

    start: pd.Timestamp
    end: pd.Timestamp

    def overlaps_or_touches(self, other: Coverage) -> bool:
        return self.start <= other.end and other.start <= self.end


@dataclass
class CacheMeta:
    """Provenance for a cached series.

    `feed` and `adjustment` are recorded because they silently change the
    numbers. SIP and IEX bars for the same minute are different bars; split-
    adjusted and raw prices differ by a factor across a split date. Mixing
    them inside one cached series would produce a discontinuity that looks
    exactly like a real price move.
    """

    symbol: str
    timeframe: str
    feed: str
    adjustment: str
    covered: list[Coverage]

    def to_json(self) -> str:
        return json.dumps(
            {
                "symbol": self.symbol,
                "timeframe": self.timeframe,
                "feed": self.feed,
                "adjustment": self.adjustment,
                "covered": [
                    {"start": c.start.isoformat(), "end": c.end.isoformat()}
                    for c in self.covered
                ],
            },
            indent=2,
        )

    @classmethod
    def from_json(cls, text: str) -> CacheMeta:
        raw = json.loads(text)
        return cls(
            symbol=raw["symbol"],
            timeframe=raw["timeframe"],
            feed=raw["feed"],
            adjustment=raw["adjustment"],
            covered=[
                Coverage(
                    pd.Timestamp(c["start"]).tz_convert(NY),
                    pd.Timestamp(c["end"]).tz_convert(NY),
                )
                for c in raw["covered"]
            ],
        )


class ParquetCache:
    """Read/write bars on local disk, keyed by symbol and timeframe."""

    def __init__(self, root: Path | str = DEFAULT_CACHE_ROOT) -> None:
        self.root = Path(root)

    # -- paths ---------------------------------------------------------

    def bars_path(self, symbol: str, timeframe: str) -> Path:
        return self.root / symbol.upper() / f"{timeframe}.parquet"

    def meta_path(self, symbol: str, timeframe: str) -> Path:
        return self.root / symbol.upper() / f"{timeframe}.json"

    # -- read ----------------------------------------------------------

    def load(self, symbol: str, timeframe: str) -> pd.DataFrame:
        """Everything cached for this series, or an empty canonical frame."""
        path = self.bars_path(symbol, timeframe)
        if not path.exists():
            from .schema import empty_bars

            return empty_bars()
        df = pd.read_parquet(path)
        df = to_canonical(df)
        validate_bars(df, name=f"cache:{symbol}/{timeframe}")
        return df

    def load_meta(self, symbol: str, timeframe: str) -> CacheMeta | None:
        path = self.meta_path(symbol, timeframe)
        if not path.exists():
            return None
        return CacheMeta.from_json(path.read_text(encoding="utf-8"))

    def missing_ranges(
        self,
        symbol: str,
        timeframe: str,
        start: pd.Timestamp,
        end: pd.Timestamp,
    ) -> list[Coverage]:
        """Which parts of [start, end] have never been requested.

        Returning the gaps rather than a yes/no lets a caller extend a cached
        series at either end without re-downloading the middle.
        """
        meta = self.load_meta(symbol, timeframe)
        wanted = Coverage(_as_ny(start), _as_ny(end))
        if meta is None or not meta.covered:
            return [wanted]

        gaps: list[Coverage] = []
        cursor = wanted.start
        for block in sorted(meta.covered, key=lambda c: c.start):
            if block.end < cursor:
                continue
            if block.start > wanted.end:
                break
            if block.start > cursor:
                gaps.append(Coverage(cursor, min(block.start, wanted.end)))
            cursor = max(cursor, block.end)
            if cursor >= wanted.end:
                return gaps
        if cursor < wanted.end:
            gaps.append(Coverage(cursor, wanted.end))
        return gaps

    # -- write ---------------------------------------------------------

    def store(
        self,
        df: pd.DataFrame,
        *,
        symbol: str,
        timeframe: str,
        feed: str,
        adjustment: str,
        requested: Coverage,
    ) -> pd.DataFrame:
        """Merge `df` into the cache and return the full merged series.

        Refuses to mix feeds or adjustments within one series — see
        `CacheMeta`. Change either and the cached file must be rebuilt.
        """
        existing_meta = self.load_meta(symbol, timeframe)
        if existing_meta is not None:
            if existing_meta.feed != feed:
                raise ValueError(
                    f"{symbol}/{timeframe} is cached from feed {existing_meta.feed!r}; "
                    f"refusing to merge {feed!r}. Delete the cached file to rebuild."
                )
            if existing_meta.adjustment != adjustment:
                raise ValueError(
                    f"{symbol}/{timeframe} is cached with adjustment "
                    f"{existing_meta.adjustment!r}; refusing to merge {adjustment!r}. "
                    "Delete the cached file to rebuild."
                )

        incoming = to_canonical(df)
        validate_bars(incoming, name=f"incoming:{symbol}/{timeframe}")

        current = self.load(symbol, timeframe)
        merged = pd.concat([current, incoming]) if len(current) else incoming
        merged = to_canonical(merged)  # de-dupes, keeping the newer record
        validate_bars(merged, name=f"merged:{symbol}/{timeframe}")

        covered = list(existing_meta.covered) if existing_meta else []
        covered.append(Coverage(_as_ny(requested.start), _as_ny(requested.end)))
        meta = CacheMeta(
            symbol=symbol.upper(),
            timeframe=timeframe,
            feed=feed,
            adjustment=adjustment,
            covered=_merge_coverage(covered),
        )

        path = self.bars_path(symbol, timeframe)
        path.parent.mkdir(parents=True, exist_ok=True)
        merged.to_parquet(path, engine="pyarrow", index=True)
        self.meta_path(symbol, timeframe).write_text(meta.to_json(), encoding="utf-8")
        return merged


def _merge_coverage(blocks: list[Coverage]) -> list[Coverage]:
    """Collapse overlapping or adjacent intervals into a minimal set."""
    if not blocks:
        return []
    ordered = sorted(blocks, key=lambda c: c.start)
    out = [ordered[0]]
    for block in ordered[1:]:
        last = out[-1]
        if block.start <= last.end:
            out[-1] = Coverage(last.start, max(last.end, block.end))
        else:
            out.append(block)
    return out


def _as_ny(ts: pd.Timestamp) -> pd.Timestamp:
    ts = pd.Timestamp(ts)
    if ts.tz is None:
        raise ValueError(
            f"coverage bounds must be timezone-aware, got naive {ts}. "
            "A naive bound is ambiguous across DST and would corrupt the "
            "cache's record of what has been fetched."
        )
    return ts.tz_convert(NY)
