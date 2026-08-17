"""Backfill the full available history for a symbol into the Parquet cache.

Read-only market data. Places no orders.

Run:  .venv\\Scripts\\python.exe scripts\\backfill.py [SYMBOL] [MINUTES]
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ictbt import quality
from ictbt.calendar import sessions
from ictbt.fetch import HISTORY_START, AlpacaBarSource, load_credentials
from ictbt.schema import NY


def main(argv: list[str]) -> int:
    symbol = (argv[0] if argv else "SPY").upper()
    minutes = int(argv[1]) if len(argv) > 1 else 15

    load_credentials()
    source = AlpacaBarSource()

    start = HISTORY_START
    end = pd.Timestamp.now(tz=NY)
    print(f"backfilling {symbol} {minutes}m from {start.date()} to {end.date()}")
    print("this paginates internally; expect a few minutes\n")

    t0 = time.monotonic()
    bars = source.get_bars(symbol, minutes=minutes, start=start, end=end)
    elapsed = time.monotonic() - t0

    if bars.empty:
        print("no bars returned")
        return 1

    print(f"done in {elapsed:,.0f}s")
    print(f"{len(bars):,} bars, {bars.index[0]} -> {bars.index[-1]}")

    sess = sessions(bars)
    print(f"{len(sess):,} regular-hours sessions")

    report = quality.check(bars, symbol=symbol, timeframe_minutes=minutes)
    print(f"\n{report.summary()}")

    opens = pd.Series([t.strftime("%H:%M") for t in sess["first_bar"]])
    odd = opens[opens != "09:30"]
    print(f"\nsessions not opening at 09:30 New York: {len(odd):,}")
    if len(odd):
        print(odd.value_counts().head().to_string())

    path = source.cache.bars_path(symbol, f"{minutes}m")
    print(f"\ncached to {path} ({path.stat().st_size / 1e6:,.1f} MB)")

    # Year-by-year bar counts: a year that is dramatically short is a coverage
    # hole worth knowing about before it silently skews a backtest.
    per_year = sess.groupby([d.year for d in sess.index]).size()
    print("\nsessions per year:")
    print(per_year.to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
