"""Quality report for cached bars. No network.

Run:  .venv\\Scripts\\python.exe scripts\\report.py [SYMBOL] [MINUTES]
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ictbt import quality
from ictbt.cache import ParquetCache
from ictbt.calendar import rth_only, sessions


def main(argv: list[str]) -> int:
    symbol = (argv[0] if argv else "SPY").upper()
    minutes = int(argv[1]) if len(argv) > 1 else 15

    bars = ParquetCache().load(symbol, f"{minutes}m")
    if bars.empty:
        print(f"nothing cached for {symbol} {minutes}m — run scripts/backfill.py")
        return 1

    rth = rth_only(bars)
    sess = sessions(bars)

    print(f"{len(bars):,} bars total, {len(rth):,} inside regular hours")
    print(f"{bars.index[0].date()} -> {bars.index[-1].date()}\n")
    print(quality.check(bars, symbol=symbol, timeframe_minutes=minutes).summary())

    print("\nbar-count distribution across sessions:")
    print(sess["bars"].value_counts().sort_index().to_string())

    early = sess[sess["early_close"]]
    print(f"\n{len(early)} early-close sessions, most recent 5:")
    print(early.tail(5)[["bars", "last_bar", "scheduled_close"]].to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
