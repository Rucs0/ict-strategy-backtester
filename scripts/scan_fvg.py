"""Run FVG detection over cached bars and describe what it found.

Descriptive only. This measures how often the pattern occurs and how often it
fills — it says nothing about whether trading it makes money. That question
belongs to Phase 3, after costs.

Run:  .venv\\Scripts\\python.exe scripts\\scan_fvg.py [SYMBOL] [MINUTES]
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ictbt.cache import ParquetCache
from ictbt.calendar import rth_only, sessions
from ictbt.signals import find_fvgs


def main(argv: list[str]) -> int:
    symbol = (argv[0] if argv else "SPY").upper()
    minutes = int(argv[1]) if len(argv) > 1 else 15

    bars = ParquetCache().load(symbol, f"{minutes}m")
    if bars.empty:
        print(f"nothing cached for {symbol} {minutes}m — run scripts/backfill.py")
        return 1

    rth = rth_only(bars)
    sess = sessions(bars)
    gaps = find_fvgs(bars, rth=True, fill_mode="touch")

    print(f"{symbol} {minutes}m — {bars.index[0].date()} to {bars.index[-1].date()}")
    print(f"{len(rth):,} regular-hours bars across {len(sess):,} sessions\n")

    print(f"fair value gaps found: {len(gaps):,}")
    print(f"  per session: {len(gaps) / len(sess):.2f}")
    print(f"  as a share of bars: {len(gaps) / len(rth):.1%}")

    by_dir = gaps["direction"].value_counts()
    print("\nby direction:")
    for name, count in by_dir.items():
        print(f"  {name:<9} {count:>7,}  ({count / len(gaps):.1%})")

    filled = gaps["filled"]
    print(f"\nfilled within the same session: {filled.sum():,} ({filled.mean():.1%})")
    print(f"unfilled at session close:      {(~filled).sum():,} ({(~filled).mean():.1%})")

    print("\nbars to fill (of those that filled):")
    print(gaps.loc[filled, "bars_to_fill"].describe().to_string())

    print("\ngap size as % of price:")
    print(gaps["size_pct"].describe().to_string())

    tradeable = gaps["tradeable_from"].notna()
    print(f"\ntradeable (candle 4 exists in session): {tradeable.sum():,} "
          f"({tradeable.mean():.1%})")

    per_year = gaps.groupby(gaps.index.year).size()
    print("\ngaps per year:")
    print(per_year.to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
