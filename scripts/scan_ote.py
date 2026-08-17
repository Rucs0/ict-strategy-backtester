"""OTE band entry rates, against equal-width bands at other depths.

The comparison is the point. A band is easier to hit the wider it is and the
closer it sits to price, so a raw hit rate for 62%-79% says nothing on its
own. Every row below uses a band of identical width; only the depth changes.
If the conventional band is special, it should stand out from its neighbours.

Run:  .venv\\Scripts\\python.exe scripts\\scan_ote.py [SYMBOL] [MINUTES]
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ictbt.cache import ParquetCache
from ictbt.calendar import sessions
from ictbt.signals import find_ote_zones

WIDTH = 0.17  # 0.79 - 0.62, held fixed across every comparison


def main(argv: list[str]) -> int:
    symbol = (argv[0] if argv else "SPY").upper()
    minutes = int(argv[1]) if len(argv) > 1 else 15

    bars = ParquetCache().load(symbol, f"{minutes}m")
    if bars.empty:
        print(f"nothing cached for {symbol} {minutes}m — run scripts/backfill.py")
        return 1

    n_sessions = len(sessions(bars))
    print(f"{symbol} {minutes}m — {n_sessions:,} sessions\n")

    print(f"{'n':>3}{'legs':>8}{'per session':>13}{'entered':>9}{'entry %':>9}"
          f"{'invalidated':>13}{'neither':>9}")
    for n in (1, 2, 3, 4, 5):
        zones = find_ote_zones(bars, n=n, rth=True, validate=False)
        if zones.empty:
            print(f"{n:>3}{0:>8}")
            continue
        entered = int(zones["entered_at"].notna().sum())
        invalid = int(zones["invalidated_at"].notna().sum())
        neither = len(zones) - entered - invalid
        print(f"{n:>3}{len(zones):>8,}{len(zones) / n_sessions:>13.2f}"
              f"{entered:>9,}{entered / len(zones):>9.1%}"
              f"{invalid:>13,}{neither:>9,}")

    print(f"\nequal-width ({WIDTH:.2f}) bands at different retracement depths, n=2")
    print("if 0.62-0.79 is special it should not sit on a smooth curve")
    print(f"{'band':>14}{'legs':>8}{'entered':>9}{'entry %':>9}")
    for low in (0.10, 0.20, 0.30, 0.40, 0.50, 0.62, 0.70, 0.80):
        high = low + WIDTH
        if high >= 1.0:
            continue
        zones = find_ote_zones(
            bars, n=2, rth=True, low_ratio=low, high_ratio=high, validate=False
        )
        if zones.empty:
            continue
        entered = int(zones["entered_at"].notna().sum())
        marker = "  <- conventional OTE" if abs(low - 0.62) < 1e-9 else ""
        print(f"{low:>7.2f}-{high:<6.2f}{len(zones):>8,}{entered:>9,}"
              f"{entered / len(zones):>9.1%}{marker}")

    zones = find_ote_zones(bars, n=2, rth=True, validate=False)
    print("\nat n=2, direction split:")
    print(zones["direction"].value_counts().to_string())
    print("\nbars from leg end to entry:")
    print(zones["bars_to_entry"].describe().to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
