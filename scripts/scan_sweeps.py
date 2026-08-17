"""Liquidity sweep frequency across n, the inherited free parameter.

Descriptive only. Whether a sweep predicts anything is a Phase 3 question.

Run:  .venv\\Scripts\\python.exe scripts\\scan_sweeps.py [SYMBOL] [MINUTES]
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ictbt.cache import ParquetCache
from ictbt.calendar import rth_only, sessions
from ictbt.signals import find_sweeps, find_swings


def main(argv: list[str]) -> int:
    symbol = (argv[0] if argv else "SPY").upper()
    minutes = int(argv[1]) if len(argv) > 1 else 15

    bars = ParquetCache().load(symbol, f"{minutes}m")
    if bars.empty:
        print(f"nothing cached for {symbol} {minutes}m — run scripts/backfill.py")
        return 1

    rth = rth_only(bars)
    n_sessions = len(sessions(bars))
    print(f"{symbol} {minutes}m — {len(rth):,} RTH bars, {n_sessions:,} sessions\n")

    print(f"{'n':>3}{'swings':>10}{'sweeps':>9}{'per session':>13}"
          f"{'swept %':>10}{'bullish':>9}{'bearish':>9}{'med pen %':>11}")
    for n in (1, 2, 3, 4, 5, 6, 8):
        swings = find_swings(bars, n=n, scope="session", rth=True, validate=False)
        sweeps = find_sweeps(bars, n=n, rth=True, validate=False)
        if sweeps.empty:
            print(f"{n:>3}{len(swings):>10,}{0:>9}")
            continue
        bull = int((sweeps["direction"] == "bullish").sum())
        bear = int((sweeps["direction"] == "bearish").sum())
        swept_share = len(sweeps) / len(swings) if len(swings) else float("nan")
        print(f"{n:>3}{len(swings):>10,}{len(sweeps):>9,}"
              f"{len(sweeps) / n_sessions:>13.2f}{swept_share:>10.1%}"
              f"{bull:>9,}{bear:>9,}{sweeps['penetration_pct'].median():>11.4f}")

    print("\nat n=2, how deep do sweeps penetrate? (% of level)")
    sweeps = find_sweeps(bars, n=2, rth=True, validate=False)
    print(sweeps["penetration_pct"].describe().to_string())

    print("\nbars between the swing and the sweep (n=2):")
    print(sweeps["bars_since_swing"].describe().to_string())

    print("\nmin_penetration sensitivity at n=2 (SPY ~ $500, so 0.01 = 1 cent)")
    print(f"{'threshold':>12}{'sweeps':>10}{'retained':>11}")
    base = len(sweeps)
    for threshold in (0.0, 0.01, 0.02, 0.05, 0.10, 0.25):
        found = find_sweeps(
            bars, n=2, rth=True, min_penetration=threshold, validate=False
        )
        print(f"{threshold:>12.2f}{len(found):>10,}{len(found) / base:>11.0%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
