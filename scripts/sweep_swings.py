"""Sensitivity of swing detection to n, the free parameter.

`n` has no derivation. This sweep exists so its effect is visible before
anything is built on top of it.

Run:  .venv\\Scripts\\python.exe scripts\\sweep_swings.py [SYMBOL] [MINUTES]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ictbt.cache import ParquetCache
from ictbt.calendar import rth_only, sessions
from ictbt.signals import find_swings


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

    print(f"{'n':>3}{'swings':>10}{'per session':>14}{'highs':>9}{'lows':>9}"
          f"{'confirm lag':>13}{'kept from n-1':>15}")

    previous: set | None = None
    for n in (1, 2, 3, 4, 5, 6, 8, 10):
        swings = find_swings(bars, n=n, scope="session", rth=True)
        if swings.empty:
            print(f"{n:>3}{0:>10}")
            continue
        highs = int((swings["kind"] == "high").sum())
        lows = int((swings["kind"] == "low").sum())
        current = set(swings.index)
        overlap = (
            f"{len(current & previous) / len(previous):.0%}"
            if previous else "—"
        )
        print(f"{n:>3}{len(swings):>10,}{len(swings) / n_sessions:>14.2f}"
              f"{highs:>9,}{lows:>9,}{float(n):>13.0f}{overlap:>15}")
        previous = current

    print("\nstability of the swing set as n changes")
    print("(how much of the n=2 swing set survives at each other n)")
    baseline = set(find_swings(bars, n=2, scope="session", rth=True).index)
    for n in (1, 3, 4, 5):
        other = set(find_swings(bars, n=n, scope="session", rth=True).index)
        jaccard = len(baseline & other) / len(baseline | other)
        retained = len(baseline & other) / len(baseline)
        print(f"  n={n}: {retained:.0%} of n=2 swings retained, "
              f"Jaccard {jaccard:.2f}")

    print("\nscope comparison at n=2 (session vs continuous)")
    per_session = find_swings(bars, n=2, scope="session", rth=True)
    continuous = find_swings(bars, n=2, scope="continuous", rth=True)
    print(f"  session:    {len(per_session):,}")
    print(f"  continuous: {len(continuous):,}")
    shared = len(set(per_session.index) & set(continuous.index))
    print(f"  shared:     {shared:,} "
          f"({shared / max(len(per_session), 1):.0%} of session-scoped)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
