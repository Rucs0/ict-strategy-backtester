"""Compare real FVG statistics against two null models.

Run:  .venv\\Scripts\\python.exe scripts\\null_test.py [SYMBOL] [MINUTES] [REPS]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ictbt.cache import ParquetCache
from ictbt.calendar import rth_only, sessions
from ictbt.nullmodel import matched_touch_rate, shuffled_bars
from ictbt.signals import find_fvgs

SEED = 20260817


def main(argv: list[str]) -> int:
    symbol = (argv[0] if argv else "SPY").upper()
    minutes = int(argv[1]) if len(argv) > 1 else 15
    reps = int(argv[2]) if len(argv) > 2 else 20

    bars = ParquetCache().load(symbol, f"{minutes}m")
    if bars.empty:
        print(f"nothing cached for {symbol} {minutes}m — run scripts/backfill.py")
        return 1

    rng = np.random.default_rng(SEED)
    rth = rth_only(bars)
    n_sessions = len(sessions(bars))

    real = find_fvgs(bars, rth=True)
    real_count = len(real)
    real_fill = real["filled"].mean()

    print(f"{symbol} {minutes}m — {len(rth):,} RTH bars, {n_sessions:,} sessions\n")
    print("REAL DATA")
    print(f"  gaps:              {real_count:,}  ({real_count / n_sessions:.2f} per session)")
    print(f"  fill rate:         {real_fill:.1%}")
    print(f"  median bars->fill: {real.loc[real['filled'], 'bars_to_fill'].median():.0f}")

    print(f"\nNULL A — within-session bar shuffle, {reps} repetitions")
    print("  (same bars, same jumps, random order)")
    counts, fills, medians = [], [], []
    for i in range(reps):
        synth = shuffled_bars(rth, rng, rth=False)
        gaps = find_fvgs(synth, rth=False)
        counts.append(len(gaps))
        fills.append(gaps["filled"].mean() if len(gaps) else np.nan)
        medians.append(
            gaps.loc[gaps["filled"], "bars_to_fill"].median() if len(gaps) else np.nan
        )
        print(f"    rep {i + 1:>2}: {len(gaps):>6,} gaps, fill {fills[-1]:.1%}", end="\r")

    counts = np.array(counts, dtype=float)
    fills = np.array(fills, dtype=float)
    print(" " * 60, end="\r")
    print(f"  gaps:              {counts.mean():,.0f} +/- {counts.std():,.0f}")
    print(f"  fill rate:         {fills.mean():.1%} +/- {fills.std():.1%}")
    print(f"  median bars->fill: {np.nanmedian(medians):.0f}")

    z_count = (real_count - counts.mean()) / counts.std() if counts.std() else np.nan
    z_fill = (real_fill - fills.mean()) / fills.std() if fills.std() else np.nan
    print(f"\n  real vs shuffled, gap count: z = {z_count:+.1f}")
    print(f"  real vs shuffled, fill rate: z = {z_fill:+.1f}")

    print("\nNULL B — matched-position touch rate")
    print("  (arbitrary bar, same time of day: does price revisit its extreme?)")
    baseline = matched_touch_rate(rth, real, rng, rth=False)
    print(f"  baseline touch rate: {baseline:.1%}")
    print(f"  real FVG fill rate:  {real_fill:.1%}")
    print(f"  difference:          {real_fill - baseline:+.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
