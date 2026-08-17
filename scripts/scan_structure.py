"""Structure event frequency, and the sweep-then-MSS sequence.

The sequence matters because it is the setup ICT material actually describes:
liquidity taken, then structure shifts against it. Counting how often that
sequence occurs is descriptive; whether it predicts anything is Phase 3.

Run:  .venv\\Scripts\\python.exe scripts\\scan_structure.py [SYMBOL] [MINUTES]
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ictbt.cache import ParquetCache
from ictbt.calendar import rth_only, session_date, sessions
from ictbt.signals import find_structure_events, find_sweeps


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

    print(f"{'n':>3}{'events':>9}{'bos':>8}{'mss':>8}{'mss/session':>13}"
          f"{'mss share':>11}")
    for n in (1, 2, 3, 4, 5, 6):
        events = find_structure_events(bars, n=n, rth=True, validate=False)
        if events.empty:
            print(f"{n:>3}{0:>9}")
            continue
        bos = int((events["event"] == "bos").sum())
        mss = int((events["event"] == "mss").sum())
        print(f"{n:>3}{len(events):>9,}{bos:>8,}{mss:>8,}"
              f"{mss / n_sessions:>13.2f}{mss / len(events):>11.1%}")

    n = 2
    events = find_structure_events(bars, n=n, rth=True, validate=False)
    sweeps = find_sweeps(bars, n=n, rth=True, validate=False)

    print(f"\nat n={n}: {len(sweeps):,} sweeps, "
          f"{int((events['event'] == 'mss').sum()):,} MSS")

    # The canonical setup: a sweep, then a shift against it, same session,
    # within a short window. The window is a choice; it is swept below.
    mss = events[events["event"] == "mss"]

    # A single bar can emit both a bullish and a bearish event, so these
    # indexes contain duplicates. Everything below works positionally.
    sweep_day = session_date(sweeps.index).to_numpy()
    sweep_dir = sweeps["direction"].to_numpy()
    sweep_pos = rth.index.get_indexer(sweeps.index)

    mss_day = session_date(mss.index).to_numpy()
    mss_dir = mss["direction"].to_numpy()
    mss_pos = rth.index.get_indexer(mss.index)

    print("\nsweep followed by a same-direction MSS in the same session")
    print("(liquidity taken, then structure shifts the way the sweep implied)")
    print(f"{'within':>8}{'pairs':>9}{'per session':>13}{'% of sweeps':>13}")
    for window in (1, 2, 3, 4, 6, 8, 12):
        pairs = 0
        for k in range(len(sweeps)):
            match = (
                (mss_day == sweep_day[k])
                & (mss_dir == sweep_dir[k])
                & (mss_pos > sweep_pos[k])
                & (mss_pos <= sweep_pos[k] + window)
            )
            if match.any():
                pairs += 1
        print(f"{window:>8}{pairs:>9,}{pairs / n_sessions:>13.3f}"
              f"{pairs / len(sweeps):>13.1%}")

    print("\nbars between the broken swing and the break (n=2):")
    print(events["bars_since_swing"].describe().to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
