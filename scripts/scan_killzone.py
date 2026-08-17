"""FVG density by killzone, plus a window-shift sensitivity check.

The sensitivity check is the point. If a window's apparent importance survives
sliding it 15 or 30 minutes either way, the window boundary is not doing the
work. If the number moves a lot, the boundary was fitted to noise.

Run:  .venv\\Scripts\\python.exe scripts\\scan_killzone.py [SYMBOL] [MINUTES]
"""

from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ictbt.cache import ParquetCache
from ictbt.calendar import rth_only
from ictbt.nullmodel import session_positions
from ictbt.signals import KILLZONES, Killzone, describe_killzones, in_killzone
from ictbt.signals import find_fvgs


def shift(kz: Killzone, minutes: int) -> Killzone:
    """Slide a window by `minutes`, keeping its length."""
    base = dt.datetime(2000, 1, 1)
    start = (base.replace(hour=kz.start.hour, minute=kz.start.minute)
             + dt.timedelta(minutes=minutes)).time()
    end = (base.replace(hour=kz.end.hour, minute=kz.end.minute)
           + dt.timedelta(minutes=minutes)).time()
    return Killzone(f"{kz.name}{minutes:+d}m", start, end)


def main(argv: list[str]) -> int:
    symbol = (argv[0] if argv else "SPY").upper()
    minutes = int(argv[1]) if len(argv) > 1 else 15

    bars = ParquetCache().load(symbol, f"{minutes}m")
    if bars.empty:
        print(f"nothing cached for {symbol} {minutes}m — run scripts/backfill.py")
        return 1

    rth_all = rth_only(bars)
    gaps = find_fvgs(bars, rth=True)

    # Only bars at session position >= 2 can be candle 3 of a three-candle
    # pattern. Leaving the first two bars of each session in the denominator
    # makes any window overlapping the open look artificially barren — it was
    # what made the 07:00-10:00 window report zero gaps, since its entire RTH
    # overlap is exactly those two ineligible bars.
    positions = session_positions(rth_all, rth_all.index)
    rth = rth_all.loc[positions >= 2]

    print("killzone registry (windows are quoted, not derived):")
    print(describe_killzones().to_string())

    print(f"\n{symbol} {minutes}m — {len(rth):,} RTH bars, {len(gaps):,} gaps\n")
    print("FVG density inside each window vs outside")
    print(f"{'window':<18}{'bars in':>10}{'gaps in':>10}{'per bar in':>13}"
          f"{'per bar out':>13}{'ratio':>8}")

    overall = len(gaps) / len(rth)
    for name, kz in KILLZONES.items():
        if not kz.overlaps_rth:
            print(f"{name:<18}{'—':>10}{'—':>10}{'no RTH overlap':>26}")
            continue
        bar_mask = in_killzone(rth.index, kz)
        gap_mask = in_killzone(gaps.index, kz)
        n_bars = int(bar_mask.sum())
        n_gaps = int(gap_mask.sum())
        if not n_bars:
            continue
        inside = n_gaps / n_bars
        out_bars = len(rth) - n_bars
        outside = (len(gaps) - n_gaps) / out_bars if out_bars else float("nan")
        ratio = inside / outside if outside else float("nan")
        print(f"{name:<18}{n_bars:>10,}{n_gaps:>10,}{inside:>13.3f}"
              f"{outside:>13.3f}{ratio:>8.2f}")

    print(f"\noverall gap rate: {overall:.3f} per bar")

    print("\nsensitivity — sliding each testable window in 15-minute steps")
    print(f"{'window':<20}{'shift':>8}{'gaps in':>10}{'per bar':>10}{'ratio':>8}")
    for name in ("silver_bullet_am", "ny_pm"):
        kz = KILLZONES[name]
        for offset in (-30, -15, 0, 15, 30):
            moved = shift(kz, offset)
            bar_mask = in_killzone(rth.index, moved)
            gap_mask = in_killzone(gaps.index, moved)
            n_bars = int(bar_mask.sum())
            n_gaps = int(gap_mask.sum())
            if not n_bars:
                continue
            inside = n_gaps / n_bars
            out_bars = len(rth) - n_bars
            outside = (len(gaps) - n_gaps) / out_bars if out_bars else float("nan")
            ratio = inside / outside if outside else float("nan")
            marker = "  <- as quoted" if offset == 0 else ""
            print(f"{name:<20}{offset:>+8d}{n_gaps:>10,}{inside:>10.3f}"
                  f"{ratio:>8.2f}{marker}")
        print()

    print("gap rate by half hour (the intraday volatility profile):")
    half_hour = pd.Series(
        [t.hour + (0 if t.minute < 30 else 0.5) for t in rth.index], index=rth.index
    )
    gap_half = pd.Series(
        [t.hour + (0 if t.minute < 30 else 0.5) for t in gaps.index], index=gaps.index
    )
    bars_by = half_hour.value_counts().sort_index()
    gaps_by = gap_half.value_counts().sort_index().reindex(bars_by.index, fill_value=0)
    profile = (gaps_by / bars_by).round(3)
    for slot, rate in profile.items():
        hour = int(slot)
        minute = 30 if slot % 1 else 0
        bar = "#" * int(rate * 60)
        print(f"  {hour:02d}:{minute:02d}  {rate:.3f}  {bar}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
