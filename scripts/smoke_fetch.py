"""First contact with the Alpaca API: pull real bars and inspect them.

Read-only. Fetches market data and writes to the local Parquet cache.
Places no orders and touches no account endpoint.

Run:  .venv\\Scripts\\python.exe scripts\\smoke_fetch.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ictbt import quality
from ictbt.calendar import sessions
from ictbt.fetch import AlpacaBarSource, load_credentials
from ictbt.schema import NY

SYMBOL = "SPY"
MINUTES = 15
LOOKBACK_DAYS = 30


def main() -> int:
    load_credentials()

    end = pd.Timestamp.now(tz=NY)
    start = end - pd.Timedelta(days=LOOKBACK_DAYS)

    source = AlpacaBarSource()
    print(f"fetching {SYMBOL} {MINUTES}m from {start.date()} to {end.date()} ...")
    bars = source.get_bars(SYMBOL, minutes=MINUTES, start=start, end=end)

    if bars.empty:
        print("no bars returned — check the symbol and the date window")
        return 1

    print(f"\n{len(bars):,} bars, {bars.index[0]} -> {bars.index[-1]}")
    print(f"index tz: {bars.index.tz}")
    print("\nfirst 3 bars:")
    print(bars.head(3).to_string())

    sess = sessions(bars)
    print(f"\n{len(sess)} sessions")
    print(sess.head(5).to_string())

    report = quality.check(bars, symbol=SYMBOL, timeframe_minutes=MINUTES)
    print(f"\n{report.summary()}")

    # The session-open bar is the single best check that timezone handling is
    # right end to end: it must be 09:30 New York on every session, in both
    # daylight-saving regimes.
    opens = {t.strftime("%H:%M") for t in sess["first_bar"]}
    print(f"\nsession-open times seen: {sorted(opens)}")
    if opens != {"09:30"}:
        print("  WARNING: expected every session to open at 09:30 New York")

    cached = source.cache.bars_path(SYMBOL, "15m")
    print(f"\ncached to {cached} ({cached.stat().st_size:,} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
