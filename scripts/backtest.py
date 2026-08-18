"""Phase 3: run the strategy in-sample and sweep its parameters.

In-sample only. The holdout is not touched here, and the guard enforces it.

Run:  .venv\\Scripts\\python.exe scripts\\backtest.py [SYMBOL] [MINUTES]
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ictbt.backtest import metrics, split_bars
from ictbt.backtest.costs import ZERO_COSTS, TransactionCosts
from ictbt.backtest.engine import SignalCache, StrategyConfig, run_backtest
from ictbt.backtest.split import require_in_sample
from ictbt.cache import ParquetCache


def main(argv: list[str]) -> int:
    symbol = (argv[0] if argv else "SPY").upper()
    minutes = int(argv[1]) if len(argv) > 1 else 15

    bars = ParquetCache().load(symbol, f"{minutes}m")
    if bars.empty:
        print(f"nothing cached for {symbol} {minutes}m — run scripts/backfill.py")
        return 1

    data = split_bars(bars)
    print(data.describe())
    print(f"  ratio: {data.ratio:.0%} in-sample\n")

    train = data.in_sample
    require_in_sample(train)  # tripwire, not decoration

    costs = TransactionCosts()
    print(f"costs: {costs.describe()}\n")

    baseline = StrategyConfig()
    print("=" * 62)
    print(f"BASELINE  {baseline.label()}")
    print("=" * 62)
    trades = run_backtest(train, baseline, costs)
    perf = metrics.evaluate(
        trades, capital=baseline.capital, risk_per_trade=baseline.risk_per_trade
    )
    print(perf.summary())

    if not trades.empty:
        print("\nexit reasons:")
        print(trades["exit_reason"].value_counts().to_string())
        print("\ndirection:")
        print(trades["direction"].value_counts().to_string())

    print("\n" + "=" * 62)
    print("PARAMETER SWEEP (in-sample)")
    print("=" * 62)
    print("every row net of costs; a strategy that only works at one setting")
    print("of an underived parameter is noise\n")
    print(f"{'config':<30}{'trades':>8}{'net P&L':>11}{'win%':>7}"
          f"{'exp R':>8}{'t':>7}{'gross':>11}")

    cache = SignalCache()
    rows = []
    for n, window, target_r in itertools.product(
        (1, 2, 3, 4), (2, 4, 6, 8), (1.0, 2.0, 3.0)
    ):
        config = StrategyConfig(n=n, sweep_to_mss_bars=window, target_r=target_r)
        t = run_backtest(train, config, costs, validate=False, cache=cache)
        p = metrics.evaluate(
            t, capital=config.capital, risk_per_trade=config.risk_per_trade
        )
        g = metrics.evaluate(
            run_backtest(train, config, ZERO_COSTS, validate=False, cache=cache),
            capital=config.capital, risk_per_trade=config.risk_per_trade,
        )
        rows.append(
            {
                "n": n, "window": window, "target_r": target_r,
                "trades": p.trade_count, "net": p.net_pnl,
                "gross": g.net_pnl, "win": p.win_rate,
                "exp_r": p.expectancy_r, "t": p.t_stat,
            }
        )
        print(f"{config.label():<30}{p.trade_count:>8,}{p.net_pnl:>11,.0f}"
              f"{p.win_rate:>7.1%}{p.expectancy_r:>8.3f}{p.t_stat:>7.2f}"
              f"{g.net_pnl:>11,.0f}")

    frame = pd.DataFrame(rows)
    print("\n" + "-" * 62)
    print(f"configurations tested:      {len(frame)}")
    print(f"profitable net of costs:    {(frame['net'] > 0).sum()}")
    print(f"profitable gross:           {(frame['gross'] > 0).sum()}")
    print(f"with |t| > 2:               {(frame['t'].abs() > 2).sum()}")
    print(f"median net P&L:             {frame['net'].median():,.0f}")
    print(f"median expectancy (R):      {frame['exp_r'].median():+.3f}")

    expected_by_chance = 0.05 * len(frame)
    print(f"\nconfigurations with |t| > 2 expected from noise alone: "
          f"{expected_by_chance:.1f}")

    print("\ncost drag: median gross minus net = "
          f"{(frame['gross'] - frame['net']).median():,.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
