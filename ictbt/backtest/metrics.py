"""Performance metrics, reported gross and net side by side.

Every headline figure appears twice: before costs and after. The gap between
them is the cost drag, and for an intraday strategy it is usually the most
informative number on the page. A gross figure on its own is a diagnostic,
never a result.

**On the t-statistic.** It is included because with a setup that fires
monthly, "the strategy made money" and "the strategy has edge" are different
claims, and only the second one matters. A rough rule: |t| below about 2 means
the result is indistinguishable from luck at the usual threshold. That
threshold is itself generous here, because it assumes a single hypothesis was
tested — and a parameter sweep tests hundreds. If a sweep of 500
configurations produces a handful with t above 2, that is the expected yield
from noise, not a discovery.

**On Sharpe.** Computed from daily net returns including flat days, which is
the standard convention and the conservative one for a strategy that trades
rarely. A per-trade Sharpe would look far better and would not be comparable
to anything.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

#: US equity trading days per year, for annualization.
TRADING_DAYS = 252


@dataclass
class Performance:
    """Summary statistics for one backtest run."""

    trade_count: int
    win_rate: float
    gross_pnl: float
    net_pnl: float
    total_cost: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    expectancy: float
    expectancy_r: float
    sharpe: float
    gross_sharpe: float
    max_drawdown: float
    t_stat: float
    capital: float

    @property
    def cost_drag_pct(self) -> float:
        """Costs as a share of gross profit, when gross was positive."""
        if self.gross_pnl <= 0:
            return float("nan")
        return self.total_cost / self.gross_pnl

    @property
    def net_return_pct(self) -> float:
        return self.net_pnl / self.capital if self.capital else float("nan")

    def summary(self) -> str:
        if not self.trade_count:
            return "no trades"

        lines = [
            f"trades           {self.trade_count:>12,}",
            f"win rate         {self.win_rate:>12.1%}",
            "",
            f"{'':17}{'gross':>14}{'net':>14}",
            f"{'P&L':17}{self.gross_pnl:>14,.2f}{self.net_pnl:>14,.2f}",
            f"{'Sharpe':17}{self.gross_sharpe:>14.2f}{self.sharpe:>14.2f}",
            "",
            f"transaction cost {self.total_cost:>12,.2f}",
        ]
        if np.isfinite(self.cost_drag_pct):
            lines.append(f"cost / gross P&L {self.cost_drag_pct:>12.1%}")
        lines += [
            f"return on capital{self.net_return_pct:>12.1%}",
            f"max drawdown     {self.max_drawdown:>12,.2f}",
            "",
            f"avg win          {self.avg_win:>12,.2f}",
            f"avg loss         {self.avg_loss:>12,.2f}",
            f"profit factor    {self.profit_factor:>12.2f}",
            f"expectancy       {self.expectancy:>12,.2f}  "
            f"({self.expectancy_r:+.3f} R)",
            f"t-stat           {self.t_stat:>12.2f}",
        ]
        if abs(self.t_stat) < 2.0:
            lines.append("  |t| < 2: not distinguishable from luck")
        return "\n".join(lines)


def evaluate(
    trades: pd.DataFrame, *, capital: float, risk_per_trade: float
) -> Performance:
    """Compute performance statistics from a trade ledger."""
    if trades.empty:
        return Performance(
            trade_count=0,
            win_rate=float("nan"),
            gross_pnl=0.0,
            net_pnl=0.0,
            total_cost=0.0,
            avg_win=float("nan"),
            avg_loss=float("nan"),
            profit_factor=float("nan"),
            expectancy=float("nan"),
            expectancy_r=float("nan"),
            sharpe=float("nan"),
            gross_sharpe=float("nan"),
            max_drawdown=0.0,
            t_stat=float("nan"),
            capital=capital,
        )

    net = trades["net_pnl"].to_numpy(dtype="float64")
    gross = trades["gross_pnl"].to_numpy(dtype="float64")
    cost = trades["cost"].to_numpy(dtype="float64")

    wins = net[net > 0]
    losses = net[net < 0]

    gross_profit = float(wins.sum())
    gross_loss = float(-losses.sum())

    return Performance(
        trade_count=len(trades),
        win_rate=float(len(wins) / len(net)),
        gross_pnl=float(gross.sum()),
        net_pnl=float(net.sum()),
        total_cost=float(cost.sum()),
        avg_win=float(wins.mean()) if len(wins) else float("nan"),
        avg_loss=float(losses.mean()) if len(losses) else float("nan"),
        profit_factor=(
            gross_profit / gross_loss if gross_loss > 0 else float("inf")
        ),
        expectancy=float(net.mean()),
        expectancy_r=float(net.mean() / risk_per_trade) if risk_per_trade else float("nan"),
        sharpe=_sharpe(trades, "net_pnl", capital),
        gross_sharpe=_sharpe(trades, "gross_pnl", capital),
        max_drawdown=_max_drawdown(net),
        t_stat=_t_stat(net),
        capital=capital,
    )


def _t_stat(pnl: np.ndarray) -> float:
    """One-sample t against a null of zero mean P&L per trade."""
    if len(pnl) < 2:
        return float("nan")
    sd = pnl.std(ddof=1)
    if sd == 0:
        return float("nan")
    return float(pnl.mean() / (sd / np.sqrt(len(pnl))))


def _sharpe(trades: pd.DataFrame, column: str, capital: float) -> float:
    """Annualized Sharpe from daily returns, counting flat days as zero.

    Flat days are included deliberately. Dropping them would report the
    Sharpe of the days the strategy chose to trade, which flatters any rare
    strategy and is not comparable to a buy-and-hold figure.
    """
    if trades.empty or not capital:
        return float("nan")

    daily = trades.groupby(trades["exit_at"].dt.date)[column].sum()
    if daily.empty:
        return float("nan")

    calendar = pd.date_range(
        min(daily.index), max(daily.index), freq="B"
    ).date
    series = daily.reindex(calendar, fill_value=0.0) / capital

    sd = series.std(ddof=1)
    if sd == 0 or not np.isfinite(sd):
        return float("nan")
    return float(series.mean() / sd * np.sqrt(TRADING_DAYS))


def _max_drawdown(pnl: np.ndarray) -> float:
    """Largest peak-to-trough decline of the cumulative P&L curve."""
    equity = np.cumsum(pnl)
    peak = np.maximum.accumulate(equity)
    return float(np.max(peak - equity)) if len(equity) else 0.0
