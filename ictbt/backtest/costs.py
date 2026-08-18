"""Transaction costs.

`CLAUDE.md` calls omitting these the single most common way retail backtests
lie, and the reason is arithmetic rather than rhetoric. A strategy taking a
few trades a session on a move of a few tenths of a percent is operating on a
margin comparable to the cost of trading. Costs are not a correction applied
at the end; for intraday strategies they frequently *are* the result.

Three components, charged separately because they behave differently:

- **Spread.** You buy at the ask and sell at the bid. Crossing the spread
  costs half of it per side, the full spread per round trip. Modelled per
  share.
- **Commission.** Alpaca charges nothing for US equities, so the default is
  zero — but zero is a property of one broker at one moment, not of trading,
  and a result that only survives at zero commission should say so.
- **Slippage.** The gap between the price you expected and the price you got.
  Charged per side, on top of the spread, since a marketable order in a fast
  market often does worse than the touch.

**SPY is the best case in the entire US equity market.** Its spread is
routinely a single cent on a price in the hundreds — a few tenths of a basis
point. Any result that fails to clear costs here fails worse everywhere else,
which is exactly what makes it a fair first test rather than a flattering one.

The defaults model a penny spread and a further half-cent of slippage per
side. That is deliberately not the most optimistic reading available; a
backtest that only works at zero slippage is not describing a tradeable
strategy.
"""

from __future__ import annotations

from dataclasses import dataclass

#: A penny-wide market, which is typical for SPY in regular hours.
DEFAULT_SPREAD_PER_SHARE = 0.01

#: Alpaca charges nothing for US equities. Kept as a parameter so results can
#: be reported against a broker that does.
DEFAULT_COMMISSION_PER_SHARE = 0.0

#: Half a cent per side beyond the touch. A judgement, and on the
#: conservative side of what marketable orders actually experience.
DEFAULT_SLIPPAGE_PER_SHARE = 0.005


@dataclass(frozen=True)
class TransactionCosts:
    """Per-share trading costs, charged per side."""

    spread_per_share: float = DEFAULT_SPREAD_PER_SHARE
    commission_per_share: float = DEFAULT_COMMISSION_PER_SHARE
    slippage_per_share: float = DEFAULT_SLIPPAGE_PER_SHARE

    def __post_init__(self) -> None:
        for name in (
            "spread_per_share",
            "commission_per_share",
            "slippage_per_share",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be >= 0")

    @property
    def per_side(self) -> float:
        """Cost of one fill, per share.

        Half the spread, because crossing a two-sided market costs half of it
        to get in and half to get out.
        """
        return (
            self.spread_per_share / 2.0
            + self.slippage_per_share
            + self.commission_per_share
        )

    @property
    def round_trip(self) -> float:
        """Cost of a complete trade, per share."""
        return 2.0 * self.per_side

    def fill_price(self, price: float, *, side: str) -> float:
        """Adjust an intended price to what a market order would actually get.

        Buys fill above the reference, sells below it. Sign errors here would
        turn a cost into a subsidy, which is why this is one function rather
        than an adjustment scattered through the engine.
        """
        if side == "buy":
            return price + self.per_side
        if side == "sell":
            return price - self.per_side
        raise ValueError(f"side must be 'buy' or 'sell', got {side!r}")

    def describe(self) -> str:
        return (
            f"spread {self.spread_per_share:.4f}/share, "
            f"slippage {self.slippage_per_share:.4f}/side, "
            f"commission {self.commission_per_share:.4f}/side "
            f"= {self.round_trip:.4f}/share round trip"
        )


#: Costs switched off. For measuring the cost drag by difference only — a
#: gross figure is a diagnostic, never a result.
ZERO_COSTS = TransactionCosts(0.0, 0.0, 0.0)
