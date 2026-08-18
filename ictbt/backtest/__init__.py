"""Backtesting: cost model, event loop, metrics, and the data split.

Engine choice, recorded here because it constrains everything downstream.
A custom event loop was chosen over `vectorbt` and `backtrader`:

- **Against vectorbt**: it is the fastest option and is built for large
  parameter sweeps, which this project needs. But it assumes signals can be
  expressed as vectors computed up front, and two of the primitives
  (`sweeps`, `structure`) are stateful walks that retire levels as price
  interacts with them. Forcing them into a vectorized form would mean either
  precomputing state that depends on the path, or reimplementing them twice.
  The sweep sizes here are thousands of trades, not millions, so its speed
  advantage buys little.
- **Against backtrader**: mature and event-driven, but its broker and order
  abstractions carry assumptions about fills that are hard to audit from
  outside. The single thing this project cannot afford to get wrong is the
  fill model, so it should be code we can read in full.
- **For a custom loop**: about two hundred lines, every fill assumption
  visible and testable, and it can express the conservative intrabar rule
  that the OHLC data actually justifies.

The cost is that it is our own code and can be wrong in ways a mature library
would not be. Mitigated by tests and mutation checks, not by hope.
"""

from .split import DataSplit, SPLIT_DATE, split_bars

__all__ = ["DataSplit", "SPLIT_DATE", "split_bars"]
