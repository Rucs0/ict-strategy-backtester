# ICT Strategy Backtester

## What this project is

A research tool that mechanizes ICT (Inner Circle Trader) day-trading concepts into
testable code and measures whether they have edge net of transaction costs.

**The deliverable is the backtest, not a trading bot.** A rigorous negative result
("these concepts do not survive realistic costs") is a successful outcome and should
be reported as such. Do not tune parameters until the result turns positive.

Paper trading only. No live capital, no real brokerage execution.

## Build order

Do not skip ahead. Each phase gates the next.

### Phase 1 — Data layer
- Source intraday OHLCV with enough history for real testing.
- `yfinance` is insufficient: ~60 days of 15m bars, ~7 days of 1m. Evaluate
  Polygon.io free tier or Alpaca's market data API instead.
- Cache locally to Parquet. Do not re-fetch on every run.
- Write tests for gap handling, session boundaries, and timezone correctness
  (ICT concepts are timezone-sensitive — killzones are defined in New York time).

### Phase 2 — Signal primitives
Implement each as a pure function: `(DataFrame) -> Series[bool]` or a signal object.
Unit test each against hand-constructed fixtures before moving on.

Start with the mechanically well-defined ones:
- **Fair value gap** — 3-candle pattern, candle 1 high < candle 3 low (bullish) or
  candle 1 low > candle 3 high (bearish), gap not yet filled.
- **Killzone filter** — time-of-day windows, New York time.
- **Swing points** — fractal highs/lows with N bars either side. `N` is a free
  parameter; expose it and document that it was chosen, not derived.
- **Liquidity sweep** — wick penetrates a prior swing point, candle closes back inside.
- **Market structure shift** — break of a swing point in the opposite direction.
- **OTE** — Fibonacci retracement band between swing points.

Defer or flag: **order blocks**. Definition depends on "impulsive move," which has no
mechanical definition. If implemented, pick an explicit threshold, document it as
arbitrary, and test sensitivity to it.

### Phase 3 — Strategy assembly and backtest
- Compose primitives into entry/exit rules. Keep composition declarative and
  configurable so variants are cheap to test.
- **Split out-of-sample data before looking at any results.** Hold it back.
- **Model transaction costs.** Spread plus commission, applied per trade. This is not
  optional — high-frequency intraday strategies commonly show gross profit and net
  loss, and omitting costs is the single most common way retail backtests lie.
- Report: net P&L, Sharpe, max drawdown, win rate, trade count, and P&L *gross vs net*
  side by side so the cost drag is visible.
- Test parameter sensitivity. If results collapse when `N` changes from 5 to 6, the
  edge is noise.

### Phase 4 — Forward paper testing (only if Phase 3 shows net edge)
- TradingView Pine Script alert → webhook → local handler → Alpaca paper endpoint.
- Log every signal, decision, and simulated fill to a reviewable file.
- Requires an Alpaca account; check current age requirements in their terms first.

## Constraints

- **Owner is 16.** Do not implement live trading, real-money execution, or anything
  requiring an account whose terms he may not meet. Flag it instead of building it.
- **Equities and equity index futures data only.** Avoid building around leveraged
  spot forex or CFDs — overnight swap financing is interest-based, and the owner
  follows Hanafi rulings on this.
- No leverage modeling, no margin, no overnight financing logic.

## Honest framing to preserve

There is no independently verified public track record for ICT methodology, and
community evidence is overwhelmingly retrospective chart-marking. Build the tooling
to find out, not to confirm. If the code starts accumulating special cases that make
historical trades look better, stop and say so.

## Stack

Python. pandas for data, pytest for tests. Choose a backtesting approach in Phase 3
(custom event loop vs. `backtrader` vs. `vectorbt`) and document the tradeoff.
