# ICT Strategy Backtester

A research tool that mechanizes six ICT (Inner Circle Trader) day-trading concepts
into testable code and measures whether they hold edge net of transaction costs,
against ten years of SPY 15-minute bars.

**Result: negative.** The canonical setup lost money net of costs at baseline
(61 trades, −0.072R expectancy), and no configuration in a 48-point parameter
sweep showed significant positive edge. Two of the methodology's most commonly
cited statistics (the 74% fair-value-gap fill rate, the 62–79% "optimal trade
entry" band) turn out to be fully explained by baselines that have nothing to do
with the pattern.

**Read the full report: [`docs/findings.md`](docs/findings.md).**

Paper trading only. No live capital, no real brokerage execution.

## Why this exists

There is no independently verified public track record for ICT methodology.
The community evidence for it is overwhelmingly retrospective chart-marking,
setups identified after the outcome is already visible. This project asks a
narrower question: do the concepts survive being written down as code and run
forward through data they have not seen. It answers that question honestly,
including when the answer doesn't flatter the premise.

## Method, briefly

- Out-of-sample holdout fixed **before** any backtest ran, guarded in code
- Transaction costs modelled on every fill
- Every undefined parameter swept, results reported across the full range,
  no single flattering value
- Every load-bearing invariant checked by mutation testing (deliberately
  breaking it and confirming the test suite catches it)
- A concept whose definition was too vague to mechanize (order blocks) was
  left unimplemented rather than invented

Full detail, including the parameter sweep table and both null-model results:
[`docs/findings.md`](docs/findings.md).

## Reproducing this

256 tests run offline against hand-built fixtures.

```
.venv/Scripts/python.exe scripts/backfill.py   SPY 15   # rebuild the cache
.venv/Scripts/python.exe scripts/backtest.py   SPY 15   # strategy + sweep
.venv/Scripts/python.exe scripts/null_test.py  SPY 15   # null models
.venv/Scripts/python.exe -m pytest -q                   # 256 tests
```

Requires an Alpaca market-data API key in `.env`, see `.env.example`.

## License

MIT, see [`LICENSE`](LICENSE).
