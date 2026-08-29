# ICT day-trading concepts do not survive costs

**Falsification report: SPY 15-minute bars, 2016–2022**

256 tests · 167,215 bars · 2,669 sessions · paper only

Six ICT primitives were mechanized, tested against ten years of consolidated-tape
data, and assembled into the canonical setup the methodology describes. Across 48
parameter configurations, none showed statistically significant edge. The only
result clearing the usual threshold was a loss.

> **Result: negative.** The sweep-then-shift setup lost money net of costs at
> baseline parameters (61 trades, −0.072R expectancy, t = −0.70), and the
> parameter sweep produced no configuration with significant positive edge.
>
> The out-of-sample holdout reserved for validation was never used. Nothing here
> earned it.

---

## What was actually tested

ICT (Inner Circle Trader) is a widely taught body of day-trading concepts with no
independently verified public track record. The community evidence for it is
overwhelmingly retrospective chart-marking: setups identified after the fact, on
charts where the outcome is already visible.

This project asks a narrower question than "does ICT work": whether the concepts
survive being written down as code and run forward through data they have not
seen. That question can be answered honestly.

The instrument is SPY, the S&P 500 ETF, chosen because it is the friendliest
possible test: the tightest spread in the US equity market, the deepest
liquidity, and the instrument the methodology is most often demonstrated on. A
strategy that fails here fails worse everywhere else.

---

## How it was built

Three phases, each gating the next. No phase began before the previous one had
its tests passing.

### 01. Data

Ten years of 15-minute bars from a consolidated-tape feed, cached locally. All
timestamps carry New York wall-clock time, because ICT concepts are defined in it
and the UTC offset changes twice a year.

Session boundaries come from the real NYSE calendar rather than a fixed
09:30–16:00 window. That correction mattered. The exchange closes at 13:00 on
about two days a year, but the feed keeps returning bars until 16:00 because the
ETF still trades after hours elsewhere. A fixed window swallowed 251 thin
after-hours bars across 21 sessions as though they were regular trading, and the
wide price gaps between those sparse prints are indistinguishable from the very
pattern this project was about to go measure.

### 02. Signal primitives

Fair value gap, killzone filter, swing points, liquidity sweep, market structure
shift, and optimal trade entry, each a pure function over bars, each unit tested
against hand-built fixtures before anything was assembled.

Order blocks were deliberately not built. Their definition depends on an
"impulsive move," which has no mechanical meaning. Implementing one would have
meant inventing a threshold and then testing the invention rather than the stated
concept.

### 03. Backtest

A custom event loop, chosen over off-the-shelf libraries so every fill assumption
stays readable. Entry is at the next bar's open, never the signal bar's close. A
bar containing both stop and target resolves as a stop, because OHLC data cannot
order the high and the low within a bar and the optimistic reading is worth three
risk units per ambiguous trade.

---

## What was fixed in advance

A backtest is evidence only if its rules were set before the results were
visible. These were.

| | |
|---|---|
| **Fixed first** | **Holdout split at 2023-01-01.** Chosen before a single backtest ran and guarded in code: touching it raises an exception unless explicitly overridden. It remains unused. |
| **Modelled** | **Transaction costs on every fill.** A penny spread plus half a cent of slippage per side. Costs are not a footnote for an intraday strategy; they are frequently the whole result. |
| **Swept** | **Every undefined parameter.** ICT does not specify the swing lookback, the killzone boundaries, the penetration threshold or the retracement band. Results are reported across ranges, not at the flattering value. |
| **Not done** | **No tuning toward a positive result.** The exit rule looks mis-specified. It was recorded and left alone, because testing variants until one works is the failure this design exists to prevent. |

---

## Primary result: the strategy loses money

The setup is the canonical ICT sequence: price sweeps a prior swing point,
structure shifts in the same direction within a few bars, enter, stop beyond the
sweep, target a multiple of that risk, flat by the close.

**Baseline: n=2, window=4 bars, 2R target, in-sample 2016–2022**

| Measure | Gross | Net of costs |
|---|---:|---:|
| Trades | 61 | 61 |
| Win rate | 44.3% | 44.3% |
| P&L | −257 | −437 |
| **Expectancy per trade** | **−0.042 R** | **−0.072 R** |
| Transaction cost | — | 180 |
| Max drawdown | — | 797 |
| t-statistic | — | −0.70 |

A t-statistic of −0.70 does not establish that the strategy loses. It establishes
that 61 trades cannot tell a small loss apart from noise. That is the more
important finding, and it recurs everywhere below: **the setup is too rare to
test decisively.** Over ten full years, the complete sequence fires about once a
month.

---

## Parameter sweep: nothing survives moving the knobs

Forty-eight configurations across the swing lookback `n`, the sweep-to-shift
window, and the target multiple. Expectancy is average profit per trade in risk
units, net of costs.

| n | window | R=1 | R=2 | R=3 | trades |
|---:|---:|---:|---:|---:|---:|
| 1 | 2 | +0.095 | +0.096 | +0.109 | 150 |
| 1 | 4 | +0.051 | +0.054 | +0.070 | 317 |
| 1 | 6 | −0.009 | +0.005 | +0.001 | 490 |
| 1 | 8 | −0.071 | −0.038 | −0.043 | 594 |
| 2 | 2 | −0.104 | −0.112 | −0.141 | 23 |
| 2 | 4 | −0.086 | −0.072 | −0.099 | 61 |
| 2 | 6 | −0.010 | +0.012 | +0.008 | 94 |
| 2 | 8 | −0.005 | +0.021 | −0.015 | 127 |
| 3 | 2 | −0.095 | −0.241 | −0.241 | 3 |
| 3 | 4 | −0.132 | −0.168 | −0.168 | 12 |
| 3 | 6 | −0.039 | −0.091 | −0.091 | 25 |
| 3 | 8 | −0.035 | −0.049 | −0.049 | 35 |
| 4 | 2 | — | — | — | 0 |
| 4 | 4 | +0.153 | +0.153 | +0.153 | 2 |
| 4 | 6 | +0.061 | +0.005 | +0.005 | 9 |
| 4 | 8 | −0.085 | −0.131 | −0.131 | 11 |

The values straddle zero rather than sitting above it. Two things matter more
than the scatter.

**The best-looking rows are the least trustworthy.** Every positive cluster sits
at `n = 1`, the loosest possible swing definition: a bar that merely beats its
two immediate neighbours. Hold `n` at 1 and widen the window from 2 bars to 8,
and that same family becomes the largest loss in the sweep. A result whose sign
depends on a parameter nobody has ever justified is noise.

**Costs decide the marginal cases.** Twenty-three configurations are profitable
before costs; seventeen after. The median cost drag is larger than the median
result, on the tightest-spread instrument available.

**Sweep summary: 48 configurations**

| Measure | Value | Reading |
|---|---:|---|
| Profitable gross | 23 / 48 | Coin-flip before costs |
| Profitable net | 17 / 48 | Costs flip six |
| **With \|t\| > 2** | **1** | **And it is a loss** |
| Expected from noise | 2.4 | Fewer than chance predicts |
| Median net P&L | −82 | |
| Median expectancy | −0.038 R | |

The last two rows are the cleanest statement of the result. Testing 48
configurations against pure noise would be expected to throw up about 2.4 that
clear the conventional significance threshold by luck alone. One did. It was
negative.

---

## Secondary results: two headline statistics dissolve under a baseline

Before the strategy was assembled, two individual concepts were measured against
null models. Both produce impressive-sounding numbers that turn out to be exactly
what an unremarkable baseline already predicts.

### Fair value gaps fill 74% of the time, and so does everything else

13,156 gaps were detected; 73.9% filled within the session. That is the figure
usually cited as evidence the pattern means something. Two independent baselines
say it does not.

Shuffling the bars within each session (preserving every bar's shape and every
jump between bars, destroying only their order) produces gaps that fill at
73.4% ± 0.3%. Randomly reordered price data yields the same statistic.
Separately, asking the same question of an arbitrary bar at the same time of day
gives 79.7%: real gaps fill *less* often than an arbitrary price level, which is
what geometry alone predicts, since a gap only forms after a decisive move.

### The Fibonacci retracement band is not special

The optimal-entry band at 62–79% retracement is entered 46.4% of the time.
Holding the band width fixed at 0.17 and sliding only its depth produces a
smooth, monotone curve:

| Band | Entered |
|---|---:|
| 0.10–0.27 | 69.1% |
| 0.20–0.37 | 67.5% |
| 0.30–0.47 | 64.3% |
| 0.40–0.57 | 60.4% |
| 0.50–0.67 | 54.9% |
| **0.62–0.79** | **46.4%** ← conventional OTE |
| 0.70–0.87 | 40.1% |
| 0.80–0.97 | 30.1% |

The conventional band sits on that curve, not above it. Its hit rate is fully
explained by how deep it sits: deeper retracements happen less often.

### Killzones show no concentration

Gap density is flat across the trading day (0.175–0.226 per bar). Sliding the
most-cited window thirty minutes in either direction moves its concentration
ratio between 1.05 and 1.11, with the quoted boundary sitting at 1.08. The
boundary is doing no work.

---

## What this does not show

A negative result is worth only as much as its limits are stated.

- **It does not prove ICT cannot work.** It shows that one mechanized reading of
  it, on one instrument, at one timeframe, over one decade, did not clear costs.
  A discretionary trader reading context off a chart is doing something this code
  does not do.
- **The sample is thin.** The complete setup fires roughly monthly. With 61
  trades at baseline, only a large edge would have been detectable. A small real
  edge would be invisible here, and so would a small real loss.
- **The mechanization involved judgement calls.** "Prevailing structure" has no
  mechanical definition in the source material; the rule used here is the
  narrowest one that needs no second definition. Someone reading structure off a
  chart would disagree with it sometimes.
- **The timeframe limits resolution.** Fifteen-minute bars cannot order the high
  and the low within a bar. Ambiguous trades were resolved conservatively, which
  is an assumption rather than a measurement.
- **Index futures were not tested.** The data source carries no futures, so
  several concepts leaning on the overnight session (which an ETF simply does
  not have) could not be examined at all.

---

## The one loose thread, left deliberately loose

At baseline, of 61 trades: 44 exited at the session close, 15 at the stop, and
only **2 reached the 2R target**.

The target is almost never reachable in the bars remaining after entry. That may
mean the exit rule is mis-specified rather than the signal being worthless: a 2R
target on 15-minute bars with a hard session-close exit may simply not have room
to resolve.

It was not pursued. Testing exit variants until one produces a profit is
precisely the process that generates convincing backtests of nothing. If it is
investigated, the honest form is to commit in advance to a single alternative,
run it once, and report whatever comes back.

---

## Reproducing this

The result rests on the code being correct, so the code is tested rather than
trusted. 256 tests run without network access against hand-built fixtures. Every
load-bearing invariant was additionally verified by mutation: deliberately
breaking it and confirming the suite fails.

That practice caught a real defect in this project's own sweep detection. A
lookahead guard appeared harmless by argument; a mutation showed it silently
changed 68 of 3,059 sweeps, in both directions. The written reasoning had been
wrong, and only the mechanical check found it.

```
.venv/Scripts/python.exe scripts/backfill.py   SPY 15   # rebuild the cache
.venv/Scripts/python.exe scripts/backtest.py   SPY 15   # strategy + sweep
.venv/Scripts/python.exe scripts/null_test.py  SPY 15   # null models
.venv/Scripts/python.exe -m pytest -q                   # 256 tests
```

---

*Research tool. Paper only. No part of this project places orders or touches
real capital.*

*In-sample 2016-01-04 to 2022-12-30 · holdout 2023-01-01 onward, unused.*
