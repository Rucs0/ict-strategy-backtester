"""The strategy and its event loop.

**The strategy.** The canonical ICT day-trade sequence, mechanized:

1. Price sweeps a prior confirmed swing point — liquidity taken.
2. Within `sweep_to_mss_bars`, structure shifts the same way — an MSS.
3. Enter in that direction.
4. Stop beyond the sweep's extreme; target a fixed multiple of that risk.
5. Flat by the session close regardless.

Optional filters (killzone, OTE, fair value gap) are off by default, so the
base case is the sequence alone and each filter's contribution is measurable
rather than assumed.

**Fill assumptions, which decide whether any of this means anything.**

- *Entry is at the next bar's open.* The MSS is only knowable once its bar has
  closed, so the earliest tradeable price is the following bar's open. Filling
  at the signal bar's close is the most common way a backtest quietly grants
  itself information it did not have.
- *A bar that contains both stop and target is resolved as a stop.* OHLC data
  does not record the order of the high and the low within a bar, so this is
  genuinely undecidable. The optimistic reading inflates results badly — on a
  2R target the two assumptions differ by three risk units on every ambiguous
  trade. `optimistic_intrabar` exists to measure that gap, not to be used.
- *Costs are charged on both fills*, at the rate in `TransactionCosts`.
- *Session close exits at the last bar's close*, without slippage beyond the
  standard per-side charge.

**Position sizing is risk-based**: shares are chosen so a stop-out loses
`risk_per_trade`. This normalizes every trade to 1R and makes the R multiples
comparable across instruments and volatility regimes. Fractional shares are
permitted, matching Alpaca; whole-share rounding would perturb small accounts
slightly and is not modelled.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..calendar import rth_only, session_date
from ..schema import validate_bars
from ..signals import (
    find_fvgs,
    find_ote_zones,
    find_structure_events,
    find_sweeps,
    in_killzone,
)
from .costs import TransactionCosts

TRADE_COLUMNS = (
    "exit_at",
    "direction",
    "entry_price",
    "exit_price",
    "shares",
    "stop",
    "target",
    "gross_pnl",
    "cost",
    "net_pnl",
    "r_multiple",
    "exit_reason",
    "sweep_at",
    "mss_at",
)


@dataclass(frozen=True)
class StrategyConfig:
    """Every knob in one place, so a sweep is a loop over configs.

    Defaults are the plainest reading of the setup, not a tuned combination.
    """

    n: int = 2
    sweep_to_mss_bars: int = 4
    min_penetration: float = 0.0
    target_r: float = 2.0
    stop_buffer: float = 0.0
    killzones: tuple[str, ...] = ()
    require_ote: bool = False
    require_fvg: bool = False
    risk_per_trade: float = 100.0
    capital: float = 10_000.0
    optimistic_intrabar: bool = False

    def __post_init__(self) -> None:
        if self.n < 1:
            raise ValueError(f"n must be >= 1, got {self.n}")
        if self.sweep_to_mss_bars < 1:
            raise ValueError(
                f"sweep_to_mss_bars must be >= 1, got {self.sweep_to_mss_bars}"
            )
        if self.target_r <= 0:
            raise ValueError(f"target_r must be > 0, got {self.target_r}")
        if self.risk_per_trade <= 0:
            raise ValueError(f"risk_per_trade must be > 0, got {self.risk_per_trade}")

    def label(self) -> str:
        parts = [f"n={self.n}", f"win={self.sweep_to_mss_bars}", f"R={self.target_r}"]
        if self.min_penetration:
            parts.append(f"pen={self.min_penetration}")
        if self.killzones:
            parts.append("kz=" + "+".join(self.killzones))
        if self.require_ote:
            parts.append("ote")
        if self.require_fvg:
            parts.append("fvg")
        return " ".join(parts)


class SignalCache:
    """Memoizes the stateful scans across configurations in a sweep.

    Sweeps and structure events depend only on `n` and `min_penetration`, not
    on the window, target or filters. Recomputing them for every config makes
    a sweep quadratically wasteful for no benefit — a 48-config sweep over ten
    years spends almost all of its time redoing identical work.

    Keyed on the parameters that actually affect the result, so this is a
    memo, not an approximation: a cached run and a cold run return the same
    trades.
    """

    def __init__(self) -> None:
        self._sweeps: dict[tuple, pd.DataFrame] = {}
        self._events: dict[tuple, pd.DataFrame] = {}
        self._ote: dict[tuple, dict] = {}
        self._fvg: dict | None = None

    def sweeps(self, df: pd.DataFrame, n: int, min_penetration: float):
        key = (n, min_penetration)
        if key not in self._sweeps:
            self._sweeps[key] = find_sweeps(
                df, n=n, rth=True, min_penetration=min_penetration, validate=False
            )
        return self._sweeps[key]

    def events(self, df: pd.DataFrame, n: int):
        if n not in self._events:
            self._events[n] = find_structure_events(
                df, n=n, rth=True, validate=False
            )
        return self._events[n]

    def ote(self, df: pd.DataFrame, n: int) -> dict:
        if n not in self._ote:
            zones = find_ote_zones(df, n=n, rth=True, validate=False)
            self._ote[n] = {
                ts: True for ts in zones["entered_at"].dropna()
            }
        return self._ote[n]

    def fvg(self, df: pd.DataFrame) -> dict:
        if self._fvg is None:
            gaps = find_fvgs(df, rth=True, validate=False)
            self._fvg = {ts: True for ts in gaps["tradeable_from"].dropna()}
        return self._fvg


def run_backtest(
    df: pd.DataFrame,
    config: StrategyConfig,
    costs: TransactionCosts,
    *,
    validate: bool = True,
    cache: SignalCache | None = None,
) -> pd.DataFrame:
    """Run `config` over `df` and return the trade ledger.

    Pass a shared `SignalCache` when sweeping parameters; results are
    identical either way.
    """
    if validate:
        validate_bars(df, name="backtest input")

    frame = rth_only(df)
    if len(frame) < 3:
        return _empty_trades()

    setups = _find_setups(df, frame, config, cache or SignalCache())
    if not setups:
        return _empty_trades()

    # Precomputed once: session_date builds a Series per call, which is
    # ruinous inside a per-bar exit loop.
    dates = session_date(frame.index).to_numpy()

    trades = [
        trade
        for setup in setups
        if (trade := _simulate(frame, dates, setup, config, costs)) is not None
    ]
    if not trades:
        return _empty_trades()

    out = pd.DataFrame.from_records(trades).set_index("entry_at")
    out.index.name = "entry_at"
    return out.sort_index()


def _find_setups(
    df: pd.DataFrame,
    frame: pd.DataFrame,
    config: StrategyConfig,
    cache: SignalCache,
) -> list[dict]:
    """Locate sweep-then-MSS pairs that survive the configured filters."""
    sweeps = cache.sweeps(df, config.n, config.min_penetration)
    events = cache.events(df, config.n)
    if sweeps.empty or events.empty:
        return []

    mss = events[events["event"] == "mss"]
    if mss.empty:
        return []

    positions = {ts: i for i, ts in enumerate(frame.index)}
    sweep_day = session_date(sweeps.index).to_numpy()
    mss_day = session_date(mss.index).to_numpy()
    mss_pos = np.array([positions.get(ts, -1) for ts in mss.index])
    mss_dir = mss["direction"].to_numpy()

    ote_ok = cache.ote(df, config.n) if config.require_ote else None
    fvg_ok = cache.fvg(df) if config.require_fvg else None

    setups: list[dict] = []
    used_mss: set[int] = set()

    for k, (sweep_at, sweep) in enumerate(sweeps.iterrows()):
        start = positions.get(sweep_at)
        if start is None:
            continue
        window = (
            (mss_day == sweep_day[k])
            & (mss_dir == sweep["direction"])
            & (mss_pos > start)
            & (mss_pos <= start + config.sweep_to_mss_bars)
        )
        candidates = np.flatnonzero(window)
        if not len(candidates):
            continue

        pick = candidates[0]
        if pick in used_mss:
            continue  # one trade per structure shift

        entry_pos = mss_pos[pick] + 1
        if entry_pos >= len(frame):
            continue  # MSS on the last bar of the session: never tradeable
        if session_date(frame.index[[entry_pos]]).iloc[0] != sweep_day[k]:
            continue  # entry would fall into the next session

        entry_at = frame.index[entry_pos]
        if config.killzones and not _in_any_killzone(entry_at, config.killzones):  # noqa: E501
            continue
        if ote_ok is not None and not ote_ok.get(entry_at, False):
            continue
        if fvg_ok is not None and not fvg_ok.get(entry_at, False):
            continue

        used_mss.add(pick)
        setups.append(
            {
                "direction": sweep["direction"],
                "sweep_at": sweep_at,
                "sweep_extreme": _sweep_extreme(frame, start, sweep["direction"]),
                "mss_at": mss.index[pick],
                "entry_pos": entry_pos,
            }
        )

    return setups


def _sweep_extreme(frame: pd.DataFrame, pos: int, direction: str) -> float:
    """The wick extreme of the sweeping bar — where the stop goes behind."""
    if direction == "bullish":
        return float(frame["low"].to_numpy()[pos])
    return float(frame["high"].to_numpy()[pos])


def _simulate(
    frame: pd.DataFrame,
    dates: np.ndarray,
    setup: dict,
    config: StrategyConfig,
    costs: TransactionCosts,
) -> dict | None:
    """Walk one trade from entry to exit."""
    highs = frame["high"].to_numpy(dtype="float64")
    lows = frame["low"].to_numpy(dtype="float64")
    opens = frame["open"].to_numpy(dtype="float64")
    closes = frame["close"].to_numpy(dtype="float64")
    index = frame.index

    entry_pos = setup["entry_pos"]
    long = setup["direction"] == "bullish"
    reference = opens[entry_pos]

    stop = (
        setup["sweep_extreme"] - config.stop_buffer
        if long
        else setup["sweep_extreme"] + config.stop_buffer
    )
    risk_per_share = (reference - stop) if long else (stop - reference)
    if risk_per_share <= 0:
        # Price already through the stop at entry; there is no trade to take.
        return None

    entry_price = costs.fill_price(reference, side="buy" if long else "sell")
    shares = config.risk_per_trade / risk_per_share
    target = (
        reference + config.target_r * risk_per_share
        if long
        else reference - config.target_r * risk_per_share
    )

    day = dates[entry_pos]
    exit_pos, exit_reference, reason = _walk(
        highs, lows, closes, dates, entry_pos, day, stop, target, long, config
    )

    exit_price = costs.fill_price(exit_reference, side="sell" if long else "buy")
    direction_sign = 1.0 if long else -1.0

    gross = direction_sign * (exit_reference - reference) * shares
    net = direction_sign * (exit_price - entry_price) * shares
    return {
        "entry_at": index[entry_pos],
        "exit_at": index[exit_pos],
        "direction": setup["direction"],
        "entry_price": entry_price,
        "exit_price": exit_price,
        "shares": shares,
        "stop": stop,
        "target": target,
        "gross_pnl": gross,
        "cost": gross - net,
        "net_pnl": net,
        "r_multiple": net / config.risk_per_trade,
        "exit_reason": reason,
        "sweep_at": setup["sweep_at"],
        "mss_at": setup["mss_at"],
    }


def _walk(
    highs, lows, closes, dates, entry_pos, day, stop, target, long, config
) -> tuple[int, float, str]:
    """Find the exit bar, price and reason."""
    for pos in range(entry_pos, len(dates)):
        if dates[pos] != day:
            break

        hit_stop = lows[pos] <= stop if long else highs[pos] >= stop
        hit_target = highs[pos] >= target if long else lows[pos] <= target

        if hit_stop and hit_target:
            # Undecidable from OHLC. The conservative reading is the stop.
            if config.optimistic_intrabar:
                return pos, target, "target"
            return pos, stop, "stop"
        if hit_stop:
            return pos, stop, "stop"
        if hit_target:
            return pos, target, "target"

        last_of_session = pos + 1 >= len(dates) or dates[pos + 1] != day
        if last_of_session:
            return pos, closes[pos], "session_close"

    return entry_pos, closes[entry_pos], "session_close"


def _in_any_killzone(timestamp: pd.Timestamp, names: tuple[str, ...]) -> bool:
    single = pd.DatetimeIndex([timestamp])
    return any(bool(in_killzone(single, name).iloc[0]) for name in names)


def _empty_trades() -> pd.DataFrame:
    from ..schema import NY

    stamp = f"datetime64[ns, {NY}]"
    return pd.DataFrame(
        {
            "exit_at": pd.Series(dtype=stamp),
            "direction": pd.Series(dtype="object"),
            "entry_price": pd.Series(dtype="float64"),
            "exit_price": pd.Series(dtype="float64"),
            "shares": pd.Series(dtype="float64"),
            "stop": pd.Series(dtype="float64"),
            "target": pd.Series(dtype="float64"),
            "gross_pnl": pd.Series(dtype="float64"),
            "cost": pd.Series(dtype="float64"),
            "net_pnl": pd.Series(dtype="float64"),
            "r_multiple": pd.Series(dtype="float64"),
            "exit_reason": pd.Series(dtype="object"),
            "sweep_at": pd.Series(dtype=stamp),
            "mss_at": pd.Series(dtype=stamp),
        },
        index=pd.DatetimeIndex([], tz=NY, name="entry_at"),
    )
