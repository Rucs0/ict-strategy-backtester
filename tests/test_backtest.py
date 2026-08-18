"""Backtest engine, costs, split and metrics.

The fill assumptions get the most attention here, because they are what
separates a backtest from a wish. Entry timing, the intrabar tie-break and
the cost sign are each pinned individually.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ictbt.backtest import metrics
from ictbt.backtest.costs import ZERO_COSTS, TransactionCosts
from ictbt.backtest.engine import StrategyConfig, run_backtest
from ictbt.backtest.split import (
    SPLIT_DATE,
    HoldoutViolation,
    require_in_sample,
    split_bars,
)

from .conftest import make_bars, make_session


class TestCosts:
    def test_per_side_is_half_the_spread_plus_slippage(self):
        costs = TransactionCosts(
            spread_per_share=0.02, slippage_per_share=0.005,
            commission_per_share=0.001,
        )
        assert costs.per_side == pytest.approx(0.016)
        assert costs.round_trip == pytest.approx(0.032)

    def test_buys_fill_above_and_sells_below(self):
        """A sign error here would turn a cost into a subsidy."""
        costs = TransactionCosts(spread_per_share=0.02, slippage_per_share=0.0,
                                 commission_per_share=0.0)

        assert costs.fill_price(100.0, side="buy") == pytest.approx(100.01)
        assert costs.fill_price(100.0, side="sell") == pytest.approx(99.99)

    def test_invalid_side_is_rejected(self):
        with pytest.raises(ValueError, match="side"):
            TransactionCosts().fill_price(100.0, side="hold")

    def test_negative_costs_are_rejected(self):
        with pytest.raises(ValueError, match="spread_per_share"):
            TransactionCosts(spread_per_share=-0.01)

    def test_zero_costs_are_free(self):
        assert ZERO_COSTS.round_trip == 0.0
        assert ZERO_COSTS.fill_price(100.0, side="buy") == 100.0


class TestSplit:
    def test_the_boundary_belongs_to_the_holdout(self):
        bars = pd.concat(
            [
                make_session("2022-12-30", minutes=15, bars=4),
                make_session("2023-01-03", minutes=15, bars=4),
            ]
        )
        split = split_bars(bars)

        assert split.in_sample.index[-1].year == 2022
        assert split.out_of_sample.index[0].year == 2023

    def test_the_split_date_is_pinned(self):
        """Moving this to improve a result would defeat its purpose."""
        assert SPLIT_DATE.date().isoformat() == "2023-01-01"

    def test_in_sample_data_passes_the_guard(self):
        require_in_sample(make_session("2022-06-15", minutes=15, bars=4))

    def test_holdout_data_trips_the_guard(self):
        with pytest.raises(HoldoutViolation, match="holdout boundary"):
            require_in_sample(make_session("2024-06-17", minutes=15, bars=4))

    def test_the_guard_can_be_bypassed_explicitly(self):
        require_in_sample(
            make_session("2024-06-17", minutes=15, bars=4), allow=True
        )


TIMES = [
    "09:30", "09:45", "10:00", "10:15", "10:30", "10:45", "11:00",
    "11:15", "11:30", "11:45", "12:00", "12:15", "12:30", "12:45",
]

# A complete bullish setup. It needs a prior *bearish* break, because an MSS
# is by definition a break against prevailing structure â€” so the first break
# of any session is classified BOS and can never trigger a trade. That is a
# real constraint of the strategy, not a quirk of this fixture, and it cuts
# the tradeable population noticeably.
#
#   10:00  swing low 100 forms, confirmed 10:30
#   10:45  closes at 96, below 100  -> bearish break, structure now bearish
#          and its own low of 95 becomes a swing low, confirmed 11:15
#   11:00  swing high 110 forms, confirmed 11:30
#   11:30  wicks to 94 through the 95 low and closes back above -> sweep
#   11:45  closes at 114, above 110 -> bullish break against bearish
#          structure -> MSS
#   12:00  entry at the open
#
# Entry 114, stop at the sweep low of 94, so risk is 20 and a 2R target
# sits at 154.
BASE = [
    (105.0, 108.0, 103.0, 106.0),   # 09:30
    (106.0, 109.0, 102.0, 104.0),   # 09:45
    (104.0, 109.0, 100.0, 108.0),   # 10:00  swing low 100
    (108.0, 109.0, 104.0, 106.0),   # 10:15
    (106.0, 108.0, 105.0, 107.0),   # 10:30  confirms low 100
    (107.0, 107.0, 95.0, 96.0),     # 10:45  bearish break; swing low 95
    (98.0, 110.0, 97.0, 109.0),     # 11:00  swing high 110
    (105.0, 106.0, 98.0, 100.0),    # 11:15  confirms low 95
    (100.0, 105.0, 94.0, 100.0),    # 11:30  sweeps 95; confirms high 110
    (101.0, 115.0, 100.0, 114.0),   # 11:45  bullish break -> MSS
]


#: Neutral bars: no new swing (their extremes tie), no break, no sweep.
FILLER = (100.0, 104.0, 96.0, 101.0)


def setup_bars(tail: list[tuple[float, float, float, float]]) -> pd.DataFrame:
    rows = BASE + tail
    return make_bars(
        [(t, o, h, low, c) for t, (o, h, low, c) in zip(TIMES, rows)],
        date="2022-03-09",
    )


def delayed_setup_bars(
    tail: list[tuple[float, float, float, float]]
) -> pd.DataFrame:
    """Same setup with two neutral bars between the sweep and the MSS.

    Pushes the gap from one bar to three, so the window parameter has
    something to actually exclude.
    """
    rows = BASE[:9] + [FILLER, FILLER] + [BASE[9]] + tail
    return make_bars(
        [(t, o, h, low, c) for t, (o, h, low, c) in zip(TIMES, rows)],
        date="2022-03-09",
    )


class TestEntryTiming:
    def test_entry_is_at_the_bar_after_the_signal(self):
        """Filling at the signal bar's close would use unavailable information.

        The MSS prints at 11:45; the earliest price obtainable is the 12:00
        open of 114.
        """
        bars = setup_bars([(114.0, 155.0, 113.0, 154.0)])
        trades = run_backtest(bars, StrategyConfig(), ZERO_COSTS)

        assert len(trades) == 1
        assert trades.index[0].strftime("%H:%M") == "12:00"
        assert trades.iloc[0]["entry_price"] == pytest.approx(114.0)

    def test_no_trade_when_the_signal_lands_on_the_last_bar(self):
        """An MSS with no following bar was never tradeable."""
        bars = setup_bars([])
        assert run_backtest(bars, StrategyConfig(), ZERO_COSTS).empty

    def test_the_window_between_sweep_and_mss_is_enforced(self):
        """Sweep at 11:30, MSS three bars later. A window of 2 excludes it."""
        bars = delayed_setup_bars([(114.0, 155.0, 113.0, 154.0)])

        assert len(
            run_backtest(bars, StrategyConfig(sweep_to_mss_bars=3), ZERO_COSTS)
        ) == 1
        assert run_backtest(
            bars, StrategyConfig(sweep_to_mss_bars=2), ZERO_COSTS
        ).empty


class TestExits:
    def test_a_target_hit_exits_at_the_target(self):
        """Entry 114, stop at the sweep low of 94, risk 20, 2R target 154."""
        bars = setup_bars([(114.0, 155.0, 113.0, 154.0)])
        trade = run_backtest(bars, StrategyConfig(), ZERO_COSTS).iloc[0]

        assert trade["exit_reason"] == "target"
        assert trade["stop"] == pytest.approx(94.0)
        assert trade["target"] == pytest.approx(154.0)
        assert trade["r_multiple"] == pytest.approx(2.0)

    def test_a_stop_hit_exits_at_the_stop(self):
        bars = setup_bars([(114.0, 115.0, 93.0, 95.0)])
        trade = run_backtest(bars, StrategyConfig(), ZERO_COSTS).iloc[0]

        assert trade["exit_reason"] == "stop"
        assert trade["r_multiple"] == pytest.approx(-1.0)

    def test_an_open_trade_exits_at_the_session_close(self):
        bars = setup_bars([(114.0, 116.0, 113.0, 115.0)])
        trade = run_backtest(bars, StrategyConfig(), ZERO_COSTS).iloc[0]

        assert trade["exit_reason"] == "session_close"
        assert trade["exit_price"] == pytest.approx(115.0)

    def test_a_position_is_never_carried_overnight(self):
        monday = setup_bars([(114.0, 116.0, 113.0, 115.0)])
        tuesday = make_bars(
            [("09:30", 115.0, 160.0, 114.0, 159.0)], date="2022-03-10"
        )
        trade = run_backtest(
            pd.concat([monday, tuesday]), StrategyConfig(), ZERO_COSTS
        ).iloc[0]

        assert trade["exit_at"].date().isoformat() == "2022-03-09"
        assert trade["exit_reason"] == "session_close"


class TestIntrabarAmbiguity:
    """A bar containing both stop and target is undecidable from OHLC."""

    def test_the_conservative_reading_takes_the_stop(self):
        bars = setup_bars([(114.0, 155.0, 93.0, 120.0)])
        trade = run_backtest(bars, StrategyConfig(), ZERO_COSTS).iloc[0]

        assert trade["exit_reason"] == "stop"
        assert trade["r_multiple"] == pytest.approx(-1.0)

    def test_the_optimistic_reading_differs_by_three_r(self):
        """Why the assumption matters: same bars, 3R apart."""
        bars = setup_bars([(114.0, 155.0, 93.0, 120.0)])

        pessimistic = run_backtest(bars, StrategyConfig(), ZERO_COSTS).iloc[0]
        optimistic = run_backtest(
            bars, StrategyConfig(optimistic_intrabar=True), ZERO_COSTS
        ).iloc[0]

        assert optimistic["exit_reason"] == "target"
        gap = optimistic["r_multiple"] - pessimistic["r_multiple"]
        assert gap == pytest.approx(3.0)


class TestCostApplication:
    def test_costs_reduce_a_winning_trade(self):
        bars = setup_bars([(114.0, 155.0, 113.0, 154.0)])
        costs = TransactionCosts(spread_per_share=0.02, slippage_per_share=0.0,
                                 commission_per_share=0.0)

        free = run_backtest(bars, StrategyConfig(), ZERO_COSTS).iloc[0]
        charged = run_backtest(bars, StrategyConfig(), costs).iloc[0]

        assert charged["net_pnl"] < free["net_pnl"]
        assert charged["cost"] > 0

    def test_costs_worsen_a_losing_trade(self):
        """A cost that helped a loser would be a sign error."""
        bars = setup_bars([(114.0, 115.0, 93.0, 95.0)])
        costs = TransactionCosts(spread_per_share=0.02, slippage_per_share=0.0,
                                 commission_per_share=0.0)

        free = run_backtest(bars, StrategyConfig(), ZERO_COSTS).iloc[0]
        charged = run_backtest(bars, StrategyConfig(), costs).iloc[0]

        assert charged["net_pnl"] < free["net_pnl"]
        assert charged["cost"] > 0

    def test_cost_equals_round_trip_times_shares(self):
        bars = setup_bars([(114.0, 155.0, 113.0, 154.0)])
        costs = TransactionCosts(spread_per_share=0.02, slippage_per_share=0.005,
                                 commission_per_share=0.0)
        trade = run_backtest(bars, StrategyConfig(), costs).iloc[0]

        assert trade["cost"] == pytest.approx(
            costs.round_trip * trade["shares"]
        )


class TestPositionSizing:
    def test_shares_are_sized_so_a_stop_loses_exactly_the_risk(self):
        bars = setup_bars([(114.0, 115.0, 93.0, 95.0)])
        config = StrategyConfig(risk_per_trade=250.0)
        trade = run_backtest(bars, config, ZERO_COSTS).iloc[0]

        assert trade["net_pnl"] == pytest.approx(-250.0)

    def test_a_wider_stop_buys_fewer_shares(self):
        bars = setup_bars([(114.0, 155.0, 113.0, 154.0)])

        tight = run_backtest(bars, StrategyConfig(), ZERO_COSTS).iloc[0]
        wide = run_backtest(
            bars, StrategyConfig(stop_buffer=10.0), ZERO_COSTS
        ).iloc[0]

        assert wide["shares"] < tight["shares"]

    def test_invalid_config_is_rejected(self):
        with pytest.raises(ValueError, match="target_r"):
            StrategyConfig(target_r=0.0)
        with pytest.raises(ValueError, match="risk_per_trade"):
            StrategyConfig(risk_per_trade=0.0)


class TestSignalCache:
    def test_a_cached_run_matches_a_cold_run(self):
        """The cache is a memo, not an approximation."""
        from ictbt.backtest.engine import SignalCache

        bars = setup_bars([(114.0, 155.0, 113.0, 154.0)])
        cache = SignalCache()

        cold = run_backtest(bars, StrategyConfig(), ZERO_COSTS)
        warm = run_backtest(bars, StrategyConfig(), ZERO_COSTS, cache=cache)
        again = run_backtest(bars, StrategyConfig(), ZERO_COSTS, cache=cache)

        pd.testing.assert_frame_equal(cold, warm)
        pd.testing.assert_frame_equal(warm, again)

    def test_a_shared_cache_still_distinguishes_configs(self):
        """Keyed on the parameters that affect the scans, so n is respected."""
        from ictbt.backtest.engine import SignalCache

        bars = setup_bars([(114.0, 155.0, 113.0, 154.0)])
        cache = SignalCache()

        at_2 = run_backtest(bars, StrategyConfig(n=2), ZERO_COSTS, cache=cache)
        at_4 = run_backtest(bars, StrategyConfig(n=4), ZERO_COSTS, cache=cache)

        assert len(at_2) == 1
        assert at_4.empty


class TestMetrics:
    def _ledger(self, pnls: list[float], cost: float = 1.0) -> pd.DataFrame:
        dates = pd.date_range(
            "2022-03-09", periods=len(pnls), freq="B", tz="America/New_York"
        )
        return pd.DataFrame(
            {
                "exit_at": dates,
                "gross_pnl": [p + cost for p in pnls],
                "cost": [cost] * len(pnls),
                "net_pnl": pnls,
            },
            index=pd.DatetimeIndex(dates, name="entry_at"),
        )

    def test_win_rate_and_totals(self):
        perf = metrics.evaluate(
            self._ledger([100.0, -50.0, 100.0, -50.0]),
            capital=10_000.0, risk_per_trade=50.0,
        )

        assert perf.trade_count == 4
        assert perf.win_rate == pytest.approx(0.5)
        assert perf.net_pnl == pytest.approx(100.0)
        assert perf.total_cost == pytest.approx(4.0)

    def test_gross_and_net_differ_by_the_cost(self):
        perf = metrics.evaluate(
            self._ledger([100.0, -50.0], cost=2.0),
            capital=10_000.0, risk_per_trade=50.0,
        )
        assert perf.gross_pnl - perf.net_pnl == pytest.approx(perf.total_cost)

    def test_max_drawdown_measures_peak_to_trough(self):
        perf = metrics.evaluate(
            self._ledger([100.0, -30.0, -40.0, 20.0]),
            capital=10_000.0, risk_per_trade=50.0,
        )
        assert perf.max_drawdown == pytest.approx(70.0)

    def test_t_stat_is_near_zero_for_a_coin_flip(self):
        perf = metrics.evaluate(
            self._ledger([50.0, -50.0] * 20),
            capital=10_000.0, risk_per_trade=50.0,
        )
        assert abs(perf.t_stat) < 0.5

    def test_an_empty_ledger_reports_no_trades(self):
        perf = metrics.evaluate(
            pd.DataFrame({"net_pnl": [], "gross_pnl": [], "cost": []}),
            capital=10_000.0, risk_per_trade=50.0,
        )
        assert perf.trade_count == 0
        assert "no trades" in perf.summary()

