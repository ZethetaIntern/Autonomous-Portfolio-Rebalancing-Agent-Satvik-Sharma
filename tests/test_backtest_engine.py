"""Unit tests for Backtesting Engine, Performance Analyser, and Historical Scenario Runner."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.backtesting.performance_analyser import PerformanceAnalyser
from src.backtesting.backtest_engine import BacktestEngine
from src.backtesting.strategy_comparator import StrategyComparator
from src.backtesting.scenario_runner import ScenarioRunner


class TestPerformanceAnalyser:
    """Test suite for institutional risk and return attribution metrics."""

    @pytest.fixture
    def analyser(self) -> PerformanceAnalyser:
        return PerformanceAnalyser(risk_free_rate=0.065)

    def test_basic_metrics_computation(self, analyser: PerformanceAnalyser):
        # 100 days of constant 0.1% daily returns
        returns = np.full(100, 0.001)
        res = analyser.compute_metrics(returns)

        assert res["annualized_return"] == pytest.approx(0.252, rel=1e-2)
        assert res["annualized_volatility"] == pytest.approx(0.0, abs=1e-5)
        assert res["max_drawdown_pct"] == 0.0

    def test_drawdown_and_sharpe_with_volatility(self, analyser: PerformanceAnalyser):
        np.random.seed(42)
        # Moderate daily returns with 15% annualized vol
        daily_rets = np.random.normal(0.0005, 0.01, size=252)
        res = analyser.compute_metrics(daily_rets)

        assert res["annualized_volatility"] > 0.10
        assert res["max_drawdown_pct"] < 0.0  # drawdown is negative percentage
        assert isinstance(res["sharpe_ratio"], float)
        assert isinstance(res["sortino_ratio"], float)

    def test_tracking_error_and_information_ratio(self, analyser: PerformanceAnalyser):
        np.random.seed(42)
        portfolio_rets = np.random.normal(0.0006, 0.01, size=252)
        bench_rets = np.random.normal(0.0004, 0.01, size=252)

        res = analyser.compute_metrics(portfolio_rets, benchmark_returns=bench_rets)
        assert "tracking_error" in res
        assert "information_ratio" in res
        assert res["tracking_error"] > 0.0

    def test_comprehensive_metrics(self, analyser: PerformanceAnalyser):
        wealth = [10_000_000.0, 10_050_000.0, 9_980_000.0, 10_120_000.0, 10_250_000.0]
        res = analyser.compute_comprehensive_metrics(
            portfolio_values=wealth,
            turnover_history=[0.05, 0.03],
            transaction_costs_inr=15_000.0,
            tax_shield_inr=25_000.0,
        )

        assert res["initial_capital_inr"] == 10_000_000.0
        assert res["final_capital_inr"] == 10_250_000.0
        assert res["total_return_pct"] == 2.5
        assert res["tax_alpha_bps"] == pytest.approx(25.0, rel=1e-2)  # 25000 / 10M * 10000 = 25 bps
        assert res["cost_drag_bps"] == pytest.approx(15.0, rel=1e-2)


class TestBacktestEngine:
    """Test suite for event-driven multi-strategy simulation engine."""

    @pytest.fixture
    def engine(self) -> BacktestEngine:
        return BacktestEngine(initial_capital=10_000_000.0, risk_free_rate=0.065)

    def test_synthetic_price_history_generation(self, engine: BacktestEngine):
        df = engine.generate_synthetic_price_history(n_days=100, seed=42)
        assert len(df) >= 100
        assert "NIFTY_50_EQUITY" in df.columns
        assert "G_SEC_BONDS" in df.columns or "G_SEC_10Y_BOND" in df.columns
        assert (df.values > 0.0).all()

    def test_buy_and_hold_simulation(self, engine: BacktestEngine):
        # Short 30-day simulation
        price_df = engine.generate_synthetic_price_history(n_days=30, seed=123)
        res = engine.run_simulation(price_history=price_df, rebalance_rule="BUY_AND_HOLD")

        assert res["strategy"] == "BUY_AND_HOLD"
        assert res["rebalances_executed"] == 0
        assert len(res["portfolio_values"]) == len(price_df)
        assert res["total_costs_inr"] == 0.0

    def test_calendar_quarterly_simulation(self, engine: BacktestEngine):
        # 140 days allows 2 quarterly rebalances (Day 63 and Day 126)
        price_df = engine.generate_synthetic_price_history(n_days=140, seed=123)
        res = engine.run_simulation(price_history=price_df, rebalance_rule="CALENDAR_QUARTERLY")

        assert res["strategy"] == "CALENDAR_QUARTERLY"
        assert res["rebalances_executed"] == 2
        assert len(res["rebalance_events"]) == 2
        assert res["total_costs_inr"] > 0.0

    def test_ai_agent_simulation(self, engine: BacktestEngine):
        # 65 days with March tax-loss harvesting trigger on Day 60
        price_df = engine.generate_synthetic_price_history(n_days=65, seed=123)
        res = engine.run_simulation(
            price_history=price_df,
            rebalance_rule="AI_AGENT",
            turnover_budget=0.15,
        )

        assert res["strategy"] == "AI_AGENT"
        assert len(res["portfolio_values"]) == len(price_df)
        assert res["final_capital"] > 0.0
        assert "performance_metrics" in res


class TestStrategyComparator:
    """Test suite for comparative multi-strategy backtest tournaments."""

    def test_strategy_comparison(self):
        comparator = StrategyComparator()
        # 70 days run
        engine = BacktestEngine(initial_capital=5_000_000.0)
        price_df = engine.generate_synthetic_price_history(n_days=70, seed=42)

        comparison = comparator.run_comparison(
            price_history=price_df,
            seed=42,
        )

        assert "winner" in comparison
        assert len(comparison["ranked_strategies"]) == 4
        strategies_tested = {s["strategy"] for s in comparison["ranked_strategies"]}
        assert strategies_tested == {"AI_AGENT", "CALENDAR_QUARTERLY", "THRESHOLD_ONLY", "BUY_AND_HOLD"}
        assert "sharpe_improvement" in comparison
        assert "tax_drag_reduction_bps" in comparison


class TestScenarioRunner:
    """Test suite for crisis stress tests and circuit breaker integration."""

    def test_covid_crash_scenario(self):
        runner = ScenarioRunner()
        res = runner.run_crisis_scenario("COVID_CRASH_2020", test_kill_switch=True)

        assert res["scenario"] == "COVID_CRASH_2020"
        assert res["simulated_peak_vix"] == 75.0
        assert res["kill_switch_tripped"] is True
        assert "AUTOMATED TRIP" in res["kill_switch_reason"]
        assert res["agent_mitigation_score"] >= 75.0
        assert res["simulated_days"] == 60

    def test_flash_crash_scenario(self):
        runner = ScenarioRunner()
        res = runner.run_crisis_scenario("FLASH_CRASH", test_kill_switch=True)

        assert res["scenario"] == "FLASH_CRASH"
        assert res["kill_switch_tripped"] is True
        assert res["simulated_days"] == 10
