"""Strategy Comparator contrasting Rebalancing Paradigms for WealthPilot AI.

Executes parallel or comparative backtests across 4 strategies:
1. 'AI_AGENT': Dynamic multi-metric trigger + QP optimization + Tax-loss harvesting
2. 'CALENDAR_QUARTERLY': Legacy quarterly calendar baseline
3. 'THRESHOLD_ONLY': Simple fixed 5% corridor rebalancing
4. 'BUY_AND_HOLD': Zero rebalancing baseline

Evaluates Sharpe enhancement, turnover efficiency, transaction cost savings, and tax alpha.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import pandas as pd

from src.backtesting.backtest_engine import BacktestEngine


class StrategyComparator:
    """Runs parallel simulations and performs comparative risk/return/tax attribution."""

    def __init__(self, backtest_engine: Optional[BacktestEngine] = None) -> None:
        self.engine = backtest_engine or BacktestEngine()

    def run_comparison(
        self,
        price_history: Optional[pd.DataFrame] = None,
        target_weights: Optional[Dict[str, float]] = None,
        drift_threshold: float = 0.05,
        turnover_budget: float = 0.20,
        seed: int = 42,
    ) -> Dict[str, Any]:
        """Executes all 4 strategies on the identical price path and produces comparative rankings."""
        if price_history is None:
            price_history = self.engine.generate_synthetic_price_history(n_days=252, seed=seed)

        strategies = ["AI_AGENT", "CALENDAR_QUARTERLY", "THRESHOLD_ONLY", "BUY_AND_HOLD"]
        results: Dict[str, Dict[str, Any]] = {}

        for strat in strategies:
            res = self.engine.run_simulation(
                price_history=price_history,
                target_weights=target_weights,
                rebalance_rule=strat,
                drift_threshold=drift_threshold,
                turnover_budget=turnover_budget,
                seed=seed,
            )
            results[strat] = res

        return self.compare_strategies(results)

    def compare_strategies(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyzes comparative performance metrics across simulated strategies.
        
        Args:
            results: Dict mapping strategy names to backtest result dictionaries.
        """
        summary_table: List[Dict[str, Any]] = []

        for strat, res in results.items():
            metrics = res.get("performance_metrics", {})
            summary_table.append({
                "strategy": strat,
                "total_return_pct": res.get("total_return_pct", metrics.get("total_return_pct", 0.0)),
                "sharpe_ratio": res.get("annualized_sharpe", metrics.get("sharpe_ratio", 0.0)),
                "annualized_volatility": res.get("annualized_volatility", metrics.get("annualized_volatility", 0.0)),
                "max_drawdown_pct": res.get("max_drawdown_pct", metrics.get("max_drawdown_pct", 0.0)),
                "rebalances_count": res.get("rebalances_executed", 0),
                "annualized_turnover_pct": metrics.get("annualized_turnover_pct", 0.0),
                "total_costs_inr": res.get("total_costs_inr", 0.0),
                "tax_alpha_bps": res.get("tax_alpha_bps", 0.0),
            })

        # Sort strategies by Sharpe ratio descending
        ranked_table = sorted(summary_table, key=lambda x: x["sharpe_ratio"], reverse=True)
        winner = ranked_table[0]["strategy"]

        # Benchmarks comparison
        ai_metrics = next((s for s in summary_table if s["strategy"] == "AI_AGENT"), ranked_table[0])
        calendar_metrics = next((s for s in summary_table if s["strategy"] == "CALENDAR_QUARTERLY"), None)
        bh_metrics = next((s for s in summary_table if s["strategy"] == "BUY_AND_HOLD"), None)

        sharpe_imp_vs_calendar = 0.0
        if calendar_metrics:
            sharpe_imp_vs_calendar = round(ai_metrics["sharpe_ratio"] - calendar_metrics["sharpe_ratio"], 3)

        sharpe_imp_vs_bh = 0.0
        if bh_metrics:
            sharpe_imp_vs_bh = round(ai_metrics["sharpe_ratio"] - bh_metrics["sharpe_ratio"], 3)

        cost_savings_vs_calendar_inr = 0.0
        if calendar_metrics:
            cost_savings_vs_calendar_inr = round(calendar_metrics["total_costs_inr"] - ai_metrics["total_costs_inr"], 2)

        return {
            "winner": winner,
            "ranked_strategies": ranked_table,
            "sharpe_improvement": sharpe_imp_vs_calendar,
            "sharpe_improvement_vs_buy_and_hold": sharpe_imp_vs_bh,
            "cost_savings_vs_calendar_inr": cost_savings_vs_calendar_inr,
            "tax_drag_reduction_bps": ai_metrics.get("tax_alpha_bps", 0.0),
            "turnover_reduction_pct": round(
                ((calendar_metrics["annualized_turnover_pct"] - ai_metrics["annualized_turnover_pct"])
                 / max(calendar_metrics["annualized_turnover_pct"], 1e-4) * 100.0) if calendar_metrics else 0.0,
                2,
            ),
            "raw_results": results,
        }
