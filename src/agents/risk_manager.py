"""Risk Manager Agent for WealthPilot AI.

Role: Chief Risk Officer / Risk Manager
Calculates pre- and post-trade 95% VaR, tracking error, evaluates liquidity schedules,
computes market impact, and stress tests proposed trades against market shocks.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional
import numpy as np

from src.optimisation.liquidity_scorer import LiquidityScorer, ExecutionSchedule
from src.optimisation.cost_estimator import CostEstimator
from src.data.market_data_simulator import MarketDataSimulator

try:
    from crewai import Agent
    HAS_CREWAI = True
    import warnings
    _cw = warnings.warn
    def _compat_crew_warn(message, category=None, stacklevel=1, source=None, *args, **kwargs):
        kwargs.pop("skip_file_prefixes", None)
        return _cw(message, category=category, stacklevel=stacklevel, source=source, *args, **kwargs)
    warnings.warn = _compat_crew_warn
except ImportError:
    HAS_CREWAI = False


class RiskManagerAgent:
    """Specialized agent stress-testing allocations, quantifying VaR, and planning execution schedules."""

    def __init__(
        self,
        liquidity_scorer: Optional[LiquidityScorer] = None,
        cost_estimator: Optional[CostEstimator] = None,
        market_simulator: Optional[MarketDataSimulator] = None,
    ) -> None:
        self.role = "Chief Risk Officer / Risk Manager"
        self.goal = "Quantify portfolio risk (VaR, Tracking Error), stress-test allocations, and schedule multi-day executions."
        self.backstory = (
            "A vigilant quantitative risk manager with deep experience in Indian and global financial crises. "
            "Expert in parametric VaR modeling, stress-testing under market shocks, and minimizing liquidity slippage."
        )
        self.liquidity_scorer = liquidity_scorer or LiquidityScorer()
        self.cost_estimator = cost_estimator or CostEstimator()
        self.sim = market_simulator or MarketDataSimulator(seed=42)
        self.cov = self.sim.get_annual_covariance()

    def as_crewai_agent(self, tools: Optional[List[Any]] = None) -> Optional[Any]:
        if not HAS_CREWAI:
            return None
        return Agent(
            role=self.role,
            goal=self.goal,
            backstory=self.backstory,
            verbose=True,
            tools=tools or [],
            allow_delegation=False,
        )

    def execute_task(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Calculates risk parameters, stress tests allocations, and schedules order lines."""
        curr_w = context.get("current_weights", {})
        prop_w = context.get("proposed_weights", {})
        trades = context.get("tax_adjusted_trades", context.get("candidate_trades", []))

        assets = self.sim.asset_names
        w_curr_vec = np.array([curr_w.get(a, 0.0) for a in assets], dtype=np.float64)
        w_prop_vec = np.array([prop_w.get(a, 0.0) for a in assets], dtype=np.float64)

        var_curr = float(np.dot(np.dot(w_curr_vec, self.cov), w_curr_vec))
        var_prop = float(np.dot(np.dot(w_prop_vec, self.cov), w_prop_vec))

        vol_curr = math.sqrt(max(0.0, var_curr))
        vol_prop = math.sqrt(max(0.0, var_prop))

        var_95_curr = 1.645 * vol_curr * 100.0
        var_95_prop = 1.645 * vol_prop * 100.0

        schedules: List[ExecutionSchedule] = self.liquidity_scorer.schedule_trade_list(trades)
        multi_day_count = sum(1 for s in schedules if s.is_multi_day)

        cost_analysis = self.cost_estimator.estimate_portfolio_trades_cost(trades)

        stress_shock = np.array([-0.20, -0.05, -0.06, 0.05, 0.00])
        shock_loss_curr = float(np.dot(w_curr_vec, stress_shock)) * 100.0
        shock_loss_prop = float(np.dot(w_prop_vec, stress_shock)) * 100.0

        risk_approved = vol_prop <= (vol_curr * 1.25)

        aum = float(context.get("portfolio_aum", 10_000_000.0))
        pre_var_inr = (var_95_curr / 100.0) * aum
        post_var_inr = (var_95_prop / 100.0) * aum

        return {
            "status": "APPROVED" if risk_approved else "RISK_WARNING",
            "volatility_pre_trade_pct": round(vol_curr * 100.0, 2),
            "volatility_post_trade_pct": round(vol_prop * 100.0, 2),
            "var_95_pre_trade_pct": round(var_95_curr, 2),
            "var_95_post_trade_pct": round(var_95_prop, 2),
            "pre_trade_var_95_inr": round(pre_var_inr, 2),
            "post_trade_var_95_inr": round(post_var_inr, 2),
            "execution_schedules": schedules,
            "multi_day_schedules_count": multi_day_count,
            "cost_analysis": cost_analysis,
            "market_impact_summary": {"total_trading_cost_inr": cost_analysis["total_cost_inr"]},
            "total_transaction_cost_inr": cost_analysis["total_cost_inr"],
            "market_impact_inr": cost_analysis["total_market_impact_inr"],
            "stress_test_drawdown_pct": round(shock_loss_prop, 2),
            "stress_test_covid_shock_pre_pct": round(shock_loss_curr, 2),
            "stress_test_covid_shock_post_pct": round(shock_loss_prop, 2),
            "risk_verdict": "APPROVED_FOR_EXECUTION" if risk_approved else "ELEVATED_VOLATILITY",
        }
