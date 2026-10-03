"""Portfolio Analyst Agent for WealthPilot AI.

Role: Senior Quantitative Portfolio Analyst
Solves constrained quadratic programming allocations using CVXPY and synthesizes
executable security-level trade lists with joint round-lot optimization.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import numpy as np

from src.optimisation.portfolio_optimiser import PortfolioOptimiser
from src.optimisation.trade_list_generator import TradeListGenerator
from src.optimisation import TradeOrder

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


class PortfolioAnalystAgent:
    """Specialized agent solving constrained QP rebalancing and discrete order generation."""

    def __init__(
        self,
        portfolio_optimiser: Optional[PortfolioOptimiser] = None,
        trade_list_generator: Optional[TradeListGenerator] = None,
    ) -> None:
        self.role = "Senior Quantitative Portfolio Analyst"
        self.goal = "Solve constrained tracking error optimization and generate executable discrete order tickets."
        self.backstory = (
            "An elite quantitative portfolio manager specializing in Indian capital markets. "
            "Expert in Markowitz convex quadratic programming, tracking error minimization, "
            "and discrete round-lot knapsack execution."
        )
        self.optimiser = portfolio_optimiser or PortfolioOptimiser()
        self.generator = trade_list_generator or TradeListGenerator()

    def as_crewai_agent(self, tools: Optional[List[Any]] = None) -> Optional[Any]:
        """Returns a configured CrewAI Agent if the crewai library is installed."""
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
        """Executes portfolio optimization and trade ticket synthesis.
        
        Args:
            context: Dict containing:
                - portfolio_id: str
                - current_weights: Dict[str, float]
                - target_weights: Dict[str, float]
                - portfolio_aum: float
                - turnover_budget: float (optional)
                - min_cash_buffer: float (optional)
                - sebi_issuer_limit: float (optional)
                - sebi_sector_limits: Dict (optional)
                - asset_sector_map: Dict (optional)
                - min_trade_size: float (optional)

        Returns:
            Dict containing optimal_weights, trade_weights, candidate_trades, and tracking error metrics.
        """
        p_id = context.get("portfolio_id", "PORTFOLIO")
        curr_w = context.get("current_weights", {})
        targ_w = context.get("target_weights", {})
        aum = float(context.get("portfolio_aum", 10_000_000.0))

        turnover_budget = context.get("turnover_budget", 0.20)
        min_cash = context.get("min_cash_buffer", 0.02)
        issuer_limit = context.get("sebi_issuer_limit")
        sector_limits = context.get("sebi_sector_limits")
        sector_map = context.get("asset_sector_map")
        min_trade_size = context.get("min_trade_size", 0.0)

        # 1. Solve QP continuous allocation
        opt_res = self.optimiser.optimize_allocation(
            current_weights=curr_w,
            target_weights=targ_w,
            turnover_budget=turnover_budget,
            min_cash_buffer=min_cash,
            sebi_issuer_limit=issuer_limit,
            sebi_sector_limits=sector_limits,
            asset_sector_map=sector_map,
            min_trade_size=min_trade_size,
        )

        opt_w = opt_res["optimal_weights"]

        # 2. Generate discrete executable trade tickets with joint round lots
        plan = self.generator.generate_trade_plan(
            portfolio_id=p_id,
            current_weights=curr_w,
            optimal_weights=opt_w,
            portfolio_aum=aum,
            min_cash_buffer_pct=min_cash,
        )

        return {
            "status": "SUCCESS" if opt_res["status"] == "OPTIMAL" else "OPTIMIZER_WARNING",
            "optimal_weights": opt_w,
            "discrete_weights": plan["discrete_weights"],
            "candidate_trades": plan["orders"],
            "cash_ledger": plan["cash_ledger"],
            "tracking_error_variance": opt_res["tracking_error_variance"],
            "predicted_tracking_error": opt_res["predicted_tracking_error"],
            "turnover": opt_res["turnover"],
            "rounding_error_inr": plan["rounding_error_inr"],
            "solver": opt_res["solver"],
        }
