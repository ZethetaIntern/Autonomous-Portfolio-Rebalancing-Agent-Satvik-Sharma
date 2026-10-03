"""Portfolio Optimisation & Execution Planning Package for WealthPilot AI."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import numpy as np


@dataclass
class TradeOrder:
    """Executable security or asset-level order ticket."""
    portfolio_id: str
    asset_class: str
    action: str  # BUY or SELL
    target_units: float
    estimated_price: float
    trade_value_inr: float
    expected_impact_bps: float = 0.0
    tax_implication_inr: float = 0.0


@dataclass
class OptimizationResult:
    """Comprehensive portfolio rebalancing optimization result."""
    portfolio_id: str
    optimal_weights: Dict[str, float]
    expected_tracking_error: float
    total_turnover: float
    trades: List[TradeOrder]
    estimated_transaction_cost_inr: float
    estimated_tax_cost_inr: float
    status: str
    pre_trade_report: Optional[Any] = None


from src.optimisation.portfolio_optimiser import PortfolioOptimiser
from src.optimisation.trade_list_generator import TradeListGenerator
from src.optimisation.cost_estimator import CostEstimator
from src.optimisation.constraint_manager import ConstraintManager, PreTradeValidationReport
from src.optimisation.tax_lot_manager import TaxLot, TaxLotManager
from src.optimisation.tax_optimiser import TaxOptimiser
from src.optimisation.liquidity_scorer import LiquidityScorer, ExecutionSchedule, ExecutionSlice
from src.optimisation.tax_harvesting_scanner import (
    TaxHarvestingScanner,
    TaxHarvestingOpportunity,
    TaxHarvestingBatchReport,
)

__all__ = [
    "TradeOrder",
    "OptimizationResult",
    "PortfolioOptimiser",
    "TradeListGenerator",
    "CostEstimator",
    "ConstraintManager",
    "PreTradeValidationReport",
    "TaxLot",
    "TaxLotManager",
    "TaxOptimiser",
    "LiquidityScorer",
    "ExecutionSchedule",
    "ExecutionSlice",
    "TaxHarvestingScanner",
    "TaxHarvestingOpportunity",
    "TaxHarvestingBatchReport",
]
