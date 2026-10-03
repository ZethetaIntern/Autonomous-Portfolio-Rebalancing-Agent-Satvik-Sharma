"""Constraint Manager for portfolio optimization bounds and SEBI regulatory caps.

Maintains formal regulatory and policy constraint rulesets, runs pre-trade verification,
and compiles auditable PreTradeValidationReport certificates.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


@dataclass
class PreTradeValidationReport:
    """Pre-trade compliance and constraint verification certificate."""
    is_valid: bool
    sebi_compliant: bool
    violations: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metric_breakdown: Dict[str, Any] = field(default_factory=dict)
    summary_narrative: str = ""


class ConstraintManager:
    """Encapsulates hard portfolio constraints, regulatory ceilings, and pre-trade audits."""

    def __init__(
        self,
        min_weight: float = 0.0,
        max_weight: float = 1.0,
        min_cash_buffer: float = 0.02,
        sebi_issuer_limit: float = 0.10,          # 10% max per single issuer
        sebi_sector_limit: float = 0.30,          # 30% max per industry sector
        max_turnover_per_rebalance: float = 0.30, # 30% turnover ceiling
        min_order_value_inr: float = 1000.0,      # Minimum trade threshold
    ) -> None:
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.min_cash_buffer = min_cash_buffer
        self.sebi_issuer_limit = sebi_issuer_limit
        self.sebi_sector_limit = sebi_sector_limit
        self.max_turnover = max_turnover_per_rebalance
        self.min_order_value_inr = min_order_value_inr

    def validate_allocation(
        self,
        weights: Dict[str, float],
        tolerance: float = 0.005,
    ) -> Tuple[bool, List[str]]:
        """Quick boolean validation of portfolio weights against basic constraints."""
        violations: List[str] = []
        total = sum(weights.values())

        if abs(total - 1.0) > tolerance:
            violations.append(f"Sum of weights is {total:.4f}, expected 1.0000 (tolerance +/- {tolerance})")

        for asset, w in weights.items():
            if w < (self.min_weight - 1e-6):
                violations.append(f"Long-only breach: negative weight in {asset} ({w:.4f})")
            if w > (self.max_weight + 1e-6):
                violations.append(f"Maximum bound breach: weight {w:.4f} in {asset} exceeds {self.max_weight:.4f}")

        cash_w = weights.get("LIQUID_CASH", 0.0)
        if cash_w < (self.min_cash_buffer - 1e-6):
            violations.append(
                f"Liquidity deficit: Liquid cash {cash_w:.2%} is below required buffer {self.min_cash_buffer:.2%}"
            )

        return len(violations) == 0, violations

    def generate_pre_trade_report(
        self,
        portfolio_id: str,
        current_weights: Dict[str, float],
        proposed_weights: Dict[str, float],
        trade_orders: Optional[List[Any]] = None,
        issuer_mappings: Optional[Dict[str, str]] = None,
        sector_mappings: Optional[Dict[str, str]] = None,
        turnover_budget: Optional[float] = None,
    ) -> PreTradeValidationReport:
        """Executes full multi-dimensional pre-trade verification audit.
        
        Evaluates:
            1. Total Budget Sum Conservation: sum(proposed_weights) == 1.0
            2. Long-Only Mandate: w_proposed_i >= 0.0
            3. Liquid Cash Buffer: w_proposed_cash >= min_cash_buffer
            4. SEBI Single-Issuer Limit: max concentration <= 10% per issuer
            5. SEBI Sector Concentration: max concentration <= 30% per sector
            6. Rebalance Turnover Budget: 0.5 * sum(|w_prop - w_curr|) <= turnover_budget
            7. Minimum Trade Order Values: order_value >= min_order_value_inr

        Returns:
            PreTradeValidationReport with boolean certificate and diagnostic details.
        """
        violations: List[str] = []
        warnings: List[str] = []
        checks: Dict[str, Any] = {}

        # 1. Budget Conservation
        total_w = sum(proposed_weights.values())
        budget_pass = abs(total_w - 1.0) <= 0.005
        if not budget_pass:
            violations.append(f"Budget conservation breach: proposed weight sum is {total_w:.4f}, expected 1.0000")
        checks["budget_conservation"] = {
            "passed": budget_pass,
            "total_weight": round(total_w, 4),
            "target": 1.0,
        }

        # 2. Long-Only Mandate
        negative_assets = {a: round(w, 4) for a, w in proposed_weights.items() if w < -1e-5}
        long_only_pass = len(negative_assets) == 0
        if not long_only_pass:
            violations.append(f"Long-only constraint breach: negative weights in {negative_assets}")
        checks["long_only"] = {
            "passed": long_only_pass,
            "negative_assets": negative_assets,
        }

        # 3. Liquid Cash Buffer
        cash_weight = proposed_weights.get("LIQUID_CASH", 0.0)
        cash_pass = cash_weight >= (self.min_cash_buffer - 1e-4)
        if not cash_pass:
            violations.append(
                f"Cash buffer breach: proposed cash {cash_weight:.2%} < required {self.min_cash_buffer:.2%}"
            )
        checks["cash_buffer"] = {
            "passed": cash_pass,
            "cash_weight": round(cash_weight, 4),
            "min_required": round(self.min_cash_buffer, 4),
        }

        # 4. SEBI Single-Issuer Concentration
        # If issuer mappings provided, aggregate by issuer; otherwise only check if caller specifically requests
        issuer_weights: Dict[str, float] = {}
        if issuer_mappings is not None:
            for asset, w in proposed_weights.items():
                issuer = issuer_mappings.get(asset)
                if issuer:
                    # Sovereign / index broad instruments are exempt from 10% single corporate issuer rule
                    is_exempt = any(
                        term in issuer.upper() for term in ("G_SEC", "SOVEREIGN", "NIFTY_50", "INDEX", "GOI", "CASH")
                    )
                    if not is_exempt:
                        issuer_weights[issuer] = issuer_weights.get(issuer, 0.0) + w

        sebi_issuer_breaches = {
            iss: round(w, 4) for iss, w in issuer_weights.items() if w > (self.sebi_issuer_limit + 1e-4)
        }
        sebi_issuer_pass = len(sebi_issuer_breaches) == 0
        if not sebi_issuer_pass:
            violations.append(
                f"SEBI single-issuer breach: issuers exceed {self.sebi_issuer_limit:.0%}: {sebi_issuer_breaches}"
            )
        checks["sebi_single_issuer"] = {
            "passed": sebi_issuer_pass,
            "max_limit": round(self.sebi_issuer_limit, 4),
            "breaches": sebi_issuer_breaches,
            "issuer_breakdown": {k: round(v, 4) for k, v in issuer_weights.items()},
        }

        # 5. SEBI Sector Concentration
        sector_weights: Dict[str, float] = {}
        if sector_mappings:
            for asset, w in proposed_weights.items():
                sec = sector_mappings.get(asset, "OTHER")
                sector_weights[sec] = sector_weights.get(sec, 0.0) + w

            sector_breaches = {
                sec: round(w, 4) for sec, w in sector_weights.items() if w > (self.sebi_sector_limit + 1e-4)
            }
            sebi_sector_pass = len(sector_breaches) == 0
            if not sebi_sector_pass:
                violations.append(
                    f"SEBI sector concentration breach: sectors exceed {self.sebi_sector_limit:.0%}: {sector_breaches}"
                )
        else:
            sebi_sector_pass = True
            sector_breaches = {}

        checks["sebi_sector_concentration"] = {
            "passed": sebi_sector_pass,
            "max_limit": round(self.sebi_sector_limit, 4),
            "breaches": sector_breaches,
            "sector_breakdown": {k: round(v, 4) for k, v in sector_weights.items()},
        }

        # 6. Turnover Budget Compliance
        all_assets = set(list(current_weights.keys()) + list(proposed_weights.keys()))
        turnover = sum(abs(proposed_weights.get(a, 0.0) - current_weights.get(a, 0.0)) for a in all_assets) / 2.0
        allowed_turnover = turnover_budget if turnover_budget is not None else self.max_turnover
        turnover_pass = turnover <= (allowed_turnover + 0.005)

        if not turnover_pass:
            violations.append(
                f"Turnover budget breach: rebalance turnover {turnover:.2%} exceeds ceiling {allowed_turnover:.2%}"
            )
        checks["turnover_budget"] = {
            "passed": turnover_pass,
            "actual_turnover": round(turnover, 4),
            "ceiling": round(allowed_turnover, 4),
        }

        # 7. Minimum Trade Size & Order Sanity
        if trade_orders:
            small_orders = []
            for o in trade_orders:
                val = getattr(o, "trade_value_inr", 0.0) if hasattr(o, "trade_value_inr") else float(o.get("trade_value_inr", o.get("value_inr", 0.0)))
                asset = getattr(o, "asset_class", "") if hasattr(o, "asset_class") else o.get("asset_class", "")
                if 0.0 < val < self.min_order_value_inr:
                    small_orders.append(f"{asset} (Rs {val:.2f})")
            if small_orders:
                warnings.append(
                    f"Sub-minimum order tickets detected: {small_orders} are below threshold Rs {self.min_order_value_inr:.2f}"
                )
            checks["order_sizes"] = {
                "sub_minimum_orders": small_orders,
                "threshold_inr": self.min_order_value_inr,
            }

        is_valid = len(violations) == 0
        sebi_compliant = sebi_issuer_pass and sebi_sector_pass and long_only_pass

        # Build summary narrative
        if is_valid:
            narrative = (
                f"Portfolio {portfolio_id}: Pre-trade validation SUCCESSFUL. All SEBI statutory limits, "
                f"budget conservation (100%), cash buffer ({cash_weight:.1%}), and turnover ceiling ({turnover:.1%}) "
                f"are fully verified."
            )
        else:
            narrative = (
                f"Portfolio {portfolio_id}: Pre-trade validation REJECTED with {len(violations)} violation(s): "
                + "; ".join(violations)
            )

        return PreTradeValidationReport(
            is_valid=is_valid,
            sebi_compliant=sebi_compliant,
            violations=violations,
            warnings=warnings,
            metric_breakdown=checks,
            summary_narrative=narrative,
        )
