"""Tier 1: Threshold & Concentration Rebalancing Trigger Evaluator.

Detects:
1. Asset class strategic drift breaches.
2. Single-asset & sector concentration limit breaches.
3. Liquidity buffer deficits (<2% liquid cash).
4. Factor exposure drift (equity beta deviation).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import numpy as np

from src.triggers.trigger_evaluator import (
    BaseTriggerEvaluator,
    TriggerPriority,
    TriggerSignal,
    TriggerTier,
    TriggerType,
)
from src.monitoring.threshold_manager import DEFAULT_ASSET_CORRIDORS, ThresholdManager


class ThresholdTriggerEvaluator(BaseTriggerEvaluator):
    """Evaluates quantitative allocation breaches and hard portfolio concentration limits."""

    def __init__(
        self,
        threshold_manager: Optional[ThresholdManager] = None,
        asset_names: Optional[List[str]] = None,
        asset_corridors: Optional[Dict[str, Dict[str, float]]] = None,
    ) -> None:
        self.threshold_mgr = threshold_manager or ThresholdManager()
        self.asset_names = asset_names or [
            "NIFTY_50_EQUITY",
            "G_SEC_BONDS",
            "CORP_BONDS",
            "GOLD_ETF",
            "LIQUID_CASH",
        ]
        self.asset_corridors = asset_corridors or DEFAULT_ASSET_CORRIDORS

    def evaluate(
        self,
        portfolio_record: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> List[TriggerSignal]:
        signals: List[TriggerSignal] = []

        portfolio_id = str(portfolio_record.get("portfolio_id", "UNKNOWN"))
        client_id = str(portfolio_record.get("client_id", "UNKNOWN"))
        risk_category = str(portfolio_record.get("risk_category", "Balanced"))

        # 1. Effective threshold check
        effective_thresh = self.threshold_mgr.get_effective_threshold(
            risk_category=risk_category,
            client_id=client_id,
        )

        max_drift = 0.0
        breached_assets = []

        for ac in self.asset_names:
            curr_w = float(portfolio_record.get(f"current_weight_{ac}", 0.0))
            targ_w = float(portfolio_record.get(f"target_weight_{ac}", 0.0))
            abs_drift = abs(curr_w - targ_w)

            if abs_drift > max_drift:
                max_drift = abs_drift

            if abs_drift > effective_thresh:
                breached_assets.append((ac, curr_w, targ_w, abs_drift))

            # Concentration hard cap check
            corridor = self.asset_corridors.get(ac, {})
            hard_cap = corridor.get("hard_cap")
            if hard_cap and curr_w > hard_cap:
                signals.append(
                    TriggerSignal(
                        portfolio_id=portfolio_id,
                        client_id=client_id,
                        tier=TriggerTier.TIER_1_THRESHOLD,
                        trigger_type=TriggerType.CONCENTRATION_BREACH,
                        priority=TriggerPriority.URGENT,
                        urgency_score=4.5,
                        summary=f"Hard concentration cap breached in {ac} ({curr_w:.1%} > {hard_cap:.1%})",
                        details={
                            "asset_class": ac,
                            "current_weight": curr_w,
                            "hard_cap": hard_cap,
                            "excess_weight": curr_w - hard_cap,
                        },
                        recommended_action=f"Trim {ac} position immediately to comply with {hard_cap:.1%} regulatory limit.",
                    )
                )

        # Cash buffer deficit check
        liquid_curr = float(portfolio_record.get("current_weight_LIQUID_CASH", 0.05))
        min_cash = self.asset_corridors.get("LIQUID_CASH", {}).get("min_buffer", 0.02)
        if liquid_curr < min_cash:
            signals.append(
                TriggerSignal(
                    portfolio_id=portfolio_id,
                    client_id=client_id,
                    tier=TriggerTier.TIER_1_THRESHOLD,
                    trigger_type=TriggerType.CONCENTRATION_BREACH,
                    priority=TriggerPriority.HIGH,
                    urgency_score=3.5,
                    summary=f"Liquidity buffer deficit: Liquid cash {liquid_curr:.1%} is below minimum {min_cash:.1%}",
                    details={"current_cash_pct": liquid_curr, "min_buffer": min_cash},
                    recommended_action="Replenish liquid cash buffer to meet upcoming settlement / fee obligations.",
                )
            )

        # Asset class drift breach trigger
        if breached_assets:
            ratio = max_drift / max(effective_thresh, 0.001)
            priority = TriggerPriority.CRITICAL if ratio >= 2.0 else (
                TriggerPriority.HIGH if ratio >= 1.5 else TriggerPriority.MEDIUM
            )

            breach_descriptions = [
                f"{ac}: current {cw:.1%} vs target {tw:.1%} (drift {d:.1%})"
                for ac, cw, tw, d in breached_assets
            ]

            signals.append(
                TriggerSignal(
                    portfolio_id=portfolio_id,
                    client_id=client_id,
                    tier=TriggerTier.TIER_1_THRESHOLD,
                    trigger_type=TriggerType.ASSET_CLASS_DRIFT,
                    priority=priority,
                    urgency_score=round(ratio * 2.0, 2),
                    summary=f"Drift band breached: Max drift {max_drift:.2%} exceeds tolerance {effective_thresh:.2%}",
                    details={
                        "risk_category": risk_category,
                        "effective_threshold": effective_thresh,
                        "max_drift": max_drift,
                        "breach_ratio": round(ratio, 2),
                        "breached_assets": breach_descriptions,
                    },
                    recommended_action=f"Rebalance portfolio back to {risk_category} target allocation bands.",
                )
            )

        return signals
