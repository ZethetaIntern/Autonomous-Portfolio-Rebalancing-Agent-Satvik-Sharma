"""Tier 2: Calendar-based Rebalancing Trigger Evaluator.

Evaluates scheduled periodic rebalancing cadences:
- Ultra-Conservative & Conservative: Quarterly (90 days)
- Balanced & Aggressive: Semi-Annually (180 days)
- Ultra-Aggressive: Annually (365 days)
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from src.triggers.trigger_evaluator import (
    BaseTriggerEvaluator,
    TriggerPriority,
    TriggerSignal,
    TriggerTier,
    TriggerType,
)


DEFAULT_CADENCE_DAYS: Dict[str, int] = {
    "Ultra-Conservative": 90,
    "Conservative": 90,
    "Balanced": 180,
    "Aggressive": 180,
    "Ultra-Aggressive": 365,
}


class CalendarTriggerEvaluator(BaseTriggerEvaluator):
    """Evaluates time elapsed since last portfolio rebalancing against calendar mandate."""

    def __init__(self, cadence_days: Optional[Dict[str, int]] = None) -> None:
        self.cadence_days = cadence_days or DEFAULT_CADENCE_DAYS

    def evaluate(
        self,
        portfolio_record: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> List[TriggerSignal]:
        signals: List[TriggerSignal] = []

        portfolio_id = str(portfolio_record.get("portfolio_id", "UNKNOWN"))
        client_id = str(portfolio_record.get("client_id", "UNKNOWN"))
        risk_category = str(portfolio_record.get("risk_category", "Balanced"))

        days_since_rebalance = int(portfolio_record.get("days_since_rebalance", 0))
        cadence_target = self.cadence_days.get(risk_category, 180)

        # Allow mandate override in context or record
        mandate_frequency = portfolio_record.get("mandate_frequency")
        if mandate_frequency:
            override_map = {"MONTHLY": 30, "QUARTERLY": 90, "SEMI_ANNUALLY": 180, "ANNUALLY": 365}
            cadence_target = override_map.get(str(mandate_frequency).upper(), cadence_target)

        # Trigger fires if elapsed time meets or exceeds target
        if days_since_rebalance >= cadence_target:
            overdue_days = days_since_rebalance - cadence_target
            priority = (
                TriggerPriority.HIGH
                if overdue_days > 45
                else (TriggerPriority.MEDIUM if overdue_days > 15 else TriggerPriority.LOW)
            )

            signals.append(
                TriggerSignal(
                    portfolio_id=portfolio_id,
                    client_id=client_id,
                    tier=TriggerTier.TIER_2_CALENDAR,
                    trigger_type=TriggerType.CALENDAR_SCHEDULE,
                    priority=priority,
                    urgency_score=round(1.0 + (overdue_days / 60.0), 2),
                    summary=f"Calendar schedule due: {days_since_rebalance} days since last rebalance (Mandate: {cadence_target} days)",
                    details={
                        "risk_category": risk_category,
                        "days_since_rebalance": days_since_rebalance,
                        "cadence_days": cadence_target,
                        "overdue_days": overdue_days,
                    },
                    recommended_action=f"Execute scheduled {risk_category} calendar rebalance review.",
                )
            )

        return signals
