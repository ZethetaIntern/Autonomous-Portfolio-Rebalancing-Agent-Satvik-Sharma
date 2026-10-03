"""Trigger Consolidation Engine for WealthPilot AI.

Consolidates multiple simultaneous triggers (Tier 1 Threshold, Tier 2 Calendar,
Tier 3 Event) into a single unified rebalancing event with:
1. Highest priority arbitration and timeline SLA
2. Consolidated composite urgency score
3. Unified audit record explaining all contributing drivers
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional
import uuid

import pandas as pd

from src.triggers.trigger_evaluator import (
    BaseTriggerEvaluator,
    PRIORITY_WEIGHTS,
    TriggerPriority,
    TriggerSignal,
    TriggerTier,
    TriggerType,
)
from src.triggers.threshold_trigger import ThresholdTriggerEvaluator
from src.triggers.calendar_trigger import CalendarTriggerEvaluator
from src.triggers.event_trigger import EventTriggerEvaluator


TIMELINE_SLA_MAP: Dict[TriggerPriority, str] = {
    TriggerPriority.CRITICAL: "IMMEDIATE_SAME_DAY (T+0)",
    TriggerPriority.URGENT: "NEXT_DAY_OPEN (T+1)",
    TriggerPriority.HIGH: "WITHIN_48_HOURS (T+2)",
    TriggerPriority.MEDIUM: "NEXT_WEEKLY_BATCH (T+5)",
    TriggerPriority.LOW: "NEXT_MONTHLY_CYCLE (T+15)",
}


@dataclass
class ConsolidatedRebalanceEvent:
    """Unified rebalancing event payload synthesizing all active triggers."""
    portfolio_id: str
    client_id: str
    effective_priority: TriggerPriority
    highest_tier: TriggerTier
    primary_driver: TriggerType
    composite_urgency_score: float
    execution_timeline_sla: str
    audit_rationale: str
    contributing_triggers: List[TriggerSignal]
    event_id: str = field(default_factory=lambda: f"REBAL-{uuid.uuid4().hex[:10].upper()}")
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "portfolio_id": self.portfolio_id,
            "client_id": self.client_id,
            "effective_priority": self.effective_priority.value,
            "highest_tier": self.highest_tier.value,
            "primary_driver": self.primary_driver.value,
            "composite_urgency_score": self.composite_urgency_score,
            "execution_timeline_sla": self.execution_timeline_sla,
            "audit_rationale": self.audit_rationale,
            "timestamp": self.timestamp.isoformat(),
            "contributing_trigger_count": len(self.contributing_triggers),
            "contributing_triggers": [t.to_dict() for t in self.contributing_triggers],
        }


class TriggerConsolidator:
    """Aggregates sub-evaluator signals and produces unified rebalancing events."""

    def __init__(
        self,
        evaluators: Optional[List[BaseTriggerEvaluator]] = None,
    ) -> None:
        if evaluators is None:
            self.evaluators: List[BaseTriggerEvaluator] = [
                ThresholdTriggerEvaluator(),
                CalendarTriggerEvaluator(),
                EventTriggerEvaluator(),
            ]
        else:
            self.evaluators = evaluators

    def evaluate_portfolio(
        self,
        portfolio_record: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> Optional[ConsolidatedRebalanceEvent]:
        """Evaluate all triggers for a single portfolio and consolidate if any trigger fires.

        Args:
            portfolio_record: Dict containing portfolio metadata and weights.
            context: Macro, market shock, and date context.

        Returns:
            ConsolidatedRebalanceEvent if one or more triggers fired, else None.
        """
        all_signals: List[TriggerSignal] = []
        for eval_inst in self.evaluators:
            signals = eval_inst.evaluate(portfolio_record, context=context)
            all_signals.extend(signals)

        if not all_signals:
            return None

        return self.consolidate_signals(all_signals, portfolio_record)

    def consolidate_signals(
        self,
        signals: List[TriggerSignal],
        portfolio_record: Dict[str, Any],
    ) -> ConsolidatedRebalanceEvent:
        """Consolidate multiple signals into a single unified event."""
        sorted_signals = sorted(
            signals,
            key=lambda s: (s.priority_rank, s.urgency_score),
            reverse=True,
        )

        primary_signal = sorted_signals[0]
        effective_priority = primary_signal.priority
        highest_tier = primary_signal.tier
        primary_driver = primary_signal.trigger_type

        max_urgency = primary_signal.urgency_score
        other_urgencies = sum(s.urgency_score for s in sorted_signals[1:])
        composite_urgency = round(max_urgency + (0.15 * other_urgencies), 2)

        sla_timeline = TIMELINE_SLA_MAP.get(effective_priority, "NEXT_WEEKLY_BATCH (T+5)")

        portfolio_id = str(portfolio_record.get("portfolio_id", primary_signal.portfolio_id))
        client_id = str(portfolio_record.get("client_id", primary_signal.client_id))
        risk_cat = str(portfolio_record.get("risk_category", "Unknown"))

        rationale_lines = [
            f"Portfolio {portfolio_id} flagged for rebalancing [Priority: {effective_priority.value}, SLA: {sla_timeline}].",
            f"Primary Driver: {primary_driver.value} - {primary_signal.summary}",
        ]

        if len(sorted_signals) > 1:
            rationale_lines.append(f"Additional Co-occurring Triggers ({len(sorted_signals)-1}):")
            for sub_s in sorted_signals[1:]:
                rationale_lines.append(f" - [{sub_s.tier.value}] {sub_s.trigger_type.value}: {sub_s.summary}")

        audit_rationale = "\n".join(rationale_lines)

        return ConsolidatedRebalanceEvent(
            portfolio_id=portfolio_id,
            client_id=client_id,
            effective_priority=effective_priority,
            highest_tier=highest_tier,
            primary_driver=primary_driver,
            composite_urgency_score=composite_urgency,
            execution_timeline_sla=sla_timeline,
            audit_rationale=audit_rationale,
            contributing_triggers=sorted_signals,
        )

    def batch_evaluate(
        self,
        portfolios_df: pd.DataFrame,
        context: Optional[Dict[str, Any]] = None,
    ) -> List[ConsolidatedRebalanceEvent]:
        """Evaluate and consolidate triggers across a DataFrame of portfolios.

        Args:
            portfolios_df: DataFrame of portfolio records.
            context: External macro/market context.

        Returns:
            List of ConsolidatedRebalanceEvent objects for all flagged portfolios.
        """
        events: List[ConsolidatedRebalanceEvent] = []
        for _, row in portfolios_df.iterrows():
            record = row.to_dict()
            event = self.evaluate_portfolio(record, context=context)
            if event is not None:
                events.append(event)
        return events
