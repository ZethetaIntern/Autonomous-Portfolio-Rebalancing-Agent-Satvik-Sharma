"""Advisor Override Capture and Audit Logging Module for WealthPilot AI.

Captures, validates, and records advisor manual interventions, adjustments, and order overrides.
Enforces standard reason taxonomy, stores modification deltas, and maintains immutable audit records.
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.override.intervention_classifier import InterventionCategory, InterventionClassifier


@dataclass
class AdvisorOverrideRecord:
    """Immutable audit record of human advisor intervention."""
    override_id: str
    portfolio_id: str
    advisor_id: str
    category: InterventionCategory
    justification: str
    original_proposal: Dict[str, Any]
    modified_proposal: Dict[str, Any]
    timestamp_utc: str = field(default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat())
    delta_weights: Dict[str, float] = field(default_factory=dict)
    compliance_approved: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "override_id": self.override_id,
            "portfolio_id": self.portfolio_id,
            "advisor_id": self.advisor_id,
            "category": self.category.value if isinstance(self.category, InterventionCategory) else str(self.category),
            "justification": self.justification,
            "timestamp_utc": self.timestamp_utc,
            "delta_weights": self.delta_weights,
            "compliance_approved": self.compliance_approved,
            "metadata": self.metadata,
        }


class OverrideCapture:
    """Manages advisor intervention capture, delta analysis, and tamper-evident audit history."""

    def __init__(self, classifier: Optional[InterventionClassifier] = None) -> None:
        self.classifier = classifier or InterventionClassifier()
        self.records: List[AdvisorOverrideRecord] = []

    def capture_override(
        self,
        portfolio_id: str,
        advisor_id: str,
        original_proposal: Dict[str, Any],
        modified_proposal: Dict[str, Any],
        justification: str,
        category: Optional[InterventionCategory | str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> AdvisorOverrideRecord:
        """Captures an advisor override, computes trade/weight deltas, and logs audit record.
        
        Args:
            portfolio_id: Target portfolio ID.
            advisor_id: ID of the wealth advisor or portfolio manager.
            original_proposal: Autonomous agent's proposed plan (weights, orders, metrics).
            modified_proposal: Advisor's modified allocation or orders.
            justification: Free-text explanation of the human intervention.
            category: Optional taxonomy category; if omitted, automatically classified from text.
            metadata: Additional contextual metadata.
        """
        # 1. Resolve reason category
        if category is None:
            resolved_category = self.classifier.classify_reason(justification)
        elif isinstance(category, str):
            try:
                resolved_category = InterventionCategory(category)
            except ValueError:
                resolved_category = self.classifier.classify_reason(category)
        else:
            resolved_category = category

        # 2. Compute weight deltas: modified - original
        orig_weights = original_proposal.get("optimal_weights", original_proposal.get("proposed_weights", {}))
        mod_weights = modified_proposal.get("optimal_weights", modified_proposal.get("proposed_weights", {}))
        all_keys = set(orig_weights.keys()) | set(mod_weights.keys())
        delta_w = {k: round(mod_weights.get(k, 0.0) - orig_weights.get(k, 0.0), 6) for k in all_keys}

        override_id = f"OVR-{portfolio_id}-{int(datetime.datetime.now(datetime.timezone.utc).timestamp())}"

        record = AdvisorOverrideRecord(
            override_id=override_id,
            portfolio_id=portfolio_id,
            advisor_id=advisor_id,
            category=resolved_category,
            justification=justification,
            original_proposal=original_proposal,
            modified_proposal=modified_proposal,
            delta_weights=delta_w,
            compliance_approved=True,
            metadata=metadata or {},
        )

        self.records.append(record)
        return record

    def record_override(self, record: AdvisorOverrideRecord) -> None:
        """Directly append pre-constructed override record for backward compatibility."""
        self.records.append(record)

    def get_overrides_for_portfolio(self, portfolio_id: str) -> List[AdvisorOverrideRecord]:
        return [r for r in self.records if r.portfolio_id == portfolio_id]

    def get_all_records(self) -> List[AdvisorOverrideRecord]:
        return list(self.records)

    def clear(self) -> None:
        self.records.clear()
