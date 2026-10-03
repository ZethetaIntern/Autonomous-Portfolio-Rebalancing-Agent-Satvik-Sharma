"""Escalation Manager routing high-risk rebalancing decisions to senior leadership / IC.

Provides automated escalation evaluation, human-in-the-loop review state management,
and generates comprehensive executive briefing documents for Investment Committee / CRO review.
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class EscalationStatus(str, Enum):
    PENDING = "PENDING"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"


@dataclass
class EscalationCase:
    """Active escalation case requiring senior human intervention."""
    case_id: str
    portfolio_id: str
    client_id: str
    risk_category: str
    reasons: List[str]
    proposed_plan: Dict[str, Any]
    status: EscalationStatus = EscalationStatus.PENDING
    created_at_utc: str = field(default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat())
    resolved_at_utc: Optional[str] = None
    reviewer_id: Optional[str] = None
    reviewer_notes: Optional[str] = None
    briefing_document: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "case_id": self.case_id,
            "portfolio_id": self.portfolio_id,
            "client_id": self.client_id,
            "risk_category": self.risk_category,
            "reasons": self.reasons,
            "status": self.status.value,
            "created_at_utc": self.created_at_utc,
            "resolved_at_utc": self.resolved_at_utc,
            "reviewer_id": self.reviewer_id,
            "reviewer_notes": self.reviewer_notes,
            "briefing_document": self.briefing_document,
        }


class EscalationManager:
    """Manages escalation workflows when proposals exceed risk, compliance, or turnover thresholds."""

    def __init__(
        self,
        extreme_turnover_limit: float = 0.40,
        large_aum_escalation_limit: float = 250_000_000.0,
        large_aum_turnover_limit: float = 0.25,
    ) -> None:
        self.extreme_turnover = extreme_turnover_limit
        self.large_aum = large_aum_escalation_limit
        self.large_aum_turnover = large_aum_turnover_limit
        self.cases: Dict[str, EscalationCase] = {}

    def check_escalation_required(self, turnover: float, aum: float) -> bool:
        """Determines if automatic IC escalation is required based on turnover and AUM."""
        return turnover > self.extreme_turnover or (aum > self.large_aum and turnover > self.large_aum_turnover)

    def compile_briefing_document(
        self,
        portfolio_id: str,
        client_id: str,
        reasons: List[str],
        proposed_plan: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Compiles a comprehensive structured briefing document for human executive review."""
        ctx = context or {}
        aum = float(proposed_plan.get("portfolio_aum", ctx.get("portfolio_aum", 10_000_000.0)))
        turnover = float(proposed_plan.get("turnover", ctx.get("turnover", 0.0)))
        trades = proposed_plan.get("candidate_trades", proposed_plan.get("trades", []))
        risk_metrics = ctx.get("risk_metrics", proposed_plan.get("risk_metrics", {}))
        comp_report = ctx.get("compliance_report", proposed_plan.get("compliance_report", {}))
        tax_summary = ctx.get("tax_summary", proposed_plan.get("tax_summary", {}))

        return {
            "title": f"URGENT ESCALATION BRIEFING - Portfolio {portfolio_id}",
            "generated_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "target_reviewers": ["Investment Committee", "Chief Risk Officer", "Compliance Head"],
            "portfolio_overview": {
                "portfolio_id": portfolio_id,
                "client_id": client_id,
                "portfolio_aum_inr": aum,
                "turnover_pct": turnover * 100.0,
            },
            "breach_summary": reasons,
            "quantitative_risk_assessment": {
                "pre_trade_var_95_inr": risk_metrics.get("pre_trade_var_95_inr", 0.0),
                "post_trade_var_95_inr": risk_metrics.get("post_trade_var_95_inr", 0.0),
                "tracking_error_predicted": proposed_plan.get("predicted_tracking_error", 0.0),
                "stress_test_drawdown_pct": risk_metrics.get("stress_test_drawdown_pct", 0.0),
            },
            "compliance_findings": {
                "status": comp_report.get("status", "INVESTIGATION_REQUIRED"),
                "failed_rules": comp_report.get("reasons", []),
                "sebi_circular_reference": "SEBI/HO/MRD/DOP1/CIR/P/2024/69",
            },
            "tax_implications": {
                "realized_pnl_inr": tax_summary.get("net_realized_pnl_inr", 0.0),
                "tax_liability_inr": tax_summary.get("total_tax_liability_inr", 0.0),
                "harvested_loss_inr": tax_summary.get("harvest_realized_loss_inr", 0.0),
            },
            "orders_count": len(trades),
            "recommendation": "Manual human approval required. Autonomous trading suspended for this portfolio until reviewed.",
        }

    def create_escalation_case(
        self,
        portfolio_id: str,
        client_id: str,
        risk_category: str,
        reasons: List[str],
        proposed_plan: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> EscalationCase:
        """Initializes and registers a new escalation case with briefing document."""
        case_id = f"ESC-{portfolio_id}-{int(datetime.datetime.now(datetime.timezone.utc).timestamp())}"
        briefing = self.compile_briefing_document(portfolio_id, client_id, reasons, proposed_plan, context)

        case = EscalationCase(
            case_id=case_id,
            portfolio_id=portfolio_id,
            client_id=client_id,
            risk_category=risk_category,
            reasons=reasons,
            proposed_plan=proposed_plan,
            status=EscalationStatus.PENDING,
            briefing_document=briefing,
        )
        self.cases[case_id] = case
        return case

    def review_case(
        self,
        case_id: str,
        decision: EscalationStatus,
        reviewer_id: str,
        reviewer_notes: str,
    ) -> EscalationCase:
        """Applies human sign-off or rejection to an escalated case."""
        if case_id not in self.cases:
            raise KeyError(f"Escalation case {case_id} not found.")

        case = self.cases[case_id]
        case.status = decision
        case.reviewer_id = reviewer_id
        case.reviewer_notes = reviewer_notes
        case.resolved_at_utc = datetime.datetime.now(datetime.timezone.utc).isoformat()
        return case

    def get_pending_cases(self) -> List[EscalationCase]:
        return [c for c in self.cases.values() if c.status == EscalationStatus.PENDING]
