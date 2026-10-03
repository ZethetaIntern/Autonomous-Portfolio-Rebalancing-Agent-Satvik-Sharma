"""Compliance Officer Agent for WealthPilot AI.

Role: Chief Compliance & SEBI Regulatory Officer
Performs SEBI pre-trade rule validation, enforces single-issuer (10%) and sector (30%) limits,
and compiles immutable regulatory audit records.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from src.optimisation.constraint_manager import ConstraintManager, PreTradeValidationReport
from src.explainability.compliance_explainer import ComplianceExplainer, ComplianceExplanationPayload

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


class ComplianceOfficerAgent:
    """Specialized agent auditing SEBI regulatory compliance and generating immutable audit logs."""

    def __init__(
        self,
        constraint_manager: Optional[ConstraintManager] = None,
        compliance_explainer: Optional[ComplianceExplainer] = None,
    ) -> None:
        self.role = "Chief Compliance & SEBI Regulatory Officer"
        self.goal = "Enforce SEBI regulatory limits, ensure 100% policy adherence, and certify audit trails."
        self.backstory = (
            "A seasoned SEBI regulatory compliance officer with zero tolerance for mandate breaches. "
            "Expert in SEBI circulars, single-issuer limits, sector exposure caps, and algorithmic audit logs."
        )
        self.constraint_manager = constraint_manager or ConstraintManager()
        self.compliance_explainer = compliance_explainer or ComplianceExplainer()

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
        """Audits proposed weights and trades against SEBI rulesets and generates certification report."""
        p_id = context.get("portfolio_id", "PORTFOLIO")
        prop_w = context.get("proposed_weights", {})
        curr_w = context.get("current_weights") or prop_w
        trades = context.get("tax_adjusted_trades", context.get("candidate_trades", []))
        issuer_mappings = context.get("issuer_mappings")
        sector_mappings = context.get("sector_mappings")
        turnover_budget = context.get("turnover_budget")

        # Adapt cash buffer if LIQUID_CASH is not in proposed weights
        if "LIQUID_CASH" not in prop_w:
            self.constraint_manager.min_cash_buffer = 0.0
        else:
            self.constraint_manager.min_cash_buffer = float(context.get("cash_buffer_pct", 0.02))

        # 1. Run Pre-trade constraint verification
        report: PreTradeValidationReport = self.constraint_manager.generate_pre_trade_report(
            portfolio_id=p_id,
            current_weights=curr_w,
            proposed_weights=prop_w,
            trade_orders=trades,
            issuer_mappings=issuer_mappings,
            sector_mappings=sector_mappings,
            turnover_budget=turnover_budget,
        )

        # 2. Extract constraint matrix
        matrix = {
            k: v["passed"] for k, v in report.metric_breakdown.items() if isinstance(v, dict) and "passed" in v
        }

        # 3. Create compliance audit record
        audit_record: ComplianceExplanationPayload = self.compliance_explainer.generate_compliance_audit(
            decision_id=context.get("decision_id", f"DEC_{p_id}"),
            portfolio_id=p_id,
            client_id=context.get("client_id", "CLIENT_001"),
            risk_category=context.get("risk_category", "Balanced"),
            trigger_category=context.get("trigger_category", "THRESHOLD"),
            trigger_severity=context.get("trigger_severity", "HIGH"),
            current_weights=curr_w,
            target_weights=context.get("target_weights", {}),
            proposed_weights=prop_w,
            trades=trades,
            constraint_matrix=matrix,
            sebi_compliant=report.sebi_compliant,
            shap_evidence=context.get("shap_evidence"),
            lime_evidence=context.get("lime_evidence"),
            counterfactual_evidence=context.get("counterfactual_evidence"),
        )

        status_str = "PASS" if report.is_valid else "COMPLIANCE_REJECTED"

        return {
            "status": status_str,
            "is_valid": report.is_valid,
            "sebi_compliant": report.sebi_compliant,
            "violations": report.violations,
            "warnings": report.warnings,
            "reasons": report.violations,
            "report": report,
            "pre_trade_report": report,
            "compliance_audit_record": audit_record,
            "digital_signature_hash": audit_record.digital_signature_hash,
            "summary_narrative": report.summary_narrative,
        }
