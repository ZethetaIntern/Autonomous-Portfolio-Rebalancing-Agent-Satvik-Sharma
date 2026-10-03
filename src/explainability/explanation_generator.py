"""Central Explanation Orchestration Framework for WealthPilot AI.

Orchestrates audience-tailored explanations across a 3x3 template matrix:
- Tiers: Client, Financial Advisor, Compliance Auditor
- Triggers: Threshold, Calendar, Event

Integrates LangChain structured output parsing (PydanticOutputParser), Flesch-Kincaid
readability validation, and surrogate explainability payloads (SHAP, LIME, Counterfactuals).
"""

from __future__ import annotations

import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate

from src.explainability.client_explainer import ClientExplainer, ClientExplanationPayload
from src.explainability.advisor_explainer import AdvisorExplainer, AdvisorExplanationPayload
from src.explainability.compliance_explainer import ComplianceExplainer, ComplianceExplanationPayload
from src.explainability.shap_integration import ShapExplainer
from src.explainability.lime_integration import LimeExplainer
from src.explainability.counterfactual_generator import CounterfactualGenerator


class FullExplanationPacket(BaseModel):
    """Unified multi-audience explainability packet."""
    portfolio_id: str
    decision_id: str
    timestamp_utc: str
    trigger_category: str
    client_explanation: ClientExplanationPayload
    advisor_dossier: AdvisorExplanationPayload
    compliance_audit: ComplianceExplanationPayload
    shap_attribution: Dict[str, float]
    lime_top_features: List[Dict[str, Any]]
    counterfactual_scenario: Dict[str, Any]


# 9-Template Matrix definitions for LangChain Prompt Templates
EXPLANATION_PROMPT_TEMPLATES: Dict[str, str] = {
    "CLIENT_THRESHOLD": (
        "You are an empathetic personal wealth advisor explaining an automatic portfolio rebalance to a retail client.\n"
        "Portfolio: {portfolio_id}\n"
        "Asset: {primary_asset}, Drift: {drift_direction} (Current: {current_equity_pct}%, Target: {target_equity_pct}%)\n"
        "Fees: INR {total_costs_inr}, Tax: INR {tax_impact_inr}\n"
        "Requirement: Grade 8 readability, under 200 words, cost transparency, zero jargon.\n"
        "{format_instructions}"
    ),
    "CLIENT_CALENDAR": (
        "Explain a routine scheduled quarterly/annual portfolio health check to a retail investor.\n"
        "Portfolio: {portfolio_id}, Target: {target_equity_pct}%\n"
        "Requirement: Friendly, reassuring, Grade 8 readability, under 200 words.\n"
        "{format_instructions}"
    ),
    "CLIENT_EVENT": (
        "Explain an event-driven tax-loss harvesting rebalance to a retail client.\n"
        "Tax Savings: INR {tax_shield_inr}\n"
        "Requirement: Highlight tax savings and keeping investments steady for growth, under 200 words.\n"
        "{format_instructions}"
    ),
    "ADVISOR_THRESHOLD": (
        "You are a Senior Quantitative Strategist preparing an executive briefing for a Financial Advisor.\n"
        "Portfolio: {portfolio_id}, Mandate: {risk_category}\n"
        "SAD: {sad}%, RMSD: {rmsd}%, Tracking Error Reduction: {tracking_error_reduction_bps} bps\n"
        "Turnover: {turnover_pct}%, VaR: {var_95_pct}%, Sharpe: {ex_ante_sharpe}\n"
        "Include before/after breakdown, 3 alternative policies, and 24h override deadline. Under 400 words.\n"
        "{format_instructions}"
    ),
    "ADVISOR_CALENDAR": (
        "Prepare an advisor dossier for a scheduled calendar rebalance review under policy mandate.\n"
        "Portfolio: {portfolio_id}, Risk Category: {risk_category}\n"
        "Turnover: {turnover_pct}%, Tracking Error: {tracking_error_reduction_bps} bps. Under 400 words.\n"
        "{format_instructions}"
    ),
    "ADVISOR_EVENT": (
        "Prepare an advisor briefing on market shock response or March tax-loss harvesting execution.\n"
        "Tax Alpha: INR {tax_shield_inr}, Portfolio: {portfolio_id}. Under 400 words.\n"
        "{format_instructions}"
    ),
    "COMPLIANCE_THRESHOLD": (
        "Generate a formal regulatory audit record citing SEBI Circular SEBI/HO/MRD/DOP1/CIR/P/2024/69.\n"
        "Verify all hard constraints, SEBI single-issuer (10%) and sector (30%) limits, and SHAP evidence.\n"
        "{format_instructions}"
    ),
    "COMPLIANCE_CALENDAR": (
        "Generate a regulatory compliance record verifying calendar mandate adherence against client IPS.\n"
        "{format_instructions}"
    ),
    "COMPLIANCE_EVENT": (
        "Generate a compliance audit record for event-driven autonomous rebalancing with cryptographic hash.\n"
        "{format_instructions}"
    ),
}


class ExplanationGenerator:
    """Primary orchestration framework generating audience-tailored explanations across the 9-template matrix."""

    def __init__(
        self,
        client_explainer: Optional[ClientExplainer] = None,
        advisor_explainer: Optional[AdvisorExplainer] = None,
        compliance_explainer: Optional[ComplianceExplainer] = None,
        shap_explainer: Optional[ShapExplainer] = None,
        lime_explainer: Optional[LimeExplainer] = None,
        counterfactual_generator: Optional[CounterfactualGenerator] = None,
        llm: Optional[Any] = None,
    ) -> None:
        self.shap_explainer = shap_explainer or ShapExplainer()
        self.lime_explainer = lime_explainer or LimeExplainer(surrogate_shap_explainer=self.shap_explainer)
        self.counterfactual_generator = counterfactual_generator or CounterfactualGenerator(shap_explainer=self.shap_explainer)

        self.client_explainer = client_explainer or ClientExplainer()
        self.advisor_explainer = advisor_explainer or AdvisorExplainer()
        self.compliance_explainer = compliance_explainer or ComplianceExplainer()
        self.llm = llm

        # LangChain Parsers
        self.client_parser = PydanticOutputParser(pydantic_object=ClientExplanationPayload)
        self.advisor_parser = PydanticOutputParser(pydantic_object=AdvisorExplanationPayload)
        self.compliance_parser = PydanticOutputParser(pydantic_object=ComplianceExplanationPayload)

    def get_template_key(self, audience: str, trigger_category: str) -> str:
        """Resolves template key from 3x3 matrix (e.g. 'CLIENT_THRESHOLD')."""
        aud = audience.upper()
        trig = trigger_category.upper()
        return f"{aud}_{trig}"

    def generate_full_explanation_packet(self, context: Dict[str, Any]) -> FullExplanationPacket:
        """Orchestrates comprehensive multi-tier explanations and surrogate explainability evidence.
        
        Args:
            context: Dictionary containing:
                - portfolio_id: str
                - decision_id: str
                - client_id: str
                - risk_category: str
                - trigger_category: "THRESHOLD", "CALENDAR", or "EVENT"
                - trigger_severity: "CRITICAL", "HIGH", "MEDIUM"
                - current_weights: Dict[str, float]
                - target_weights: Dict[str, float]
                - proposed_weights: Dict[str, float]
                - trades: List[TradeOrder or Dict]
                - sad: float
                - rmsd: float
                - tracking_error_reduction_bps: float
                - total_costs_inr: float
                - tax_impact_inr: float
                - constraint_matrix: Dict[str, bool]
                - sebi_compliant: bool
                - explainability_features: Dict[str, float]

        Returns:
            FullExplanationPacket containing structured payloads for Client, Advisor, and Compliance.
        """
        portfolio_id = str(context.get("portfolio_id", "PORT_UNKNOWN"))
        decision_id = str(context.get("decision_id", f"DEC_{portfolio_id}_{int(datetime.datetime.now().timestamp())}"))
        client_id = str(context.get("client_id", "CLIENT_001"))
        risk_cat = str(context.get("risk_category", "Balanced"))
        trigger_cat = str(context.get("trigger_category", "THRESHOLD")).upper()
        trigger_sev = str(context.get("trigger_severity", "HIGH"))

        curr_w = context.get("current_weights", {})
        targ_w = context.get("target_weights", {})
        prop_w = context.get("proposed_weights", {})
        trades = context.get("trades", [])

        sad = float(context.get("sad", 0.05))
        rmsd = float(context.get("rmsd", 0.025))
        te_red_bps = float(context.get("tracking_error_reduction_bps", 35.0))
        costs_inr = float(context.get("total_costs_inr", 1500.0))
        tax_inr = float(context.get("tax_impact_inr", 0.0))
        tax_shield_inr = float(context.get("tax_shield_inr", 0.0))

        constraint_matrix = context.get("constraint_matrix", {
            "budget_conserved": True,
            "long_only_valid": True,
            "cash_buffer_satisfied": True,
            "sebi_single_issuer_pass": True,
            "sebi_sector_limit_pass": True,
            "turnover_budget_pass": True,
        })
        sebi_compliant = bool(context.get("sebi_compliant", True))

        # 1. Compute surrogate explainability: SHAP, LIME, Counterfactuals
        features = context.get("explainability_features", {
            "equity_drift_pct": float(sad * 50.0),
            "vix_level": 16.5,
            "days_since_rebalance": 95.0,
            "client_risk_score": 3.0,
            "tax_lot_maturity_days": 240.0,
            "sector_concentration_pct": 25.0,
        })

        shap_res = self.shap_explainer.explain_instance(features)
        lime_res = self.lime_explainer.explain_instance(features)
        cf_res = self.counterfactual_generator.generate_counterfactual(features)

        # 2. Generate Tier 1: Client Plain English Explanation (Grade <= 8.0, <= 200 words)
        primary_asset = "NIFTY_50_EQUITY"
        curr_eq = curr_w.get(primary_asset, 0.50) * 100.0
        targ_eq = targ_w.get(primary_asset, 0.50) * 100.0
        drift_dir = "OVERWEIGHT" if curr_eq > targ_eq else "UNDERWEIGHT"

        client_payload = self.client_explainer.generate_explanation(
            portfolio_id=portfolio_id,
            decision_id=decision_id,
            trigger_category=trigger_cat,
            primary_asset=primary_asset,
            drift_direction=drift_dir,
            current_equity_pct=curr_eq,
            target_equity_pct=targ_eq,
            total_costs_inr=costs_inr,
            tax_impact_inr=tax_inr,
            client_goal=f"{risk_cat} Wealth Mandate",
        )

        # 3. Generate Tier 2: Advisor Quantitative Dossier (<= 400 words)
        advisor_payload = self.advisor_explainer.generate_dossier(
            portfolio_id=portfolio_id,
            decision_id=decision_id,
            risk_category=risk_cat,
            trigger_category=trigger_cat,
            current_weights=curr_w,
            target_weights=targ_w,
            proposed_weights=prop_w,
            trades=trades,
            sad=sad,
            rmsd=rmsd,
            tracking_error_reduction_bps=te_red_bps,
            turnover_pct=float(context.get("turnover", 0.05)) * 100.0,
            tax_shield_inr=tax_shield_inr,
        )

        # 4. Generate Tier 3: Compliance Immutable Audit Log
        compliance_payload = self.compliance_explainer.generate_compliance_audit(
            decision_id=decision_id,
            portfolio_id=portfolio_id,
            client_id=client_id,
            risk_category=risk_cat,
            trigger_category=trigger_cat,
            trigger_severity=trigger_sev,
            current_weights=curr_w,
            target_weights=targ_w,
            proposed_weights=prop_w,
            trades=trades,
            constraint_matrix=constraint_matrix,
            sebi_compliant=sebi_compliant,
            shap_evidence=shap_res,
            lime_evidence=lime_res,
            counterfactual_evidence=cf_res,
        )

        now_utc = datetime.datetime.now(datetime.timezone.utc).isoformat()

        return FullExplanationPacket(
            portfolio_id=portfolio_id,
            decision_id=decision_id,
            timestamp_utc=now_utc,
            trigger_category=trigger_cat,
            client_explanation=client_payload,
            advisor_dossier=advisor_payload,
            compliance_audit=compliance_payload,
            shap_attribution=shap_res["feature_attributions"],
            lime_top_features=lime_res["top_features"],
            counterfactual_scenario=cf_res,
        )
