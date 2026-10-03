"""Explanation Writer Agent for WealthPilot AI.

Role: Financial Narrative & Explainability Specialist
Generates structured multi-audience explanations across Client, Advisor, and Compliance tiers
using the 9-template matrix, Flesch-Kincaid readability scoring, and surrogate explainability.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from src.explainability.explanation_generator import ExplanationGenerator, FullExplanationPacket

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


class ExplanationWriterAgent:
    """Specialized agent synthesizing structured explanations across multi-stakeholder audiences."""

    def __init__(
        self,
        explanation_generator: Optional[ExplanationGenerator] = None,
    ) -> None:
        self.role = "Financial Narrative & Explainability Specialist"
        self.goal = (
            "Produce clear, audience-tailored rebalancing explanations (Grade 8 for clients, quantitative for advisors, "
            "and statutory regulatory audit packets for compliance officers)."
        )
        self.backstory = (
            "An expert in financial communication, regulatory transparency, and machine learning interpretability. "
            "Skilled at translating complex convex optimization math into clear plain-English narratives while providing "
            "rock-solid quantitative evidence and SHAP/LIME surrogate explanations for auditors."
        )
        self.generator = explanation_generator or ExplanationGenerator()

    def as_crewai_agent(self, tools: Optional[List[Any]] = None) -> Optional[Any]:
        """Returns a configured CrewAI Agent if crewai library is installed."""
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
        """Synthesizes structured explanations across all 3 tiers.
        
        Args:
            context: Dictionary containing portfolio, trade, constraint, and feature data.

        Returns:
            Dict containing client, advisor, compliance payloads, and surrogate attributions.
        """
        packet: FullExplanationPacket = self.generator.generate_full_explanation_packet(context)

        client_dict = packet.client_explanation.model_dump()
        client_dict["plain_english_summary"] = client_dict.get("narrative", "")

        comp_dict = packet.compliance_audit.model_dump()
        comp_dict["legal_attestation"] = (
            f"Adherent to SEBI Circular {packet.compliance_audit.regulatory_circular}. "
            f"Certified compliant: {packet.compliance_audit.sebi_compliance_certified}."
        )

        return {
            "status": "SUCCESS",
            "portfolio_id": packet.portfolio_id,
            "decision_id": packet.decision_id,
            "timestamp_utc": packet.timestamp_utc,
            "trigger_category": packet.trigger_category,
            "client_explanation": client_dict,
            "advisor_dossier": packet.advisor_dossier.model_dump(),
            "compliance_audit": comp_dict,
            "shap_attribution": packet.shap_attribution,
            "lime_top_features": packet.lime_top_features,
            "counterfactual_scenario": packet.counterfactual_scenario,
            "full_packet": packet,
        }

    def draft_explanations(self, decision_context: Dict[str, Any]) -> Dict[str, str]:
        res = self.execute_task(decision_context)
        return {
            "client_explanation": res["client_explanation"].get("plain_english_summary", ""),
            "advisor_explanation": res["advisor_dossier"].get("executive_summary", ""),
            "compliance_explanation": res["compliance_audit"].get("legal_attestation", ""),
        }


ExplanationWriter = ExplanationWriterAgent
