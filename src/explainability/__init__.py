"""Explainability Engine for WealthPilot AI (Client, Advisor, Compliance Auditor).

Modules:
    - explanation_generator: Orchestration framework for 3x3 template matrix.
    - client_explainer: Flesch-Kincaid Grade 8 retail plain-language narratives.
    - advisor_explainer: Quantitative briefings, tracking error reduction, override protocol.
    - compliance_explainer: Immutable audit records with SEBI circular citation.
    - shap_integration: XGBoost surrogate model & Tree SHAP feature attributions.
    - lime_integration: Local linear surrogate models & top-5 contributing features.
    - counterfactual_generator: Minimal decision-boundary feature perturbation solver.
"""

from __future__ import annotations

from src.explainability.client_explainer import (
    ClientExplainer,
    ClientExplanationPayload,
    calculate_flesch_kincaid_grade,
)
from src.explainability.advisor_explainer import (
    AdvisorExplainer,
    AdvisorExplanationPayload,
    AllocationRow,
)
from src.explainability.compliance_explainer import (
    ComplianceExplainer,
    ComplianceExplanationPayload,
    REGULATORY_CIRCULAR_SEBI,
)
from src.explainability.shap_integration import (
    ShapExplainer,
    EXPLAINABILITY_FEATURES,
)
from src.explainability.lime_integration import LimeExplainer
from src.explainability.counterfactual_generator import CounterfactualGenerator
from src.explainability.explanation_generator import (
    ExplanationGenerator,
    FullExplanationPacket,
    EXPLANATION_PROMPT_TEMPLATES,
)

__all__ = [
    "ExplanationGenerator",
    "FullExplanationPacket",
    "EXPLANATION_PROMPT_TEMPLATES",
    "ClientExplainer",
    "ClientExplanationPayload",
    "calculate_flesch_kincaid_grade",
    "AdvisorExplainer",
    "AdvisorExplanationPayload",
    "AllocationRow",
    "ComplianceExplainer",
    "ComplianceExplanationPayload",
    "REGULATORY_CIRCULAR_SEBI",
    "ShapExplainer",
    "EXPLAINABILITY_FEATURES",
    "LimeExplainer",
    "CounterfactualGenerator",
]
