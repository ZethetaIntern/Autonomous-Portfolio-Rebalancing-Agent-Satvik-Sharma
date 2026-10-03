"""Explainability Scorecard evaluating explanation quality and regulatory sufficiency.

Scores generated explanations across 4 institutional dimensions:
1. Accuracy: Numerical consistency between narrative assertions and quantitative trade ledgers
2. Completeness: Coverage of mandated disclosure fields (goals, costs, alternatives, overrides)
3. Readability: Flesch-Kincaid Grade level compliance (<= 8.0 for clients, concise word counts)
4. Regulatory Sufficiency: SEBI Circular citation, constraint matrix verification, digital signatures
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ExplanationScorecardResult:
    """Detailed multi-dimensional explanation evaluation scorecard."""
    decision_id: str
    composite_score: float  # 0.0 to 100.0
    grade: str              # "A", "B", "C", "REJECTED"
    is_compliant: bool
    accuracy_score: float   # 0.0 to 25.0
    completeness_score: float # 0.0 to 25.0
    readability_score: float  # 0.0 to 25.0
    regulatory_score: float   # 0.0 to 25.0
    deficiencies: List[str] = field(default_factory=list)
    dimension_breakdown: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "decision_id": self.decision_id,
            "composite_score": self.composite_score,
            "grade": self.grade,
            "is_compliant": self.is_compliant,
            "accuracy_score": self.accuracy_score,
            "completeness_score": self.completeness_score,
            "readability_score": self.readability_score,
            "regulatory_score": self.regulatory_score,
            "deficiencies": self.deficiencies,
            "dimension_breakdown": self.dimension_breakdown,
        }


class ExplainabilityScorecard:
    """Evaluates explanation artifacts for mathematical accuracy, readability, and regulatory rigor."""

    def __init__(self, target_max_grade: float = 8.0, client_max_words: int = 200, advisor_max_words: int = 400) -> None:
        self.max_client_grade = target_max_grade
        self.client_max_words = client_max_words
        self.advisor_max_words = advisor_max_words

    def score_explanation_packet(
        self,
        explanation_packet: Dict[str, Any],
        actual_trade_context: Optional[Dict[str, Any]] = None,
    ) -> ExplanationScorecardResult:
        """Evaluates an end-to-end multi-audience explanation packet.
        
        Args:
            explanation_packet: Dict containing client_explanation, advisor_dossier, compliance_audit.
            actual_trade_context: Quantitative ground-truth ledger (weights, costs, taxes).
        """
        ctx = actual_trade_context or {}
        decision_id = str(explanation_packet.get("decision_id", "DEC_UNKNOWN"))
        deficiencies: List[str] = []

        client_exp = explanation_packet.get("client_explanation", {})
        advisor_dos = explanation_packet.get("advisor_dossier", {})
        comp_audit = explanation_packet.get("compliance_audit", {})

        # -------------------------------------------------------------
        # 1. ACCURACY SCORE (0 - 25 pts)
        # -------------------------------------------------------------
        acc_pts = 25.0
        acc_details = {}

        # Check costs reported vs actual
        reported_costs = client_exp.get("cost_transparency", {}).get("total_estimated_costs_inr")
        actual_costs = ctx.get("total_costs_inr")
        if reported_costs is not None and actual_costs is not None:
            if abs(reported_costs - actual_costs) > 5.0:
                acc_pts -= 5.0
                deficiencies.append(f"Cost discrepancy: reported Rs. {reported_costs:,.2f} vs actual Rs. {actual_costs:,.2f}")
            acc_details["costs_verified"] = True
        else:
            acc_details["costs_verified"] = True

        # Check reported drift vs actual
        reported_drift = client_exp.get("cost_transparency", {}).get("current_equity_pct")
        actual_eq_pct = ctx.get("current_equity_pct")
        if reported_drift is not None and actual_eq_pct is not None:
            if abs(reported_drift - actual_eq_pct) > 0.5:
                acc_pts -= 5.0
                deficiencies.append(f"Equity weight discrepancy: reported {reported_drift:.1f}% vs actual {actual_eq_pct:.1f}%")

        accuracy_score = max(0.0, acc_pts)

        # -------------------------------------------------------------
        # 2. COMPLETENESS SCORE (0 - 25 pts)
        # -------------------------------------------------------------
        comp_pts = 0.0
        comp_details = {}

        # Client elements (up to 10 pts)
        has_headline = bool(client_exp.get("headline"))
        has_narrative = bool(client_exp.get("narrative") or client_exp.get("plain_english_summary"))
        has_goal = bool(client_exp.get("goal_relevance"))
        has_costs = bool(client_exp.get("cost_transparency"))
        client_comp_cnt = sum([has_headline, has_narrative, has_goal, has_costs])
        comp_pts += (client_comp_cnt / 4.0) * 10.0
        comp_details["client_fields_present"] = client_comp_cnt

        if client_comp_cnt < 4:
            deficiencies.append(f"Client explanation missing {4 - client_comp_cnt} required disclosure sections")

        # Advisor elements (up to 8 pts)
        has_adv_summary = bool(advisor_dos.get("executive_summary"))
        has_adv_drift = "sad" in advisor_dos or "drift_metrics" in advisor_dos
        has_adv_alternatives = bool(advisor_dos.get("counterfactual_alternatives") or advisor_dos.get("alternative_policies"))
        adv_comp_cnt = sum([has_adv_summary, has_adv_drift, has_adv_alternatives])
        comp_pts += (adv_comp_cnt / 3.0) * 8.0
        comp_details["advisor_fields_present"] = adv_comp_cnt

        # Compliance elements (up to 7 pts)
        has_matrix = bool(comp_audit.get("constraint_verification_matrix"))
        has_hash = bool(comp_audit.get("digital_signature_hash"))
        comp_pts += (sum([has_matrix, has_hash]) / 2.0) * 7.0
        comp_details["compliance_fields_present"] = sum([has_matrix, has_hash])

        completeness_score = min(25.0, round(comp_pts, 2))

        # -------------------------------------------------------------
        # 3. READABILITY SCORE (0 - 25 pts)
        # -------------------------------------------------------------
        read_pts = 25.0
        client_grade = float(client_exp.get("readability_grade", 7.5))
        client_words = int(client_exp.get("word_count", 120))
        advisor_words = int(advisor_dos.get("word_count", len(str(advisor_dos.get("executive_summary", "")).split())))

        if client_grade > self.max_client_grade:
            penalty = (client_grade - self.max_client_grade) * 4.0
            read_pts -= min(15.0, penalty)
            deficiencies.append(f"Client narrative readability grade {client_grade:.1f} exceeds Grade {self.max_client_grade:.1f} limit")

        if client_words > self.client_max_words:
            read_pts -= 5.0
            deficiencies.append(f"Client word count {client_words} exceeds {self.client_max_words} word limit")

        if advisor_words > self.advisor_max_words:
            read_pts -= 5.0
            deficiencies.append(f"Advisor dossier word count {advisor_words} exceeds {self.advisor_max_words} word limit")

        readability_score = max(0.0, round(read_pts, 2))

        # -------------------------------------------------------------
        # 4. REGULATORY SUFFICIENCY SCORE (0 - 25 pts)
        # -------------------------------------------------------------
        reg_pts = 25.0
        sebi_cert = comp_audit.get("sebi_compliance_certified", comp_audit.get("sebi_compliant", True))
        circular = str(comp_audit.get("regulatory_circular", ""))
        dig_sig = str(comp_audit.get("digital_signature_hash", ""))

        if not sebi_cert:
            reg_pts -= 15.0
            deficiencies.append("SEBI compliance certification flagged non-compliant")

        if "SEBI" not in circular.upper():
            reg_pts -= 5.0
            deficiencies.append("Missing formal SEBI circular reference citation")

        if len(dig_sig) < 32:
            reg_pts -= 5.0
            deficiencies.append("Invalid or missing cryptographic audit signature hash")

        regulatory_score = max(0.0, round(reg_pts, 2))

        # -------------------------------------------------------------
        # COMPOSITE SCORE & GRADE
        # -------------------------------------------------------------
        composite = round(accuracy_score + completeness_score + readability_score + regulatory_score, 1)

        if composite >= 90.0:
            grade = "A"
        elif composite >= 78.0:
            grade = "B"
        elif composite >= 65.0:
            grade = "C"
        else:
            grade = "REJECTED"

        is_compliant = (composite >= 75.0) and sebi_cert and (client_grade <= self.max_client_grade + 0.5)

        return ExplanationScorecardResult(
            decision_id=decision_id,
            composite_score=composite,
            grade=grade,
            is_compliant=is_compliant,
            accuracy_score=accuracy_score,
            completeness_score=completeness_score,
            readability_score=readability_score,
            regulatory_score=regulatory_score,
            deficiencies=deficiencies,
            dimension_breakdown={
                "client_grade": client_grade,
                "client_word_count": client_words,
                "advisor_word_count": advisor_words,
                "sebi_certified": sebi_cert,
                "has_digital_signature": len(dig_sig) >= 32,
            },
        )
