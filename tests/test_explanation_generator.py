"""Comprehensive Unit & Integration Test Suite for Phase 3 (Days 6–7).

Tests:
    - Pydantic schema validation for Client, Advisor, and Compliance explanation payloads.
    - Flesch-Kincaid Grade 8 readability index and 200-word conciseness bounds.
    - Advisor quantitative dossier (< 400 words), before/after tables, and override protocols.
    - Compliance audit log immutability, SEBI circular citation, and SHA-256 digital signature.
    - XGBoost surrogate model training, Tree SHAP feature attributions, and waterfall structures.
    - LIME local linear approximations and top-5 contributing features.
    - Counterfactual minimal distance feature perturbations across the decision boundary.
    - 3x3 template matrix orchestration in ExplanationGenerator.
"""

import math
import pytest

from src.explainability import (
    AdvisorExplainer,
    AdvisorExplanationPayload,
    ClientExplainer,
    ClientExplanationPayload,
    ComplianceExplainer,
    ComplianceExplanationPayload,
    CounterfactualGenerator,
    ExplanationGenerator,
    FullExplanationPacket,
    LimeExplainer,
    REGULATORY_CIRCULAR_SEBI,
    ShapExplainer,
    calculate_flesch_kincaid_grade,
)


@pytest.fixture
def sample_context():
    return {
        "portfolio_id": "PORT_1D_TEST",
        "decision_id": "DEC_2026_10_03_001",
        "client_id": "CLI_RAJESH_SHARMA",
        "risk_category": "Balanced",
        "trigger_category": "THRESHOLD",
        "trigger_severity": "HIGH",
        "current_weights": {
            "NIFTY_50_EQUITY": 0.62,
            "G_SEC_BONDS": 0.18,
            "CORP_BONDS": 0.12,
            "GOLD_ETF": 0.05,
            "LIQUID_CASH": 0.03,
        },
        "target_weights": {
            "NIFTY_50_EQUITY": 0.50,
            "G_SEC_BONDS": 0.25,
            "CORP_BONDS": 0.15,
            "GOLD_ETF": 0.05,
            "LIQUID_CASH": 0.05,
        },
        "proposed_weights": {
            "NIFTY_50_EQUITY": 0.54,
            "G_SEC_BONDS": 0.21,
            "CORP_BONDS": 0.15,
            "GOLD_ETF": 0.05,
            "LIQUID_CASH": 0.05,
        },
        "trades": [
            {
                "asset": "NIFTY_50_EQUITY",
                "action": "SELL",
                "units": 33.0,
                "price": 24000.0,
                "val_inr": 792000.0,
            },
            {
                "asset": "G_SEC_BONDS",
                "action": "BUY",
                "units": 7920.0,
                "price": 100.0,
                "val_inr": 792000.0,
            },
        ],
        "sad": 0.08,
        "rmsd": 0.04,
        "tracking_error_reduction_bps": 42.5,
        "total_costs_inr": 1820.0,
        "tax_impact_inr": 2500.0,
        "tax_shield_inr": 3750.0,
        "turnover": 0.06,
        "constraint_matrix": {
            "budget_conserved": True,
            "long_only_valid": True,
            "cash_buffer_satisfied": True,
            "sebi_single_issuer_pass": True,
            "sebi_sector_limit_pass": True,
            "turnover_budget_pass": True,
        },
        "sebi_compliant": True,
        "explainability_features": {
            "equity_drift_pct": 4.8,
            "vix_level": 18.2,
            "days_since_rebalance": 98.0,
            "client_risk_score": 3.0,
            "tax_lot_maturity_days": 210.0,
            "sector_concentration_pct": 28.0,
        },
    }


class TestClientExplainer:
    """Tests retail client plain-English narrative, readability, and cost transparency."""

    def test_client_narrative_readability_grade_and_length(self):
        explainer = ClientExplainer()
        payload = explainer.generate_explanation(
            portfolio_id="PORT_01",
            decision_id="DEC_01",
            trigger_category="THRESHOLD",
            primary_asset="NIFTY_50_EQUITY",
            drift_direction="OVERWEIGHT",
            current_equity_pct=62.0,
            target_equity_pct=50.0,
            total_costs_inr=1500.0,
            tax_impact_inr=2000.0,
            client_goal="Balanced Growth",
        )

        assert isinstance(payload, ClientExplanationPayload)
        # Readability must be at or below Grade 8
        assert payload.readability_grade <= 8.0
        # Word count strictly <= 200 words
        assert payload.word_count <= 200
        assert payload.word_count > 10

        # Cost transparency check
        assert payload.cost_transparency["brokerage_and_fees_inr"] == 1500.0
        assert payload.cost_transparency["tax_impact_inr"] == 2000.0
        assert payload.cost_transparency["net_total_inr"] == 3500.0

    def test_flesch_kincaid_formula(self):
        simple_text = "The dog sat on the red mat. It was a good day."
        grade = calculate_flesch_kincaid_grade(simple_text)
        assert grade <= 4.0  # Simple text is 1st - 4th grade


class TestAdvisorExplainer:
    """Tests financial advisor quantitative dossier, before/after tables, and override limits."""

    def test_advisor_dossier_structure_and_limits(self, sample_context):
        explainer = AdvisorExplainer()
        ctx = sample_context

        payload = explainer.generate_dossier(
            portfolio_id=ctx["portfolio_id"],
            decision_id=ctx["decision_id"],
            risk_category=ctx["risk_category"],
            trigger_category=ctx["trigger_category"],
            current_weights=ctx["current_weights"],
            target_weights=ctx["target_weights"],
            proposed_weights=ctx["proposed_weights"],
            trades=ctx["trades"],
            sad=ctx["sad"],
            rmsd=ctx["rmsd"],
            tracking_error_reduction_bps=ctx["tracking_error_reduction_bps"],
            turnover_pct=ctx["turnover"] * 100.0,
            tax_shield_inr=ctx["tax_shield_inr"],
        )

        assert isinstance(payload, AdvisorExplanationPayload)
        # Length under 400 words
        assert payload.word_count <= 400
        assert payload.word_count > 50

        # Allocation table rows must match asset classes
        assert len(payload.allocation_table) == 5
        equity_row = next(r for r in payload.allocation_table if r.asset_class == "NIFTY_50_EQUITY")
        assert equity_row.current_pct == 62.0
        assert equity_row.target_pct == 50.0
        assert equity_row.proposed_pct == 54.0

        # Override protocol present
        assert payload.override_protocol["override_eligible"] is True
        assert "deadline_utc" in payload.override_protocol

        # 3 alternative policies considered
        assert len(payload.alternative_strategies_considered) == 3


class TestComplianceExplainer:
    """Tests regulatory audit logs, SEBI circular citation, and digital signature immutability."""

    def test_compliance_audit_and_digital_signature(self, sample_context):
        explainer = ComplianceExplainer()
        ctx = sample_context

        payload = explainer.generate_compliance_audit(
            decision_id=ctx["decision_id"],
            portfolio_id=ctx["portfolio_id"],
            client_id=ctx["client_id"],
            risk_category=ctx["risk_category"],
            trigger_category=ctx["trigger_category"],
            trigger_severity=ctx["trigger_severity"],
            current_weights=ctx["current_weights"],
            target_weights=ctx["target_weights"],
            proposed_weights=ctx["proposed_weights"],
            trades=ctx["trades"],
            constraint_matrix=ctx["constraint_matrix"],
            sebi_compliant=ctx["sebi_compliant"],
        )

        assert isinstance(payload, ComplianceExplanationPayload)
        # Check SEBI Circular
        assert payload.regulatory_circular == REGULATORY_CIRCULAR_SEBI
        assert payload.sebi_compliance_certified is True
        assert len(payload.digital_signature_hash) == 64  # SHA-256 length

        # Verify integrity
        assert explainer.verify_integrity(payload) is True

        # Tampering detection
        payload.decision_id = "TAMPERED_ID"
        assert explainer.verify_integrity(payload) is False


class TestShapIntegration:
    """Tests Tree SHAP surrogate model training and feature attributions."""

    def test_shap_training_and_explanation(self, sample_context):
        shap_engine = ShapExplainer()
        features = sample_context["explainability_features"]

        res = shap_engine.explain_instance(features)

        assert res["decision"] in ("TRIGGER", "NO_TRIGGER")
        assert 0.0 <= res["predicted_trigger_probability"] <= 1.0
        assert "equity_drift_pct" in res["feature_attributions"]

        # Waterfall structure verification
        wf = res["waterfall_structure"]
        assert len(wf["values"]) == len(shap_engine.feature_names)
        assert len(wf["data"]) == len(shap_engine.feature_names)
        assert "base_value" in wf

        # Ranked drivers
        assert len(res["ranked_drivers"]) == 6
        assert res["top_driver"] in shap_engine.feature_names


class TestLimeIntegration:
    """Tests LIME local surrogate linear approximations and top-5 contributing features."""

    def test_lime_local_explanation(self, sample_context):
        lime_engine = LimeExplainer()
        features = sample_context["explainability_features"]

        res = lime_engine.explain_instance(features, num_features=5)

        assert 0.0 <= res["local_prediction"] <= 1.0
        assert len(res["top_features"]) <= 5
        assert len(res["top_features"]) >= 1

        top = res["top_features"][0]
        assert "feature_rule" in top
        assert "weight" in top
        assert "direction" in top
        assert top["direction"] in (
            "INCREASED_TRIGGER_PROBABILITY",
            "DECREASED_TRIGGER_PROBABILITY",
        )


class TestCounterfactualGenerator:
    """Tests minimal feature perturbation across autonomous decision boundary."""

    def test_counterfactual_generation(self, sample_context):
        cf_engine = CounterfactualGenerator()
        features = sample_context["explainability_features"]

        res = cf_engine.generate_counterfactual(features)

        assert res["original_decision"] != res["target_decision"]
        assert res["l1_distance"] >= 0.0
        assert len(res["minimal_perturbations"]) >= 1
        assert "counterfactual_narrative" in res
        assert "the agent would NOT have triggered a rebalance" in res["counterfactual_narrative"]


class TestExplanationGeneratorOrchestration:
    """Tests full orchestration across 3x3 template matrix and Pydantic validation."""

    def test_full_explanation_packet_generation(self, sample_context):
        generator = ExplanationGenerator()
        packet = generator.generate_full_explanation_packet(sample_context)

        assert isinstance(packet, FullExplanationPacket)
        assert packet.portfolio_id == sample_context["portfolio_id"]
        assert packet.decision_id == sample_context["decision_id"]

        # Client tier checks
        assert packet.client_explanation.readability_grade <= 8.0
        assert packet.client_explanation.word_count <= 200

        # Advisor tier checks
        assert packet.advisor_dossier.word_count <= 400
        assert len(packet.advisor_dossier.allocation_table) == 5

        # Compliance tier checks
        assert packet.compliance_audit.regulatory_circular == REGULATORY_CIRCULAR_SEBI
        assert len(packet.compliance_audit.digital_signature_hash) == 64

        # Explainability embeddings
        assert "equity_drift_pct" in packet.shap_attribution
        assert len(packet.lime_top_features) >= 1
        assert packet.counterfactual_scenario["target_decision"] == "NO_TRIGGER"
