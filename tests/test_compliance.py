"""Unit and integration tests for Compliance Audit Engine, Bias Detector, and Scorecard."""

import json
import pytest
from src.compliance.explainability_scorecard import ExplainabilityScorecard, ExplanationScorecardResult
from src.compliance.bias_detector import BiasDetector, BiasReport
from src.compliance.compliance_auditor import ComplianceAuditor, AuditRunResult
from src.compliance.regulatory_reporter import RegulatoryReporter


@pytest.fixture
def scorecard():
    return ExplainabilityScorecard(target_max_grade=8.0, client_max_words=200, advisor_max_words=400)


@pytest.fixture
def bias_detector():
    return BiasDetector(max_frequency_disparity_ratio=1.60, max_cost_underestimation_pct=10.0)


@pytest.fixture
def compliance_auditor(scorecard):
    return ComplianceAuditor(scorecard=scorecard, min_audit_pass_rate_pct=90.0)


@pytest.fixture
def regulatory_reporter():
    return RegulatoryReporter()


@pytest.fixture
def compliant_explanation_packet():
    return {
        "decision_id": "DEC-2026-001",
        "client_explanation": {
            "headline": "Portfolio Rebalanced to Maintain Target Risk",
            "narrative": "Your portfolio was rebalanced after equity gains pushed stock allocation higher. We trimmed equities safely.",
            "goal_relevance": "Protects your retirement horizon from uncompensated volatility.",
            "cost_transparency": {
                "total_estimated_costs_inr": 250.0,
                "current_equity_pct": 55.0,
            },
            "readability_grade": 7.2,
            "word_count": 85,
        },
        "advisor_dossier": {
            "executive_summary": "Rebalanced Large-Cap equity drift of +5.0% into sovereign debt with tracking error under 0.5%.",
            "drift_metrics": {"large_cap_drift": 0.05},
            "counterfactual_alternatives": ["Hold position and incur tracking error penalty"],
            "word_count": 180,
        },
        "compliance_audit": {
            "sebi_compliance_certified": True,
            "regulatory_circular": "SEBI/HO/MRD/DOP1/CIR/P/2024/69",
            "constraint_verification_matrix": {"single_issuer_ok": True, "sector_cap_ok": True},
            "digital_signature_hash": "a1b2c3d4e5f60718293a4b5c6d7e8f90a1b2c3d4e5f60718293a4b5c6d7e8f90",
        },
    }


@pytest.fixture
def sample_decision_universe(compliant_explanation_packet):
    """Generates 120 diverse decisions across trigger categories and risk profiles."""
    decisions = []
    risk_profiles = ["Conservative", "Moderately Conservative", "Balanced", "Growth", "Aggressive"]
    triggers = ["THRESHOLD", "CALENDAR", "EVENT"]

    for i in range(120):
        risk = risk_profiles[i % len(risk_profiles)]
        trigger = triggers[i % len(triggers)]
        aum = 500000.0 * (1 + (i % 8))

        dec = {
            "decision_id": f"DEC-AUDIT-{i:04d}",
            "portfolio_id": f"PORT-{i:05d}",
            "risk_category": risk,
            "trigger_category": trigger,
            "aum_inr": aum,
            "total_costs_inr": 250.0,
            "current_equity_pct": 55.0,
            "sebi_compliant": True,
            "rebalance_executed": True,
            "estimated_cost_inr": 250.0,
            "realized_cost_inr": 252.0,
            "explanation_packet": compliant_explanation_packet,
        }
        decisions.append(dec)
    return decisions


def test_scorecard_compliant_packet(scorecard, compliant_explanation_packet):
    """Verifies that an authentic, complete explanation packet receives Grade A and passes."""
    context = {"total_costs_inr": 250.0, "current_equity_pct": 55.0}
    res = scorecard.score_explanation_packet(compliant_explanation_packet, actual_trade_context=context)
    assert isinstance(res, ExplanationScorecardResult)
    assert res.composite_score >= 85.0
    assert res.grade in ("A", "B")
    assert res.is_compliant is True
    assert res.accuracy_score == 25.0
    assert res.completeness_score == 25.0
    assert len(res.deficiencies) == 0


def test_scorecard_deficient_packet(scorecard):
    """Verifies that an explanation missing disclosure items and circular reference fails."""
    bad_packet = {
        "decision_id": "DEC-FAIL-01",
        "client_explanation": {"headline": "Trade"},
        "advisor_dossier": {},
        "compliance_audit": {"sebi_compliance_certified": False},
    }
    res = scorecard.score_explanation_packet(bad_packet)
    assert res.composite_score < 75.0
    assert res.is_compliant is False
    assert len(res.deficiencies) > 0


def test_bias_detector_fair_cohort(bias_detector, sample_decision_universe):
    """Verifies that balanced rebalancing decision patterns show no algorithmic bias."""
    report = bias_detector.analyze_decisions(sample_decision_universe)
    assert isinstance(report, BiasReport)
    assert report.evaluated_decisions_count == len(sample_decision_universe)
    assert report.overall_bias_detected is False
    assert len(report.flags) == 0


def test_bias_detector_flags_cost_underestimation(bias_detector):
    """Verifies that systematic cost underestimation is flagged."""
    biased_records = []
    for i in range(40):
        biased_records.append({
            "decision_id": f"DEC-UNDER-{i}",
            "risk_category": "Balanced",
            "aum_inr": 1000000.0,
            "estimated_cost_inr": 100.0,
            "realized_cost_inr": 300.0,  # 200% underestimation
        })
    report = bias_detector.analyze_decisions(biased_records)
    assert report.overall_bias_detected is True
    assert any("underestimation" in f.lower() for f in report.flags)


def test_compliance_auditor_sampling_and_run(compliance_auditor, sample_decision_universe):
    """Verifies stratified sampling of 100 decisions, audit scoring, and SHA-256 digital seal."""
    # Test sampling
    sample = compliance_auditor.draw_stratified_sample(sample_decision_universe, sample_size=100)
    assert len(sample) == 100

    # Test full audit run
    audit_res = compliance_auditor.run_quarterly_audit(sample_decision_universe, sample_size=100, quarter="Q4-2024")
    assert isinstance(audit_res, AuditRunResult)
    assert audit_res.total_sampled == 100
    assert audit_res.overall_verdict == "AUDIT_PASSED"
    assert audit_res.audit_pass_rate_pct >= 90.0
    assert len(audit_res.audit_signature_hash) == 64
    assert len(audit_res.trigger_stratification) == 3
    assert len(audit_res.risk_stratification) == 5


def test_regulatory_reporter_package_and_export(compliance_auditor, regulatory_reporter, sample_decision_universe):
    """Verifies generation and serialization of the SEBI audit package."""
    audit_res = compliance_auditor.run_quarterly_audit(sample_decision_universe, sample_size=50)

    sample_trades = [
        {"trade_id": "T1", "order_value_inr": 500000.0, "stt_inr": 500.0},
        {"trade_id": "T2", "order_value_inr": 250000.0, "stt_inr": 250.0},
    ]

    pkg = regulatory_reporter.generate_sebi_audit_package(
        audit_result=audit_res,
        transaction_logs=sample_trades,
        reporting_period="Q4-FY2025-26",
    )
    assert isinstance(pkg, dict)
    assert pkg["reporting_entity"] == "WealthPilot AI Portfolio Management Services Pvt Ltd"
    assert "SEBI Circular SEBI/HO/MRD/DOP1/CIR/P/2024/69" in pkg["statutory_references"]
    assert pkg["executive_summary"]["overall_verdict"] == "AUDIT_PASSED"
    assert len(pkg["cryptographic_provenance_seal"]) == 64

    # Test Markdown formatting
    md = regulatory_reporter.export_audit_package_markdown(pkg)
    assert "# SEBI STATUTORY ALGORITHMIC REBALANCING AUDIT REPORT" in md
    assert pkg["cryptographic_provenance_seal"] in md

    # Test JSON serialization
    json_str = regulatory_reporter.export_audit_package_json(pkg)
    parsed = json.loads(json_str)
    assert parsed["package_id"] == pkg["package_id"]
