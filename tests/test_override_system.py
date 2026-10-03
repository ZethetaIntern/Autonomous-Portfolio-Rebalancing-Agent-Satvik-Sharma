"""Unit tests for Human-in-the-Loop Override & Escalation Governance System."""

from __future__ import annotations

import pytest

from src.override.intervention_classifier import (
    InterventionClassifier,
    InterventionTier,
    InterventionCategory,
    InterventionClassificationResult,
)
from src.override.override_capture import OverrideCapture, AdvisorOverrideRecord
from src.override.escalation_manager import EscalationManager, EscalationStatus
from src.override.kill_switch import KillSwitch, CircuitBreakerState


class TestInterventionClassifier:
    """Test suite for 4-tier graduated intervention model and reason taxonomy."""

    @pytest.fixture
    def classifier(self) -> InterventionClassifier:
        return InterventionClassifier()

    def test_informational_tier(self, classifier: InterventionClassifier):
        proposal = {
            "turnover": 0.03,
            "portfolio_aum": 2_000_000.0,
            "total_trade_value_inr": 60_000.0,
            "sad": 0.02,
        }
        res = classifier.classify_proposal(proposal, agent_confidence=0.95, compliance_passed=True)
        assert res.tier == InterventionTier.INFORMATIONAL
        assert res.waiting_period_hours == 0.0
        assert not res.requires_human_signoff
        assert res.escalation_target is None

    def test_advisory_tier(self, classifier: InterventionClassifier):
        proposal = {
            "turnover": 0.08,
            "portfolio_aum": 5_000_000.0,
            "sad": 0.06,
        }
        res = classifier.classify_proposal(proposal, agent_confidence=0.85, compliance_passed=True)
        assert res.tier == InterventionTier.ADVISORY
        assert res.waiting_period_hours in [4.0, 24.0]
        assert not res.requires_human_signoff

    def test_approval_required_tier_high_turnover(self, classifier: InterventionClassifier):
        proposal = {
            "turnover": 0.22,
            "portfolio_aum": 5_000_000.0,
            "sad": 0.10,
        }
        res = classifier.classify_proposal(proposal, agent_confidence=0.88, compliance_passed=True)
        assert res.tier == InterventionTier.APPROVAL_REQUIRED
        assert res.requires_human_signoff
        assert res.escalation_target == "RELATIONSHIP_MANAGER"

    def test_approval_required_large_trade_value(self, classifier: InterventionClassifier):
        proposal = {
            "turnover": 0.06,
            "portfolio_aum": 100_000_000.0,  # 10 Crores -> trade value = 60 Lakhs > 50L threshold
            "total_trade_value_inr": 6_000_000.0,
            "sad": 0.04,
        }
        res = classifier.classify_proposal(proposal, agent_confidence=0.92, compliance_passed=True)
        assert res.tier == InterventionTier.APPROVAL_REQUIRED
        assert res.requires_human_signoff

    def test_escalation_tier_compliance_failure(self, classifier: InterventionClassifier):
        proposal = {"turnover": 0.04, "portfolio_aum": 2_000_000.0, "sad": 0.03}
        res = classifier.classify_proposal(
            proposal,
            agent_confidence=0.95,
            compliance_passed=False,
            constraint_breaches=["Single-issuer limit exceeded (12.5% > 10.0%)"],
        )
        assert res.tier == InterventionTier.ESCALATION
        assert res.requires_human_signoff
        assert res.escalation_target == "CHIEF_RISK_OFFICER_AND_COMPLIANCE_HEAD"
        assert "SEBI/Mandate compliance" in res.rationale

    def test_escalation_tier_severe_drift(self, classifier: InterventionClassifier):
        proposal = {"turnover": 0.10, "portfolio_aum": 5_000_000.0, "sad": 0.18}
        res = classifier.classify_proposal(proposal, agent_confidence=0.90, compliance_passed=True)
        assert res.tier == InterventionTier.ESCALATION
        assert "Severe portfolio drift" in res.rationale

    def test_reason_taxonomy_classification(self, classifier: InterventionClassifier):
        assert classifier.classify_reason("Harvest short-term capital losses before March 31") == InterventionCategory.TAX_OPTIMIZATION
        assert classifier.classify_reason("Client requested cash withdrawal for real estate down payment") == InterventionCategory.LIQUIDITY_NEEDS
        assert classifier.classify_reason("Client preference to avoid pharmaceutical stocks") == InterventionCategory.CLIENT_PREFERENCE
        assert classifier.classify_reason("Anticipating IT sector outperformance next quarter") == InterventionCategory.TACTICAL_VIEW
        assert classifier.classify_reason("Investment Committee special policy exception") == InterventionCategory.POLICY_EXCEPTION
        assert classifier.classify_reason("Reducing portfolio duration due to interest rate risk") == InterventionCategory.RISK_REDUCTION


class TestOverrideCapture:
    """Test suite for advisor override capture, delta analysis, and audit records."""

    def test_capture_and_delta_calculation(self):
        capture = OverrideCapture()
        original = {
            "optimal_weights": {"NIFTY_50_EQUITY": 0.60, "G_SEC_10Y_BOND": 0.40},
            "portfolio_aum": 10_000_000.0,
        }
        modified = {
            "optimal_weights": {"NIFTY_50_EQUITY": 0.55, "G_SEC_10Y_BOND": 0.45},
            "portfolio_aum": 10_000_000.0,
        }

        record = capture.capture_override(
            portfolio_id="PORT_101",
            advisor_id="ADV_42",
            original_proposal=original,
            modified_proposal=modified,
            justification="Tactical defensive shift ahead of RBI MPC policy announcement",
        )

        assert record.portfolio_id == "PORT_101"
        assert record.advisor_id == "ADV_42"
        assert record.category == InterventionCategory.TACTICAL_VIEW
        assert record.delta_weights["NIFTY_50_EQUITY"] == -0.05
        assert record.delta_weights["G_SEC_10Y_BOND"] == 0.05

        records = capture.get_overrides_for_portfolio("PORT_101")
        assert len(records) == 1
        assert records[0].override_id.startswith("OVR-PORT_101")


class TestEscalationManager:
    """Test suite for escalation case management, review lifecycle, and executive briefing dossiers."""

    def test_escalation_workflow(self):
        manager = EscalationManager()
        proposed_plan = {
            "portfolio_aum": 300_000_000.0,  # 30 Crores
            "turnover": 0.30,
            "candidate_trades": [{"symbol": "INFY", "quantity": 5000}],
            "predicted_tracking_error": 0.0085,
        }

        # Check threshold trigger
        assert manager.check_escalation_required(turnover=0.30, aum=300_000_000.0)

        # Create case
        case = manager.create_escalation_case(
            portfolio_id="PORT_HNI_01",
            client_id="CLIENT_HNI",
            risk_category="Aggressive",
            reasons=["Turnover 30% on 30 Crore AUM exceeds standard autonomous threshold"],
            proposed_plan=proposed_plan,
        )

        assert case.status == EscalationStatus.PENDING
        assert "URGENT ESCALATION BRIEFING" in case.briefing_document["title"]
        assert case.briefing_document["portfolio_overview"]["portfolio_aum_inr"] == 300_000_000.0

        # Review and approve case
        updated = manager.review_case(
            case_id=case.case_id,
            decision=EscalationStatus.APPROVED,
            reviewer_id="CRO_SARAH",
            reviewer_notes="Approved following client verbal mandate confirmation.",
        )

        assert updated.status == EscalationStatus.APPROVED
        assert updated.reviewer_id == "CRO_SARAH"
        assert updated.resolved_at_utc is not None
        assert len(manager.get_pending_cases()) == 0


class TestKillSwitch:
    """Test suite for automated circuit breakers, manual kill switches, and execution isolation."""

    @pytest.fixture
    def kill_switch(self) -> KillSwitch:
        return KillSwitch(vix_halt_threshold=40.0, vix_throttle_threshold=30.0)

    def test_normal_state(self, kill_switch: KillSwitch):
        allowed, msg = kill_switch.is_execution_allowed("PORT_001")
        assert allowed
        assert kill_switch.state == CircuitBreakerState.NORMAL

    def test_vix_automated_trip(self, kill_switch: KillSwitch):
        # VIX elevated -> THROTTLED
        state, reason = kill_switch.evaluate_automated_triggers(vix_level=32.0)
        assert state == CircuitBreakerState.THROTTLED
        allowed, _ = kill_switch.is_execution_allowed("PORT_001")
        assert allowed  # Throttled still permits execution with caution

        # VIX extreme spike (>= 40.0) -> HALTED
        state, reason = kill_switch.evaluate_automated_triggers(vix_level=42.5)
        assert state == CircuitBreakerState.HALTED
        assert kill_switch.is_globally_halted
        allowed, msg = kill_switch.is_execution_allowed("PORT_001")
        assert not allowed
        assert "Global trading halt active" in msg

    def test_error_rate_automated_trip(self, kill_switch: KillSwitch):
        # 3 failures out of 150 executions = 2.0% > 1.0% threshold
        state, reason = kill_switch.evaluate_automated_triggers(failed_executions=3, total_executions=150)
        assert state == CircuitBreakerState.HALTED
        assert "Operational error rate" in reason

    def test_market_flash_drop_automated_trip(self, kill_switch: KillSwitch):
        # -6.0% daily market drop <= -5.0% threshold
        state, reason = kill_switch.evaluate_automated_triggers(daily_market_return=-0.06)
        assert state == CircuitBreakerState.HALTED
        assert "Severe market flash drop" in reason

    def test_manual_kill_and_reset(self, kill_switch: KillSwitch):
        kill_switch.activate_global_halt(reason="Emergency maintenance", source="SYS_ADMIN")
        assert kill_switch.is_globally_halted
        allowed, _ = kill_switch.is_execution_allowed()
        assert not allowed

        kill_switch.deactivate_global_halt(reset_by="HEAD_OF_OPERATIONS", justification="Maintenance complete")
        assert not kill_switch.is_globally_halted
        allowed, _ = kill_switch.is_execution_allowed()
        assert allowed

    def test_portfolio_level_isolation(self, kill_switch: KillSwitch):
        kill_switch.halt_portfolio("PORT_SUSPECT", reason="Court attachment order pending")
        allowed_normal, _ = kill_switch.is_execution_allowed("PORT_NORMAL")
        allowed_suspect, msg = kill_switch.is_execution_allowed("PORT_SUSPECT")

        assert allowed_normal
        assert not allowed_suspect
        assert "PORT_SUSPECT is individually frozen" in msg

        kill_switch.resume_portfolio("PORT_SUSPECT")
        allowed_resumed, _ = kill_switch.is_execution_allowed("PORT_SUSPECT")
        assert allowed_resumed
