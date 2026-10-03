"""Unit tests for Multi-Agent CrewAI Orchestration Framework."""

from __future__ import annotations

import pytest

from src.agents.orchestrator import OrchestratorAgent, SharedMemoryStore
from src.agents.portfolio_analyst import PortfolioAnalystAgent
from src.agents.tax_specialist import TaxSpecialistAgent
from src.agents.risk_manager import RiskManagerAgent
from src.agents.compliance_officer import ComplianceOfficerAgent
from src.agents.explanation_writer import ExplanationWriterAgent


class TestMultiAgentOrchestration:
    """Test suite verifying agent task pipelines, consensus, retries, and shared memory."""

    @pytest.fixture
    def orchestrator(self) -> OrchestratorAgent:
        return OrchestratorAgent()

    def test_priority_queue(self, orchestrator: OrchestratorAgent):
        queue = [
            {"portfolio_id": "P_LOW", "trigger_severity": "LOW", "sad": 0.03, "portfolio_aum": 1_000_000.0},
            {"portfolio_id": "P_CRIT", "trigger_severity": "CRITICAL", "sad": 0.12, "portfolio_aum": 50_000_000.0},
            {"portfolio_id": "P_HIGH", "trigger_severity": "HIGH", "sad": 0.08, "portfolio_aum": 10_000_000.0},
        ]

        prioritized = orchestrator.prioritize_queue(queue)
        assert prioritized[0]["portfolio_id"] == "P_CRIT"
        assert prioritized[1]["portfolio_id"] == "P_HIGH"
        assert prioritized[2]["portfolio_id"] == "P_LOW"

    def test_full_orchestration_workflow_success(self, orchestrator: OrchestratorAgent):
        portfolio = {
            "portfolio_id": "PORT_IND_01",
            "client_id": "CLIENT_001",
            "risk_category": "Balanced",
            "portfolio_aum": 10_000_000.0,
            "current_weights": {
                "NIFTY_50_EQUITY": 0.62,
                "G_SEC_10Y_BOND": 0.38,
            },
            "target_weights": {
                "NIFTY_50_EQUITY": 0.50,
                "G_SEC_10Y_BOND": 0.50,
            },
            "turnover_budget": 0.25,
            "min_cash_buffer": 0.02,
        }
        trigger = {
            "trigger_category": "THRESHOLD",
            "trigger_severity": "MEDIUM",
            "sad": 0.12,
            "rmsd": 0.06,
        }

        res = orchestrator.coordinate_rebalance(portfolio, trigger, max_retries=3)

        assert res["status"] == "APPROVED_FOR_REVIEW"
        assert res["consensus_reached"] is True
        assert res["attempt_count"] == 1
        assert len(res["candidate_trades"]) > 0
        assert "pre_trade_var_95_inr" in res["risk_metrics"]
        assert res["compliance_report"]["status"] == "PASS"
        assert "client_explanation" in res["explanation_packet"]
        assert "advisor_dossier" in res["explanation_packet"]
        assert "compliance_audit" in res["explanation_packet"]
        assert len(res["audit_trail"]) >= 5

    def test_orchestration_retry_and_escalation_on_breach(self, orchestrator: OrchestratorAgent):
        # Extremely restrictive turnover budget that will fail compliance
        portfolio = {
            "portfolio_id": "PORT_FAIL_01",
            "client_id": "CLIENT_RESTRICTED",
            "risk_category": "Conservative",
            "portfolio_aum": 5_000_000.0,
            "current_weights": {"NIFTY_50_EQUITY": 0.80, "G_SEC_10Y_BOND": 0.20},
            "target_weights": {"NIFTY_50_EQUITY": 0.30, "G_SEC_10Y_BOND": 0.70},
            "turnover_budget": 0.01,  # Unrealistically tight turnover budget
            "min_cash_buffer": 0.02,
        }
        trigger = {
            "trigger_category": "THRESHOLD",
            "trigger_severity": "CRITICAL",
            "sad": 0.50,
        }

        # Mock compliance check to always fail for this test to verify 3 retries and escalation
        class FailingCompliance(ComplianceOfficerAgent):
            def execute_task(self, context):
                return {"status": "FAIL", "reasons": ["Strict mandate violation in stress scenario"]}

        orchestrator.compliance_officer = FailingCompliance()
        res = orchestrator.coordinate_rebalance(portfolio, trigger, max_retries=3)

        assert res["status"] == "ESCALATED"
        assert res["consensus_reached"] is False
        assert res["attempt_count"] == 3
        assert "Compliance check failed" in res["error_detail"]


class TestSpecialistAgents:
    """Test suite for individual specialist agents."""

    def test_portfolio_analyst_agent(self):
        analyst = PortfolioAnalystAgent()
        out = analyst.execute_task({
            "portfolio_id": "PORT_A",
            "current_weights": {"NIFTY_50_EQUITY": 0.60, "G_SEC_10Y_BOND": 0.40},
            "target_weights": {"NIFTY_50_EQUITY": 0.50, "G_SEC_10Y_BOND": 0.50},
            "portfolio_aum": 10_000_000.0,
        })
        assert out["status"] == "SUCCESS"
        assert "optimal_weights" in out
        assert "discrete_weights" in out
        assert len(out["candidate_trades"]) > 0

    def test_tax_specialist_agent(self):
        tax_agent = TaxSpecialistAgent()
        out = tax_agent.execute_task({
            "portfolio_id": "PORT_T",
            "candidate_trades": [
                {"symbol": "INFY", "action": "SELL", "quantity": 100, "estimated_price": 1800.0}
            ],
            "existing_tax_lots": [
                {"lot_id": "LOT-1", "symbol": "INFY", "quantity": 100, "purchase_price": 2000.0, "purchase_date": "2024-11-01"}
            ],
        })
        assert out["status"] == "SUCCESS"
        assert out["net_realized_pnl_inr"] < 0.0  # Harvested loss
        assert out["harvest_realized_loss_inr"] > 0.0

    def test_risk_manager_agent(self):
        risk_agent = RiskManagerAgent()
        out = risk_agent.execute_task({
            "portfolio_id": "PORT_R",
            "portfolio_aum": 10_000_000.0,
            "current_weights": {"NIFTY_50_EQUITY": 0.60, "G_SEC_10Y_BOND": 0.40},
            "proposed_weights": {"NIFTY_50_EQUITY": 0.50, "G_SEC_10Y_BOND": 0.50},
            "candidate_trades": [
                {"symbol": "NIFTY_50_EQUITY", "trade_value_inr": 1_000_000.0}
            ],
        })
        assert out["pre_trade_var_95_inr"] > 0.0
        assert out["post_trade_var_95_inr"] > 0.0
        assert out["stress_test_drawdown_pct"] <= 0.0
        assert "market_impact_summary" in out

    def test_compliance_officer_agent(self):
        comp_agent = ComplianceOfficerAgent()
        out = comp_agent.execute_task({
            "portfolio_id": "PORT_C",
            "client_id": "CLI_1",
            "proposed_weights": {"NIFTY_50_EQUITY": 0.50, "G_SEC_10Y_BOND": 0.50},
            "candidate_trades": [],
            "portfolio_aum": 10_000_000.0,
            "cash_buffer_pct": 0.02,
            "turnover_budget": 0.20,
            "turnover_incurred": 0.08,
            "issuer_limit": 0.10,
            "sector_limit": 0.30,
        })
        assert out["status"] == "PASS"
        assert out["report"].sebi_compliant is True
        assert out["report"].is_valid is True

    def test_explanation_writer_agent(self):
        writer = ExplanationWriterAgent()
        out = writer.execute_task({
            "portfolio_id": "PORT_EXP",
            "client_id": "CLIENT_001",
            "risk_category": "Growth",
            "trigger_category": "THRESHOLD",
            "trigger_severity": "MEDIUM",
            "current_weights": {"NIFTY_50_EQUITY": 0.60, "G_SEC_10Y_BOND": 0.40},
            "target_weights": {"NIFTY_50_EQUITY": 0.50, "G_SEC_10Y_BOND": 0.50},
            "proposed_weights": {"NIFTY_50_EQUITY": 0.50, "G_SEC_10Y_BOND": 0.50},
            "trades": [],
            "sad": 0.10,
            "rmsd": 0.05,
            "total_costs_inr": 1200.0,
            "tax_impact_inr": 0.0,
        })
        assert out["status"] == "SUCCESS"
        assert "plain_english_summary" in out["client_explanation"]
        assert "executive_summary" in out["advisor_dossier"]
        assert "legal_attestation" in out["compliance_audit"]
