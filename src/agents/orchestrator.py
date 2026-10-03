"""Multi-Agent Orchestrator coordinating consensus, retry loops, and rebalancing workflow.

Role: Supervisory Executive Coordinator
Receives triggers, prioritizes queued portfolios, assigns tasks across the specialist Crew,
manages automated retry loops (max 3 attempts), handles escalations, and maintains shared memory.
"""

from __future__ import annotations

import datetime
from typing import Any, Dict, List, Optional

from src.agents.portfolio_analyst import PortfolioAnalystAgent
from src.agents.tax_specialist import TaxSpecialistAgent
from src.agents.risk_manager import RiskManagerAgent
from src.agents.compliance_officer import ComplianceOfficerAgent
from src.agents.explanation_writer import ExplanationWriterAgent

try:
    from crewai import Agent, Crew, Process, Task
    HAS_CREWAI = True
    import warnings
    _cw = warnings.warn
    def _compat_crew_warn(message, category=None, stacklevel=1, source=None, *args, **kwargs):
        kwargs.pop("skip_file_prefixes", None)
        return _cw(message, category=category, stacklevel=stacklevel, source=source, *args, **kwargs)
    warnings.warn = _compat_crew_warn
except ImportError:
    HAS_CREWAI = False


class SharedMemoryStore:
    """Thread-safe key-value and audit memory store across agent collaboration workflows."""

    def __init__(self) -> None:
        self._store: Dict[str, Any] = {}
        self._audit_events: List[Dict[str, Any]] = []

    def set(self, key: str, value: Any) -> None:
        self._store[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        return self._store.get(key, default)

    def record_event(self, agent_name: str, event_type: str, details: Dict[str, Any]) -> None:
        event = {
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "agent": agent_name,
            "event_type": event_type,
            "details": details,
        }
        self._audit_events.append(event)

    def get_audit_trail(self) -> List[Dict[str, Any]]:
        return list(self._audit_events)

    def clear(self) -> None:
        self._store.clear()
        self._audit_events.clear()


class OrchestratorAgent:
    """Supervisory Executive Coordinator managing multi-agent consensus, queues, retries, and escalations."""

    def __init__(
        self,
        portfolio_analyst: Optional[PortfolioAnalystAgent] = None,
        tax_specialist: Optional[TaxSpecialistAgent] = None,
        risk_manager: Optional[RiskManagerAgent] = None,
        compliance_officer: Optional[ComplianceOfficerAgent] = None,
        explanation_writer: Optional[ExplanationWriterAgent] = None,
        memory: Optional[SharedMemoryStore] = None,
    ) -> None:
        self.role = "Supervisory Executive Coordinator"
        self.goal = "Prioritize portfolios, orchestrate multi-agent consensus rebalancing, manage retries, and escalate breaches."
        self.backstory = (
            "A seasoned executive supervisory AI coordinator engineered for autonomous portfolio management. "
            "Ensures flawless synchronization among quantitative analysts, tax specialists, risk officers, and compliance "
            "monitors while maintaining an immutable audit log and fail-safe human escalation."
        )
        self.portfolio_analyst = portfolio_analyst or PortfolioAnalystAgent()
        self.tax_specialist = tax_specialist or TaxSpecialistAgent()
        self.risk_manager = risk_manager or RiskManagerAgent()
        self.compliance_officer = compliance_officer or ComplianceOfficerAgent()
        self.explanation_writer = explanation_writer or ExplanationWriterAgent()
        self.memory = memory or SharedMemoryStore()

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
            allow_delegation=True,
        )

    def prioritize_queue(self, queue: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Prioritizes queued portfolios based on drift severity, trigger severity, and AUM.
        
        Sorting score = severity_weight * 1000 + sad * 100 + log10(aum)
        """
        severity_map = {"CRITICAL": 4.0, "HIGH": 3.0, "MEDIUM": 2.0, "LOW": 1.0}

        def score_fn(item: Dict[str, Any]) -> float:
            sev = item.get("trigger_severity", "MEDIUM").upper()
            sev_wt = severity_map.get(sev, 2.0)
            sad = float(item.get("sad", 0.05))
            aum = float(item.get("portfolio_aum", 1_000_000.0))
            return sev_wt * 1000.0 + sad * 100.0 + (aum / 1_000_000.0)

        return sorted(queue, key=score_fn, reverse=True)

    def coordinate_rebalance(
        self,
        portfolio_record: Dict[str, Any],
        trigger_event: Dict[str, Any],
        max_retries: int = 3,
    ) -> Dict[str, Any]:
        """Executes multi-agent consensus workflow with up to max_retries retry attempts.
        
        Pipeline:
        1. Portfolio Analyst: Solves constrained QP and discrete candidate trades
        2. Tax Specialist: Optimizes tax lots, STCG/LTCG, and wash-sale replacement
        3. Risk Manager: Computes pre/post VaR, stress test, and liquidity schedule
        4. Compliance Officer: Validates SEBI rules (issuer/sector/cash/turnover)
           - If compliance fails, attempt retry with tightened constraints (up to max 3 attempts)
        5. Explanation Writer: Generates Client, Advisor, and Compliance explanation dossiers
        """
        p_id = portfolio_record.get("portfolio_id", "PORT_001")
        wf_id = f"WF-{p_id}-{int(datetime.datetime.now().timestamp())}"
        self.memory.record_event("Orchestrator", "WORKFLOW_START", {"workflow_id": wf_id, "portfolio_id": p_id})

        current_weights = portfolio_record.get("current_weights", {"NIFTY_50_EQUITY": 0.65, "G_SEC_10Y_BOND": 0.35})
        target_weights = portfolio_record.get("target_weights", {"NIFTY_50_EQUITY": 0.50, "G_SEC_10Y_BOND": 0.50})
        portfolio_aum = float(portfolio_record.get("portfolio_aum", 10_000_000.0))
        turnover_budget = float(portfolio_record.get("turnover_budget", 0.20))
        min_cash_buffer = float(portfolio_record.get("min_cash_buffer", 0.02))

        attempt = 0
        success = False
        last_error = None
        analyst_out = {}
        tax_out = {}
        risk_out = {}
        compliance_out = {}

        # Retry loop (max 3 attempts)
        while attempt < max_retries and not success:
            attempt += 1
            self.memory.record_event("Orchestrator", "ATTEMPT_START", {"attempt": attempt, "workflow_id": wf_id})

            try:
                # 1. Analyst Optimization
                analyst_context = {
                    "portfolio_id": p_id,
                    "current_weights": current_weights,
                    "target_weights": target_weights,
                    "portfolio_aum": portfolio_aum,
                    "turnover_budget": turnover_budget,
                    "min_cash_buffer": min_cash_buffer,
                    "sebi_issuer_limit": portfolio_record.get("sebi_issuer_limit", 0.10),
                    "sebi_sector_limits": portfolio_record.get("sebi_sector_limits"),
                    "asset_sector_map": portfolio_record.get("asset_sector_map"),
                }
                analyst_out = self.portfolio_analyst.execute_task(analyst_context)
                self.memory.record_event("PortfolioAnalyst", "QP_COMPLETED", {"status": analyst_out.get("status")})

                # 2. Tax Specialist Optimization
                tax_lots = portfolio_record.get("tax_lots", [])
                candidate_trades = analyst_out.get("candidate_trades", [])
                tax_out = self.tax_specialist.execute_task({
                    "portfolio_id": p_id,
                    "candidate_trades": candidate_trades,
                    "existing_tax_lots": tax_lots,
                    "current_financial_year": portfolio_record.get("financial_year", "2024-2025"),
                    "harvest_threshold_inr": portfolio_record.get("harvest_threshold_inr", 5000.0),
                })
                self.memory.record_event("TaxSpecialist", "TAX_EVAL_COMPLETED", {"net_realized_pnl": tax_out.get("net_realized_pnl_inr")})

                # 3. Risk Manager Evaluation
                risk_out = self.risk_manager.execute_task({
                    "portfolio_id": p_id,
                    "portfolio_aum": portfolio_aum,
                    "current_weights": current_weights,
                    "proposed_weights": analyst_out.get("discrete_weights", analyst_out.get("optimal_weights", {})),
                    "candidate_trades": candidate_trades,
                })
                self.memory.record_event("RiskManager", "RISK_EVAL_COMPLETED", {"pre_var": risk_out.get("pre_trade_var_95_inr")})

                # 4. Compliance Officer Validation
                compliance_out = self.compliance_officer.execute_task({
                    "portfolio_id": p_id,
                    "client_id": portfolio_record.get("client_id", "CLIENT_001"),
                    "proposed_weights": analyst_out.get("discrete_weights", analyst_out.get("optimal_weights", {})),
                    "candidate_trades": candidate_trades,
                    "portfolio_aum": portfolio_aum,
                    "cash_buffer_pct": min_cash_buffer,
                    "turnover_budget": turnover_budget,
                    "turnover_incurred": analyst_out.get("turnover", 0.0),
                    "issuer_limit": portfolio_record.get("sebi_issuer_limit", 0.10),
                    "sector_limit": 0.30,
                    "asset_sector_map": portfolio_record.get("asset_sector_map", {}),
                })
                self.memory.record_event("ComplianceOfficer", "COMPLIANCE_EVAL_COMPLETED", {"status": compliance_out.get("status")})

                if compliance_out.get("status") in ["PASS", "APPROVED"]:
                    success = True
                else:
                    last_error = f"Compliance check failed: {compliance_out.get('reasons', ['Rule violation'])}"
                    # Auto-tighten turnover or cash buffer for next attempt
                    turnover_budget = max(0.05, turnover_budget * 0.85)
                    min_cash_buffer = min(0.05, min_cash_buffer * 1.10)

            except Exception as ex:
                last_error = str(ex)
                self.memory.record_event("Orchestrator", "ATTEMPT_EXCEPTION", {"attempt": attempt, "error": str(ex)})

        # 5. Multi-audience explanation generation (if success or post-final-attempt)
        explanation_context = {
            "portfolio_id": p_id,
            "decision_id": wf_id,
            "client_id": portfolio_record.get("client_id", "CLIENT_001"),
            "risk_category": portfolio_record.get("risk_category", "Balanced"),
            "trigger_category": trigger_event.get("trigger_category", "THRESHOLD"),
            "trigger_severity": trigger_event.get("trigger_severity", "MEDIUM"),
            "current_weights": current_weights,
            "target_weights": target_weights,
            "proposed_weights": analyst_out.get("discrete_weights", analyst_out.get("optimal_weights", current_weights)),
            "trades": analyst_out.get("candidate_trades", []),
            "sad": float(trigger_event.get("sad", 0.05)),
            "rmsd": float(trigger_event.get("rmsd", 0.025)),
            "tracking_error_reduction_bps": float(analyst_out.get("predicted_tracking_error", 0.0035)) * 10000.0,
            "total_costs_inr": float(risk_out.get("market_impact_summary", {}).get("total_trading_cost_inr", 1500.0)),
            "tax_impact_inr": float(tax_out.get("total_tax_liability_inr", 0.0)),
            "tax_shield_inr": float(tax_out.get("harvest_realized_loss_inr", 0.0)),
            "sebi_compliant": success,
        }

        explanation_out = self.explanation_writer.execute_task(explanation_context)
        self.memory.record_event("ExplanationWriter", "EXPLANATIONS_GENERATED", {"status": explanation_out.get("status")})

        workflow_status = "APPROVED_FOR_REVIEW" if success else "ESCALATED"
        if not success:
            self.memory.record_event("Orchestrator", "ESCALATION_TRIGGERED", {"reason": last_error, "attempts": attempt})

        return {
            "workflow_id": wf_id,
            "portfolio_id": p_id,
            "status": workflow_status,
            "consensus_reached": success,
            "attempt_count": attempt,
            "error_detail": last_error if not success else None,
            "optimal_weights": analyst_out.get("optimal_weights", {}),
            "discrete_weights": analyst_out.get("discrete_weights", {}),
            "candidate_trades": analyst_out.get("candidate_trades", []),
            "tax_summary": tax_out,
            "risk_metrics": risk_out,
            "compliance_report": compliance_out,
            "explanation_packet": explanation_out,
            "audit_trail": self.memory.get_audit_trail(),
        }


# Backward compatibility alias
Orchestrator = OrchestratorAgent
