"""Multi-Agent Rebalancing Collaboration Framework for WealthPilot AI.

Role Archetypes:
1. OrchestratorAgent: Coordinates agent lifecycle, prioritization, consensus, retry loops, and delegation
2. PortfolioAnalystAgent: Evaluates drift metrics, returns, convex QP optimization, and discrete knapsack orders
3. TaxSpecialistAgent: Evaluates capital gains tax optimization, tax-loss harvesting, and wash-sale compliance
4. RiskManagerAgent: Enforces volatility budgets, 95% VaR, stress-testing, and liquidity execution scheduling
5. ComplianceOfficerAgent: Ensures SEBI regulatory compliance and investment mandate adherence
6. ExplanationWriterAgent: Synthesizes multi-audience tailored narratives (Client, Advisor, Compliance)
"""

import warnings

# Ensure warnings.warn remains compatible across Python 3.13 and third-party libraries
_orig_warn = warnings.warn
def _compat_warn(message, category=None, stacklevel=1, source=None, *args, **kwargs):
    kwargs.pop("skip_file_prefixes", None)
    return _orig_warn(message, category=category, stacklevel=stacklevel, source=source, *args, **kwargs)
warnings.warn = _compat_warn

from typing import Any, Dict

from src.agents.portfolio_analyst import PortfolioAnalystAgent
from src.agents.tax_specialist import TaxSpecialistAgent
from src.agents.risk_manager import RiskManagerAgent
from src.agents.compliance_officer import ComplianceOfficerAgent
from src.agents.explanation_writer import ExplanationWriterAgent, ExplanationWriter
from src.agents.orchestrator import OrchestratorAgent, Orchestrator, SharedMemoryStore


class BaseAgent:
    """Base class for WealthPilot specialized intelligence agents."""

    def __init__(self, name: str, role: str) -> None:
        self.name = name
        self.role = role

    def process(self, task_payload: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError


__all__ = [
    "BaseAgent",
    "OrchestratorAgent",
    "Orchestrator",
    "SharedMemoryStore",
    "PortfolioAnalystAgent",
    "TaxSpecialistAgent",
    "RiskManagerAgent",
    "ComplianceOfficerAgent",
    "ExplanationWriterAgent",
    "ExplanationWriter",
]
