"""Human-in-the-Loop Override & Escalation Governance Package for WealthPilot AI.

Components:
- InterventionClassifier: 4-tier graduated intervention model (Informational, Advisory, Approval Required, Escalation)
- OverrideCapture: Advisor modification tracking, reason taxonomy, and audit records
- EscalationManager: Senior human review, IC briefing documents, and lifecycle status tracking
- KillSwitch: Automated and manual multi-tier circuit breaker (NORMAL, THROTTLED, HALTED)
"""

from src.override.intervention_classifier import (
    InterventionClassifier,
    InterventionTier,
    InterventionCategory,
    InterventionClassificationResult,
)
from src.override.override_capture import (
    OverrideCapture,
    AdvisorOverrideRecord,
)
from src.override.escalation_manager import (
    EscalationManager,
    EscalationCase,
    EscalationStatus,
)
from src.override.kill_switch import (
    KillSwitch,
    CircuitBreakerState,
)

__all__ = [
    "InterventionClassifier",
    "InterventionTier",
    "InterventionCategory",
    "InterventionClassificationResult",
    "OverrideCapture",
    "AdvisorOverrideRecord",
    "EscalationManager",
    "EscalationCase",
    "EscalationStatus",
    "KillSwitch",
    "CircuitBreakerState",
]
