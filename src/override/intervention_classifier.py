"""Advisor Intervention Classifier for WealthPilot AI.

Classifies rebalancing decisions into 4 graduated intervention tiers based on
trade impact, turnover, asset size, and agent model confidence:
- Tier 1: INFORMATIONAL (Turnover <= 5%, Confidence >= 0.90; auto-execute, notification logged)
- Tier 2: ADVISORY (Turnover 5-15%, Confidence 0.75-0.90; 4-24h waiting period window)
- Tier 3: APPROVAL_REQUIRED (Turnover > 15%, Trade Value > 50L INR, or Confidence < 0.75; explicit approval needed)
- Tier 4: ESCALATION (SEBI/Mandate breaches, SAD > 15%, Model conflict, or Confidence < 0.50; IC / CRO review)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional


class InterventionTier(str, Enum):
    INFORMATIONAL = "INFORMATIONAL"
    ADVISORY = "ADVISORY"
    APPROVAL_REQUIRED = "APPROVAL_REQUIRED"
    ESCALATION = "ESCALATION"


class InterventionCategory(str, Enum):
    TACTICAL_VIEW = "TACTICAL_VIEW"
    CLIENT_PREFERENCE = "CLIENT_PREFERENCE"
    LIQUIDITY_NEEDS = "LIQUIDITY_NEEDS"
    TAX_OPTIMIZATION = "TAX_OPTIMIZATION"
    POLICY_EXCEPTION = "POLICY_EXCEPTION"
    RISK_REDUCTION = "RISK_REDUCTION"
    OTHER = "OTHER"


@dataclass
class InterventionClassificationResult:
    """Detailed graduated intervention classification outcome."""
    tier: InterventionTier
    waiting_period_hours: float
    requires_human_signoff: bool
    escalation_target: Optional[str]
    rationale: str
    confidence_score: float
    factors: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tier": self.tier.value,
            "waiting_period_hours": self.waiting_period_hours,
            "requires_human_signoff": self.requires_human_signoff,
            "escalation_target": self.escalation_target,
            "rationale": self.rationale,
            "confidence_score": self.confidence_score,
            "factors": self.factors,
        }


class InterventionClassifier:
    """Classifies rebalancing proposals into 4 graduated intervention tiers."""

    def __init__(
        self,
        high_turnover_threshold: float = 0.15,
        medium_turnover_threshold: float = 0.05,
        large_trade_value_inr: float = 5_000_000.0,
        high_confidence_threshold: float = 0.90,
        medium_confidence_threshold: float = 0.75,
        low_confidence_threshold: float = 0.50,
        severe_drift_sad: float = 0.15,
    ) -> None:
        self.high_turnover = high_turnover_threshold
        self.medium_turnover = medium_turnover_threshold
        self.large_trade_value = large_trade_value_inr
        self.high_conf = high_confidence_threshold
        self.med_conf = medium_confidence_threshold
        self.low_conf = low_confidence_threshold
        self.severe_drift_sad = severe_drift_sad

    def classify_proposal(
        self,
        proposal: Dict[str, Any],
        agent_confidence: float = 0.92,
        compliance_passed: bool = True,
        constraint_breaches: Optional[List[str]] = None,
    ) -> InterventionClassificationResult:
        """Evaluates proposal metrics against graduated intervention rules."""
        turnover = float(proposal.get("turnover", proposal.get("turnover_pct", 0.0) / 100.0 if "turnover_pct" in proposal else 0.0))
        portfolio_aum = float(proposal.get("portfolio_aum", 10_000_000.0))
        trade_value_inr = float(proposal.get("total_trade_value_inr", turnover * portfolio_aum))
        sad = float(proposal.get("sad", 0.04))
        has_client_restriction = bool(proposal.get("has_client_restriction", False))
        model_conflict = bool(proposal.get("model_conflict", False))

        breaches = constraint_breaches or []
        factors = {
            "turnover": turnover,
            "portfolio_aum": portfolio_aum,
            "trade_value_inr": trade_value_inr,
            "sad": sad,
            "agent_confidence": agent_confidence,
            "compliance_passed": compliance_passed,
            "constraint_breaches": breaches,
            "model_conflict": model_conflict,
        }

        if (
            not compliance_passed
            or len(breaches) > 0
            or sad > self.severe_drift_sad
            or model_conflict
            or agent_confidence < self.low_conf
        ):
            reasons = []
            if not compliance_passed:
                reasons.append("SEBI/Mandate compliance check failed")
            if breaches:
                reasons.append(f"Hard constraint breaches: {', '.join(breaches)}")
            if sad > self.severe_drift_sad:
                reasons.append(f"Severe portfolio drift SAD ({sad:.1%}) exceeds {self.severe_drift_sad:.1%}")
            if model_conflict:
                reasons.append("Conflicting recommendations between quantitative and risk models")
            if agent_confidence < self.low_conf:
                reasons.append(f"Low agent confidence score ({agent_confidence:.2f}) below {self.low_conf:.2f}")

            return InterventionClassificationResult(
                tier=InterventionTier.ESCALATION,
                waiting_period_hours=0.0,
                requires_human_signoff=True,
                escalation_target="CHIEF_RISK_OFFICER_AND_COMPLIANCE_HEAD",
                rationale="; ".join(reasons),
                confidence_score=agent_confidence,
                factors=factors,
            )

        if (
            turnover > self.high_turnover
            or trade_value_inr >= self.large_trade_value
            or agent_confidence < self.med_conf
            or has_client_restriction
        ):
            reasons = []
            if turnover > self.high_turnover:
                reasons.append(f"Portfolio turnover ({turnover:.1%}) exceeds threshold ({self.high_turnover:.1%})")
            if trade_value_inr >= self.large_trade_value:
                reasons.append(f"Aggregate trade value (INR {trade_value_inr:,.0f}) exceeds threshold (INR {self.large_trade_value:,.0f})")
            if agent_confidence < self.med_conf:
                reasons.append(f"Moderate agent confidence ({agent_confidence:.2f}) requires advisor sign-off")
            if has_client_restriction:
                reasons.append("Client-specific investment restriction active on portfolio")

            return InterventionClassificationResult(
                tier=InterventionTier.APPROVAL_REQUIRED,
                waiting_period_hours=0.0,
                requires_human_signoff=True,
                escalation_target="RELATIONSHIP_MANAGER",
                rationale="; ".join(reasons),
                confidence_score=agent_confidence,
                factors=factors,
            )

        if turnover > self.medium_turnover or agent_confidence < self.high_conf:
            waiting_hours = 24.0 if turnover > 0.10 else 4.0
            rationale = (
                f"Moderate rebalance (Turnover: {turnover:.1%}, Confidence: {agent_confidence:.2f}). "
                f"Advisory notification dispatched; queued for auto-execution in {waiting_hours:.0f} hours unless overridden."
            )
            return InterventionClassificationResult(
                tier=InterventionTier.ADVISORY,
                waiting_period_hours=waiting_hours,
                requires_human_signoff=False,
                escalation_target=None,
                rationale=rationale,
                confidence_score=agent_confidence,
                factors=factors,
            )

        return InterventionClassificationResult(
            tier=InterventionTier.INFORMATIONAL,
            waiting_period_hours=0.0,
            requires_human_signoff=False,
            escalation_target=None,
            rationale="Routine low-impact rebalancing; high agent confidence. Auto-executed with notification logging.",
            confidence_score=agent_confidence,
            factors=factors,
        )

    def classify_reason(self, reason_text: str) -> InterventionCategory:
        """Classifies human advisor override reason text into standard taxonomy."""
        text = reason_text.lower()
        if "tax" in text or "harvest" in text or "stcg" in text or "ltcg" in text:
            return InterventionCategory.TAX_OPTIMIZATION
        elif "cash" in text or "withdraw" in text or "liquidity" in text or "margin" in text:
            return InterventionCategory.LIQUIDITY_NEEDS
        elif "client" in text or "preference" in text or "avoid" in text:
            return InterventionCategory.CLIENT_PREFERENCE
        elif "tactical" in text or "view" in text or "sector" in text or "outperform" in text or "macro" in text:
            return InterventionCategory.TACTICAL_VIEW
        elif "risk" in text or "volatil" in text or "var" in text or "duration" in text or "drawdown" in text:
            return InterventionCategory.RISK_REDUCTION
        elif "policy exception" in text or "board" in text or "mandate exception" in text:
            return InterventionCategory.POLICY_EXCEPTION
        return InterventionCategory.OTHER

    def classify(self, reason_text: str) -> InterventionCategory:
        return self.classify_reason(reason_text)
