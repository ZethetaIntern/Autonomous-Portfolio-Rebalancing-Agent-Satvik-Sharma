"""Base Trigger Evaluation Architecture and Core Data Models for WealthPilot AI.

Supports a three-tier rebalancing trigger taxonomy:
- Tier 1: Threshold & Concentration Triggers (Asset Drift, Concentration breach, Factor drift)
- Tier 2: Calendar Triggers (Periodic rebalance schedule: monthly, quarterly, annual)
- Tier 3: Event-Driven Triggers (Market shocks >10%, Regulatory shifts, Client life events, March FY Tax Harvesting)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
import uuid


class TriggerTier(str, Enum):
    TIER_1_THRESHOLD = "TIER_1_THRESHOLD"
    TIER_2_CALENDAR = "TIER_2_CALENDAR"
    TIER_3_EVENT = "TIER_3_EVENT"


class TriggerPriority(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    URGENT = "URGENT"
    CRITICAL = "CRITICAL"


class TriggerType(str, Enum):
    ASSET_CLASS_DRIFT = "ASSET_CLASS_DRIFT"
    CONCENTRATION_BREACH = "CONCENTRATION_BREACH"
    FACTOR_EXPOSURE_DRIFT = "FACTOR_EXPOSURE_DRIFT"
    CALENDAR_SCHEDULE = "CALENDAR_SCHEDULE"
    MARKET_SHOCK = "MARKET_SHOCK"
    REGULATORY_CHANGE = "REGULATORY_CHANGE"
    CLIENT_LIFE_EVENT = "CLIENT_LIFE_EVENT"
    TAX_LOSS_HARVESTING = "TAX_LOSS_HARVESTING"


PRIORITY_WEIGHTS: Dict[TriggerPriority, int] = {
    TriggerPriority.LOW: 1,
    TriggerPriority.MEDIUM: 2,
    TriggerPriority.HIGH: 3,
    TriggerPriority.URGENT: 4,
    TriggerPriority.CRITICAL: 5,
}


@dataclass
class TriggerSignal:
    """Individual rebalancing trigger detected for a portfolio."""
    portfolio_id: str
    client_id: str
    tier: TriggerTier
    trigger_type: TriggerType
    priority: TriggerPriority
    summary: str
    details: Dict[str, Any]
    recommended_action: str
    trigger_id: str = field(default_factory=lambda: f"TRG-{uuid.uuid4().hex[:8].upper()}")
    timestamp: datetime = field(default_factory=datetime.utcnow)
    urgency_score: float = 1.0

    @property
    def priority_rank(self) -> int:
        return PRIORITY_WEIGHTS.get(self.priority, 1)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trigger_id": self.trigger_id,
            "portfolio_id": self.portfolio_id,
            "client_id": self.client_id,
            "tier": self.tier.value,
            "trigger_type": self.trigger_type.value,
            "priority": self.priority.value,
            "urgency_score": self.urgency_score,
            "summary": self.summary,
            "details": self.details,
            "recommended_action": self.recommended_action,
            "timestamp": self.timestamp.isoformat(),
        }


class BaseTriggerEvaluator(ABC):
    """Abstract base class for all trigger evaluators."""

    @abstractmethod
    def evaluate(self, portfolio_record: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> List[TriggerSignal]:
        """Evaluate trigger rules against a portfolio record and market context.

        Args:
            portfolio_record: Dictionary or Series with portfolio weights, risk band, AUM, etc.
            context: Additional market data, current date, macro shocks, or regulatory flags.

        Returns:
            List of triggered signals (empty if no triggers fire).
        """
        pass
