"""Tier 3: Event-Driven Rebalancing Trigger Evaluator.

Evaluates high-impact external and client-specific events:
1. Market shocks & crashes (>10% sudden asset drawdown)
2. Client life events (retirement transition, liquidity withdrawal requests)
3. FY-end March Tax Harvesting window (Indian Section 112A LTCG exemption ₹1.25L)
4. Regulatory & mandate shifts (SEBI asset reclassifications)
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from src.triggers.trigger_evaluator import (
    BaseTriggerEvaluator,
    TriggerPriority,
    TriggerSignal,
    TriggerTier,
    TriggerType,
)


class EventTriggerEvaluator(BaseTriggerEvaluator):
    """Evaluates macro shocks, regulatory updates, tax windows, and client life transitions."""

    def __init__(
        self,
        crash_drawdown_threshold: float = 0.10,
        march_tax_harvesting_active: bool = True,
    ) -> None:
        self.crash_threshold = crash_drawdown_threshold
        self.march_tax_active = march_tax_harvesting_active

    def evaluate(
        self,
        portfolio_record: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> List[TriggerSignal]:
        signals: List[TriggerSignal] = []
        ctx = context or {}

        portfolio_id = str(portfolio_record.get("portfolio_id", "UNKNOWN"))
        client_id = str(portfolio_record.get("client_id", "UNKNOWN"))
        aum = float(portfolio_record.get("aum_inr", 1_000_000.0))

        # 1. Market Shock / Crash Evaluation
        market_drawdowns = ctx.get("market_drawdowns", {})
        equity_drawdown = float(market_drawdowns.get("NIFTY_50_EQUITY", 0.0))
        if abs(equity_drawdown) >= self.crash_threshold and equity_drawdown < 0:
            signals.append(
                TriggerSignal(
                    portfolio_id=portfolio_id,
                    client_id=client_id,
                    tier=TriggerTier.TIER_3_EVENT,
                    trigger_type=TriggerType.MARKET_SHOCK,
                    priority=TriggerPriority.CRITICAL,
                    urgency_score=5.0,
                    summary=f"Market Shock detected: Equity drawdown of {equity_drawdown:.1%} breached {self.crash_threshold:.1%} circuit breaker.",
                    details={
                        "asset_class": "NIFTY_50_EQUITY",
                        "drawdown": equity_drawdown,
                        "threshold": self.crash_threshold,
                    },
                    recommended_action="Execute risk-containment rebalance: reallocate defensively or tactically re-enter equities at depressed valuations.",
                )
            )

        # 2. Client Life Event Evaluation
        life_event = str(portfolio_record.get("life_event", "NONE")).upper()
        if life_event in ("RETIREMENT_PLANNED", "RETIREMENT"):
            signals.append(
                TriggerSignal(
                    portfolio_id=portfolio_id,
                    client_id=client_id,
                    tier=TriggerTier.TIER_3_EVENT,
                    trigger_type=TriggerType.CLIENT_LIFE_EVENT,
                    priority=TriggerPriority.URGENT,
                    urgency_score=4.0,
                    summary="Client life event: Retirement transition requires de-risking toward sovereign bonds & liquid income.",
                    details={"event_type": "RETIREMENT", "portfolio_aum": aum},
                    recommended_action="Transition asset allocation from capital appreciation to capital preservation and regular dividend yield.",
                )
            )
        elif life_event in ("LIQUIDITY_DEMAND", "WITHDRAWAL"):
            signals.append(
                TriggerSignal(
                    portfolio_id=portfolio_id,
                    client_id=client_id,
                    tier=TriggerTier.TIER_3_EVENT,
                    trigger_type=TriggerType.CLIENT_LIFE_EVENT,
                    priority=TriggerPriority.HIGH,
                    urgency_score=3.5,
                    summary="Client liquidity demand: Upcoming planned redemption requires raising liquid cash buffer.",
                    details={"event_type": "LIQUIDITY_DEMAND", "portfolio_aum": aum},
                    recommended_action="Sell highest tax-efficient tranches to fund cash buffer.",
                )
            )

        # 3. March Financial Year-End Tax Harvesting Window
        # Active in March (month 3) or if explicitly passed in context
        current_month = ctx.get("current_month", datetime.utcnow().month)
        tax_sensitive = bool(portfolio_record.get("tax_sensitive", True))

        if (current_month == 3 or ctx.get("force_tax_harvesting_window", False)) and tax_sensitive:
            signals.append(
                TriggerSignal(
                    portfolio_id=portfolio_id,
                    client_id=client_id,
                    tier=TriggerTier.TIER_3_EVENT,
                    trigger_type=TriggerType.TAX_LOSS_HARVESTING,
                    priority=TriggerPriority.HIGH,
                    urgency_score=3.8,
                    summary="March FY-End Tax Window: Opportunity to harvest capital gains up to ₹1,25,000 Section 112A annual exemption.",
                    details={
                        "exemption_limit_inr": 125000,
                        "fiscal_year_ending": True,
                        "tax_sensitive": tax_sensitive,
                    },
                    recommended_action="Harvest unrealized LTCG up to ₹1.25L tax-free and offset any eligible short-term capital losses.",
                )
            )

        # 4. Regulatory Mandate Shift
        if ctx.get("regulatory_reclassification_active", False):
            signals.append(
                TriggerSignal(
                    portfolio_id=portfolio_id,
                    client_id=client_id,
                    tier=TriggerTier.TIER_3_EVENT,
                    trigger_type=TriggerType.REGULATORY_CHANGE,
                    priority=TriggerPriority.HIGH,
                    urgency_score=3.2,
                    summary="Regulatory Notice: SEBI categorisation update requires portfolio alignment.",
                    details={"regulator": "SEBI", "mandate_shift": ctx.get("regulatory_details", {})},
                    recommended_action="Align portfolio limits to new regulatory guidelines.",
                )
            )

        return signals
