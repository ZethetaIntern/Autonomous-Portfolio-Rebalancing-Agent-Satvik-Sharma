"""Unit tests for Three-Tier Trigger Engine and Trigger Consolidator."""

import pytest

from src.triggers.trigger_evaluator import (
    TriggerPriority,
    TriggerTier,
    TriggerType,
)
from src.triggers.threshold_trigger import ThresholdTriggerEvaluator
from src.triggers.calendar_trigger import CalendarTriggerEvaluator
from src.triggers.event_trigger import EventTriggerEvaluator
from src.triggers.trigger_consolidator import (
    TriggerConsolidator,
    ConsolidatedRebalanceEvent,
)


def test_threshold_trigger_drift_and_cash():
    evaluator = ThresholdTriggerEvaluator()

    record = {
        "portfolio_id": "WP-PF-000001",
        "client_id": "WP-CL-000001",
        "risk_category": "Balanced",  # default threshold 3.0%
        "target_weight_NIFTY_50_EQUITY": 0.50,
        "current_weight_NIFTY_50_EQUITY": 0.55,  # 5% drift -> breach!
        "target_weight_G_SEC_BONDS": 0.25,
        "current_weight_G_SEC_BONDS": 0.23,
        "target_weight_CORP_BONDS": 0.15,
        "current_weight_CORP_BONDS": 0.14,
        "target_weight_GOLD_ETF": 0.05,
        "current_weight_GOLD_ETF": 0.07,
        "target_weight_LIQUID_CASH": 0.05,
        "current_weight_LIQUID_CASH": 0.01,  # 1% cash < 2% minimum buffer -> breach!
    }

    signals = evaluator.evaluate(record)
    assert len(signals) >= 2

    types = [s.trigger_type for s in signals]
    assert TriggerType.ASSET_CLASS_DRIFT in types
    assert TriggerType.CONCENTRATION_BREACH in types


def test_threshold_trigger_hard_cap_breach():
    evaluator = ThresholdTriggerEvaluator()

    record = {
        "portfolio_id": "WP-PF-000002",
        "client_id": "WP-CL-000002",
        "risk_category": "Ultra-Aggressive",
        "target_weight_NIFTY_50_EQUITY": 0.85,
        "current_weight_NIFTY_50_EQUITY": 0.94,  # > 90% hard cap
        "target_weight_G_SEC_BONDS": 0.00,
        "current_weight_G_SEC_BONDS": 0.00,
        "target_weight_CORP_BONDS": 0.05,
        "current_weight_CORP_BONDS": 0.01,
        "target_weight_GOLD_ETF": 0.05,
        "current_weight_GOLD_ETF": 0.02,
        "target_weight_LIQUID_CASH": 0.05,
        "current_weight_LIQUID_CASH": 0.03,
    }

    signals = evaluator.evaluate(record)
    hard_cap_signals = [s for s in signals if s.trigger_type == TriggerType.CONCENTRATION_BREACH and "Hard concentration" in s.summary]
    assert len(hard_cap_signals) == 1
    assert hard_cap_signals[0].priority == TriggerPriority.URGENT


def test_calendar_trigger():
    evaluator = CalendarTriggerEvaluator()

    # Balanced portfolio rebalance cadence is 180 days
    # Case 1: 30 days -> No trigger
    rec_recent = {"portfolio_id": "P1", "risk_category": "Balanced", "days_since_rebalance": 30}
    assert len(evaluator.evaluate(rec_recent)) == 0

    # Case 2: 200 days -> Calendar trigger fired
    rec_overdue = {"portfolio_id": "P2", "risk_category": "Balanced", "days_since_rebalance": 200}
    signals = evaluator.evaluate(rec_overdue)
    assert len(signals) == 1
    assert signals[0].trigger_type == TriggerType.CALENDAR_SCHEDULE
    assert signals[0].tier == TriggerTier.TIER_2_CALENDAR


def test_event_trigger_market_shock_and_life_event():
    evaluator = EventTriggerEvaluator()

    record = {
        "portfolio_id": "P3",
        "client_id": "C3",
        "life_event": "RETIREMENT_PLANNED",
        "tax_sensitive": True,
    }

    # Market shock context: NIFTY down 12%
    context = {
        "market_drawdowns": {"NIFTY_50_EQUITY": -0.12},
        "current_month": 3,  # March FY-end tax harvesting window
    }

    signals = evaluator.evaluate(record, context=context)

    types = [s.trigger_type for s in signals]
    assert TriggerType.MARKET_SHOCK in types
    assert TriggerType.CLIENT_LIFE_EVENT in types
    assert TriggerType.TAX_LOSS_HARVESTING in types

    # Market shock should be CRITICAL
    shock_signal = next(s for s in signals if s.trigger_type == TriggerType.MARKET_SHOCK)
    assert shock_signal.priority == TriggerPriority.CRITICAL


def test_trigger_consolidator_single_and_multiple():
    consolidator = TriggerConsolidator()

    # Scenario with drift breach + calendar due + march tax window
    record = {
        "portfolio_id": "WP-PF-CONSOL",
        "client_id": "WP-CL-CONSOL",
        "risk_category": "Balanced",
        "days_since_rebalance": 210,
        "tax_sensitive": True,
        "target_weight_NIFTY_50_EQUITY": 0.50,
        "current_weight_NIFTY_50_EQUITY": 0.56,  # 6% drift > 3% threshold
        "target_weight_G_SEC_BONDS": 0.25,
        "current_weight_G_SEC_BONDS": 0.21,
        "target_weight_CORP_BONDS": 0.15,
        "current_weight_CORP_BONDS": 0.14,
        "target_weight_GOLD_ETF": 0.05,
        "current_weight_GOLD_ETF": 0.05,
        "target_weight_LIQUID_CASH": 0.05,
        "current_weight_LIQUID_CASH": 0.04,
    }
    context = {"current_month": 3}

    event = consolidator.evaluate_portfolio(record, context=context)

    assert event is not None
    assert isinstance(event, ConsolidatedRebalanceEvent)
    assert event.portfolio_id == "WP-PF-CONSOL"
    assert len(event.contributing_triggers) >= 2
    # Verify effective priority and SLA are populated
    assert event.effective_priority in (TriggerPriority.CRITICAL, TriggerPriority.URGENT, TriggerPriority.HIGH)
    assert "T+" in event.execution_timeline_sla
    assert len(event.audit_rationale) > 50

    # Test serialization
    event_dict = event.to_dict()
    assert event_dict["event_id"].startswith("REBAL-")
    assert "contributing_triggers" in event_dict
