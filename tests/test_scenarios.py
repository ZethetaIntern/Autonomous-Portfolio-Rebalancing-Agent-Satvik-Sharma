"""Unit and integration tests for Scenario Runner and all 5 mandatory market scenarios."""

import pytest
from src.backtesting.scenario_runner import ScenarioRunner


@pytest.fixture
def runner():
    return ScenarioRunner()


def test_scenario_1_normal_drift(runner):
    """Verifies Scenario 1: 3-month equity rally triggers cost-efficient threshold rebalancing."""
    res = runner.run_scenario_1_normal_drift()
    assert res["scenario_id"] == "SCENARIO_1_NORMAL_DRIFT"
    assert res["test_passed"] is True
    assert res["equity_growth_realized_pct"] > 5.0
    assert res["rebalances_executed"] >= 1
    assert res["total_return_pct"] > 5.0
    assert "details" in res
    assert len(res["narrative"]) > 40


def test_scenario_2_market_crash(runner):
    """Verifies Scenario 2: 22% market crash triggers urgent circuit breaker trip and crisis communication."""
    res = runner.run_scenario_2_market_crash()
    assert res["scenario_id"] == "SCENARIO_2_MARKET_CRASH"
    assert res["test_passed"] is True
    assert res["peak_vix_observed"] >= 40.0
    assert res["equity_5_day_crash_pct"] <= -20.0
    assert res["kill_switch_tripped"] is True
    assert res["crisis_communication_generated"] is True
    assert len(res["narrative"]) > 40


def test_scenario_3_sector_rotation(runner):
    """Verifies Scenario 3: Growth-to-Value sector rotation triggers intra-equity rebalancing."""
    res = runner.run_scenario_3_sector_rotation()
    assert res["scenario_id"] == "SCENARIO_3_SECTOR_ROTATION"
    assert res["test_passed"] is True
    assert res["overall_equity_drift_pct"] == 0.0  # Overall equity weight remained flat
    assert res["intra_equity_sad_pct"] >= 15.0
    assert res["turnover_incurred_pct"] > 0.0
    assert abs(res["rebalanced_it_weight"] - 0.25) < 0.05
    assert len(res["narrative"]) > 40


def test_scenario_4_regulatory_event(runner):
    """Verifies Scenario 4: SEBI circular capping international equity at 15% generates compliant trades."""
    res = runner.run_scenario_4_regulatory_event()
    assert res["scenario_id"] == "SCENARIO_4_REGULATORY_EVENT"
    assert res["test_passed"] is True
    assert res["initial_intl_weight_pct"] > 15.0
    assert res["post_rebalance_intl_weight_pct"] <= 15.01
    assert res["compliance_achieved"] is True
    assert "SEBI/HO/MRD/DOP1/CIR/P/2024/69" in res["sebi_circular_reference"]
    assert len(res["narrative"]) > 40


def test_scenario_5_tax_harvesting(runner):
    """Verifies Scenario 5: March FY-end tax harvesting realizes losses, avoids wash-sales, and delivers tax alpha."""
    res = runner.run_scenario_5_tax_harvesting()
    assert res["scenario_id"] == "SCENARIO_5_TAX_HARVESTING"
    assert res["test_passed"] is True
    assert res["harvested_capital_loss_inr"] >= 50000.0
    assert res["wash_sale_substitutions_count"] >= 1
    assert res["tax_shield_inr"] > 0.0
    assert len(res["narrative"]) > 40


def test_run_all_scenarios(runner):
    """Verifies that all 5 market scenarios execute together and all pass validation."""
    results = runner.run_all_scenarios()
    assert results["total_scenarios"] == 5
    assert results["passed_count"] == 5
    assert results["all_passed"] is True
    assert len(results["scenarios"]) == 5
    for scen_id, res in results["scenarios"].items():
        assert res["test_passed"] is True, f"Scenario {scen_id} failed"
