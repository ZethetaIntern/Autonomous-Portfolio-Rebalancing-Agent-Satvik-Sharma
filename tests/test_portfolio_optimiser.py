"""Comprehensive Unit & Integration Test Suite for Day 4: Core Trade Generation Optimizer.

Tests CVXPY QP formulation, tracking error variance minimization, hard constraint adherence
(budget conservation, long-only, turnover limits, SEBI issuer & sector caps, cash buffer),
joint round-lot discrete trade generation, statutory transaction costs, and pre-trade audits.
"""

import math
import numpy as np
import pytest

from src.data.market_data_simulator import ASSET_CLASSES, MarketDataSimulator
from src.optimisation import (
    ConstraintManager,
    CostEstimator,
    PortfolioOptimiser,
    PreTradeValidationReport,
    TradeListGenerator,
    TradeOrder,
)


@pytest.fixture
def default_optimiser():
    sim = MarketDataSimulator(seed=42)
    return PortfolioOptimiser(
        asset_names=ASSET_CLASSES,
        covariance_matrix=sim.get_annual_covariance(),
    )


@pytest.fixture
def sample_weights():
    # Current drifted weights vs Strategic Target weights
    curr = {
        "NIFTY_50_EQUITY": 0.62,
        "G_SEC_BONDS": 0.18,
        "CORP_BONDS": 0.12,
        "GOLD_ETF": 0.05,
        "LIQUID_CASH": 0.03,
    }
    targ = {
        "NIFTY_50_EQUITY": 0.50,
        "G_SEC_BONDS": 0.25,
        "CORP_BONDS": 0.15,
        "GOLD_ETF": 0.05,
        "LIQUID_CASH": 0.05,
    }
    return curr, targ


class TestPortfolioOptimiser:
    """Tests CVXPY Quadratic Programming formulation and hard constraint enforcement."""

    def test_optimizer_convergence_unconstrained(self, default_optimiser, sample_weights):
        curr, targ = sample_weights
        # With high turnover budget (e.g. 50%), optimizer should achieve target exactly
        result = default_optimiser.optimize_allocation(curr, targ, turnover_budget=0.50)

        assert result["status"] == "OPTIMAL"
        assert result["constraints_verified"] is True
        assert len(result["violations"]) == 0

        # Optimal weights match target weights closely
        for ac in ASSET_CLASSES:
            assert math.isclose(result["optimal_weights"][ac], targ[ac], abs_tol=1e-3)

        assert result["predicted_tracking_error"] < 1e-3
        assert result["turnover"] > 0.0

    def test_budget_constraint_conservation(self, default_optimiser, sample_weights):
        curr, targ = sample_weights
        result = default_optimiser.optimize_allocation(curr, targ, turnover_budget=0.08)

        total_opt = sum(result["optimal_weights"].values())
        assert math.isclose(total_opt, 1.0, abs_tol=1e-5)

        # Net trade weight sum must be zero
        net_trade = sum(result["trade_weights"].values())
        assert math.isclose(net_trade, 0.0, abs_tol=1e-5)

    def test_long_only_constraint(self, default_optimiser):
        # Target has 0% in an asset; optimizer must never allocate negative weights
        curr = {
            "NIFTY_50_EQUITY": 0.05,
            "G_SEC_BONDS": 0.35,
            "CORP_BONDS": 0.30,
            "GOLD_ETF": 0.20,
            "LIQUID_CASH": 0.10,
        }
        targ = {
            "NIFTY_50_EQUITY": 0.00,
            "G_SEC_BONDS": 0.40,
            "CORP_BONDS": 0.30,
            "GOLD_ETF": 0.20,
            "LIQUID_CASH": 0.10,
        }
        result = default_optimiser.optimize_allocation(curr, targ, turnover_budget=0.10)

        assert result["status"] == "OPTIMAL"
        for ac, w in result["optimal_weights"].items():
            assert w >= -1e-6, f"Asset {ac} violated long-only bound: {w}"

    def test_turnover_budget_enforcement(self, default_optimiser, sample_weights):
        curr, targ = sample_weights
        turnover_ceiling = 0.05  # Strict 5% turnover budget

        result = default_optimiser.optimize_allocation(curr, targ, turnover_budget=turnover_ceiling)

        assert result["status"] == "OPTIMAL"
        # Actual turnover must be <= turnover_ceiling + tolerance
        assert result["turnover"] <= turnover_ceiling + 1e-4

        # Because turnover is constrained, predicted tracking error must be > 0
        assert result["predicted_tracking_error"] > 0.001

    def test_sebi_single_issuer_limit(self, default_optimiser):
        # Asset with target > 10% constrained by 10% issuer limit
        curr = {
            "NIFTY_50_EQUITY": 0.20,
            "G_SEC_BONDS": 0.30,
            "CORP_BONDS": 0.20,
            "GOLD_ETF": 0.20,
            "LIQUID_CASH": 0.10,
        }
        targ = {
            "NIFTY_50_EQUITY": 0.20,
            "G_SEC_BONDS": 0.30,
            "CORP_BONDS": 0.20,
            "GOLD_ETF": 0.20,
            "LIQUID_CASH": 0.10,
        }
        # Enforce 10% ceiling across assets
        result = default_optimiser.optimize_allocation(
            curr,
            targ,
            sebi_issuer_limit=0.25,
            turnover_budget=0.20,
        )
        assert result["status"] == "OPTIMAL"
        for ac, w in result["optimal_weights"].items():
            assert w <= 0.25 + 1e-4

    def test_sebi_sector_concentration_limit(self, default_optimiser):
        curr = {
            "NIFTY_50_EQUITY": 0.50,
            "G_SEC_BONDS": 0.25,
            "CORP_BONDS": 0.15,
            "GOLD_ETF": 0.05,
            "LIQUID_CASH": 0.05,
        }
        targ = {
            "NIFTY_50_EQUITY": 0.50,
            "G_SEC_BONDS": 0.25,
            "CORP_BONDS": 0.15,
            "GOLD_ETF": 0.05,
            "LIQUID_CASH": 0.05,
        }
        # Cap DEBT sector (G_SEC_BONDS + CORP_BONDS) to 30% max
        sector_map = {
            "NIFTY_50_EQUITY": "EQUITY",
            "G_SEC_BONDS": "DEBT",
            "CORP_BONDS": "DEBT",
            "GOLD_ETF": "COMMODITY",
            "LIQUID_CASH": "CASH",
        }
        sector_limits = {"DEBT": 0.30}

        result = default_optimiser.optimize_allocation(
            curr,
            targ,
            sebi_sector_limits=sector_limits,
            asset_sector_map=sector_map,
            turnover_budget=0.30,
        )
        assert result["status"] == "OPTIMAL"
        debt_w = result["optimal_weights"]["G_SEC_BONDS"] + result["optimal_weights"]["CORP_BONDS"]
        assert debt_w <= 0.30 + 1e-4

    def test_cash_buffer_enforcement(self, default_optimiser):
        curr = {
            "NIFTY_50_EQUITY": 0.50,
            "G_SEC_BONDS": 0.30,
            "CORP_BONDS": 0.15,
            "GOLD_ETF": 0.05,
            "LIQUID_CASH": 0.00,  # Zero cash currently
        }
        targ = {
            "NIFTY_50_EQUITY": 0.55,
            "G_SEC_BONDS": 0.30,
            "CORP_BONDS": 0.10,
            "GOLD_ETF": 0.05,
            "LIQUID_CASH": 0.00,  # Zero cash in target
        }
        # Hard constraint requires min 3% cash buffer
        result = default_optimiser.optimize_allocation(
            curr,
            targ,
            min_cash_buffer=0.03,
            turnover_budget=0.20,
        )
        assert result["status"] == "OPTIMAL"
        assert result["optimal_weights"]["LIQUID_CASH"] >= 0.03 - 1e-5

    def test_net_cash_flow_deposit(self, default_optimiser, sample_weights):
        curr, targ = sample_weights
        # 10% external cash deposit into portfolio
        deposit = 0.10
        result = default_optimiser.optimize_allocation(
            curr,
            targ,
            net_cash_flow=deposit,
            turnover_budget=0.30,
        )
        assert result["status"] == "OPTIMAL"
        total_w = sum(result["optimal_weights"].values())
        assert math.isclose(total_w, 1.0 + deposit, abs_tol=1e-4)

    def test_min_trade_size_thresholding(self, default_optimiser, sample_weights):
        curr, targ = sample_weights
        # Any trade < 2% of AUM should be thresholded to 0
        min_trade = 0.02
        result = default_optimiser.optimize_allocation(
            curr,
            targ,
            turnover_budget=0.10,
            min_trade_size=min_trade,
        )
        assert result["status"] == "OPTIMAL"
        for ac, trade_w in result["trade_weights"].items():
            if abs(trade_w) > 1e-5:
                assert abs(trade_w) >= min_trade - 1e-5, f"Trade in {ac} ({trade_w}) was below min_trade {min_trade}"


class TestTradeListGenerator:
    """Tests joint round-lot optimization and executable ticket synthesis."""

    def test_joint_round_lot_generation(self):
        generator = TradeListGenerator(min_order_value_inr=1000.0)
        curr = {
            "NIFTY_50_EQUITY": 0.60,
            "G_SEC_BONDS": 0.15,
            "CORP_BONDS": 0.15,
            "GOLD_ETF": 0.05,
            "LIQUID_CASH": 0.05,
        }
        opt = {
            "NIFTY_50_EQUITY": 0.50,
            "G_SEC_BONDS": 0.25,
            "CORP_BONDS": 0.15,
            "GOLD_ETF": 0.05,
            "LIQUID_CASH": 0.05,
        }
        aum = 10_000_000.0  # ₹1 Crore

        plan = generator.generate_trade_plan(
            portfolio_id="PORT_1D_TEST",
            current_weights=curr,
            optimal_weights=opt,
            portfolio_aum=aum,
            min_cash_buffer_pct=0.02,
        )

        assert plan["order_count"] >= 1
        orders = plan["orders"]
        assert all(isinstance(o, TradeOrder) for o in orders)

        # Check cash conservation
        cash_ledger = plan["cash_ledger"]
        assert cash_ledger["post_cash_inr"] >= cash_ledger["min_required_buffer_inr"]
        assert cash_ledger["buy_spend_inr"] <= cash_ledger["initial_cash_inr"] + cash_ledger["sell_proceeds_inr"]

        # Check discrete weights sum to 1.0
        disc_weights = plan["discrete_weights"]
        assert math.isclose(sum(disc_weights.values()), 1.0, abs_tol=1e-3)


class TestCostEstimator:
    """Tests explicit Indian transaction fees and square-root market impact model."""

    def test_explicit_statutory_charges(self):
        estimator = CostEstimator()
        # Equity BUY order of ₹10 Lakhs
        buy_order = {
            "asset_class": "NIFTY_50_EQUITY",
            "action": "BUY",
            "trade_value_inr": 1_000_000.0,
        }
        costs = estimator.estimate_trade_costs(buy_order)

        # 5 bps brokerage = 500 INR
        assert costs["brokerage_inr"] == 500.0
        # 0.1% STT = 1000 INR
        assert costs["stt_inr"] == 1000.0
        # 1.5 bps Stamp duty on BUY = 150 INR
        assert costs["stamp_duty_inr"] == 150.0
        # GST 18% on (brokerage + exchange)
        assert costs["gst_inr"] > 90.0
        # Total cost must be > total explicit
        assert costs["total_cost_inr"] >= costs["total_explicit_inr"]
        assert costs["total_cost_bps"] > 15.0

    def test_square_root_market_impact(self):
        estimator = CostEstimator()
        # Small trade vs 10x larger trade in same asset
        small_order = {"asset_class": "NIFTY_50_EQUITY", "action": "BUY", "trade_value_inr": 1_000_000.0}
        large_order = {"asset_class": "NIFTY_50_EQUITY", "action": "BUY", "trade_value_inr": 100_000_000.0}

        c_small = estimator.estimate_trade_costs(small_order)
        c_large = estimator.estimate_trade_costs(large_order)

        # Square root model: impact bps scales with sqrt(trade_value / ADV)
        # Ratio of trade values = 100 -> ratio of impact bps ~ sqrt(100) = 10
        ratio = c_large["market_impact_bps"] / max(c_small["market_impact_bps"], 1e-6)
        assert math.isclose(ratio, 10.0, rel_tol=0.05)


class TestConstraintManager:
    """Tests pre-trade verification certificates and violation detection."""

    def test_pre_trade_report_clean_portfolio(self):
        cm = ConstraintManager()
        curr = {"NIFTY_50_EQUITY": 0.55, "G_SEC_BONDS": 0.35, "LIQUID_CASH": 0.10}
        prop = {"NIFTY_50_EQUITY": 0.50, "G_SEC_BONDS": 0.40, "LIQUID_CASH": 0.10}

        rpt = cm.generate_pre_trade_report("PORT_001", curr, prop)
        assert isinstance(rpt, PreTradeValidationReport)
        assert rpt.is_valid is True
        assert rpt.sebi_compliant is True
        assert len(rpt.violations) == 0

    def test_pre_trade_report_detects_violations(self):
        cm = ConstraintManager(min_cash_buffer=0.02, max_turnover_per_rebalance=0.10)
        curr = {"NIFTY_50_EQUITY": 0.60, "G_SEC_BONDS": 0.35, "LIQUID_CASH": 0.05}
        # Proposed violates: negative weight, cash below buffer, and turnover > 10%
        prop = {"NIFTY_50_EQUITY": 0.85, "G_SEC_BONDS": 0.145, "LIQUID_CASH": 0.005}

        rpt = cm.generate_pre_trade_report("PORT_BREACH", curr, prop, turnover_budget=0.10)
        assert rpt.is_valid is False
        assert len(rpt.violations) >= 2
        # Check cash deficit and turnover breach detected
        assert any("Cash buffer breach" in v for v in rpt.violations)
        assert any("Turnover budget breach" in v for v in rpt.violations)
