"""Comprehensive Unit & Integration Test Suite for Day 5: Tax-Aware Execution & Liquidity Scheduling.

Tests:
    - Tax lot tracking, cost basis calculation, and 365-day equity holding threshold.
    - Lot selection algorithms (FIFO, LIFO, HIFO, Tax-Minimizer).
    - Post-Budget 2024 Indian capital gains tax rates (20% STCG, 12.5% LTCG, ₹1.25L exemption).
    - Statutory set-off rules (STCL vs STCG/LTCG, LTCL vs LTCG only).
    - 30-day wash-sale avoidance window and substitute ETF beta preservation.
    - Liquidity scoring, tier classification, and multi-day TWAP/VWAP scheduling.
    - March financial year-end automated tax-harvesting scanner and tax alpha generation.
"""

import datetime
import math
import pytest

from src.optimisation import (
    ExecutionSchedule,
    LiquidityScorer,
    TaxHarvestingOpportunity,
    TaxHarvestingScanner,
    TaxLot,
    TaxLotManager,
    TaxOptimiser,
    TradeOrder,
)


@pytest.fixture
def lot_manager():
    return TaxLotManager(wash_sale_window_days=30, equity_ltcg_days=365)


@pytest.fixture
def tax_optimiser():
    return TaxOptimiser(
        stcg_rate=0.20,
        ltcg_rate=0.125,
        annual_ltcg_exemption=125000.0,
    )


@pytest.fixture
def liquidity_scorer():
    return LiquidityScorer()


class TestTaxLotManager:
    """Tests lot tracking, holding period categorization, and lot selection strategies."""

    def test_tax_lot_holding_period_and_classification(self):
        ref_date = datetime.date(2026, 3, 15)
        # Lot acquired 200 days ago -> STCG / STCL (< 365 days)
        st_date = ref_date - datetime.timedelta(days=200)
        lot_st = TaxLot("L_ST", "P_TEST", "NIFTY_50_EQUITY", 10.0, st_date, 20000.0, 24000.0)

        assert lot_st.holding_period_days(ref_date) == 200
        assert lot_st.is_ltcg(ref_date) is False
        assert lot_st.gain_category(ref_date) == "STCG"
        assert lot_st.unrealized_gain_loss == 40000.0

        # Lot acquired 400 days ago -> LTCG / LTCL (>= 365 days)
        lt_date = ref_date - datetime.timedelta(days=400)
        lot_lt = TaxLot("L_LT", "P_TEST", "NIFTY_50_EQUITY", 10.0, lt_date, 26000.0, 24000.0)

        assert lot_lt.holding_period_days(ref_date) == 400
        assert lot_lt.is_ltcg(ref_date) is True
        assert lot_lt.gain_category(ref_date) == "LTCL"
        assert lot_lt.unrealized_gain_loss == -20000.0

    def test_lot_selection_fifo_lifo_hifo(self, lot_manager):
        ref_date = datetime.date(2026, 3, 15)
        p_id = "PORT_SELECTION"

        # Three lots with different dates and costs:
        # L1: Jan 2024, Cost 18000 (oldest, low cost)
        # L2: Jan 2025, Cost 25000 (middle, high cost)
        # L3: Dec 2025, Cost 22000 (newest, medium cost)
        lot_manager.add_lots([
            TaxLot("L1", p_id, "NIFTY_50_EQUITY", 10.0, datetime.date(2024, 1, 10), 18000.0, 24000.0),
            TaxLot("L2", p_id, "NIFTY_50_EQUITY", 10.0, datetime.date(2025, 1, 10), 25000.0, 24000.0),
            TaxLot("L3", p_id, "NIFTY_50_EQUITY", 10.0, datetime.date(2025, 12, 10), 22000.0, 24000.0),
        ])

        # FIFO: should select L1 first
        fifo_lots = lot_manager.select_lots_for_sale(p_id, "NIFTY_50_EQUITY", 5.0, strategy="FIFO", as_of=ref_date)
        assert fifo_lots[0][0].lot_id == "L1"

        # LIFO: should select L3 first
        lifo_lots = lot_manager.select_lots_for_sale(p_id, "NIFTY_50_EQUITY", 5.0, strategy="LIFO", as_of=ref_date)
        assert lifo_lots[0][0].lot_id == "L3"

        # HIFO: should select L2 (cost 25000) first
        hifo_lots = lot_manager.select_lots_for_sale(p_id, "NIFTY_50_EQUITY", 5.0, strategy="HIFO", as_of=ref_date)
        assert hifo_lots[0][0].lot_id == "L2"

    def test_lot_selection_tax_minimizer(self, lot_manager):
        ref_date = datetime.date(2026, 3, 15)
        p_id = "PORT_TAX_MIN"

        # Add:
        # L_Gain: LTCG of +4000 per share (price 24k, cost 20k)
        # L_Loss: STCL of -2000 per share (price 24k, cost 26k)
        lot_manager.add_lots([
            TaxLot("L_Gain", p_id, "NIFTY_50_EQUITY", 10.0, datetime.date(2024, 1, 1), 20000.0, 24000.0),
            TaxLot("L_Loss", p_id, "NIFTY_50_EQUITY", 10.0, datetime.date(2025, 11, 1), 26000.0, 24000.0),
        ])

        # TAX_MINIMIZER must prioritize the loss lot (L_Loss) first to harvest losses and shield tax
        selected = lot_manager.select_lots_for_sale(p_id, "NIFTY_50_EQUITY", 5.0, strategy="TAX_MINIMIZER", as_of=ref_date)
        assert selected[0][0].lot_id == "L_Loss"

    def test_execute_sale_and_wash_sale_recording(self, lot_manager):
        ref_date = datetime.date(2026, 3, 15)
        p_id = "PORT_EXEC_SALE"

        lot = TaxLot("L_HARVEST", p_id, "NIFTY_50_EQUITY", 10.0, datetime.date(2025, 11, 1), 26000.0, 24000.0)
        lot_manager.add_lot(lot)

        # Sell 5 units at 24000 -> realized loss = 5 * (24000 - 26000) = -10,000 INR
        sale_result = lot_manager.execute_sale(
            p_id, "NIFTY_50_EQUITY", 5.0, strategy="TAX_MINIMIZER", as_of=ref_date
        )

        assert sale_result["realized_stcl_inr"] == 10000.0
        assert sale_result["units_sold"] == 5.0

        # Remaining units in lot must be 5.0
        active = lot_manager.get_lots(p_id, "NIFTY_50_EQUITY")
        assert len(active) == 1
        assert active[0].quantity == 5.0

        # Wash sale avoidance window must now be active for 30 days
        blocked, clear_date = lot_manager.is_in_wash_sale_window(p_id, "NIFTY_50_EQUITY", check_date=ref_date)
        assert blocked is True
        assert clear_date == ref_date + datetime.timedelta(days=30)

        # Check substitute asset mapping
        substitute = lot_manager.get_substitute_asset("NIFTY_50_EQUITY")
        assert substitute == "NIFTY_NEXT_50_EQUITY"


class TestTaxOptimiser:
    """Tests Indian Income Tax Act capital gains calculations, loss set-offs, and wash sale rerouting."""

    def test_equity_stcg_20_percent_rate(self, tax_optimiser):
        # STCG of 100,000 INR taxed @ 20% = 20,000 INR
        res = tax_optimiser.calculate_tax_liability(realized_stcg=100000.0)
        assert res["stcg_tax_inr"] == 20000.0
        assert res["ltcg_tax_inr"] == 0.0
        assert res["total_tax_inr"] == 20000.0

    def test_section_112a_ltcg_exemption_and_12_5_percent_rate(self, tax_optimiser):
        # LTCG of 200,000 INR; Section 112A exemption = 125,000 INR
        # Taxable LTCG = 75,000 INR @ 12.5% = 9,375 INR
        res = tax_optimiser.calculate_tax_liability(realized_stcg=0.0, realized_ltcg=200000.0)
        assert res["applied_ltcg_exemption"] == 125000.0
        assert res["taxable_ltcg"] == 75000.0
        assert res["ltcg_tax_inr"] == 9375.0
        assert res["total_tax_inr"] == 9375.0

    def test_stcl_can_offset_both_stcg_and_ltcg(self, tax_optimiser):
        # STCL = 80,000 INR; STCG = 30,000 INR; LTCG = 150,000 INR
        # STCL offsets all 30k STCG -> remaining STCL = 50,000 INR
        # Remaining 50k STCL offsets LTCG -> remaining LTCG = 100,000 INR
        # LTCG 100,000 is fully covered by ₹1.25L exemption -> Tax = 0!
        res = tax_optimiser.calculate_tax_liability(
            realized_stcg=30000.0,
            realized_stcl=80000.0,
            realized_ltcg=150000.0,
            realized_ltcl=0.0,
        )
        assert res["stcl_offset_against_stcg"] == 30000.0
        assert res["stcl_offset_against_ltcg"] == 50000.0
        assert res["taxable_stcg"] == 0.0
        assert res["taxable_ltcg"] == 0.0
        assert res["total_tax_inr"] == 0.0

    def test_ltcl_cannot_offset_stcg(self, tax_optimiser):
        # Crucial Indian tax law test: LTCL can ONLY offset LTCG, NOT STCG.
        # STCG = 50,000 INR; LTCL = 50,000 INR
        # Expected: STCG is NOT offset, tax = 20% of 50k = 10,000 INR.
        # LTCL is carried forward!
        res = tax_optimiser.calculate_tax_liability(
            realized_stcg=50000.0,
            realized_stcl=0.0,
            realized_ltcg=0.0,
            realized_ltcl=50000.0,
        )
        assert res["taxable_stcg"] == 50000.0
        assert res["stcg_tax_inr"] == 10000.0
        assert res["unabsorbed_ltcl_carryforward"] == 50000.0
        assert res["total_tax_inr"] == 10000.0

    def test_trade_plan_wash_sale_automatic_substitution(self, tax_optimiser, lot_manager):
        ref_date = datetime.date(2026, 3, 15)
        p_id = "PORT_WASH_TEST"

        # Harvest a loss on NIFTY_50_EQUITY
        lot_manager.record_loss_harvest(p_id, "NIFTY_50_EQUITY", ref_date, 25000.0)

        # Proposed BUY order for NIFTY_50_EQUITY
        buy_order = TradeOrder(
            portfolio_id=p_id,
            asset_class="NIFTY_50_EQUITY",
            action="BUY",
            target_units=10.0,
            estimated_price=24000.0,
            trade_value_inr=240000.0,
        )

        res = tax_optimiser.evaluate_trade_plan_taxes(
            portfolio_id=p_id,
            orders=[buy_order],
            lot_manager=lot_manager,
            as_of=ref_date,
            apply_wash_sale_substitutes=True,
        )

        assert len(res["wash_sale_substitutions"]) == 1
        sub_info = res["wash_sale_substitutions"][0]
        assert sub_info["original_asset"] == "NIFTY_50_EQUITY"
        assert sub_info["substitute_asset"] == "NIFTY_NEXT_50_EQUITY"

        # The processed order must have been replaced with the substitute ETF
        assert res["orders"][0].asset_class == "NIFTY_NEXT_50_EQUITY"


class TestLiquidityScorer:
    """Tests ADV depth scoring, spread scoring, and multi-day TWAP/VWAP scheduling."""

    def test_liquidity_score_and_tier(self, liquidity_scorer):
        # Liquid asset (NIFTY 50)
        nifty_score = liquidity_scorer.score_security("NIFTY_50_EQUITY")
        assert nifty_score["liquidity_score"] >= 0.75
        assert nifty_score["liquidity_tier"] == "TIER_1_HIGH_LIQUIDITY"

        # Illiquid corporate bond with wide spread (30 bps) and lower ADV
        bond_score = liquidity_scorer.score_security("CORP_BONDS", adv_inr=100_000_000.0, bid_ask_spread_bps=30.0)
        assert bond_score["liquidity_score"] < nifty_score["liquidity_score"]

    def test_single_day_schedule_for_small_trade(self, liquidity_scorer):
        order = TradeOrder("P1", "NIFTY_50_EQUITY", "BUY", 10.0, 24000.0, 240000.0)
        sched = liquidity_scorer.generate_execution_schedule(order)

        assert sched.duration_days == 1
        assert sched.is_multi_day is False
        assert len(sched.slices) == 1

    def test_multi_day_twap_schedule_for_illiquid_position(self, liquidity_scorer):
        # ₹50 Crore trade in CORP_BONDS (where ADV is ₹50 Crore, max daily cap is 5% = ₹2.5 Crore)
        # Required duration = ceil(50 Cr / 2.5 Cr) = 20 trading days!
        trade_val = 500_000_000.0
        units = 500_000.0
        order = TradeOrder("P1", "CORP_BONDS", "SELL", units, 1000.0, trade_val)

        sched = liquidity_scorer.generate_execution_schedule(order)
        assert sched.is_multi_day is True
        assert sched.duration_days > 1
        assert len(sched.slices) == sched.duration_days
        # Total value of slices equals total trade value
        total_slice_val = sum(s.target_value_inr for s in sched.slices)
        assert math.isclose(total_slice_val, trade_val, abs_tol=10.0)


class TestTaxHarvestingScanner:
    """Tests automated March financial year-end scanning and tax alpha generation."""

    def test_march_window_detection(self):
        scanner = TaxHarvestingScanner()
        march_date = datetime.date(2026, 3, 20)
        oct_date = datetime.date(2026, 10, 15)

        assert scanner.is_march_financial_year_end(march_date) is True
        assert scanner.is_march_financial_year_end(oct_date) is False

    def test_tax_loss_and_gain_harvesting_generation(self, lot_manager):
        march_date = datetime.date(2026, 3, 25)
        p_id = "PORT_MARCH_HARVEST"

        # Add a loss lot and an LTCG lot
        # Lot 1: NIFTY 50 bought at 26,000, current price 24,000 -> 20,000 loss (STCL)
        # Lot 2: G-SEC BONDS bought at 90, current price 100 -> 10,000 LTCG
        lot_manager.add_lots([
            TaxLot("L_EQUITY_LOSS", p_id, "NIFTY_50_EQUITY", 10.0, datetime.date(2025, 11, 1), 26000.0, 24000.0),
            TaxLot("L_GSEC_GAIN", p_id, "G_SEC_BONDS", 1000.0, datetime.date(2024, 1, 1), 90.0, 100.0),
        ])

        scanner = TaxHarvestingScanner(min_harvestable_loss_inr=10000.0)
        opp = scanner.scan_portfolio(p_id, lot_manager, as_of=march_date)

        assert opp is not None
        assert opp.is_march_window is True
        assert opp.harvestable_stcl_inr == 20000.0
        assert opp.estimated_tax_shield_inr > 0.0

        # Must have paired SELL + substitute BUY orders
        orders = opp.recommended_orders
        assert len(orders) >= 2

        # Check substitute mapping for NIFTY_50_EQUITY
        assert opp.substitute_mappings.get("NIFTY_50_EQUITY") == "NIFTY_NEXT_50_EQUITY"
        sub_buys = [o for o in orders if o.asset_class == "NIFTY_NEXT_50_EQUITY" and o.action == "BUY"]
        assert len(sub_buys) == 1
