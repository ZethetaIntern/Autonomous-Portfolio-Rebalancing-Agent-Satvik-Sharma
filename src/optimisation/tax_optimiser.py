"""Tax Optimization Engine for WealthPilot AI.

Calculates capital gains tax liabilities under the Indian Income Tax Act (Sections 111A, 112A),
enforces legal set-off rules, utilizes the ₹1.25 Lakh LTCG exemption, and applies wash-sale
avoidance rerouting.
"""

from __future__ import annotations

import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

from src.optimisation import TradeOrder
from src.optimisation.tax_lot_manager import TaxLotManager, DEFAULT_SUBSTITUTE_ASSET_MAP


class TaxOptimiser:
    """Computes statutory Indian capital gains tax liabilities, set-offs, and tax-loss harvesting alpha."""

    def __init__(
        self,
        stcg_rate: float = 0.20,              # 20.0% STCG under Section 111A (Budget 2024)
        ltcg_rate: float = 0.125,             # 12.5% LTCG under Section 112A (Budget 2024)
        annual_ltcg_exemption: float = 125000.0, # ₹1.25 Lakh Section 112A exemption
        wash_sale_window_days: int = 30,
        substitute_map: Optional[Dict[str, str]] = None,
    ) -> None:
        self.stcg_rate = stcg_rate
        self.ltcg_rate = ltcg_rate
        self.annual_ltcg_exemption = annual_ltcg_exemption
        self.wash_sale_window_days = wash_sale_window_days
        self.substitute_map = substitute_map or DEFAULT_SUBSTITUTE_ASSET_MAP.copy()

    def calculate_tax_liability(
        self,
        realized_stcg: float,
        realized_stcl: float = 0.0,
        realized_ltcg: float = 0.0,
        realized_ltcl: float = 0.0,
        claimed_ltcg_exemption: float = 0.0,
    ) -> Dict[str, float]:
        """Calculates net capital gains tax liability under Indian set-off and exemption provisions.
        
        Set-off Rules under Income Tax Act:
            1. STCL can be set off against both STCG and LTCG.
            2. LTCL can ONLY be set off against LTCG (cannot offset STCG).
            3. Net LTCG after loss set-offs is eligible for ₹1.25L exemption (Section 112A).
            4. Net STCG is taxed at 20%.
            5. Net taxable LTCG is taxed at 12.5%.
        """
        stcg = max(0.0, float(realized_stcg))
        stcl = max(0.0, float(realized_stcl))
        ltcg = max(0.0, float(realized_ltcg))
        ltcl = max(0.0, float(realized_ltcl))

        # Step 1: Set off LTCL against LTCG
        ltcl_offset_against_ltcg = min(ltcg, ltcl)
        ltcg_remaining = ltcg - ltcl_offset_against_ltcg
        unabsorbed_ltcl = ltcl - ltcl_offset_against_ltcg

        # Step 2: Set off STCL against STCG first
        stcl_offset_against_stcg = min(stcg, stcl)
        stcg_remaining = stcg - stcl_offset_against_stcg
        remaining_stcl = stcl - stcl_offset_against_stcg

        # Step 3: Set off remaining STCL against remaining LTCG
        stcl_offset_against_ltcg = min(ltcg_remaining, remaining_stcl)
        ltcg_after_stcl = ltcg_remaining - stcl_offset_against_ltcg
        unabsorbed_stcl = remaining_stcl - stcl_offset_against_ltcg

        # Step 4: Apply Section 112A annual exemption (₹1.25 Lakh) to remaining LTCG
        remaining_exemption = max(0.0, self.annual_ltcg_exemption - claimed_ltcg_exemption)
        applied_exemption = min(ltcg_after_stcl, remaining_exemption)
        taxable_ltcg = max(0.0, ltcg_after_stcl - applied_exemption)
        taxable_stcg = max(0.0, stcg_remaining)

        # Step 5: Compute tax
        ltcg_tax = taxable_ltcg * self.ltcg_rate
        stcg_tax = taxable_stcg * self.stcg_rate
        total_tax = ltcg_tax + stcg_tax

        return {
            "gross_stcg": round(stcg, 2),
            "gross_stcl": round(stcl, 2),
            "gross_ltcg": round(ltcg, 2),
            "gross_ltcl": round(ltcl, 2),
            "stcl_offset_against_stcg": round(stcl_offset_against_stcg, 2),
            "stcl_offset_against_ltcg": round(stcl_offset_against_ltcg, 2),
            "ltcl_offset_against_ltcg": round(ltcl_offset_against_ltcg, 2),
            "unabsorbed_stcl_carryforward": round(unabsorbed_stcl, 2),
            "unabsorbed_ltcl_carryforward": round(unabsorbed_ltcl, 2),
            "applied_ltcg_exemption": round(applied_exemption, 2),
            "taxable_stcg": round(taxable_stcg, 2),
            "taxable_ltcg": round(taxable_ltcg, 2),
            "stcg_tax_inr": round(stcg_tax, 2),
            "ltcg_tax_inr": round(ltcg_tax, 2),
            "total_tax_inr": round(total_tax, 2),
        }

    def evaluate_trade_plan_taxes(
        self,
        portfolio_id: str,
        orders: List[TradeOrder],
        lot_manager: TaxLotManager,
        ytd_claimed_exemption: float = 0.0,
        strategy: str = "TAX_MINIMIZER",
        as_of: Optional[datetime.date] = None,
        apply_wash_sale_substitutes: bool = True,
    ) -> Dict[str, Any]:
        """Simulates tax consequences across proposed trade orders and checks wash-sale substitutions.
        
        Args:
            portfolio_id: Identifier of portfolio.
            orders: Proposed TradeOrder tickets from TradeListGenerator.
            lot_manager: Active TaxLotManager instance holding current portfolio lots.
            ytd_claimed_exemption: Section 112A LTCG exemption already utilized this FY.
            strategy: Lot selection rule ("TAX_MINIMIZER", "FIFO", "LIFO", "HIFO").
            as_of: Evaluation date.
            apply_wash_sale_substitutes: If True, automatically substitutes buy orders in wash window.

        Returns:
            Dict containing total tax liability, per-order tax implications, updated orders with substitutes,
            and wash-sale warnings.
        """
        ref_date = as_of or datetime.date.today()

        total_stcg = 0.0
        total_stcl = 0.0
        total_ltcg = 0.0
        total_ltcl = 0.0
        processed_orders: List[TradeOrder] = []
        wash_sale_substitutions: List[Dict[str, Any]] = []

        for order in orders:
            order_copy = TradeOrder(
                portfolio_id=order.portfolio_id,
                asset_class=order.asset_class,
                action=order.action,
                target_units=order.target_units,
                estimated_price=order.estimated_price,
                trade_value_inr=order.trade_value_inr,
                expected_impact_bps=order.expected_impact_bps,
                tax_implication_inr=0.0,
            )

            if order.action.upper() == "SELL":
                # Evaluate simulated lot depletion
                allocations = lot_manager.select_lots_for_sale(
                    portfolio_id=portfolio_id,
                    asset_class=order.asset_class,
                    units_to_sell=order.target_units,
                    strategy=strategy,
                    as_of=ref_date,
                )

                order_stcg = 0.0
                order_stcl = 0.0
                order_ltcg = 0.0
                order_ltcl = 0.0

                for lot, units in allocations:
                    cost = units * lot.cost_per_share
                    proceeds = units * order.estimated_price
                    diff = proceeds - cost
                    is_lt = lot.is_ltcg(ref_date)

                    if diff >= 0:
                        if is_lt:
                            order_ltcg += diff
                        else:
                            order_stcg += diff
                    else:
                        if is_lt:
                            order_ltcl += abs(diff)
                        else:
                            order_stcl += abs(diff)

                # Estimate isolated tax for this ticket (marginal)
                ticket_tax = self.calculate_tax_liability(
                    realized_stcg=order_stcg,
                    realized_stcl=order_stcl,
                    realized_ltcg=order_ltcg,
                    realized_ltcl=order_ltcl,
                    claimed_ltcg_exemption=self.annual_ltcg_exemption, # conservative marginal
                )["total_tax_inr"]

                order_copy.tax_implication_inr = round(ticket_tax, 2)

                total_stcg += order_stcg
                total_stcl += order_stcl
                total_ltcg += order_ltcg
                total_ltcl += order_ltcl

                if (order_stcl + order_ltcl) > 0:
                    lot_manager.record_loss_harvest(portfolio_id, order.asset_class, ref_date, order_stcl + order_ltcl)

            elif order.action.upper() == "BUY":
                # Check for wash-sale restriction
                is_blocked, clear_date = lot_manager.is_in_wash_sale_window(
                    portfolio_id=portfolio_id,
                    asset_class=order.asset_class,
                    check_date=ref_date,
                )

                if is_blocked:
                    substitute = self.substitute_map.get(order.asset_class, f"{order.asset_class}_ETF")
                    wash_sale_substitutions.append({
                        "original_asset": order.asset_class,
                        "substitute_asset": substitute,
                        "blocked_until": clear_date.isoformat() if clear_date else "",
                        "reason": f"30-day wash-sale avoidance window active for {order.asset_class}",
                    })
                    if apply_wash_sale_substitutes and substitute != order.asset_class:
                        order_copy.asset_class = substitute

            processed_orders.append(order_copy)

        # Aggregate portfolio-level capital gains computation
        aggregate_tax = self.calculate_tax_liability(
            realized_stcg=total_stcg,
            realized_stcl=total_stcl,
            realized_ltcg=total_ltcg,
            realized_ltcl=total_ltcl,
            claimed_ltcg_exemption=ytd_claimed_exemption,
        )

        realized_shield = (
            (aggregate_tax["stcl_offset_against_stcg"] * self.stcg_rate)
            + (aggregate_tax["stcl_offset_against_ltcg"] * self.ltcg_rate)
            + (aggregate_tax["ltcl_offset_against_ltcg"] * self.ltcg_rate)
        )
        potential_carry_forward_shield = (total_stcl * self.stcg_rate) + (total_ltcl * self.ltcg_rate)
        effective_shield = realized_shield if realized_shield > 0 else potential_carry_forward_shield

        return {
            "portfolio_id": portfolio_id,
            "orders": processed_orders,
            "tax_summary": aggregate_tax,
            "wash_sale_substitutions": wash_sale_substitutions,
            "total_estimated_tax_inr": aggregate_tax["total_tax_inr"],
            "harvested_losses_inr": round(total_stcl + total_ltcl, 2),
            "tax_shield_inr": round(effective_shield, 2),
        }
