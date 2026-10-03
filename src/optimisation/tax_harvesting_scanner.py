"""Tax Harvesting Scanner for WealthPilot AI.

Automated scanner to identify harvestable loss lots during March financial year-end windows
and generate tax-advantaged rebalancing trades with wash-sale avoidance and beta preservation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import datetime
from typing import Any, Dict, List, Optional, Tuple

from src.optimisation import TradeOrder
from src.optimisation.tax_lot_manager import TaxLot, TaxLotManager
from src.optimisation.tax_optimiser import TaxOptimiser


@dataclass
class TaxHarvestingOpportunity:
    """Individual portfolio tax-loss and tax-gain harvesting opportunity."""
    portfolio_id: str
    is_march_window: bool
    harvestable_stcl_inr: float
    harvestable_ltcl_inr: float
    unrealized_ltcg_exemption_eligible_inr: float
    estimated_tax_shield_inr: float
    recommended_orders: List[TradeOrder] = field(default_factory=list)
    substitute_mappings: Dict[str, str] = field(default_factory=dict)
    summary_narrative: str = ""


@dataclass
class TaxHarvestingBatchReport:
    """Aggregated scan results across portfolio universe."""
    scan_date: datetime.date
    is_march_fy_end_window: bool
    total_portfolios_scanned: int
    portfolios_with_opportunities: int
    total_harvestable_losses_inr: float
    total_tax_alpha_inr: float
    opportunities: List[TaxHarvestingOpportunity] = field(default_factory=list)


class TaxHarvestingScanner:
    """Scans portfolio tax-lot holdings, detects loss harvesting opportunities, and generates rebalancing plans."""

    def __init__(
        self,
        tax_optimiser: Optional[TaxOptimiser] = None,
        min_harvestable_loss_inr: float = 10000.0,
        enable_gain_harvesting_up_to_exemption: bool = True,
    ) -> None:
        self.tax_optimiser = tax_optimiser or TaxOptimiser()
        self.min_loss_threshold = float(min_harvestable_loss_inr)
        self.enable_gain_harvesting = enable_gain_harvesting_up_to_exemption

    def is_march_financial_year_end(self, as_of: Optional[datetime.date] = None) -> bool:
        """Indian financial year ends on March 31. March 1-31 is the prime tax harvesting window."""
        ref_date = as_of or datetime.date.today()
        return ref_date.month == 3

    def scan_portfolio(
        self,
        portfolio_id: str,
        lot_manager: TaxLotManager,
        ytd_realized_stcg: float = 0.0,
        ytd_realized_ltcg: float = 0.0,
        ytd_claimed_exemption: float = 0.0,
        as_of: Optional[datetime.date] = None,
        force_scan: bool = False,
    ) -> Optional[TaxHarvestingOpportunity]:
        """Scans a single portfolio and compiles tax-loss harvesting and wash-sale replacement orders.
        
        Args:
            portfolio_id: Unique portfolio ID.
            lot_manager: Active TaxLotManager with portfolio lots.
            ytd_realized_stcg: Already realized short term capital gains this FY.
            ytd_realized_ltcg: Already realized long term capital gains this FY.
            ytd_claimed_exemption: Section 112A LTCG exemption already claimed this FY.
            as_of: Date of scan (defaults to today).
            force_scan: If True, evaluates regardless of whether date is in March.

        Returns:
            TaxHarvestingOpportunity if harvestable losses exceed threshold or offset gains, else None.
        """
        ref_date = as_of or datetime.date.today()
        is_march = self.is_march_financial_year_end(ref_date)

        if not is_march and not force_scan:
            return None

        lots = lot_manager.get_lots(portfolio_id)
        if not lots:
            return None

        # 1. Identify loss lots and eligible LTCG lots
        loss_lots: List[TaxLot] = []
        gain_lots: List[TaxLot] = []

        total_stcl = 0.0
        total_ltcl = 0.0
        total_unrealized_ltcg = 0.0

        for lot in lots:
            gain = lot.unrealized_gain_loss
            is_lt = lot.is_ltcg(ref_date)
            if gain < -1e-4:
                loss_lots.append(lot)
                if is_lt:
                    total_ltcl += abs(gain)
                else:
                    total_stcl += abs(gain)
            elif gain > 1e-4 and is_lt:
                gain_lots.append(lot)
                total_unrealized_ltcg += gain

        total_losses = total_stcl + total_ltcl
        has_taxable_gains_to_offset = (ytd_realized_stcg > 0) or (ytd_realized_ltcg > 0)

        # Trigger condition: total losses exceed threshold OR can offset existing realized taxable gains
        if total_losses < self.min_loss_threshold and not has_taxable_gains_to_offset:
            return None

        # 2. Build tax-advantaged trades
        orders: List[TradeOrder] = []
        substitutes_used: Dict[str, str] = {}

        # Step A: Harvest loss positions (SELL loss lots)
        # Group losses by asset
        asset_loss_units: Dict[str, float] = {}
        asset_prices: Dict[str, float] = {}
        for lot in loss_lots:
            asset_loss_units[lot.asset_class] = asset_loss_units.get(lot.asset_class, 0.0) + lot.quantity
            asset_prices[lot.asset_class] = lot.current_price

        for asset, units in asset_loss_units.items():
            price = asset_prices[asset]
            val = units * price
            # Sell order for loss asset
            orders.append(
                TradeOrder(
                    portfolio_id=portfolio_id,
                    asset_class=asset,
                    action="SELL",
                    target_units=round(units, 4),
                    estimated_price=price,
                    trade_value_inr=round(val, 2),
                    expected_impact_bps=0.0,
                    tax_implication_inr=0.0,  # Negative tax = tax benefit
                )
            )

            # Step B: Immediate BUY order into substitute asset to preserve portfolio beta & avoid wash sale!
            substitute = lot_manager.get_substitute_asset(asset)
            substitutes_used[asset] = substitute
            substitute_price = price  # Approximate unit price
            orders.append(
                TradeOrder(
                    portfolio_id=portfolio_id,
                    asset_class=substitute,
                    action="BUY",
                    target_units=round(units, 4),
                    estimated_price=substitute_price,
                    trade_value_inr=round(val, 2),
                    expected_impact_bps=0.0,
                    tax_implication_inr=0.0,
                )
            )

        # Step C: Optional Section 112A Tax-Free LTCG Gain Harvesting (Step-up basis)
        remaining_ltcg_exemption = max(0.0, self.tax_optimiser.annual_ltcg_exemption - ytd_claimed_exemption)
        eligible_free_ltcg = 0.0
        if self.enable_gain_harvesting and remaining_ltcg_exemption > 0 and total_unrealized_ltcg > 0:
            # Sort gain lots by lowest percentage gain (easiest to step-up)
            gain_lots.sort(key=lambda l: (l.current_price - l.cost_per_share) / max(l.cost_per_share, 1.0))
            exemption_budget = remaining_ltcg_exemption

            for lot in gain_lots:
                if exemption_budget <= 0:
                    break
                lot_gain = lot.unrealized_gain_loss
                harvest_gain = min(lot_gain, exemption_budget)
                if harvest_gain > 5000.0:  # Minimum worthwhile gain slice
                    fraction = harvest_gain / lot_gain
                    units_to_harvest = lot.quantity * fraction
                    val = units_to_harvest * lot.current_price

                    orders.append(
                        TradeOrder(
                            portfolio_id=portfolio_id,
                            asset_class=lot.asset_class,
                            action="SELL",
                            target_units=round(units_to_harvest, 4),
                            estimated_price=lot.current_price,
                            trade_value_inr=round(val, 2),
                            expected_impact_bps=0.0,
                            tax_implication_inr=0.0,  # Tax-free within exemption
                        )
                    )
                    # Immediately re-buy identical asset to step-up cost basis (gain sales are not wash sales!)
                    orders.append(
                        TradeOrder(
                            portfolio_id=portfolio_id,
                            asset_class=lot.asset_class,
                            action="BUY",
                            target_units=round(units_to_harvest, 4),
                            estimated_price=lot.current_price,
                            trade_value_inr=round(val, 2),
                            expected_impact_bps=0.0,
                            tax_implication_inr=0.0,
                        )
                    )
                    eligible_free_ltcg += harvest_gain
                    exemption_budget -= harvest_gain

        # Calculate estimated tax alpha
        # STCL saves 20% on STCG; LTCL saves 12.5% on LTCG; Free LTCG saves 12.5% future tax
        tax_shield = (
            (total_stcl * self.tax_optimiser.stcg_rate)
            + (total_ltcl * self.tax_optimiser.ltcg_rate)
            + (eligible_free_ltcg * self.tax_optimiser.ltcg_rate)
        )

        narrative = (
            f"Tax Harvesting Opportunity for {portfolio_id}: Identified INR {total_losses:,.2f} harvestable losses "
            f"(STCL: INR {total_stcl:,.2f}, LTCL: INR {total_ltcl:,.2f}) and INR {eligible_free_ltcg:,.2f} tax-free "
            f"LTCG basis step-up. Generated {len(orders)} paired trades yielding an estimated tax alpha of "
            f"INR {tax_shield:,.2f}."
        )

        return TaxHarvestingOpportunity(
            portfolio_id=portfolio_id,
            is_march_window=is_march,
            harvestable_stcl_inr=round(total_stcl, 2),
            harvestable_ltcl_inr=round(total_ltcl, 2),
            unrealized_ltcg_exemption_eligible_inr=round(eligible_free_ltcg, 2),
            estimated_tax_shield_inr=round(tax_shield, 2),
            recommended_orders=orders,
            substitute_mappings=substitutes_used,
            summary_narrative=narrative,
        )

    def scan_batch(
        self,
        portfolio_ids: List[str],
        lot_manager: TaxLotManager,
        as_of: Optional[datetime.date] = None,
        force_scan: bool = False,
    ) -> TaxHarvestingBatchReport:
        """Batch scans multiple portfolios and aggregates tax alpha."""
        ref_date = as_of or datetime.date.today()
        is_march = self.is_march_financial_year_end(ref_date)
        opportunities: List[TaxHarvestingOpportunity] = []

        total_losses = 0.0
        total_alpha = 0.0

        for p_id in portfolio_ids:
            opp = self.scan_portfolio(
                portfolio_id=p_id,
                lot_manager=lot_manager,
                as_of=ref_date,
                force_scan=force_scan,
            )
            if opp:
                opportunities.append(opp)
                total_losses += opp.harvestable_stcl_inr + opp.harvestable_ltcl_inr
                total_alpha += opp.estimated_tax_shield_inr

        return TaxHarvestingBatchReport(
            scan_date=ref_date,
            is_march_fy_end_window=is_march,
            total_portfolios_scanned=len(portfolio_ids),
            portfolios_with_opportunities=len(opportunities),
            total_harvestable_losses_inr=round(total_losses, 2),
            total_tax_alpha_inr=round(total_alpha, 2),
            opportunities=opportunities,
        )
