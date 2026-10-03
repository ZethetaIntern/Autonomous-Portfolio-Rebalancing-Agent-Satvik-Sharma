"""Tax Specialist Agent for WealthPilot AI.

Role: Chartered Accountant & Quantitative Tax Specialist
Optimizes trade lots for Indian capital gains taxation (STCG vs LTCG),
executes tax-loss harvesting, and enforces wash-sale prevention rules.
"""

from __future__ import annotations

import datetime
from typing import Any, Dict, List, Optional

from src.optimisation.tax_optimiser import TaxOptimiser
from src.optimisation.tax_lot_manager import TaxLotManager, TaxLot
from src.optimisation import TradeOrder

try:
    from crewai import Agent
    HAS_CREWAI = True
    import warnings
    _cw = warnings.warn
    def _compat_crew_warn(message, category=None, stacklevel=1, source=None, *args, **kwargs):
        kwargs.pop("skip_file_prefixes", None)
        return _cw(message, category=category, stacklevel=stacklevel, source=source, *args, **kwargs)
    warnings.warn = _compat_crew_warn
except ImportError:
    HAS_CREWAI = False


class TaxSpecialistAgent:
    """Specialized agent optimizing tax-lot harvesting and enforcing wash-sale rules."""

    def __init__(
        self,
        tax_optimiser: Optional[TaxOptimiser] = None,
        tax_lot_manager: Optional[TaxLotManager] = None,
    ) -> None:
        self.role = "Chartered Accountant & Quantitative Tax Specialist"
        self.goal = "Minimize realized capital gains tax liabilities, harvest optimal tax losses, and avoid wash-sale penalties."
        self.backstory = (
            "A seasoned Indian tax attorney and quantitative tax strategist specializing in Section 111A/112A capital gains, "
            "STCG/LTCG lot management, and sophisticated tax-loss harvesting with replacement proxy assets."
        )
        self.lot_manager = tax_lot_manager or TaxLotManager()
        self.tax_optimiser = tax_optimiser or TaxOptimiser()

    def as_crewai_agent(self, tools: Optional[List[Any]] = None) -> Optional[Any]:
        if not HAS_CREWAI:
            return None
        return Agent(
            role=self.role,
            goal=self.goal,
            backstory=self.backstory,
            verbose=True,
            tools=tools or [],
            allow_delegation=False,
        )

    def execute_task(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluates candidate trades from PortfolioAnalyst, applies tax optimization, and reroutes wash sales."""
        p_id = context.get("portfolio_id", "PORTFOLIO")
        raw_trades = context.get("candidate_trades", [])
        ytd_exemption = float(context.get("ytd_claimed_exemption", 0.0))
        strategy = context.get("lot_selection_strategy", "TAX_MINIMIZER")
        as_of = context.get("as_of_date") or datetime.date.today()

        existing_lots = context.get("existing_tax_lots", [])
        for lot in existing_lots:
            if isinstance(lot, TaxLot):
                self.lot_manager.add_lot(lot)
            elif isinstance(lot, dict):
                p_date = lot.get("purchase_date")
                if isinstance(p_date, str):
                    p_date = datetime.date.fromisoformat(p_date)
                lot_obj = TaxLot(
                    lot_id=lot.get("lot_id", f"LOT-{p_id}-{int(datetime.datetime.now().timestamp())}"),
                    portfolio_id=p_id,
                    asset_class=lot.get("symbol", lot.get("asset_class", "NIFTY_50_EQUITY")),
                    quantity=float(lot.get("quantity", lot.get("units", 100.0))),
                    cost_per_share=float(lot.get("purchase_price", lot.get("cost_per_share", 100.0))),
                    current_price=float(lot.get("current_price", lot.get("purchase_price", 100.0))),
                    acquisition_date=p_date or datetime.date(2024, 1, 1),
                )
                self.lot_manager.add_lot(lot_obj)

        trades: List[TradeOrder] = []
        for t in raw_trades:
            if isinstance(t, TradeOrder):
                trades.append(t)
            elif isinstance(t, dict):
                qty = float(t.get("target_units", t.get("quantity", t.get("units", 0.0))))
                px = float(t.get("estimated_price", t.get("price", 100.0)))
                val = float(t.get("trade_value_inr", qty * px))
                order_obj = TradeOrder(
                    portfolio_id=t.get("portfolio_id", p_id),
                    asset_class=t.get("asset_class", t.get("symbol", "NIFTY_50_EQUITY")),
                    action=str(t.get("action", "BUY")).upper(),
                    target_units=qty,
                    estimated_price=px,
                    trade_value_inr=val,
                    expected_impact_bps=float(t.get("expected_impact_bps", 0.0)),
                    tax_implication_inr=float(t.get("tax_implication_inr", 0.0)),
                )
                trades.append(order_obj)

        eval_res = self.tax_optimiser.evaluate_trade_plan_taxes(
            portfolio_id=p_id,
            orders=trades,
            lot_manager=self.lot_manager,
            ytd_claimed_exemption=ytd_exemption,
            strategy=strategy,
            as_of=as_of,
            apply_wash_sale_substitutes=True,
        )

        tax_sum = eval_res["tax_summary"]
        gross_stcg = float(tax_sum.get("gross_stcg", 0.0))
        gross_stcl = float(tax_sum.get("gross_stcl", 0.0))
        gross_ltcg = float(tax_sum.get("gross_ltcg", 0.0))
        gross_ltcl = float(tax_sum.get("gross_ltcl", 0.0))
        net_pnl = (gross_stcg - gross_stcl) + (gross_ltcg - gross_ltcl)
        harvest_loss = float(eval_res.get("harvested_losses_inr", gross_stcl + gross_ltcl))

        return {
            "status": "SUCCESS",
            "tax_adjusted_trades": eval_res["orders"],
            "tax_summary": tax_sum,
            "wash_sale_substitutions": eval_res["wash_sale_substitutions"],
            "total_estimated_tax_inr": eval_res["total_estimated_tax_inr"],
            "total_tax_liability_inr": eval_res["total_estimated_tax_inr"],
            "net_realized_pnl_inr": net_pnl,
            "harvest_realized_loss_inr": harvest_loss,
            "harvested_losses_inr": harvest_loss,
            "tax_shield_inr": eval_res["tax_shield_inr"],
        }
