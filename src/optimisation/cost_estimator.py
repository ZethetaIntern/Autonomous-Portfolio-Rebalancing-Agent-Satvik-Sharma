"""Transaction Cost Estimator for WealthPilot AI.

Computes explicit Indian market statutory charges (Brokerage, STT, GST, Stamp Duty,
Exchange Turnover, SEBI fees) and implicit market impact via square-root volatility models.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Union

from src.optimisation import TradeOrder


DEFAULT_ASSET_ADV_INR: Dict[str, float] = {
    "NIFTY_50_EQUITY": 25_000_000_000.0,
    "G_SEC_BONDS": 15_000_000_000.0,
    "CORP_BONDS": 1_000_000_000.0,
    "GOLD_ETF": 500_000_000.0,
    "LIQUID_CASH": 100_000_000_000.0,
}

DEFAULT_ANNUAL_VOLATILITIES: Dict[str, float] = {
    "NIFTY_50_EQUITY": 0.160,
    "G_SEC_BONDS": 0.045,
    "CORP_BONDS": 0.055,
    "GOLD_ETF": 0.140,
    "LIQUID_CASH": 0.005,
}


class CostEstimator:
    """Calculates comprehensive transaction costs including explicit statutory fees and square-root market impact."""

    def __init__(
        self,
        brokerage_rate: float = 0.0005,
        stt_rate_equity_delivery: float = 0.001,
        gst_rate: float = 0.18,
        stamp_duty_buy: float = 0.00015,
        exchange_turnover_rate: float = 0.0000345,
        sebi_turnover_rate: float = 0.000001,
        impact_coefficient: float = 0.60,
        asset_adv_map: Optional[Dict[str, float]] = None,
        annual_vols_map: Optional[Dict[str, float]] = None,
    ) -> None:
        self.brokerage_rate = brokerage_rate
        self.stt_rate = stt_rate_equity_delivery
        self.gst_rate = gst_rate
        self.stamp_duty_buy = stamp_duty_buy
        self.exchange_rate = exchange_turnover_rate
        self.sebi_rate = sebi_turnover_rate
        self.impact_k = impact_coefficient
        self.adv_map = asset_adv_map or DEFAULT_ASSET_ADV_INR.copy()
        self.vols_map = annual_vols_map or DEFAULT_ANNUAL_VOLATILITIES.copy()

    def estimate_trade_costs(
        self,
        order: Union[TradeOrder, Dict[str, Any]],
        custom_adv_inr: Optional[float] = None,
        custom_daily_vol: Optional[float] = None,
    ) -> Dict[str, float]:
        """Calculates explicit and implicit execution costs for a single trade ticket.
        
        Explicit Costs:
            - Brokerage: trade_value * brokerage_rate
            - STT: 0.1% on equity trades
            - Exchange Turnover Charges: trade_value * exchange_rate
            - SEBI Turnover Charges: trade_value * sebi_rate
            - GST: 18% on (brokerage + exchange charges)
            - Stamp Duty: 1.5 bps on BUY orders

        Implicit Costs (Square-Root Market Impact Model):
            Impact_i = k_i * sigma_daily_i * sqrt(V_trade_i / ADV_i)
            Impact_Cost_INR = Impact_i * V_trade_i

        Returns:
            Dict containing detailed cost breakdown in INR and basis points.
        """
        if isinstance(order, TradeOrder):
            asset = order.asset_class
            action = order.action.upper()
            val = float(order.trade_value_inr)
        else:
            asset = str(order.get("asset_class", "NIFTY_50_EQUITY"))
            action = str(order.get("action", "BUY")).upper()
            val = float(order.get("trade_value_inr", order.get("value_inr", 0.0)))

        if val <= 0:
            return {
                "brokerage_inr": 0.0,
                "stt_inr": 0.0,
                "exchange_fee_inr": 0.0,
                "sebi_fee_inr": 0.0,
                "gst_inr": 0.0,
                "stamp_duty_inr": 0.0,
                "total_explicit_inr": 0.0,
                "market_impact_bps": 0.0,
                "market_impact_inr": 0.0,
                "total_cost_inr": 0.0,
                "total_cost_bps": 0.0,
            }

        is_cash = asset == "LIQUID_CASH"
        brokerage = 0.0 if is_cash else val * self.brokerage_rate
        exchange_fee = 0.0 if is_cash else val * self.exchange_rate
        sebi_fee = 0.0 if is_cash else val * self.sebi_rate
        gst = (brokerage + exchange_fee) * self.gst_rate

        is_equity = "EQUITY" in asset.upper() or asset == "NIFTY_50_EQUITY"
        stt = (val * self.stt_rate) if is_equity else 0.0

        stamp_duty = (val * self.stamp_duty_buy) if (action == "BUY" and not is_cash) else 0.0

        total_explicit = brokerage + exchange_fee + sebi_fee + gst + stt + stamp_duty

        if is_cash:
            impact_bps = 0.0
            impact_cost = 0.0
        else:
            adv = custom_adv_inr or self.adv_map.get(asset, 10_000_000_000.0)
            if custom_daily_vol is not None:
                daily_vol = custom_daily_vol
            else:
                ann_vol = self.vols_map.get(asset, 0.15)
                daily_vol = ann_vol / math.sqrt(252.0)

            participation = val / max(adv, 1.0)
            impact_fraction = self.impact_k * daily_vol * math.sqrt(participation)
            impact_bps = impact_fraction * 10000.0
            impact_cost = val * impact_fraction

        total_cost = total_explicit + impact_cost
        cost_bps = (total_cost / val) * 10000.0

        return {
            "brokerage_inr": round(brokerage, 2),
            "stt_inr": round(stt, 2),
            "exchange_fee_inr": round(exchange_fee, 2),
            "sebi_fee_inr": round(sebi_fee, 2),
            "gst_inr": round(gst, 2),
            "stamp_duty_inr": round(stamp_duty, 2),
            "total_explicit_inr": round(total_explicit, 2),
            "market_impact_bps": round(impact_bps, 2),
            "market_impact_inr": round(impact_cost, 2),
            "total_cost_inr": round(total_cost, 2),
            "total_cost_bps": round(cost_bps, 2),
        }

    def estimate_portfolio_trades_cost(
        self,
        trades: List[Union[TradeOrder, Dict[str, Any]]],
        custom_adv: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """Aggregates cost estimates across an entire rebalancing trade list."""
        adv_dict = custom_adv or self.adv_map
        total_val = 0.0
        total_explicit = 0.0
        total_impact = 0.0
        order_breakdowns: List[Dict[str, Any]] = []

        for trade in trades:
            asset = trade.asset_class if isinstance(trade, TradeOrder) else trade.get("asset_class", "")
            cost_info = self.estimate_trade_costs(trade, custom_adv_inr=adv_dict.get(asset))
            trade_val = trade.trade_value_inr if isinstance(trade, TradeOrder) else float(trade.get("trade_value_inr", trade.get("value_inr", 0.0)))

            total_val += trade_val
            total_explicit += cost_info["total_explicit_inr"]
            total_impact += cost_info["market_impact_inr"]

            order_breakdowns.append({
                "asset_class": asset,
                "trade_value_inr": trade_val,
                "cost_details": cost_info,
            })

        total_cost = total_explicit + total_impact
        aggregate_bps = (total_cost / max(total_val, 1.0)) * 10000.0

        return {
            "total_traded_value_inr": round(total_val, 2),
            "total_explicit_cost_inr": round(total_explicit, 2),
            "total_market_impact_inr": round(total_impact, 2),
            "total_cost_inr": round(total_cost, 2),
            "total_cost_bps": round(aggregate_bps, 2),
            "order_count": len(trades),
            "orders": order_breakdowns,
        }
