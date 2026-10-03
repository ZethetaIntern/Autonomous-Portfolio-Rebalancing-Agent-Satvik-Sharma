"""Trade List Generator converting continuous weights into discrete executable order tickets.

Implements joint round-lot optimization across multi-asset order lines ensuring cash
conservation, lot-size divisibility, and zero naked-shorting.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from src.optimisation import TradeOrder


DEFAULT_PRICES: Dict[str, float] = {
    "NIFTY_50_EQUITY": 24000.0,
    "G_SEC_BONDS": 100.0,
    "CORP_BONDS": 1000.0,
    "GOLD_ETF": 75.0,
    "LIQUID_CASH": 1.0,
}

DEFAULT_LOT_SIZES: Dict[str, int] = {
    "NIFTY_50_EQUITY": 1,
    "G_SEC_BONDS": 10,
    "CORP_BONDS": 5,
    "GOLD_ETF": 1,
    "LIQUID_CASH": 1,
}


class TradeListGenerator:
    """Translates continuous optimizer weights into discrete executable orders with joint round-lot optimization."""

    def __init__(
        self,
        min_order_value_inr: float = 1000.0,
        default_prices: Optional[Dict[str, float]] = None,
        default_lot_sizes: Optional[Dict[str, int]] = None,
        cash_asset_key: str = "LIQUID_CASH",
    ) -> None:
        self.min_order_value_inr = float(min_order_value_inr)
        self.prices = default_prices or DEFAULT_PRICES.copy()
        self.lot_sizes = default_lot_sizes or DEFAULT_LOT_SIZES.copy()
        self.cash_asset_key = cash_asset_key

    def generate_orders(
        self,
        portfolio_id: str,
        current_weights: Dict[str, float],
        optimal_weights: Dict[str, float],
        portfolio_aum: float,
        asset_prices: Optional[Dict[str, float]] = None,
        lot_sizes: Optional[Dict[str, int]] = None,
        min_cash_buffer_pct: float = 0.02,
    ) -> List[TradeOrder]:
        """Convenience method returning list of TradeOrder objects."""
        plan = self.generate_trade_plan(
            portfolio_id=portfolio_id,
            current_weights=current_weights,
            optimal_weights=optimal_weights,
            portfolio_aum=portfolio_aum,
            asset_prices=asset_prices,
            lot_sizes=lot_sizes,
            min_cash_buffer_pct=min_cash_buffer_pct,
        )
        return plan["orders"]

    def generate_trade_plan(
        self,
        portfolio_id: str,
        current_weights: Dict[str, float],
        optimal_weights: Dict[str, float],
        portfolio_aum: float,
        asset_prices: Optional[Dict[str, float]] = None,
        lot_sizes: Optional[Dict[str, int]] = None,
        min_cash_buffer_pct: float = 0.02,
    ) -> Dict[str, Any]:
        """Joint round-lot optimization generating executable order tickets and cash settlement ledger.
        
        Args:
            portfolio_id: Unique identifier of portfolio.
            current_weights: Pre-rebalance asset weights.
            optimal_weights: Target continuous weights from PortfolioOptimiser.
            portfolio_aum: Total portfolio AUM in INR.
            asset_prices: Per-unit market prices (defaults to calibrated Indian prices).
            lot_sizes: Exchange lot multiples (e.g. 1 for equity, 10 for debt).
            min_cash_buffer_pct: Minimum fractional liquidity buffer retained in cash.

        Returns:
            Dict containing:
                - orders: List[TradeOrder]
                - discrete_weights: Dict[str, float]
                - cash_ledger: Dict of pre_cash, sell_proceeds, buy_spend, post_cash
                - rounding_error_inr: Aggregate rupee rounding residual
        """
        prices = {**self.prices, **(asset_prices or {})}
        lots = {**self.lot_sizes, **(lot_sizes or {})}
        all_assets = list(set(list(current_weights.keys()) + list(optimal_weights.keys())))

        non_cash_assets = [a for a in all_assets if a != self.cash_asset_key]

        current_cash_inr = float(current_weights.get(self.cash_asset_key, 0.0) * portfolio_aum)
        target_cash_inr = float(optimal_weights.get(self.cash_asset_key, min_cash_buffer_pct) * portfolio_aum)
        min_cash_buffer_inr = float(min_cash_buffer_pct * portfolio_aum)

        target_deltas: Dict[str, float] = {}
        for a in non_cash_assets:
            curr_w = current_weights.get(a, 0.0)
            opt_w = optimal_weights.get(a, 0.0)
            target_deltas[a] = (opt_w - curr_w) * portfolio_aum

        sell_orders: List[Dict[str, Any]] = []
        realized_sell_proceeds = 0.0

        for a in non_cash_assets:
            delta_val = target_deltas[a]
            if delta_val >= 0:
                continue

            needed_sell_val = abs(delta_val)
            if needed_sell_val < self.min_order_value_inr:
                continue

            price = prices.get(a, 100.0)
            lot = lots.get(a, 1)

            max_shares_held = (current_weights.get(a, 0.0) * portfolio_aum) / price
            raw_sell_shares = needed_sell_val / price

            rounded_lots = round(raw_sell_shares / lot)
            sell_shares = rounded_lots * lot

            if sell_shares * price > (max_shares_held * price + 1.0):
                sell_shares = math.floor(max_shares_held / lot) * lot

            if sell_shares <= 0:
                continue

            trade_val = sell_shares * price
            realized_sell_proceeds += trade_val

            sell_orders.append({
                "asset_class": a,
                "action": "SELL",
                "units": float(sell_shares),
                "price": price,
                "trade_value": trade_val,
                "lot_size": lot,
            })

        net_spendable_cash = max(0.0, current_cash_inr + realized_sell_proceeds - min_cash_buffer_inr)

        buy_candidates: List[Dict[str, Any]] = []
        for a in non_cash_assets:
            delta_val = target_deltas[a]
            if delta_val <= 0:
                continue

            if delta_val < self.min_order_value_inr:
                continue

            price = prices.get(a, 100.0)
            lot = lots.get(a, 1)
            raw_buy_shares = delta_val / price
            base_lots = math.floor(raw_buy_shares / lot)
            base_shares = base_lots * lot
            base_cost = base_shares * price

            buy_candidates.append({
                "asset_class": a,
                "price": price,
                "lot_size": lot,
                "lot_cost": lot * price,
                "target_val": delta_val,
                "allocated_lots": base_lots,
                "allocated_shares": base_shares,
                "allocated_cost": base_cost,
            })

        total_base_cost = sum(c["allocated_cost"] for c in buy_candidates)
        remaining_cash = net_spendable_cash - total_base_cost

        if remaining_cash < 0:
            buy_candidates.sort(key=lambda c: (c["allocated_cost"] - c["target_val"]), reverse=True)
            for c in buy_candidates:
                while remaining_cash < 0 and c["allocated_lots"] > 0:
                    c["allocated_lots"] -= 1
                    c["allocated_shares"] = c["allocated_lots"] * c["lot_size"]
                    c["allocated_cost"] = c["allocated_shares"] * c["price"]
                    remaining_cash += c["lot_cost"]

        if remaining_cash > 0:
            improved = True
            while improved:
                improved = False
                best_idx = -1
                best_gain = -1e9

                for idx, c in enumerate(buy_candidates):
                    if c["lot_cost"] <= remaining_cash:
                        current_cost = c["allocated_cost"]
                        target_v = c["target_val"]
                        current_err_sq = (target_v - current_cost) ** 2
                        new_err_sq = (target_v - (current_cost + c["lot_cost"])) ** 2
                        gain = current_err_sq - new_err_sq

                        if gain > 0 and gain > best_gain:
                            best_gain = gain
                            best_idx = idx

                if best_idx >= 0 and best_gain > 0:
                    cand = buy_candidates[best_idx]
                    cand["allocated_lots"] += 1
                    cand["allocated_shares"] = cand["allocated_lots"] * cand["lot_size"]
                    cand["allocated_cost"] = cand["allocated_shares"] * cand["price"]
                    remaining_cash -= cand["lot_cost"]
                    improved = True

        buy_orders: List[Dict[str, Any]] = []
        total_buy_spend = 0.0
        for c in buy_candidates:
            if c["allocated_shares"] > 0 and c["allocated_cost"] >= self.min_order_value_inr:
                buy_orders.append({
                    "asset_class": c["asset_class"],
                    "action": "BUY",
                    "units": float(c["allocated_shares"]),
                    "price": c["price"],
                    "trade_value": c["allocated_cost"],
                    "lot_size": c["lot_size"],
                })
                total_buy_spend += c["allocated_cost"]

        orders: List[TradeOrder] = []
        for s in sell_orders:
            orders.append(
                TradeOrder(
                    portfolio_id=portfolio_id,
                    asset_class=s["asset_class"],
                    action="SELL",
                    target_units=s["units"],
                    estimated_price=s["price"],
                    trade_value_inr=round(s["trade_value"], 2),
                    expected_impact_bps=0.0,
                    tax_implication_inr=0.0,
                )
            )

        for b in buy_orders:
            orders.append(
                TradeOrder(
                    portfolio_id=portfolio_id,
                    asset_class=b["asset_class"],
                    action="BUY",
                    target_units=b["units"],
                    estimated_price=b["price"],
                    trade_value_inr=round(b["trade_value"], 2),
                    expected_impact_bps=0.0,
                    tax_implication_inr=0.0,
                )
            )

        post_cash_inr = current_cash_inr + realized_sell_proceeds - total_buy_spend
        discrete_values: Dict[str, float] = {self.cash_asset_key: post_cash_inr}

        for a in non_cash_assets:
            curr_units = (current_weights.get(a, 0.0) * portfolio_aum) / prices.get(a, 100.0)
            trade_units = 0.0
            for o in orders:
                if o.asset_class == a:
                    trade_units += o.target_units if o.action == "BUY" else -o.target_units
            post_units = max(0.0, curr_units + trade_units)
            discrete_values[a] = post_units * prices.get(a, 100.0)

        total_post_val = sum(discrete_values.values())
        discrete_weights = {
            k: round(v / total_post_val, 6) for k, v in discrete_values.items()
        }

        rounding_error_inr = sum(
            abs(discrete_values[a] - (optimal_weights.get(a, 0.0) * portfolio_aum))
            for a in all_assets
        )

        return {
            "portfolio_id": portfolio_id,
            "orders": orders,
            "discrete_weights": discrete_weights,
            "cash_ledger": {
                "initial_cash_inr": round(current_cash_inr, 2),
                "sell_proceeds_inr": round(realized_sell_proceeds, 2),
                "buy_spend_inr": round(total_buy_spend, 2),
                "post_cash_inr": round(post_cash_inr, 2),
                "post_cash_pct": round(post_cash_inr / total_post_val, 4),
                "min_required_buffer_inr": round(min_cash_buffer_inr, 2),
            },
            "rounding_error_inr": round(rounding_error_inr, 2),
            "order_count": len(orders),
        }
