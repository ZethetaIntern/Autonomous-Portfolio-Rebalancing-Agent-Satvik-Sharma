"""Tax Lot Manager for WealthPilot AI.

Tracks acquisition dates, cost basis, holding periods, and manages lot selection
strategies (FIFO, LIFO, HIFO, Tax-Minimizer) under Indian Income Tax Act rules.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import datetime
from typing import Any, Dict, List, Optional, Tuple


# Default 30-day Wash Sale Avoidance Substitute Asset Mapping
DEFAULT_SUBSTITUTE_ASSET_MAP: Dict[str, str] = {
    "NIFTY_50_EQUITY": "NIFTY_NEXT_50_EQUITY",
    "G_SEC_BONDS": "BHARAT_BOND_ETF",
    "CORP_BONDS": "BANKING_PSU_DEBT_ETF",
    "GOLD_ETF": "SOVEREIGN_GOLD_BOND",
    "LIQUID_CASH": "LIQUID_CASH",
}


@dataclass
class TaxLot:
    """Individual tax lot tracking units, cost basis, and acquisition date."""
    lot_id: str
    portfolio_id: str
    asset_class: str
    quantity: float
    acquisition_date: datetime.date
    cost_per_share: float
    current_price: float = 0.0

    @property
    def total_cost_basis(self) -> float:
        return self.quantity * self.cost_per_share

    @property
    def current_value(self) -> float:
        return self.quantity * self.current_price

    @property
    def unrealized_gain_loss(self) -> float:
        return (self.current_price - self.cost_per_share) * self.quantity

    def holding_period_days(self, as_of: Optional[datetime.date] = None) -> int:
        ref_date = as_of or datetime.date.today()
        return max(0, (ref_date - self.acquisition_date).days)

    def is_ltcg(self, as_of: Optional[datetime.date] = None, equity_threshold_days: int = 365) -> bool:
        """Determines if the lot qualifies for Long-Term Capital Gains (>12 months for equity)."""
        days = self.holding_period_days(as_of)
        # Indian equity holding threshold is 12 months (365 days)
        return days >= equity_threshold_days

    def gain_category(self, as_of: Optional[datetime.date] = None) -> str:
        """Categorizes lot into STCL, LTCL, STCG, or LTCG."""
        ltcg = self.is_ltcg(as_of)
        gain = self.unrealized_gain_loss
        if gain < 0:
            return "LTCL" if ltcg else "STCL"
        else:
            return "LTCG" if ltcg else "STCG"


class TaxLotManager:
    """Manages multi-portfolio tax lots, lot-depletion engines, and wash-sale avoidance windows."""

    def __init__(
        self,
        substitute_map: Optional[Dict[str, str]] = None,
        wash_sale_window_days: int = 30,
        equity_ltcg_days: int = 365,
    ) -> None:
        self.lots: Dict[str, List[TaxLot]] = {}  # portfolio_id -> list of TaxLots
        self.substitute_map = substitute_map or DEFAULT_SUBSTITUTE_ASSET_MAP.copy()
        self.wash_sale_window_days = wash_sale_window_days
        self.equity_ltcg_days = equity_ltcg_days
        # History of loss harvests: portfolio_id -> List[(asset_class, harvest_date, loss_inr)]
        self.harvest_history: Dict[str, List[Tuple[str, datetime.date, float]]] = {}

    def add_lot(self, lot: TaxLot) -> None:
        """Registers a new tax lot for a portfolio."""
        if lot.portfolio_id not in self.lots:
            self.lots[lot.portfolio_id] = []
        self.lots[lot.portfolio_id].append(lot)

    def add_lots(self, lots: List[TaxLot]) -> None:
        """Batch registers tax lots."""
        for lot in lots:
            self.add_lot(lot)

    def get_lots(self, portfolio_id: str, asset_class: Optional[str] = None) -> List[TaxLot]:
        """Retrieves active tax lots for a portfolio, optionally filtered by asset."""
        p_lots = self.lots.get(portfolio_id, [])
        if asset_class:
            return [l for l in p_lots if l.asset_class == asset_class and l.quantity > 1e-6]
        return [l for l in p_lots if l.quantity > 1e-6]

    def update_prices(self, price_map: Dict[str, float]) -> None:
        """Updates current market price across all active lots."""
        for p_lots in self.lots.values():
            for lot in p_lots:
                if lot.asset_class in price_map:
                    lot.current_price = price_map[lot.asset_class]

    def get_unrealized_summary(
        self,
        portfolio_id: str,
        as_of: Optional[datetime.date] = None,
    ) -> Dict[str, float]:
        """Calculates total unrealized STCG, STCL, LTCG, and LTCL for a portfolio."""
        ref_date = as_of or datetime.date.today()
        p_lots = self.get_lots(portfolio_id)

        stcg = 0.0
        stcl = 0.0
        ltcg = 0.0
        ltcl = 0.0
        total_value = 0.0
        total_cost = 0.0

        for lot in p_lots:
            cat = lot.gain_category(ref_date)
            gain = lot.unrealized_gain_loss
            total_value += lot.current_value
            total_cost += lot.total_cost_basis

            if cat == "STCG":
                stcg += gain
            elif cat == "STCL":
                stcl += abs(gain)
            elif cat == "LTCG":
                ltcg += gain
            elif cat == "LTCL":
                ltcl += abs(gain)

        return {
            "unrealized_stcg_inr": round(stcg, 2),
            "unrealized_stcl_inr": round(stcl, 2),
            "unrealized_ltcg_inr": round(ltcg, 2),
            "unrealized_ltcl_inr": round(ltcl, 2),
            "net_unrealized_inr": round(stcg - stcl + ltcg - ltcl, 2),
            "total_portfolio_value_inr": round(total_value, 2),
            "total_cost_basis_inr": round(total_cost, 2),
            "lot_count": len(p_lots),
        }

    def select_lots_for_sale(
        self,
        portfolio_id: str,
        asset_class: str,
        units_to_sell: float,
        strategy: str = "TAX_MINIMIZER",
        as_of: Optional[datetime.date] = None,
    ) -> List[Tuple[TaxLot, float]]:
        """Selects and allocates units from specific tax lots based on the chosen strategy.
        
        Strategies:
            - "FIFO": First acquired lots sold first.
            - "LIFO": Most recently acquired lots sold first.
            - "HIFO": Highest cost per share sold first (minimizes immediate capital gains).
            - "TAX_MINIMIZER":
                1. STCL lots (highest per-share loss first) to harvest valuable short-term losses.
                2. LTCL lots (highest per-share loss next).
                3. LTCG lots (lowest per-share gain first, taxed at favorable 12.5% rate).
                4. STCG lots (lowest per-share gain first, deferred to avoid 20% tax).

        Returns:
            List of (TaxLot, units_to_take_from_lot).
        """
        ref_date = as_of or datetime.date.today()
        candidate_lots = [l for l in self.get_lots(portfolio_id, asset_class) if l.quantity > 1e-6]

        if not candidate_lots:
            return []

        # Sort lots by strategy
        strategy_upper = strategy.upper()
        if strategy_upper == "FIFO":
            candidate_lots.sort(key=lambda l: l.acquisition_date)
        elif strategy_upper == "LIFO":
            candidate_lots.sort(key=lambda l: l.acquisition_date, reverse=True)
        elif strategy_upper == "HIFO":
            candidate_lots.sort(key=lambda l: l.cost_per_share, reverse=True)
        elif strategy_upper == "TAX_MINIMIZER":
            def sort_key(lot: TaxLot) -> Tuple[int, float]:
                cat = lot.gain_category(ref_date)
                per_share_gain = lot.current_price - lot.cost_per_share
                # Priority rank:
                # 0: STCL (sort by most negative per-share gain)
                # 1: LTCL (sort by most negative per-share gain)
                # 2: LTCG (sort by smallest positive per-share gain)
                # 3: STCG (sort by smallest positive per-share gain)
                if cat == "STCL":
                    return (0, per_share_gain)
                elif cat == "LTCL":
                    return (1, per_share_gain)
                elif cat == "LTCG":
                    return (2, per_share_gain)
                else:
                    return (3, per_share_gain)

            candidate_lots.sort(key=sort_key)
        else:
            raise ValueError(f"Unknown lot selection strategy: {strategy}")

        allocations: List[Tuple[TaxLot, float]] = []
        remaining_units = float(units_to_sell)

        for lot in candidate_lots:
            if remaining_units <= 1e-6:
                break
            take = min(lot.quantity, remaining_units)
            allocations.append((lot, take))
            remaining_units -= take

        return allocations

    def execute_sale(
        self,
        portfolio_id: str,
        asset_class: str,
        units_to_sell: float,
        strategy: str = "TAX_MINIMIZER",
        sale_price: Optional[float] = None,
        as_of: Optional[datetime.date] = None,
    ) -> Dict[str, Any]:
        """Depletes allocated lots from the portfolio and calculates exact realized tax gains/losses."""
        ref_date = as_of or datetime.date.today()
        allocations = self.select_lots_for_sale(
            portfolio_id=portfolio_id,
            asset_class=asset_class,
            units_to_sell=units_to_sell,
            strategy=strategy,
            as_of=ref_date,
        )

        realized_stcg = 0.0
        realized_stcl = 0.0
        realized_ltcg = 0.0
        realized_ltcl = 0.0
        total_proceeds = 0.0
        total_cost = 0.0
        lots_depleted_info: List[Dict[str, Any]] = []

        for lot, units in allocations:
            exec_price = sale_price if sale_price is not None else lot.current_price
            lot_cost = units * lot.cost_per_share
            lot_proceeds = units * exec_price
            gain_or_loss = lot_proceeds - lot_cost
            is_lt = lot.is_ltcg(ref_date, self.equity_ltcg_days)

            total_proceeds += lot_proceeds
            total_cost += lot_cost

            if gain_or_loss >= 0:
                if is_lt:
                    realized_ltcg += gain_or_loss
                else:
                    realized_stcg += gain_or_loss
            else:
                loss_val = abs(gain_or_loss)
                if is_lt:
                    realized_ltcl += loss_val
                else:
                    realized_stcl += loss_val

            # Record loss harvest if loss realized
            if gain_or_loss < 0:
                self.record_loss_harvest(portfolio_id, asset_class, ref_date, abs(gain_or_loss))

            # Reduce quantity in lot
            lot.quantity -= units

            lots_depleted_info.append({
                "lot_id": lot.lot_id,
                "units_sold": units,
                "cost_per_share": lot.cost_per_share,
                "exec_price": exec_price,
                "gain_loss_inr": round(gain_or_loss, 2),
                "is_ltcg": is_lt,
                "remaining_lot_units": max(0.0, lot.quantity),
            })

        # Remove exhausted lots
        if portfolio_id in self.lots:
            self.lots[portfolio_id] = [l for l in self.lots[portfolio_id] if l.quantity > 1e-6]

        return {
            "portfolio_id": portfolio_id,
            "asset_class": asset_class,
            "units_sold": sum(u for _, u in allocations),
            "total_proceeds_inr": round(total_proceeds, 2),
            "total_cost_basis_inr": round(total_cost, 2),
            "realized_stcg_inr": round(realized_stcg, 2),
            "realized_stcl_inr": round(realized_stcl, 2),
            "realized_ltcg_inr": round(realized_ltcg, 2),
            "realized_ltcl_inr": round(realized_ltcl, 2),
            "net_capital_gain_inr": round((realized_stcg - realized_stcl) + (realized_ltcg - realized_ltcl), 2),
            "lots_depleted": lots_depleted_info,
        }

    def record_loss_harvest(
        self,
        portfolio_id: str,
        asset_class: str,
        harvest_date: datetime.date,
        loss_inr: float,
    ) -> None:
        """Records a tax-loss harvesting event to track wash-sale avoidance windows."""
        if portfolio_id not in self.harvest_history:
            self.harvest_history[portfolio_id] = []
        self.harvest_history[portfolio_id].append((asset_class, harvest_date, float(loss_inr)))

    def is_in_wash_sale_window(
        self,
        portfolio_id: str,
        asset_class: str,
        check_date: Optional[datetime.date] = None,
    ) -> Tuple[bool, Optional[datetime.date]]:
        """Verifies if purchasing asset_class would violate the 30-day wash-sale avoidance corridor.
        
        Returns:
            (is_blocked, date_when_window_clears).
        """
        ref_date = check_date or datetime.date.today()
        history = self.harvest_history.get(portfolio_id, [])

        latest_harvest_date: Optional[datetime.date] = None
        for a, h_date, _ in history:
            if a == asset_class:
                days_since = (ref_date - h_date).days
                if 0 <= days_since < self.wash_sale_window_days:
                    if latest_harvest_date is None or h_date > latest_harvest_date:
                        latest_harvest_date = h_date

        if latest_harvest_date is not None:
            clear_date = latest_harvest_date + datetime.timedelta(days=self.wash_sale_window_days)
            return True, clear_date
        return False, None

    def get_substitute_asset(self, asset_class: str) -> str:
        """Returns the substitute ETF / asset to maintain beta exposure during wash sale window."""
        return self.substitute_map.get(asset_class, asset_class)
