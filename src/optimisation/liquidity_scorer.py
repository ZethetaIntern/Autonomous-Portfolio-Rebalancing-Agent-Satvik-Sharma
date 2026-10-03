"""Liquidity Scorer and Algorithmic Execution Scheduler for WealthPilot AI.

Evaluates security liquidity based on Average Daily Volume (ADV), bid-ask spreads,
and generates multi-day TWAP/VWAP execution schedules for positions exceeding ADV safety caps.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Dict, List, Optional, Union

from src.optimisation import TradeOrder


DEFAULT_LIQUIDITY_PROFILES: Dict[str, Dict[str, float]] = {
    "NIFTY_50_EQUITY": {
        "adv_inr": 25_000_000_000.0,
        "bid_ask_spread_bps": 2.5,
        "max_participation_rate": 0.10,
    },
    "G_SEC_BONDS": {
        "adv_inr": 15_000_000_000.0,
        "bid_ask_spread_bps": 1.5,
        "max_participation_rate": 0.15,
    },
    "CORP_BONDS": {
        "adv_inr": 500_000_000.0,
        "bid_ask_spread_bps": 25.0,
        "max_participation_rate": 0.05,
    },
    "GOLD_ETF": {
        "adv_inr": 500_000_000.0,
        "bid_ask_spread_bps": 8.0,
        "max_participation_rate": 0.08,
    },
    "LIQUID_CASH": {
        "adv_inr": 100_000_000_000.0,
        "bid_ask_spread_bps": 0.1,
        "max_participation_rate": 0.50,
    },
}


@dataclass
class ExecutionSlice:
    """Intraday or daily execution slice for TWAP/VWAP algorithms."""
    day_number: int
    slice_date_offset: int
    target_units: float
    target_value_inr: float
    algorithm: str
    time_window: str
    expected_impact_bps: float


@dataclass
class ExecutionSchedule:
    """Multi-day execution plan for large or illiquid order tickets."""
    portfolio_id: str
    asset_class: str
    action: str
    total_units: float
    total_value_inr: float
    duration_days: int
    algorithm: str
    daily_participation_pct: float
    slices: List[ExecutionSlice] = field(default_factory=list)
    is_multi_day: bool = False
    liquidity_tier: str = "TIER_1"
    summary_notes: str = ""


class LiquidityScorer:
    """Scores security and portfolio liquidity and compiles execution schedules for large trades."""

    def __init__(
        self,
        profiles: Optional[Dict[str, Dict[str, float]]] = None,
        default_max_participation: float = 0.10,
    ) -> None:
        self.profiles = profiles or DEFAULT_LIQUIDITY_PROFILES.copy()
        self.default_max_participation = default_max_participation

    def score_security(
        self,
        asset_class: str,
        adv_inr: Optional[float] = None,
        bid_ask_spread_bps: Optional[float] = None,
        trade_value_inr: float = 0.0,
    ) -> Dict[str, Any]:
        """Calculates a composite liquidity score from 0.0 (illiquid) to 1.0 (ultra-liquid).
        
        Score components:
            - ADV depth score: logarithmic scale of ADV relative to benchmark
            - Spread penalty: tight spread (< 5 bps) gets full score, wide spread (> 50 bps) penalized
            - Participation impact: penalty if trade exceeds safety threshold
        """
        profile = self.profiles.get(asset_class, {})
        adv = adv_inr if adv_inr is not None else profile.get("adv_inr", 1_000_000_000.0)
        spread = bid_ask_spread_bps if bid_ask_spread_bps is not None else profile.get("bid_ask_spread_bps", 10.0)

        log_adv = math.log10(max(adv, 1.0))
        adv_score = max(0.0, min(1.0, (log_adv - 6.0) / 4.0))

        spread_score = max(0.0, min(1.0, 1.0 - (spread / 50.0)))

        participation = (trade_value_inr / max(adv, 1.0)) if trade_value_inr > 0 else 0.0
        participation_penalty = max(0.0, min(0.3, participation * 2.0))

        raw_score = (0.60 * adv_score) + (0.40 * spread_score) - participation_penalty
        final_score = max(0.05, min(1.0, raw_score))

        if final_score >= 0.75:
            tier = "TIER_1_HIGH_LIQUIDITY"
        elif final_score >= 0.45:
            tier = "TIER_2_MODERATE_LIQUIDITY"
        else:
            tier = "TIER_3_ILLIQUID_CONSTRAINED"

        return {
            "asset_class": asset_class,
            "liquidity_score": round(final_score, 4),
            "liquidity_tier": tier,
            "adv_inr": adv,
            "spread_bps": spread,
            "participation_rate": round(participation, 6),
            "adv_score": round(adv_score, 4),
            "spread_score": round(spread_score, 4),
        }

    def score_portfolio_liquidity(self, weights: Dict[str, float]) -> float:
        """Computes weighted-average portfolio liquidity score."""
        weighted_score = 0.0
        for asset, w in weights.items():
            sec_info = self.score_security(asset)
            weighted_score += float(w) * sec_info["liquidity_score"]
        return round(float(weighted_score), 4)

    def generate_execution_schedule(
        self,
        order: Union[TradeOrder, Dict[str, Any]],
        max_daily_participation_pct: Optional[float] = None,
        algorithm: Optional[str] = None,
    ) -> ExecutionSchedule:
        """Determines execution duration and daily TWAP/VWAP scheduling for an order ticket.
        
        Safety rule:
            If trade_value_inr > max_participation * ADV:
                duration_days = ceil(trade_value_inr / (max_participation * ADV))
                Schedule multi-day TWAP/VWAP slices.
            Else:
                duration_days = 1 (Immediate or single-day execution).
        """
        if isinstance(order, TradeOrder):
            p_id = order.portfolio_id
            asset = order.asset_class
            action = order.action.upper()
            units = float(order.target_units)
            val = float(order.trade_value_inr)
            price = float(order.estimated_price)
        else:
            p_id = str(order.get("portfolio_id", "PORTFOLIO"))
            asset = str(order.get("asset_class", "NIFTY_50_EQUITY"))
            action = str(order.get("action", "BUY")).upper()
            units = float(order.get("target_units", order.get("units", 0.0)))
            val = float(order.get("trade_value_inr", order.get("value_inr", 0.0)))
            price = float(order.get("estimated_price", order.get("price", 100.0)))

        profile = self.profiles.get(asset, {})
        adv = profile.get("adv_inr", 1_000_000_000.0)
        max_part = max_daily_participation_pct or profile.get("max_participation_rate", self.default_max_participation)

        max_daily_volume_inr = adv * max_part

        if val <= max_daily_volume_inr or asset == "LIQUID_CASH":
            duration_days = 1
            is_multi_day = False
            chosen_algo = algorithm or "IMMEDIATE"
            slices = [
                ExecutionSlice(
                    day_number=1,
                    slice_date_offset=0,
                    target_units=units,
                    target_value_inr=val,
                    algorithm=chosen_algo,
                    time_window="09:15-15:30 IST",
                    expected_impact_bps=order.expected_impact_bps if isinstance(order, TradeOrder) else 0.0,
                )
            ]
            notes = f"Standard 1-day execution. Participation rate {val / max(adv, 1.0):.2%} is within safety limit {max_part:.1%}."
        else:
            duration_days = int(math.ceil(val / max_daily_volume_inr))
            is_multi_day = True
            chosen_algo = algorithm or ("VWAP" if "EQUITY" in asset else "TWAP")

            daily_units = units / duration_days
            daily_val = val / duration_days

            slices = []
            for d in range(1, duration_days + 1):
                slices.append(
                    ExecutionSlice(
                        day_number=d,
                        slice_date_offset=d - 1,
                        target_units=round(daily_units, 4),
                        target_value_inr=round(daily_val, 2),
                        algorithm=chosen_algo,
                        time_window="09:30-15:00 IST",
                        expected_impact_bps=round(5.0 / math.sqrt(duration_days), 2),
                    )
                )
            notes = (
                f"Multi-day algorithmic execution required! Trade value INR {val:,.0f} exceeds safety threshold "
                f"({max_part:.1%} of ADV = INR {max_daily_volume_inr:,.0f}). Scheduled over {duration_days} trading days via {chosen_algo}."
            )

        sec_score = self.score_security(asset, adv_inr=adv, trade_value_inr=val)

        return ExecutionSchedule(
            portfolio_id=p_id,
            asset_class=asset,
            action=action,
            total_units=units,
            total_value_inr=val,
            duration_days=duration_days,
            algorithm=chosen_algo,
            daily_participation_pct=round((val / duration_days) / max(adv, 1.0), 4),
            slices=slices,
            is_multi_day=is_multi_day,
            liquidity_tier=sec_score["liquidity_tier"],
            summary_notes=notes,
        )

    def schedule_trade_list(
        self,
        trades: List[Union[TradeOrder, Dict[str, Any]]],
    ) -> List[ExecutionSchedule]:
        """Processes an entire trade list and generates schedules for every order line."""
        return [self.generate_execution_schedule(trade) for trade in trades]
