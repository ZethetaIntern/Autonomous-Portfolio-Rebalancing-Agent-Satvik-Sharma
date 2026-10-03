"""Drift Monitoring Orchestrator and Priority Queue Manager for WealthPilot AI.

Executes vectorized scans across 50,000+ portfolios and maintains a priority queue
(`heapq`) for rebalancing intervention.
"""

from __future__ import annotations

import heapq
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from src.monitoring.drift_calculator import DriftCalculator, DriftMetricsBatch
from src.monitoring.threshold_manager import ThresholdManager


@dataclass(order=True)
class FlaggedPortfolioItem:
    """Item stored in the priority queue.

    Ordered by negative priority_score so highest priority is popped first.
    """
    priority_score: float             # Negative score for min-heap implementation of max-heap
    portfolio_id: str = field(compare=False)
    client_id: str = field(compare=False)
    risk_category: str = field(compare=False)
    aum_inr: float = field(compare=False)
    max_drift: float = field(compare=False)
    threshold: float = field(compare=False)
    drift_ratio: float = field(compare=False)
    tracking_error: float = field(compare=False)
    sad: float = field(compare=False)
    days_since_rebalance: int = field(compare=False)
    urgency_tier: str = field(compare=False)
    breaching_assets: List[str] = field(default_factory=list, compare=False)

    @property
    def raw_priority(self) -> float:
        """Returns positive priority score."""
        return -self.priority_score


class DriftMonitor:
    """Scans portfolio universe, identifies drift breaches, and maintains rebalancing queue."""

    def __init__(
        self,
        drift_calculator: Optional[DriftCalculator] = None,
        threshold_manager: Optional[ThresholdManager] = None,
        asset_names: Optional[List[str]] = None,
    ) -> None:
        self.calculator = drift_calculator or DriftCalculator(asset_names=asset_names)
        self.threshold_mgr = threshold_manager or ThresholdManager()
        self.asset_names = asset_names or [
            "NIFTY_50_EQUITY",
            "G_SEC_BONDS",
            "CORP_BONDS",
            "GOLD_ETF",
            "LIQUID_CASH",
        ]
        self._queue: List[FlaggedPortfolioItem] = []
        self._last_scan_summary: Dict[str, Any] = {}

    def calculate_priority_score(
        self,
        max_drift: float,
        threshold: float,
        aum_inr: float,
        tracking_error: float,
        days_since_rebalance: int = 30,
        urgency_multiplier: float = 1.0,
    ) -> float:
        """Compute composite priority score. Higher score = higher rebalance priority.

        Components:
        1. Breach severity ratio: (max_drift / threshold) -> weight 40%
        2. AUM log scale: log10(AUM / 100,000) -> weight 25%
        3. Tracking Error (annualized) -> weight 20%
        4. Stale allocation factor: days / 365 -> weight 15%
        """
        severity_ratio = max_drift / max(threshold, 0.001)
        aum_factor = np.log10(max(aum_inr, 100_000.0) / 100_000.0)
        te_factor = tracking_error * 10.0
        time_factor = min(2.0, days_since_rebalance / 180.0)

        raw_score = (
            (0.40 * severity_ratio)
            + (0.25 * aum_factor)
            + (0.20 * te_factor)
            + (0.15 * time_factor)
        ) * urgency_multiplier

        return float(raw_score)

    def scan_universe(
        self,
        portfolios_df: pd.DataFrame,
        clear_existing_queue: bool = True,
    ) -> Dict[str, Any]:
        """Scan entire portfolio universe with vectorized execution and enqueue breaches.

        SLA Target: Scan 50,000 portfolios in < 30 seconds.

        Args:
            portfolios_df: DataFrame containing portfolio metadata, current and target weights.
            clear_existing_queue: Whether to reset priority queue before scanning.

        Returns:
            Dictionary with scan execution metrics, breach statistics, and queue depth.
        """
        scan_start = time.perf_counter()

        if clear_existing_queue:
            self._queue = []

        n_portfolios = len(portfolios_df)
        target_cols = [f"target_weight_{ac}" for ac in self.asset_names]
        current_cols = [f"current_weight_{ac}" for ac in self.asset_names]

        # Extract matrices
        curr_mat = portfolios_df[current_cols].to_numpy(dtype=np.float64)
        targ_mat = portfolios_df[target_cols].to_numpy(dtype=np.float64)

        # 1. Vectorized Drift Calculation
        batch_metrics = self.calculator.calculate_vectorized(curr_mat, targ_mat)

        # 2. Vectorized Effective Thresholds
        risk_cats = portfolios_df["risk_category"].to_numpy()
        client_ids = portfolios_df["client_id"].to_numpy()
        custom_deltas = (
            portfolios_df["custom_band_delta"].to_numpy(dtype=np.float64)
            if "custom_band_delta" in portfolios_df
            else None
        )

        effective_thresholds = self.threshold_mgr.get_effective_thresholds_vectorized(
            risk_categories=risk_cats,
            client_ids=client_ids,
            custom_deltas=custom_deltas,
        )

        # 3. Identify Breaches
        breach_mask = batch_metrics.max_absolute_drift > effective_thresholds
        breach_indices = np.where(breach_mask)[0]

        aum_arr = portfolios_df["aum_inr"].to_numpy(dtype=np.float64)
        portfolio_id_arr = portfolios_df["portfolio_id"].to_numpy()
        days_arr = (
            portfolios_df["days_since_rebalance"].to_numpy()
            if "days_since_rebalance" in portfolios_df
            else np.full(n_portfolios, 30)
        )

        # 4. Populate Priority Queue for breached portfolios
        for idx in breach_indices:
            p_id = portfolio_id_arr[idx]
            c_id = client_ids[idx]
            r_cat = risk_cats[idx]
            aum = aum_arr[idx]
            max_d = float(batch_metrics.max_absolute_drift[idx])
            thresh = float(effective_thresholds[idx])
            ratio = max_d / thresh
            te = float(batch_metrics.tracking_error[idx])
            sad = float(batch_metrics.sum_absolute_drift[idx])
            days = int(days_arr[idx])

            # Determine breaching assets
            breaching_assets = [
                self.asset_names[j]
                for j in range(len(self.asset_names))
                if batch_metrics.absolute_drift[idx, j] > thresh
            ]

            score = self.calculate_priority_score(
                max_drift=max_d,
                threshold=thresh,
                aum_inr=aum,
                tracking_error=te,
                days_since_rebalance=days,
            )

            urgency = "CRITICAL" if ratio >= 2.0 else ("HIGH" if ratio >= 1.5 else "MEDIUM")

            item = FlaggedPortfolioItem(
                priority_score=-score,  # negative for max-heap
                portfolio_id=str(p_id),
                client_id=str(c_id),
                risk_category=str(r_cat),
                aum_inr=aum,
                max_drift=max_d,
                threshold=thresh,
                drift_ratio=ratio,
                tracking_error=te,
                sad=sad,
                days_since_rebalance=days,
                urgency_tier=urgency,
                breaching_assets=breaching_assets,
            )
            heapq.heappush(self._queue, item)

        total_elapsed = time.perf_counter() - scan_start

        self._last_scan_summary = {
            "total_portfolios_scanned": n_portfolios,
            "breached_portfolios_count": len(breach_indices),
            "breach_rate_pct": round(len(breach_indices) / max(1, n_portfolios) * 100, 2),
            "drift_calc_time_seconds": round(batch_metrics.computation_time_seconds, 4),
            "total_scan_time_seconds": round(total_elapsed, 4),
            "sla_target_seconds": 30.0,
            "sla_met": total_elapsed < 30.0,
            "queue_depth": len(self._queue),
        }

        return self._last_scan_summary

    def pop_highest_priority(self) -> Optional[FlaggedPortfolioItem]:
        """Pop the highest priority flagged portfolio from the priority queue."""
        if not self._queue:
            return None
        return heapq.heappop(self._queue)

    def peek_top_k(self, k: int = 10) -> List[FlaggedPortfolioItem]:
        """Inspect top K priority items without popping them."""
        return heapq.nsmallest(k, self._queue)

    @property
    def queue_size(self) -> int:
        """Returns the number of flagged portfolios currently in the queue."""
        return len(self._queue)

    @property
    def last_summary(self) -> Dict[str, Any]:
        """Returns the summary stats of the latest scan."""
        return self._last_scan_summary
