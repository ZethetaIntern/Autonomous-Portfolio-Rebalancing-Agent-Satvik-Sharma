"""High-performance, vectorized Drift Calculation Engine for WealthPilot AI.

Implements mathematical drift metrics across 50,000+ portfolios:
1. Absolute Drift per Asset: |w_current - w_target|
2. Sum of Absolute Drift (SAD): Σ |w_current,i - w_target,i|
3. Root Mean Square Drift (RMSD): sqrt( (1/K) * Σ (w_current,i - w_target,i)^2 )
4. Predicted Annualized Tracking Error: sqrt( Δw^T * Σ_cov * Δw )
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DriftMetricsBatch:
    """Vectorized result container for a batch of portfolio drift calculations."""
    absolute_drift: np.ndarray       # Shape: (N, K)
    max_absolute_drift: np.ndarray   # Shape: (N,)
    sum_absolute_drift: np.ndarray   # Shape: (N,) - SAD
    rmsd: np.ndarray                 # Shape: (N,) - Root Mean Square Drift
    tracking_error: np.ndarray       # Shape: (N,) - Annualized Predicted Tracking Error
    computation_time_seconds: float
    portfolio_count: int


class DriftCalculator:
    """Vectorized NumPy calculation engine for portfolio asset allocation drifts."""

    def __init__(
        self,
        asset_covariance_annual: Optional[np.ndarray] = None,
        asset_names: Optional[List[str]] = None,
    ) -> None:
        """Initialize calculator with annualized covariance matrix.

        Args:
            asset_covariance_annual: (K, K) covariance matrix of asset class annual returns.
            asset_names: Optional list of asset class names.
        """
        if asset_covariance_annual is None:
            # Default to standard Indian 5-asset covariance if not passed
            vols = np.array([0.160, 0.045, 0.055, 0.140, 0.005], dtype=np.float64)
            corr = np.array(
                [
                    [ 1.00,  0.08,  0.15, -0.12,  0.01],
                    [ 0.08,  1.00,  0.82,  0.10,  0.05],
                    [ 0.15,  0.82,  1.00,  0.08,  0.06],
                    [-0.12,  0.10,  0.08,  1.00,  0.00],
                    [ 0.01,  0.05,  0.06,  0.00,  1.00],
                ],
                dtype=np.float64,
            )
            d = np.diag(vols)
            self.cov = d @ corr @ d
        else:
            self.cov = np.asarray(asset_covariance_annual, dtype=np.float64)

        self.n_assets = self.cov.shape[0]
        self.asset_names = asset_names

    def calculate_single(
        self,
        current_weights: np.ndarray,
        target_weights: np.ndarray,
    ) -> Dict[str, Union[float, np.ndarray]]:
        """Calculate drift metrics for a single portfolio.

        Args:
            current_weights: Array of shape (K,)
            target_weights: Array of shape (K,)

        Returns:
            Dict containing SAD, RMSD, tracking error, and asset-level absolute drift.
        """
        curr = np.asarray(current_weights, dtype=np.float64)
        targ = np.asarray(target_weights, dtype=np.float64)
        delta_w = curr - targ

        abs_drift = np.abs(delta_w)
        sad = float(np.sum(abs_drift))
        rmsd = float(np.sqrt(np.mean(delta_w ** 2)))

        # Quadratic form: Δw^T * Σ * Δw
        variance = float(delta_w @ self.cov @ delta_w)
        tracking_error = float(np.sqrt(max(0.0, variance)))

        return {
            "absolute_drift": abs_drift,
            "max_absolute_drift": float(np.max(abs_drift)),
            "sum_absolute_drift": sad,
            "rmsd": rmsd,
            "tracking_error": tracking_error,
        }

    def calculate_vectorized(
        self,
        current_weights_matrix: np.ndarray,
        target_weights_matrix: np.ndarray,
    ) -> DriftMetricsBatch:
        """Compute all drift metrics across N portfolios in parallel with vectorized NumPy.

        SLA Target: Scan 50,000 portfolios in < 30 seconds (typically executes in < 0.05s).

        Args:
            current_weights_matrix: Array of shape (N, K)
            target_weights_matrix: Array of shape (N, K)

        Returns:
            DriftMetricsBatch dataclass with all computed metrics.
        """
        start_time = time.perf_counter()

        c_mat = np.asarray(current_weights_matrix, dtype=np.float64)
        t_mat = np.asarray(target_weights_matrix, dtype=np.float64)

        if c_mat.shape != t_mat.shape:
            raise ValueError(
                f"Shape mismatch: current_weights {c_mat.shape} != target_weights {t_mat.shape}"
            )

        n_portfolios, k_assets = c_mat.shape
        if k_assets != self.n_assets:
            raise ValueError(
                f"Asset dimension mismatch: weights have {k_assets} assets, covariance matrix has {self.n_assets}"
            )

        # 1. Delta weights: (N, K)
        delta_w = c_mat - t_mat

        # 2. Absolute drift per asset: (N, K)
        abs_drift = np.abs(delta_w)

        # 3. Max absolute drift per portfolio: (N,)
        max_abs_drift = np.max(abs_drift, axis=1)

        # 4. Sum of Absolute Drift (SAD): (N,)
        sad = np.sum(abs_drift, axis=1)

        # 5. Root Mean Square Drift (RMSD): (N,)
        rmsd = np.sqrt(np.mean(delta_w ** 2, axis=1))

        # 6. Vectorized Predicted Tracking Error: (N,)
        # delta_w @ cov yields (N, K)
        # element-wise product with delta_w and row sum gives Δw_i^T * Σ * Δw_i
        var_vector = np.sum((delta_w @ self.cov) * delta_w, axis=1)
        # Numerical protection against tiny floating point negatives
        tracking_error = np.sqrt(np.maximum(0.0, var_vector))

        elapsed = time.perf_counter() - start_time

        return DriftMetricsBatch(
            absolute_drift=abs_drift,
            max_absolute_drift=max_abs_drift,
            sum_absolute_drift=sad,
            rmsd=rmsd,
            tracking_error=tracking_error,
            computation_time_seconds=elapsed,
            portfolio_count=n_portfolios,
        )

    def attach_drift_metrics_to_df(
        self,
        df: pd.DataFrame,
        current_cols: Optional[List[str]] = None,
        target_cols: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Calculate drift and attach results directly as new DataFrame columns."""
        if current_cols is None or target_cols is None:
            if self.asset_names is None:
                raise ValueError("asset_names must be defined if column names are not specified.")
            current_cols = [f"current_weight_{ac}" for ac in self.asset_names]
            target_cols = [f"target_weight_{ac}" for ac in self.asset_names]

        curr_mat = df[current_cols].to_numpy(dtype=np.float64)
        targ_mat = df[target_cols].to_numpy(dtype=np.float64)

        metrics = self.calculate_vectorized(curr_mat, targ_mat)

        out_df = df.copy()
        out_df["max_absolute_drift"] = metrics.max_absolute_drift
        out_df["sum_absolute_drift"] = metrics.sum_absolute_drift
        out_df["rmsd"] = metrics.rmsd
        out_df["tracking_error"] = metrics.tracking_error

        if self.asset_names:
            for j, ac in enumerate(self.asset_names):
                out_df[f"drift_{ac}"] = metrics.absolute_drift[:, j]

        return out_df
