"""High-performance Synthetic Portfolio Generator for WealthPilot AI.

Synthesizes up to 50,000+ realistic portfolios across 5 risk categories calibrated
to Indian Capital Markets (Nifty 50, G-Sec, Corp Bonds, Gold, Liquid Cash).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from src.data.client_profile_generator import ClientProfileGenerator, RISK_CATEGORIES
from src.data.market_data_simulator import ASSET_CLASSES, MarketDataSimulator


# Strategic Asset Allocation Targets per Risk Category
DEFAULT_SAA_WEIGHTS: Dict[str, Dict[str, float]] = {
    "Ultra-Conservative": {
        "NIFTY_50_EQUITY": 0.10,
        "G_SEC_BONDS": 0.50,
        "CORP_BONDS": 0.25,
        "GOLD_ETF": 0.05,
        "LIQUID_CASH": 0.10,
    },
    "Conservative": {
        "NIFTY_50_EQUITY": 0.25,
        "G_SEC_BONDS": 0.40,
        "CORP_BONDS": 0.20,
        "GOLD_ETF": 0.05,
        "LIQUID_CASH": 0.10,
    },
    "Balanced": {
        "NIFTY_50_EQUITY": 0.50,
        "G_SEC_BONDS": 0.25,
        "CORP_BONDS": 0.15,
        "GOLD_ETF": 0.05,
        "LIQUID_CASH": 0.05,
    },
    "Aggressive": {
        "NIFTY_50_EQUITY": 0.70,
        "G_SEC_BONDS": 0.10,
        "CORP_BONDS": 0.10,
        "GOLD_ETF": 0.05,
        "LIQUID_CASH": 0.05,
    },
    "Ultra-Aggressive": {
        "NIFTY_50_EQUITY": 0.85,
        "G_SEC_BONDS": 0.00,
        "CORP_BONDS": 0.05,
        "GOLD_ETF": 0.05,
        "LIQUID_CASH": 0.05,
    },
}


class PortfolioGenerator:
    """Vectorized generator of realistic investment portfolios and drifted allocations."""

    def __init__(
        self,
        saa_weights: Optional[Dict[str, Dict[str, float]]] = None,
        asset_classes: Optional[List[str]] = None,
        seed: Optional[int] = 42,
    ) -> None:
        self.saa_weights = saa_weights or DEFAULT_SAA_WEIGHTS
        self.asset_classes = asset_classes or ASSET_CLASSES
        self.n_assets = len(self.asset_classes)
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.client_gen = ClientProfileGenerator(seed=seed)
        self.market_sim = MarketDataSimulator(seed=seed)

        # Pre-build lookup matrices for target weights
        self.risk_cat_to_idx = {cat: i for i, cat in enumerate(RISK_CATEGORIES)}
        self.saa_matrix = np.zeros((len(RISK_CATEGORIES), self.n_assets), dtype=np.float64)
        for cat_name, idx in self.risk_cat_to_idx.items():
            weights = self.saa_weights[cat_name]
            self.saa_matrix[idx] = [weights[ac] for ac in self.asset_classes]

    def generate_portfolios(
        self,
        n_portfolios: int = 50000,
        drift_intensity: float = 0.04,
        include_drift_shocks: bool = True,
    ) -> pd.DataFrame:
        """Vectorized generation of portfolios with current drifted weights and target weights.

        Args:
            n_portfolios: Number of portfolios to synthesize.
            drift_intensity: Standard deviation of simulated drift disturbance.
            include_drift_shocks: Whether to introduce market-driven drifts mimicking equity rallies/falls.

        Returns:
            DataFrame containing portfolio IDs, client metadata, target & current weights, and values.
        """
        # 1. Generate client profiles
        clients_df = self.client_gen.generate_profiles(n_clients=n_portfolios)

        # 2. Map risk categories to target weight vectors
        cat_indices = np.array([self.risk_cat_to_idx[c] for c in clients_df["risk_category"]])
        target_weights = self.saa_matrix[cat_indices]  # Shape: (N, 5)

        # 3. Simulate realistic drift
        # Drift reflects time passage, differential asset returns, and cash inflows
        # Add random perturbation with zero mean
        drift_noise = self.rng.normal(0.0, drift_intensity, size=(n_portfolios, self.n_assets))

        if include_drift_shocks:
            # Equities have had a strong run in ~30% of portfolios (bull drift)
            # and a correction in ~15% of portfolios (bear drift)
            market_regimes = self.rng.choice([-0.05, 0.0, 0.06], size=(n_portfolios, 1), p=[0.20, 0.50, 0.30])
            # Equity is index 0
            drift_noise[:, 0] += market_regimes.squeeze()

        # Raw current weights before normalization
        raw_current = target_weights + drift_noise
        # Keep non-negative (no short positions allowed in vanilla wealth management)
        raw_current = np.clip(raw_current, 0.001, 0.999)
        # Normalize each row to sum to 1.0
        row_sums = raw_current.sum(axis=1, keepdims=True)
        current_weights = raw_current / row_sums

        # 4. Generate metadata (last rebalance days, AUM)
        last_rebalance_days = self.rng.integers(15, 365, size=n_portfolios)

        # Construct DataFrame
        portfolio_ids = [f"WP-PF-{i+1:06d}" for i in range(n_portfolios)]
        df_dict = {
            "portfolio_id": portfolio_ids,
            "client_id": clients_df["client_id"],
            "risk_category": clients_df["risk_category"],
            "wealth_tier": clients_df["wealth_tier"],
            "aum_inr": clients_df["aum_inr"],
            "tax_sensitive": clients_df["tax_sensitive"],
            "has_custom_band": clients_df["has_custom_band"],
            "custom_band_delta": clients_df["custom_band_delta"],
            "life_event": clients_df["life_event"],
            "days_since_rebalance": last_rebalance_days,
        }

        # Add target and current weights for each asset class
        for j, ac in enumerate(self.asset_classes):
            df_dict[f"target_weight_{ac}"] = target_weights[:, j]
            df_dict[f"current_weight_{ac}"] = current_weights[:, j]
            df_dict[f"current_val_{ac}"] = np.round(current_weights[:, j] * clients_df["aum_inr"].to_numpy(), 2)

        return pd.DataFrame(df_dict)

    def extract_weight_matrices(
        self, df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Extract optimized NumPy arrays for target weights, current weights, and AUM.

        Args:
            df: Portfolios DataFrame.

        Returns:
            Tuple of:
              - target_weights: array of shape (N, n_assets)
              - current_weights: array of shape (N, n_assets)
              - aum: array of shape (N,)
        """
        target_cols = [f"target_weight_{ac}" for ac in self.asset_classes]
        current_cols = [f"current_weight_{ac}" for ac in self.asset_classes]

        target_weights = df[target_cols].to_numpy(dtype=np.float64)
        current_weights = df[current_cols].to_numpy(dtype=np.float64)
        aum = df["aum_inr"].to_numpy(dtype=np.float64)

        return target_weights, current_weights, aum
