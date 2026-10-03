"""Client Profile Generator synthesizing realistic investor archetypes for Indian Wealth Management."""

from __future__ import annotations

from typing import Dict, List, Optional
import numpy as np
import pandas as pd


RISK_CATEGORIES = [
    "Ultra-Conservative",
    "Conservative",
    "Balanced",
    "Aggressive",
    "Ultra-Aggressive",
]

DEFAULT_RISK_DISTRIBUTION = [0.10, 0.25, 0.35, 0.20, 0.10]


class ClientProfileGenerator:
    """Generates synthetic investor demographic, tax profile, and policy constraint data."""

    def __init__(self, seed: Optional[int] = 42) -> None:
        self.rng = np.random.default_rng(seed)

    def generate_profiles(self, n_clients: int = 50000) -> pd.DataFrame:
        """Vectorized generation of client profiles.

        Args:
            n_clients: Number of investor profiles to synthesize (default 50,000).

        Returns:
            DataFrame containing investor IDs, risk profiles, AUM, tax status, and policy overrides.
        """
        client_ids = [f"WP-CL-{i+1:06d}" for i in range(n_clients)]

        risk_cats = self.rng.choice(
            RISK_CATEGORIES,
            size=n_clients,
            p=DEFAULT_RISK_DISTRIBUTION,
        )

        mu_aum = 15.42
        sigma_aum = 1.15
        raw_aum = np.exp(self.rng.normal(loc=mu_aum, scale=sigma_aum, size=n_clients))
        aum = np.clip(raw_aum, 500_000.0, 500_000_000.0)

        wealth_tiers = np.where(
            aum >= 100_000_000.0,
            "Ultra-HNW",
            np.where(aum >= 25_000_000.0, "HNW", "Affluent/Retail"),
        )

        tax_sensitive = self.rng.random(size=n_clients) < np.where(aum > 10_000_000, 0.85, 0.40)

        has_custom_band = self.rng.random(size=n_clients) < 0.06
        custom_band_delta = np.where(
            has_custom_band,
            np.round(self.rng.uniform(-0.005, 0.010, size=n_clients), 4),
            0.0,
        )

        life_events = self.rng.choice(
            ["NONE", "RETIREMENT_PLANNED", "LIQUIDITY_DEMAND", "TAX_HARVEST_PRIORITY"],
            size=n_clients,
            p=[0.88, 0.05, 0.04, 0.03],
        )

        df = pd.DataFrame(
            {
                "client_id": client_ids,
                "risk_category": risk_cats,
                "aum_inr": np.round(aum, 2),
                "wealth_tier": wealth_tiers,
                "tax_sensitive": tax_sensitive,
                "has_custom_band": has_custom_band,
                "custom_band_delta": custom_band_delta,
                "life_event": life_events,
            }
        )
        return df
