"""Threshold and Policy Tolerance Management for WealthPilot AI.

Maintains risk-category drift bands:
- Ultra-Conservative: 2.0% (0.020)
- Conservative: 2.5% (0.025)
- Balanced: 3.0% (0.030)
- Aggressive: 4.0% (0.040)
- Ultra-Aggressive: 5.0% (0.050)

Supports client-specific overlay overrides and asset class tolerance corridors.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np
import yaml


DEFAULT_RISK_BANDS: Dict[str, float] = {
    "Ultra-Conservative": 0.020,
    "Conservative": 0.025,
    "Balanced": 0.030,
    "Aggressive": 0.040,
    "Ultra-Aggressive": 0.050,
}

DEFAULT_ASSET_CORRIDORS: Dict[str, Dict[str, float]] = {
    "NIFTY_50_EQUITY": {"inner": 0.025, "outer": 0.050, "hard_cap": 0.90},
    "G_SEC_BONDS": {"inner": 0.020, "outer": 0.040, "hard_cap": 0.75},
    "CORP_BONDS": {"inner": 0.020, "outer": 0.040, "hard_cap": 0.40},
    "GOLD_ETF": {"inner": 0.015, "outer": 0.030, "hard_cap": 0.15},
    "LIQUID_CASH": {"inner": 0.020, "outer": 0.050, "min_buffer": 0.020, "hard_cap": 0.25},
}


@dataclass
class ClientPolicyOverlay:
    """Custom policy tolerance override for an individual client or account."""
    client_id: str
    custom_band_delta: float = 0.0          # e.g., -0.005 for tighter monitoring
    fixed_band_override: Optional[float] = None
    tax_sensitive: bool = False
    lock_in_until_date: Optional[str] = None
    max_turnover_override: Optional[float] = None


class ThresholdManager:
    """Manages baseline risk bands, asset corridors, and client-level overlay overrides."""

    def __init__(
        self,
        config_path: Optional[Union[str, Path]] = None,
        risk_bands: Optional[Dict[str, float]] = None,
        asset_corridors: Optional[Dict[str, Dict[str, float]]] = None,
    ) -> None:
        self.risk_bands = risk_bands or DEFAULT_RISK_BANDS.copy()
        self.asset_corridors = asset_corridors or DEFAULT_ASSET_CORRIDORS.copy()
        self.client_overlays: Dict[str, ClientPolicyOverlay] = {}

        if config_path:
            self._load_from_yaml(Path(config_path))

    def _load_from_yaml(self, path: Path) -> None:
        """Load threshold bands from YAML configuration."""
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
                if "drift_bands" in data and "risk_category_defaults" in data["drift_bands"]:
                    self.risk_bands.update(data["drift_bands"]["risk_category_defaults"])

    def register_client_overlay(self, overlay: ClientPolicyOverlay) -> None:
        """Add or update a client-specific overlay policy."""
        self.client_overlays[overlay.client_id] = overlay

    def get_effective_threshold(
        self,
        risk_category: str,
        client_id: Optional[str] = None,
    ) -> float:
        """Get the effective drift threshold for a client portfolio."""
        base_threshold = self.risk_bands.get(risk_category, 0.030)

        if client_id and client_id in self.client_overlays:
            overlay = self.client_overlays[client_id]
            if overlay.fixed_band_override is not None:
                return overlay.fixed_band_override
            # Apply delta, bounded below by 0.005 (0.5%)
            return max(0.005, base_threshold + overlay.custom_band_delta)

        return base_threshold

    def get_effective_thresholds_vectorized(
        self,
        risk_categories: np.ndarray,
        client_ids: Optional[np.ndarray] = None,
        custom_deltas: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Vectorized computation of effective thresholds across N portfolios.

        Args:
            risk_categories: Array of shape (N,) strings
            client_ids: Optional array of shape (N,) strings
            custom_deltas: Optional array of shape (N,) float adjustments

        Returns:
            Array of shape (N,) representing threshold floats.
        """
        # Map risk categories via lookup
        base_arr = np.array([self.risk_bands.get(rc, 0.030) for rc in risk_categories], dtype=np.float64)

        if custom_deltas is not None:
            base_arr = np.maximum(0.005, base_arr + custom_deltas)

        # Check registered client overlays if present
        if client_ids is not None and self.client_overlays:
            for i, cid in enumerate(client_ids):
                if cid in self.client_overlays:
                    ov = self.client_overlays[cid]
                    if ov.fixed_band_override is not None:
                        base_arr[i] = ov.fixed_band_override
                    else:
                        base_arr[i] = max(0.005, base_arr[i] + ov.custom_band_delta)

        return base_arr

    def evaluate_breaches(
        self,
        max_absolute_drifts: np.ndarray,
        effective_thresholds: np.ndarray,
    ) -> np.ndarray:
        """Vectorized boolean mask of threshold breaches.

        Returns:
            Boolean array of shape (N,) where True indicates breach.
        """
        return max_absolute_drifts > effective_thresholds
