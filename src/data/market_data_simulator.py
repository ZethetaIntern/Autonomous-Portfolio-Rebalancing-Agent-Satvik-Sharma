"""Market Data Simulator calibrated to historical Indian (NSE/BSE) Asset Classes.

Assets:
1. NIFTY_50_EQUITY: Large Cap Equities
2. G_SEC_BONDS: Government of India Sovereign 10Y Benchmark
3. CORP_BONDS: AAA Rated Corporate Debt
4. GOLD_ETF: Domestic Gold Price Tracker
5. LIQUID_CASH: Overnight / Liquid Fund Yield
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


ASSET_CLASSES: List[str] = [
    "NIFTY_50_EQUITY",
    "G_SEC_BONDS",
    "CORP_BONDS",
    "GOLD_ETF",
    "LIQUID_CASH",
]

DEFAULT_ANNUAL_RETURNS: np.ndarray = np.array([0.125, 0.071, 0.082, 0.095, 0.062], dtype=np.float64)
DEFAULT_ANNUAL_VOLS: np.ndarray = np.array([0.160, 0.045, 0.055, 0.140, 0.005], dtype=np.float64)

DEFAULT_CORRELATION: np.ndarray = np.array(
    [
        [ 1.00,  0.08,  0.15, -0.12,  0.01],
        [ 0.08,  1.00,  0.82,  0.10,  0.05],
        [ 0.15,  0.82,  1.00,  0.08,  0.06],
        [-0.12,  0.10,  0.08,  1.00,  0.00],
        [ 0.01,  0.05,  0.06,  0.00,  1.00],
    ],
    dtype=np.float64,
)


class MarketDataSimulator:
    """Simulates correlated multi-asset return paths and provides covariance metrics."""

    def __init__(
        self,
        asset_names: Optional[List[str]] = None,
        annual_returns: Optional[np.ndarray] = None,
        annual_vols: Optional[np.ndarray] = None,
        correlation_matrix: Optional[np.ndarray] = None,
        trading_days_per_year: int = 252,
        seed: Optional[int] = 42,
    ) -> None:
        self.asset_names = asset_names or ASSET_CLASSES
        self.n_assets = len(self.asset_names)
        self.annual_returns = annual_returns if annual_returns is not None else DEFAULT_ANNUAL_RETURNS.copy()
        self.annual_vols = annual_vols if annual_vols is not None else DEFAULT_ANNUAL_VOLS.copy()
        self.correlation_matrix = correlation_matrix if correlation_matrix is not None else DEFAULT_CORRELATION.copy()
        self.trading_days = trading_days_per_year
        self.rng = np.random.default_rng(seed)

        diag_vols = np.diag(self.annual_vols)
        self.cov_matrix_annual = diag_vols @ self.correlation_matrix @ diag_vols
        self.cov_matrix_daily = self.cov_matrix_annual / self.trading_days

        self.cholesky_daily = np.linalg.cholesky(self.cov_matrix_daily)

    def get_annual_covariance(self) -> np.ndarray:
        """Returns the 5x5 annualized covariance matrix."""
        return self.cov_matrix_annual

    def simulate_returns(self, days: int = 252) -> pd.DataFrame:
        """Simulate daily correlated log-returns for all asset classes.

        Args:
            days: Number of trading days to simulate.

        Returns:
            DataFrame of daily returns with shape (days, n_assets).
        """
        daily_drift = (self.annual_returns - 0.5 * (self.annual_vols ** 2)) / self.trading_days
        standard_normals = self.rng.standard_normal(size=(days, self.n_assets))
        correlated_shocks = standard_normals @ self.cholesky_daily.T
        daily_returns = daily_drift + correlated_shocks
        return pd.DataFrame(daily_returns, columns=self.asset_names)

    def simulate_price_paths(
        self, days: int = 252, initial_prices: Optional[Dict[str, float]] = None
    ) -> pd.DataFrame:
        """Simulate daily price trajectories from starting asset levels."""
        returns_df = self.simulate_returns(days=days)
        start_prices = initial_prices or {
            "NIFTY_50_EQUITY": 24000.0,
            "G_SEC_BONDS": 100.0,
            "CORP_BONDS": 1000.0,
            "GOLD_ETF": 72.50,
            "LIQUID_CASH": 1000.0,
        }
        init_vector = np.array([start_prices[k] for k in self.asset_names])
        cum_returns = np.exp(np.cumsum(returns_df.to_numpy(), axis=0))
        prices = init_vector * np.vstack([np.ones(self.n_assets), cum_returns[:-1]])
        return pd.DataFrame(prices, columns=self.asset_names)

    def simulate_market_shock(
        self,
        asset_drawdowns: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        """Simulate an abrupt market shock event (e.g., -12% equity crash, bond rally).

        Returns:
            Dictionary of instantaneous return shocks per asset.
        """
        if asset_drawdowns is None:
            asset_drawdowns = {
                "NIFTY_50_EQUITY": -0.12,
                "G_SEC_BONDS": 0.015,
                "CORP_BONDS": -0.01,
                "GOLD_ETF": 0.045,
                "LIQUID_CASH": 0.0002,
            }
        return {asset: asset_drawdowns.get(asset, 0.0) for asset in self.asset_names}
