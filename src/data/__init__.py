"""Data Generation & Simulation Package for WealthPilot AI."""

from src.data.market_data_simulator import (
    ASSET_CLASSES,
    DEFAULT_ANNUAL_RETURNS,
    DEFAULT_ANNUAL_VOLS,
    DEFAULT_CORRELATION,
    MarketDataSimulator,
)
from src.data.client_profile_generator import (
    RISK_CATEGORIES,
    ClientProfileGenerator,
)
from src.data.portfolio_generator import (
    DEFAULT_SAA_WEIGHTS,
    PortfolioGenerator,
)

__all__ = [
    "ASSET_CLASSES",
    "DEFAULT_ANNUAL_RETURNS",
    "DEFAULT_ANNUAL_VOLS",
    "DEFAULT_CORRELATION",
    "MarketDataSimulator",
    "RISK_CATEGORIES",
    "ClientProfileGenerator",
    "DEFAULT_SAA_WEIGHTS",
    "PortfolioGenerator",
]
