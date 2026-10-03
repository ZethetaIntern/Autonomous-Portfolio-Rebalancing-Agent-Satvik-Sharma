"""Pytest fixtures for WealthPilot AI test suite."""

import warnings

# Python 3.13 / Matplotlib / CrewAI compatibility fix for warnings.warn
_orig_warn = warnings.warn
def _safe_warn(message, category=None, stacklevel=1, source=None, *args, **kwargs):
    if "skip_file_prefixes" in kwargs:
        kwargs.pop("skip_file_prefixes", None)
    return _orig_warn(message, category=category, stacklevel=stacklevel, source=source, *args, **kwargs)
warnings.warn = _safe_warn

import pytest
import numpy as np
import pandas as pd

from src.data.market_data_simulator import MarketDataSimulator, ASSET_CLASSES
from src.data.portfolio_generator import PortfolioGenerator
from src.monitoring.drift_calculator import DriftCalculator
from src.monitoring.threshold_manager import ThresholdManager
from src.monitoring.drift_monitor import DriftMonitor
from src.triggers.trigger_consolidator import TriggerConsolidator


@pytest.fixture(scope="session")
def market_simulator():
    return MarketDataSimulator(seed=42)


@pytest.fixture(scope="session")
def portfolio_generator():
    return PortfolioGenerator(seed=42)


@pytest.fixture(scope="session")
def sample_small_portfolios(portfolio_generator):
    """Small batch of 100 portfolios for fast unit tests."""
    return portfolio_generator.generate_portfolios(n_portfolios=100)


@pytest.fixture(scope="session")
def sample_50k_portfolios(portfolio_generator):
    """Full 50,000 portfolio universe for SLA benchmark testing."""
    return portfolio_generator.generate_portfolios(n_portfolios=50000)


@pytest.fixture
def drift_calculator(market_simulator):
    return DriftCalculator(
        asset_covariance_annual=market_simulator.get_annual_covariance(),
        asset_names=ASSET_CLASSES,
    )


@pytest.fixture
def threshold_manager():
    return ThresholdManager()


@pytest.fixture
def drift_monitor(drift_calculator, threshold_manager):
    return DriftMonitor(
        drift_calculator=drift_calculator,
        threshold_manager=threshold_manager,
        asset_names=ASSET_CLASSES,
    )


@pytest.fixture
def trigger_consolidator():
    return TriggerConsolidator()
