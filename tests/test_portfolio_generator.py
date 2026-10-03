"""Unit tests for synthetic portfolio generator and market data simulator."""

import numpy as np
import pandas as pd
import pytest

from src.data.market_data_simulator import MarketDataSimulator, ASSET_CLASSES
from src.data.client_profile_generator import ClientProfileGenerator, RISK_CATEGORIES
from src.data.portfolio_generator import PortfolioGenerator, DEFAULT_SAA_WEIGHTS


def test_market_data_simulator_covariance():
    sim = MarketDataSimulator(seed=123)
    cov = sim.get_annual_covariance()

    # Covariance matrix must be 5x5, symmetric, and positive semi-definite
    assert cov.shape == (5, 5)
    np.testing.assert_allclose(cov, cov.T, atol=1e-8)
    eigenvalues = np.linalg.eigvalsh(cov)
    assert np.all(eigenvalues >= 0), "Covariance matrix must be positive semi-definite"

    # Simulate returns
    returns_df = sim.simulate_returns(days=50)
    assert returns_df.shape == (50, 5)
    assert list(returns_df.columns) == ASSET_CLASSES


def test_client_profile_generator():
    gen = ClientProfileGenerator(seed=99)
    df = gen.generate_profiles(n_clients=500)

    assert len(df) == 500
    assert set(df["risk_category"].unique()).issubset(set(RISK_CATEGORIES))
    assert (df["aum_inr"] >= 500_000.0).all()
    assert (df["aum_inr"] <= 500_000_000.0).all()
    assert "wealth_tier" in df.columns
    assert "tax_sensitive" in df.columns


def test_portfolio_generator_weights_and_constraints(sample_small_portfolios):
    df = sample_small_portfolios
    assert len(df) == 100

    target_cols = [f"target_weight_{ac}" for ac in ASSET_CLASSES]
    current_cols = [f"current_weight_{ac}" for ac in ASSET_CLASSES]

    # Target weights must sum to 1.0
    targ_sums = df[target_cols].sum(axis=1)
    np.testing.assert_allclose(targ_sums, 1.0, atol=1e-5)

    # Current weights must sum to 1.0
    curr_sums = df[current_cols].sum(axis=1)
    np.testing.assert_allclose(curr_sums, 1.0, atol=1e-5)

    # All weights non-negative
    assert (df[target_cols] >= 0.0).all().all()
    assert (df[current_cols] >= 0.0).all().all()


def test_portfolio_generator_matrix_extraction(portfolio_generator, sample_small_portfolios):
    t_mat, c_mat, aum = portfolio_generator.extract_weight_matrices(sample_small_portfolios)
    assert t_mat.shape == (100, 5)
    assert c_mat.shape == (100, 5)
    assert aum.shape == (100,)
    assert (aum > 0).all()
