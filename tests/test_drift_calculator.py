"""Unit and Performance tests for Vectorized Drift Engine and Priority Queue."""

import time
import numpy as np
import pytest

from src.monitoring.drift_calculator import DriftCalculator
from src.monitoring.threshold_manager import ClientPolicyOverlay, ThresholdManager
from src.monitoring.drift_monitor import DriftMonitor


def test_drift_calculator_single_formulas():
    # 5 assets with known simple weights
    cov = np.eye(5) * 0.04  # diagonal covariance, 20% vol each
    calc = DriftCalculator(asset_covariance_annual=cov)

    target = np.array([0.50, 0.20, 0.15, 0.10, 0.05])
    current = np.array([0.55, 0.17, 0.13, 0.10, 0.05])
    # deltas: [+0.05, -0.03, -0.02, 0.00, 0.00]

    res = calc.calculate_single(current_weights=current, target_weights=target)

    # Absolute drift: [0.05, 0.03, 0.02, 0.0, 0.0]
    np.testing.assert_allclose(res["absolute_drift"], [0.05, 0.03, 0.02, 0.0, 0.0])
    assert pytest.approx(res["max_absolute_drift"], 1e-6) == 0.05
    # SAD = 0.05 + 0.03 + 0.02 = 0.10
    assert pytest.approx(res["sum_absolute_drift"], 1e-6) == 0.10
    # RMSD = sqrt( (0.0025 + 0.0009 + 0.0004 + 0 + 0) / 5 ) = sqrt( 0.0038 / 5 ) = sqrt(0.00076)
    expected_rmsd = np.sqrt(0.0038 / 5.0)
    assert pytest.approx(res["rmsd"], 1e-6) == expected_rmsd

    # Tracking error = sqrt( delta^T * cov * delta )
    # delta^T * (0.04 * I) * delta = 0.04 * sum(delta^2) = 0.04 * 0.0038 = 0.000152
    expected_te = np.sqrt(0.000152)
    assert pytest.approx(res["tracking_error"], 1e-6) == expected_te


def test_drift_calculator_vectorized_equivalence(drift_calculator):
    rng = np.random.default_rng(42)
    n = 20
    curr = rng.dirichlet(np.ones(5), size=n)
    targ = rng.dirichlet(np.ones(5), size=n)

    batch_res = drift_calculator.calculate_vectorized(curr, targ)

    assert batch_res.portfolio_count == n
    assert batch_res.absolute_drift.shape == (n, 5)
    assert batch_res.max_absolute_drift.shape == (n,)
    assert batch_res.sum_absolute_drift.shape == (n,)
    assert batch_res.rmsd.shape == (n,)
    assert batch_res.tracking_error.shape == (n,)

    # Verify matching results row-by-row
    for i in range(n):
        single_res = drift_calculator.calculate_single(curr[i], targ[i])
        assert pytest.approx(single_res["max_absolute_drift"], 1e-6) == batch_res.max_absolute_drift[i]
        assert pytest.approx(single_res["sum_absolute_drift"], 1e-6) == batch_res.sum_absolute_drift[i]
        assert pytest.approx(single_res["rmsd"], 1e-6) == batch_res.rmsd[i]
        assert pytest.approx(single_res["tracking_error"], 1e-6) == batch_res.tracking_error[i]


def test_threshold_manager_bands_and_overlays(threshold_manager):
    # Base thresholds
    assert threshold_manager.get_effective_threshold("Ultra-Conservative") == 0.020
    assert threshold_manager.get_effective_threshold("Conservative") == 0.025
    assert threshold_manager.get_effective_threshold("Balanced") == 0.030
    assert threshold_manager.get_effective_threshold("Aggressive") == 0.040
    assert threshold_manager.get_effective_threshold("Ultra-Aggressive") == 0.050

    # Register client overlay with custom delta
    overlay_delta = ClientPolicyOverlay(client_id="WP-CL-000010", custom_band_delta=-0.005)
    threshold_manager.register_client_overlay(overlay_delta)

    effective = threshold_manager.get_effective_threshold("Balanced", client_id="WP-CL-000010")
    assert pytest.approx(effective, 1e-6) == 0.025  # 0.030 - 0.005

    # Register client overlay with fixed override
    overlay_fixed = ClientPolicyOverlay(client_id="WP-CL-000020", fixed_band_override=0.015)
    threshold_manager.register_client_overlay(overlay_fixed)
    assert threshold_manager.get_effective_threshold("Aggressive", client_id="WP-CL-000020") == 0.015


def test_drift_monitor_queue_and_scan(drift_monitor, sample_small_portfolios):
    summary = drift_monitor.scan_universe(sample_small_portfolios)

    assert summary["total_portfolios_scanned"] == 100
    assert summary["sla_met"] is True
    assert summary["total_scan_time_seconds"] < 2.0
    assert drift_monitor.queue_size == summary["breached_portfolios_count"]

    if drift_monitor.queue_size > 1:
        top_item = drift_monitor.pop_highest_priority()
        second_item = drift_monitor.pop_highest_priority()
        # Ensure max-priority ordering (raw priority descending)
        assert top_item.raw_priority >= second_item.raw_priority


def test_drift_monitor_50k_portfolio_sla(drift_monitor, sample_50k_portfolios):
    """SLA Benchmark Target: 50,000 portfolios scanned in under 30.0 seconds."""
    t0 = time.perf_counter()
    summary = drift_monitor.scan_universe(sample_50k_portfolios)
    elapsed = time.perf_counter() - t0

    assert summary["total_portfolios_scanned"] == 50000
    assert summary["sla_met"] is True
    # The scan should easily beat the SLA target
    assert elapsed < 30.0, f"Scan exceeded SLA: {elapsed:.2f}s > 30.0s"
    # Even more strictly: vectorized NumPy should finish under 5 seconds
    assert summary["drift_calc_time_seconds"] < 1.0, f"Drift calculation too slow: {summary['drift_calc_time_seconds']:.3f}s"
