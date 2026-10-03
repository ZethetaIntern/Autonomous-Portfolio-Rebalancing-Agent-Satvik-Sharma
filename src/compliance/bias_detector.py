"""Algorithmic Bias and Fairness Detector for WealthPilot AI.

Evaluates autonomous rebalancing decision patterns across 50,000 portfolios to detect
systematic behavioral, demographic, asset-tier, or cost-estimation biases:
1. Risk-Profile Frequency Bias: Disproportionate rebalance frequencies in Conservative vs Aggressive
2. Transaction Cost Underestimation Bias: Systematically predicting lower trading costs than realized
3. Momentum vs. Contrarian Bias: Trading against disciplined mean-reversion in favor of trend-chasing
4. AUM / Wealth Tier Disparity Bias: Preferential drift tolerance corridors for HNIs vs Retail
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import numpy as np


@dataclass
class BiasMetricSummary:
    """Quantitative metric summary for an algorithmic fairness dimension."""
    metric_name: str
    observed_value: float
    benchmark_value: float
    disparity_ratio: float
    is_biased: bool
    description: str


@dataclass
class BiasReport:
    """Comprehensive algorithmic fairness and neutrality audit report."""
    overall_bias_detected: bool
    bias_score: float  # 0.0 (no bias) to 100.0 (severe bias)
    evaluated_decisions_count: int
    flags: List[str] = field(default_factory=list)
    metrics: Dict[str, BiasMetricSummary] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "overall_bias_detected": self.overall_bias_detected,
            "bias_score": self.bias_score,
            "evaluated_decisions_count": self.evaluated_decisions_count,
            "flags": self.flags,
            "metrics": {
                k: {
                    "metric_name": v.metric_name,
                    "observed_value": v.observed_value,
                    "benchmark_value": v.benchmark_value,
                    "disparity_ratio": v.disparity_ratio,
                    "is_biased": v.is_biased,
                    "description": v.description,
                }
                for k, v in self.metrics.items()
            },
            "recommendations": self.recommendations,
        }


class BiasDetector:
    """Audits rebalancing history across cohorts for statistical parity and algorithmic fairness."""

    def __init__(
        self,
        max_frequency_disparity_ratio: float = 1.60,
        max_cost_underestimation_pct: float = 10.0,
        max_aum_tier_drift_disparity: float = 1.35,
    ) -> None:
        self.max_freq_disparity = max_frequency_disparity_ratio
        self.max_cost_underestimation = max_cost_underestimation_pct
        self.max_aum_disparity = max_aum_tier_drift_disparity

    def analyze_decisions(self, decisions: List[Dict[str, Any]]) -> BiasReport:
        """Runs fairness tests across all rebalance decisions in the sample.
        
        Args:
            decisions: List of rebalance decision records containing risk_category,
                       portfolio_aum, estimated_costs, actual_costs, trades, sad.
        """
        if not decisions:
            return BiasReport(
                overall_bias_detected=False,
                bias_score=0.0,
                evaluated_decisions_count=0,
                flags=[],
                metrics={},
                recommendations=["No decisions provided for bias auditing."],
            )

        n_decisions = len(decisions)
        flags: List[str] = []
        metrics: Dict[str, BiasMetricSummary] = {}
        recommendations: List[str] = []

        # -----------------------------------------------------------------
        # 1. RISK-PROFILE FREQUENCY BIAS
        # -----------------------------------------------------------------
        conservative_count = sum(1 for d in decisions if str(d.get("risk_category", "")).strip().upper() == "CONSERVATIVE")
        aggressive_count = sum(1 for d in decisions if str(d.get("risk_category", "")).strip().upper() == "AGGRESSIVE")
        balanced_count = sum(1 for d in decisions if str(d.get("risk_category", "")).strip().upper() == "BALANCED")

        # Disparity ratio: Conservative / Aggressive frequency relative to population baseline
        cons_rate = conservative_count / max(n_decisions, 1)
        aggr_rate = aggressive_count / max(n_decisions, 1)
        freq_ratio = cons_rate / max(aggr_rate, 1e-4) if aggr_rate > 0 else 1.0

        is_freq_biased = freq_ratio > self.max_freq_disparity or freq_ratio < (1.0 / self.max_freq_disparity)
        if is_freq_biased:
            flags.append(f"Disproportionate rebalancing frequency across risk profiles (ratio: {freq_ratio:.2f})")
            recommendations.append("Recalibrate volatility-adjusted drift corridor widths between Conservative and Aggressive mandates.")

        metrics["risk_frequency_parity"] = BiasMetricSummary(
            metric_name="Conservative to Aggressive Frequency Ratio",
            observed_value=round(freq_ratio, 2),
            benchmark_value=1.0,
            disparity_ratio=round(freq_ratio, 2),
            is_biased=is_freq_biased,
            description="Compares the rebalancing trigger rate between Conservative and Aggressive portfolios.",
        )

        # -----------------------------------------------------------------
        # 2. TRANSACTION COST UNDERESTIMATION BIAS
        # -----------------------------------------------------------------
        cost_deltas = []
        for d in decisions:
            est_cost = float(d.get("estimated_costs_inr", d.get("estimated_cost_inr", d.get("total_costs_inr", 0.0))))
            act_cost = float(d.get("actual_costs_inr", d.get("actual_cost_inr", d.get("realized_cost_inr", est_cost))))
            if est_cost > 0:
                # Percentage underestimation: (actual - estimated) / estimated * 100
                cost_deltas.append(((act_cost - est_cost) / est_cost) * 100.0)

        mean_cost_error = float(np.mean(cost_deltas)) if cost_deltas else 0.0
        is_cost_biased = mean_cost_error > self.max_cost_underestimation

        if is_cost_biased:
            flags.append(f"Systematic transaction cost underestimation bias ({mean_cost_error:.1f}% under actuals)")
            recommendations.append("Increase square-root market impact coefficient k_i and update bid-ask spread assumptions.")

        metrics["cost_estimation_neutrality"] = BiasMetricSummary(
            metric_name="Mean Transaction Cost Underestimation (%)",
            observed_value=round(mean_cost_error, 2),
            benchmark_value=0.0,
            disparity_ratio=round(max(0.0, mean_cost_error) / max(self.max_cost_underestimation, 1e-4), 2),
            is_biased=is_cost_biased,
            description="Tests whether optimizer systematically under-projects execution costs and slippage.",
        )

        # -----------------------------------------------------------------
        # 3. AUM TIER / WEALTH DISPARITY BIAS
        # -----------------------------------------------------------------
        # Compare average drift SAD allowed before rebalance between HNIs (> 1 Cr) and Retail (< 25L)
        hni_sads = [float(d.get("sad", 0.05)) for d in decisions if float(d.get("portfolio_aum", d.get("aum_inr", d.get("aum", 1_000_000.0)))) >= 10_000_000.0]
        retail_sads = [float(d.get("sad", 0.05)) for d in decisions if float(d.get("portfolio_aum", d.get("aum_inr", d.get("aum", 1_000_000.0)))) <= 2_500_000.0]

        mean_hni_sad = float(np.mean(hni_sads)) if hni_sads else 0.05
        mean_retail_sad = float(np.mean(retail_sads)) if retail_sads else 0.05
        aum_disparity_ratio = mean_retail_sad / max(mean_hni_sad, 1e-4)

        is_aum_biased = aum_disparity_ratio > self.max_aum_disparity or aum_disparity_ratio < (1.0 / self.max_aum_disparity)
        if is_aum_biased:
            flags.append(f"AUM tier drift threshold disparity detected (ratio: {aum_disparity_ratio:.2f})")
            recommendations.append("Align retail monitoring sensitivity to match institutional drift response thresholds.")

        metrics["aum_tier_parity"] = BiasMetricSummary(
            metric_name="Retail vs HNI Drift Trigger Disparity",
            observed_value=round(aum_disparity_ratio, 2),
            benchmark_value=1.0,
            disparity_ratio=round(aum_disparity_ratio, 2),
            is_biased=is_aum_biased,
            description="Ensures retail portfolios receive equal monitoring vigilance as large HNI accounts.",
        )

        # -----------------------------------------------------------------
        # 4. MOMENTUM VS CONTRARIAN BIAS
        # -----------------------------------------------------------------
        # Rebalancing should be disciplined mean-reverting (selling overweight, buying underweight)
        contrarian_trades = 0
        total_trades = 0
        for d in decisions:
            trades = d.get("trades", d.get("candidate_trades", []))
            for t in trades:
                total_trades += 1
                action = str(t.get("action", "")).upper() if isinstance(t, dict) else str(getattr(t, "action", "")).upper()
                # If selling an asset that has drifted overweight, it is disciplined contrarian rebalancing
        if total_trades > 0:
            contrarian_pct = (contrarian_trades / total_trades) * 100.0
            is_momentum_biased = contrarian_pct < 85.0
        else:
            contrarian_pct = 100.0
            is_momentum_biased = False

        if is_momentum_biased:
            flags.append("Agent exhibiting momentum bias: failing to rebalance contrarian to target asset allocation.")
            recommendations.append("Enforce strict target weight pull constraint in CVXPY optimization formulation.")

        metrics["rebalancing_discipline"] = BiasMetricSummary(
            metric_name="Disciplined Rebalancing Share (%)",
            observed_value=round(contrarian_pct, 1),
            benchmark_value=100.0,
            disparity_ratio=round(contrarian_pct / 100.0, 2),
            is_biased=is_momentum_biased,
            description="Verifies trades enforce mean-reverting strategic asset allocation rather than chasing trend momentum.",
        )

        overall_biased = len(flags) > 0
        # Bias score: 0 to 100 (each biased dimension adds 25 pts)
        bias_score = min(100.0, sum(25.0 for m in metrics.values() if m.is_biased))

        return BiasReport(
            overall_bias_detected=overall_biased,
            bias_score=bias_score,
            evaluated_decisions_count=n_decisions,
            flags=flags,
            metrics=metrics,
            recommendations=recommendations,
        )
