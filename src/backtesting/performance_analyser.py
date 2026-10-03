"""Portfolio Performance Analyser for WealthPilot AI.

Calculates comprehensive risk-adjusted performance, drawdown dynamics, tracking error,
turnover, transaction cost drag, and tax alpha metrics according to institutional standards.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import numpy as np


class PerformanceAnalyser:
    """Computes risk-adjusted performance, drawdown dynamics, tracking error, and tax alpha."""

    def __init__(self, risk_free_rate: float = 0.065) -> None:
        """
        Args:
            risk_free_rate: Annualized Indian risk-free benchmark rate (default: 6.5% G-Sec 10Y).
        """
        self.risk_free_rate = risk_free_rate

    def compute_metrics(
        self,
        returns: np.ndarray | List[float],
        risk_free_rate: Optional[float] = None,
        benchmark_returns: Optional[np.ndarray | List[float]] = None,
    ) -> Dict[str, float]:
        """Calculates standard performance and risk metrics on a series of daily returns."""
        rf = self.risk_free_rate if risk_free_rate is None else risk_free_rate
        arr = np.asarray(returns, dtype=np.float64)
        if len(arr) == 0:
            return {
                "annualized_return": 0.0,
                "annualized_volatility": 0.0,
                "sharpe_ratio": 0.0,
                "sortino_ratio": 0.0,
                "max_drawdown_pct": 0.0,
            }

        daily_mean = float(np.mean(arr))
        ann_return = daily_mean * 252.0
        ann_vol = float(np.std(arr, ddof=1) * np.sqrt(252)) if len(arr) > 1 else 0.0

        excess_return = ann_return - rf
        sharpe = excess_return / max(ann_vol, 1e-6)

        downside = arr[arr < 0.0]
        if len(downside) > 1:
            downside_vol = float(np.std(downside, ddof=1) * np.sqrt(252))
            sortino = excess_return / max(downside_vol, 1e-6)
        else:
            sortino = sharpe

        cum_ret = np.cumprod(1.0 + arr)
        cum_max = np.maximum.accumulate(cum_ret)
        drawdowns = (cum_ret - cum_max) / cum_max
        max_dd = float(np.min(drawdowns)) if len(drawdowns) > 0 else 0.0

        res: Dict[str, float] = {
            "annualized_return": round(ann_return, 4),
            "annualized_volatility": round(ann_vol, 4),
            "sharpe_ratio": round(sharpe, 4),
            "sortino_ratio": round(sortino, 4),
            "max_drawdown_pct": round(max_dd * 100.0, 2),
        }

        if benchmark_returns is not None:
            b_arr = np.asarray(benchmark_returns, dtype=np.float64)
            if len(b_arr) == len(arr) and len(arr) > 1:
                excess_arr = arr - b_arr
                te = float(np.std(excess_arr, ddof=1) * np.sqrt(252))
                b_ann = float(np.mean(b_arr) * 252.0)
                ir = (ann_return - b_ann) / max(te, 1e-6)
                res["tracking_error"] = round(te, 4)
                res["information_ratio"] = round(ir, 4)

        return res

    def compute_comprehensive_metrics(
        self,
        portfolio_values: List[float],
        benchmark_values: Optional[List[float]] = None,
        turnover_history: Optional[List[float]] = None,
        transaction_costs_inr: float = 0.0,
        tax_shield_inr: float = 0.0,
        risk_free_rate: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Computes end-to-end institutional metrics including wealth trajectory, costs, and tax alpha.
        
        Args:
            portfolio_values: Daily portfolio valuation trajectory over backtest.
            benchmark_values: Daily benchmark valuation trajectory.
            turnover_history: List of turnover fractions incurred during rebalances.
            transaction_costs_inr: Cumulative total transaction costs in INR.
            tax_shield_inr: Cumulative tax-loss harvesting shield in INR.
            risk_free_rate: Annualized risk-free rate.
        """
        vals = np.asarray(portfolio_values, dtype=np.float64)
        if len(vals) < 2:
            return {}

        daily_returns = (vals[1:] - vals[:-1]) / vals[:-1]

        bench_returns = None
        if benchmark_values is not None:
            b_vals = np.asarray(benchmark_values, dtype=np.float64)
            if len(b_vals) == len(vals):
                bench_returns = (b_vals[1:] - b_vals[:-1]) / b_vals[:-1]

        base_metrics = self.compute_metrics(
            daily_returns,
            risk_free_rate=risk_free_rate,
            benchmark_returns=bench_returns,
        )

        initial_val = float(vals[0])
        final_val = float(vals[-1])
        total_return_pct = ((final_val - initial_val) / initial_val) * 100.0
        n_days = len(daily_returns)
        cagr = ((final_val / initial_val) ** (252.0 / max(n_days, 1)) - 1.0) * 100.0

        cum_max = np.maximum.accumulate(vals)
        drawdowns = (vals - cum_max) / cum_max
        max_dd_pct = float(np.min(drawdowns)) * 100.0

        total_turnover = float(sum(turnover_history)) if turnover_history else 0.0
        annualized_turnover = total_turnover * (252.0 / max(n_days, 1))

        tax_alpha_bps = (tax_shield_inr / max(initial_val, 1.0)) * 10000.0

        cost_drag_bps = (transaction_costs_inr / max(initial_val, 1.0)) * 10000.0

        calmar = (cagr / 100.0) / max(abs(max_dd_pct / 100.0), 1e-4)

        return {
            "initial_capital_inr": initial_val,
            "final_capital_inr": final_val,
            "total_return_pct": round(total_return_pct, 2),
            "cagr_pct": round(cagr, 2),
            "annualized_return": base_metrics.get("annualized_return", 0.0),
            "annualized_volatility": base_metrics.get("annualized_volatility", 0.0),
            "sharpe_ratio": base_metrics.get("sharpe_ratio", 0.0),
            "sortino_ratio": base_metrics.get("sortino_ratio", 0.0),
            "calmar_ratio": round(calmar, 3),
            "max_drawdown_pct": round(max_dd_pct, 2),
            "tracking_error": base_metrics.get("tracking_error", 0.0),
            "information_ratio": base_metrics.get("information_ratio", 0.0),
            "annualized_turnover_pct": round(annualized_turnover * 100.0, 2),
            "total_transaction_costs_inr": round(transaction_costs_inr, 2),
            "cost_drag_bps": round(cost_drag_bps, 2),
            "tax_shield_harvested_inr": round(tax_shield_inr, 2),
            "tax_alpha_bps": round(tax_alpha_bps, 2),
            "trading_days": n_days,
        }
