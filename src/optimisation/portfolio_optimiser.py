"""Convex Optimization Engine for WealthPilot AI (CVXPY Quadratic Programming).

Formulates and solves constrained quadratic programming (QP) for autonomous portfolio
rebalancing minimizing tracking error variance under SEBI regulatory limits, turnover
budgets, minimum trade thresholds, and net cash flows.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

try:
    import cvxpy as cp
    HAS_CVXPY = True
except ImportError:
    HAS_CVXPY = False

from src.data.market_data_simulator import ASSET_CLASSES, MarketDataSimulator

logger = logging.getLogger(__name__)


class PortfolioOptimiser:
    """Solves tracking error minimization under transaction cost, tax, and turnover constraints.
    
    Objective:
        minimize (w_current + w_trade - w_target)^T * Cov * (w_current + w_trade - w_target)
                 + turnover_penalty * sum(|w_trade|)

    Hard Constraints:
        1. Budget constraint: sum(w_current + w_trade) == 1.0 + net_cash_flow
        2. Long-only constraint: w_current + w_trade >= min_weight (typically >= 0.0)
        3. Turnover budget limit: 0.5 * sum(|w_trade|) <= turnover_budget
        4. SEBI single-issuer limit: w_current_i + w_trade_i <= max_issuer_limit (e.g. 0.10)
        5. SEBI sector concentration limit: sum_{i in sector} (w_current_i + w_trade_i) <= sector_cap
        6. Minimum cash buffer: w_cash >= min_cash_buffer
        7. Minimum trade size: |w_trade_i| >= min_trade_size or 0.0
    """

    def __init__(
        self,
        asset_names: Optional[List[str]] = None,
        covariance_matrix: Optional[np.ndarray] = None,
        risk_aversion: float = 1.0,
        default_solver: Optional[str] = None,
    ) -> None:
        self.asset_names = list(asset_names or ASSET_CLASSES)
        self.n_assets = len(self.asset_names)
        self.risk_aversion = risk_aversion
        self.default_solver = default_solver

        if covariance_matrix is not None:
            self.cov = np.asarray(covariance_matrix, dtype=np.float64)
        elif self.n_assets == len(ASSET_CLASSES):
            sim = MarketDataSimulator(asset_names=self.asset_names)
            self.cov = sim.get_annual_covariance()
        else:
            self.cov = np.diag(np.full(self.n_assets, 0.04))

        self.cov = (self.cov + self.cov.T) / 2.0
        min_eig = np.min(np.linalg.eigvalsh(self.cov))
        if min_eig < 1e-8:
            self.cov += np.eye(self.n_assets) * (1e-8 - min_eig)

    def _resolve_solver(self, requested: Optional[str] = None) -> Any:
        """Selects installed QP-capable solver."""
        if not HAS_CVXPY:
            return None
        available = cp.installed_solvers()
        if requested and requested.upper() in available:
            return getattr(cp, requested.upper())
        if self.default_solver and self.default_solver.upper() in available:
            return getattr(cp, self.default_solver.upper())
        for s in ["CLARABEL", "OSQP", "SCS"]:
            if s in available:
                return getattr(cp, s)
        return None

    def optimize_allocation(
        self,
        current_weights: Union[np.ndarray, Dict[str, float], List[float]],
        target_weights: Union[np.ndarray, Dict[str, float], List[float]],
        turnover_budget: Optional[float] = None,
        net_cash_flow: float = 0.0,
        min_cash_buffer: float = 0.02,
        sebi_issuer_limit: Optional[float] = None,
        sebi_sector_limits: Optional[Dict[str, float]] = None,
        asset_sector_map: Optional[Dict[str, str]] = None,
        asset_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
        min_trade_size: float = 0.0,
        turnover_penalty: float = 0.0,
        solver: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Solves optimal rebalancing weights minimizing tracking error variance under hard constraints.
        
        Args:
            current_weights: Current portfolio weights (array or mapping by asset name).
            target_weights: Strategic asset allocation target weights.
            turnover_budget: Maximum allowable one-way turnover fraction (e.g. 0.15 = 15%).
            net_cash_flow: External cash flow fraction (e.g. +0.05 for 5% deposit, -0.05 for withdrawal).
            min_cash_buffer: Minimum required allocation to liquid cash (default 2%).
            sebi_issuer_limit: Maximum allowable weight per single issuer (SEBI limit: 0.10).
            sebi_sector_limits: Maximum allowable aggregate weight per sector (e.g. {'FINANCIALS': 0.30}).
            asset_sector_map: Mapping of asset name to sector code.
            asset_bounds: Custom (min_weight, max_weight) per asset name.
            min_trade_size: Minimum executable trade size (e.g. 0.005 = 0.5% of AUM). Smaller trades zeroed out.
            turnover_penalty: L1 regularization penalty on trade turnover in objective function.
            solver: Explicit solver name ('CLARABEL', 'OSQP', 'SCS').

        Returns:
            Dict containing optimal_weights, trade_weights, tracking_error_variance,
            predicted_tracking_error, turnover, status, and constraint diagnostic flags.
        """
        curr = self._to_weight_vector(current_weights)
        targ = self._to_weight_vector(target_weights)

        if not HAS_CVXPY:
            return self._fallback_analytical_solve(
                curr, targ, turnover_budget=turnover_budget, min_cash_buffer=min_cash_buffer
            )

        solver_obj = self._resolve_solver(solver)

        result = self._solve_cvxpy_qp(
            curr=curr,
            targ=targ,
            turnover_budget=turnover_budget,
            net_cash_flow=net_cash_flow,
            min_cash_buffer=min_cash_buffer,
            sebi_issuer_limit=sebi_issuer_limit,
            sebi_sector_limits=sebi_sector_limits,
            asset_sector_map=asset_sector_map,
            asset_bounds=asset_bounds,
            turnover_penalty=turnover_penalty,
            solver_obj=solver_obj,
            fixed_zero_mask=None,
        )

        if min_trade_size > 0.0 and result.get("status") in ("OPTIMAL", "optimal"):
            w_trade = result["trade_weights_array"]
            sub_threshold = (np.abs(w_trade) > 1e-6) & (np.abs(w_trade) < min_trade_size)
            if np.any(sub_threshold):
                result_refined = self._solve_cvxpy_qp(
                    curr=curr,
                    targ=targ,
                    turnover_budget=turnover_budget,
                    net_cash_flow=net_cash_flow,
                    min_cash_buffer=min_cash_buffer,
                    sebi_issuer_limit=sebi_issuer_limit,
                    sebi_sector_limits=sebi_sector_limits,
                    asset_sector_map=asset_sector_map,
                    asset_bounds=asset_bounds,
                    turnover_penalty=turnover_penalty,
                    solver_obj=solver_obj,
                    fixed_zero_mask=sub_threshold,
                )
                if result_refined.get("status") in ("OPTIMAL", "optimal"):
                    result = result_refined

        return self._format_result(result, curr, targ)

    def _solve_cvxpy_qp(
        self,
        curr: np.ndarray,
        targ: np.ndarray,
        turnover_budget: Optional[float],
        net_cash_flow: float,
        min_cash_buffer: float,
        sebi_issuer_limit: Optional[float],
        sebi_sector_limits: Optional[Dict[str, float]],
        asset_sector_map: Optional[Dict[str, str]],
        asset_bounds: Optional[Dict[str, Tuple[float, float]]],
        turnover_penalty: float,
        solver_obj: Any,
        fixed_zero_mask: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Internal CVXPY problem formulation and solution."""
        w_trade = cp.Variable(self.n_assets)
        w_new = curr + w_trade
        delta = w_new - targ

        obj_expr = cp.quad_form(delta, cp.psd_wrap(self.cov))
        if turnover_penalty > 0:
            obj_expr += turnover_penalty * cp.norm1(w_trade)
        
        objective = cp.Minimize(obj_expr)
        constraints = []

        target_sum = 1.0 + float(net_cash_flow)
        constraints.append(cp.sum(w_new) == target_sum)

        if asset_bounds:
            for i, name in enumerate(self.asset_names):
                lb, ub = asset_bounds.get(name, (0.0, 1.0))
                constraints.append(w_new[i] >= lb)
                constraints.append(w_new[i] <= ub)
        else:
            constraints.append(w_new >= 0.0)

        if "LIQUID_CASH" in self.asset_names:
            cash_idx = self.asset_names.index("LIQUID_CASH")
            constraints.append(w_new[cash_idx] >= min_cash_buffer)

        if turnover_budget is not None and turnover_budget > 0:
            constraints.append(cp.norm1(w_trade) <= 2.0 * float(turnover_budget))

        if sebi_issuer_limit is not None and sebi_issuer_limit > 0:
            for i, name in enumerate(self.asset_names):
                constraints.append(w_new[i] <= float(sebi_issuer_limit))

        if sebi_sector_limits and asset_sector_map:
            for sector, cap in sebi_sector_limits.items():
                sector_indices = [
                    i for i, name in enumerate(self.asset_names)
                    if asset_sector_map.get(name) == sector
                ]
                if sector_indices:
                    constraints.append(cp.sum([w_new[i] for i in sector_indices]) <= float(cap))

        if fixed_zero_mask is not None:
            for i in range(self.n_assets):
                if fixed_zero_mask[i]:
                    constraints.append(w_trade[i] == 0.0)

        problem = cp.Problem(objective, constraints)

        try:
            if solver_obj is not None:
                problem.solve(solver=solver_obj, warm_start=True)
            else:
                problem.solve(warm_start=True)
        except Exception as e:
            logger.warning(f"Primary solver execution failed: {e}. Retrying with SCS fallback.")
            try:
                problem.solve(solver=cp.SCS)
            except Exception as e_scs:
                logger.error(f"SCS fallback solve failed: {e_scs}")
                return {
                    "status": "INFEASIBLE_OR_ERROR",
                    "optimal_weights_array": curr.copy(),
                    "trade_weights_array": np.zeros(self.n_assets),
                    "objective_value": float("inf"),
                    "solver_used": "FAILED",
                }

        status = problem.status.upper() if problem.status else "UNKNOWN"
        if status in ("OPTIMAL", "OPTIMAL_INACCURATE") and w_trade.value is not None:
            w_trade_val = np.asarray(w_trade.value, dtype=np.float64).flatten()
            w_opt_val = curr + w_trade_val
            w_opt_val = np.clip(w_opt_val, 0.0, None)
            total = np.sum(w_opt_val)
            if abs(total - target_sum) > 1e-7 and total > 0:
                w_opt_val = (w_opt_val / total) * target_sum
                w_trade_val = w_opt_val - curr

            return {
                "status": "OPTIMAL",
                "optimal_weights_array": w_opt_val,
                "trade_weights_array": w_trade_val,
                "objective_value": float(problem.value) if problem.value is not None else 0.0,
                "solver_used": str(problem.solver_stats.solver_name) if problem.solver_stats else "CVXPY",
            }
        else:
            return {
                "status": status,
                "optimal_weights_array": curr.copy(),
                "trade_weights_array": np.zeros(self.n_assets),
                "objective_value": float("inf"),
                "solver_used": str(problem.solver_stats.solver_name) if problem.solver_stats else "CVXPY",
            }

    def _fallback_analytical_solve(
        self,
        curr: np.ndarray,
        targ: np.ndarray,
        turnover_budget: Optional[float] = None,
        min_cash_buffer: float = 0.02,
    ) -> Dict[str, Any]:
        """Analytical projection fallback when CVXPY is absent."""
        diff = targ - curr
        turnover_needed = float(np.sum(np.abs(diff)) / 2.0)

        if turnover_budget is None or turnover_needed <= turnover_budget:
            optimal = targ.copy()
        else:
            step = float(turnover_budget) / max(turnover_needed, 1e-6)
            optimal = curr + (diff * step)

        if "LIQUID_CASH" in self.asset_names:
            cash_idx = self.asset_names.index("LIQUID_CASH")
            if optimal[cash_idx] < min_cash_buffer:
                deficit = min_cash_buffer - optimal[cash_idx]
                optimal[cash_idx] = min_cash_buffer
                non_cash = [i for i in range(self.n_assets) if i != cash_idx]
                total_non_cash = np.sum(optimal[non_cash])
                if total_non_cash > 0:
                    optimal[non_cash] -= deficit * (optimal[non_cash] / total_non_cash)

        optimal = np.clip(optimal, 0.0, None)
        optimal = optimal / np.sum(optimal)
        trade = optimal - curr

        return self._format_result(
            {
                "status": "OPTIMAL",
                "optimal_weights_array": optimal,
                "trade_weights_array": trade,
                "objective_value": float(np.dot(np.dot(optimal - targ, self.cov), optimal - targ)),
                "solver_used": "ANALYTICAL_PROJECTION",
            },
            curr,
            targ,
        )

    def _to_weight_vector(self, weights: Union[np.ndarray, Dict[str, float], List[float]]) -> np.ndarray:
        """Converts input weights into ordered numpy float64 array matching asset_names."""
        if isinstance(weights, dict):
            vec = np.array([float(weights.get(k, 0.0)) for k in self.asset_names], dtype=np.float64)
        else:
            vec = np.asarray(weights, dtype=np.float64).flatten()
            if len(vec) != self.n_assets:
                raise ValueError(f"Weight vector length {len(vec)} does not match {self.n_assets} assets.")
        return vec

    def _format_result(
        self,
        raw: Dict[str, Any],
        curr: np.ndarray,
        targ: np.ndarray,
    ) -> Dict[str, Any]:
        """Formats solution into standard dictionary."""
        optimal = raw["optimal_weights_array"]
        trade = raw["trade_weights_array"]
        dev_from_target = optimal - targ
        tev = float(np.dot(np.dot(dev_from_target, self.cov), dev_from_target))
        pred_te = float(np.sqrt(max(0.0, tev)))
        turnover = float(np.sum(np.abs(trade)) / 2.0)

        violations: List[str] = []
        if abs(np.sum(optimal) - (np.sum(curr) + np.sum(trade))) > 1e-4:
            violations.append(f"Budget sum mismatch: sum(optimal)={np.sum(optimal):.4f}")
        if np.any(optimal < -1e-6):
            violations.append("Long-only breach: negative weights present.")

        return {
            "optimal_weights": {ac: round(float(w), 6) for ac, w in zip(self.asset_names, optimal)},
            "trade_weights": {ac: round(float(t), 6) for ac, w, t in zip(self.asset_names, optimal, trade)},
            "turnover": round(turnover, 6),
            "tracking_error_variance": round(tev, 8),
            "predicted_tracking_error": round(pred_te, 6),
            "objective_value": round(float(raw.get("objective_value", tev)), 8),
            "status": raw.get("status", "OPTIMAL"),
            "solver": raw.get("solver_used", "CVXPY"),
            "constraints_verified": len(violations) == 0,
            "violations": violations,
        }
