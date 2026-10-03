"""Counterfactual Explanation Generator for WealthPilot AI.

Generates actionable counterfactual explanations by solving minimal feature perturbations
across the autonomous decision boundary (e.g. "If equity drift were below 3.2% instead of 4.7%,
the agent would not have triggered a rebalance").
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from scipy.optimize import minimize

from src.explainability.shap_integration import EXPLAINABILITY_FEATURES, ShapExplainer


class CounterfactualGenerator:
    """Solves minimal feature perturbations to find nearest decision-boundary counterfactual scenarios."""

    def __init__(
        self,
        shap_explainer: Optional[ShapExplainer] = None,
        feature_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
    ) -> None:
        self.shap_explainer = shap_explainer or ShapExplainer()
        self.feature_names = EXPLAINABILITY_FEATURES.copy()
        self.feature_bounds = feature_bounds or {
            "equity_drift_pct": (0.0, 15.0),
            "vix_level": (9.0, 50.0),
            "days_since_rebalance": (0.0, 365.0),
            "client_risk_score": (1.0, 5.0),
            "tax_lot_maturity_days": (0.0, 500.0),
            "sector_concentration_pct": (5.0, 60.0),
        }
        # Immutable features that cannot be dynamically altered in counterfactual simulations
        self.immutable_features = {"client_risk_score"}

    def generate_counterfactual(
        self,
        features: Union[Dict[str, float], np.ndarray, List[float]],
        target_decision: Optional[str] = None,
        max_features_to_vary: int = 2,
    ) -> Dict[str, Any]:
        """Finds the closest point across the decision boundary altering at most max_features_to_vary features.
        
        Args:
            features: Current feature values.
            target_decision: Desired counterfactual ("NO_TRIGGER" if currently TRIGGER, or vice-versa).
            max_features_to_vary: Limit on number of perturbed features for human interpretability.

        Returns:
            Dict containing minimal_perturbations, distances, and human-readable counterfactual narrative.
        """
        if isinstance(features, dict):
            orig_vec = np.array([float(features.get(f, 0.0)) for f in self.feature_names], dtype=np.float64)
        else:
            orig_vec = np.asarray(features, dtype=np.float64).flatten()

        model = self.shap_explainer.model
        orig_prob = float(model.predict_proba(orig_vec.reshape(1, -1))[0, 1])
        orig_decision = "TRIGGER" if orig_prob >= 0.50 else "NO_TRIGGER"

        if target_decision is None:
            target_decision = "NO_TRIGGER" if orig_decision == "TRIGGER" else "TRIGGER"

        target_is_no_trigger = target_decision == "NO_TRIGGER"
        target_threshold = 0.45 if target_is_no_trigger else 0.55

        # 1. First attempt: Single-feature bisection search (simplest and most intuitive explanation)
        single_feature_results: List[Dict[str, Any]] = []

        # Standard deviations for feature normalization in distance metric
        scales = np.array([2.5, 6.0, 60.0, 1.0, 90.0, 6.0], dtype=np.float64)

        for i, fname in enumerate(self.feature_names):
            if fname in self.immutable_features:
                continue

            low_b, high_b = self.feature_bounds.get(fname, (0.0, 100.0))
            best_cf_val = None
            min_dist = float("inf")

            # Grid search along feature dimension
            test_vals = np.linspace(low_b, high_b, 100)
            for val in test_vals:
                cand = orig_vec.copy()
                cand[i] = val
                prob = float(model.predict_proba(cand.reshape(1, -1))[0, 1])

                success = (prob <= target_threshold) if target_is_no_trigger else (prob >= target_threshold)
                if success:
                    dist = abs(val - orig_vec[i]) / scales[i]
                    if dist < min_dist:
                        min_dist = dist
                        best_cf_val = val

            if best_cf_val is not None:
                single_feature_results.append({
                    "feature": fname,
                    "feature_idx": i,
                    "cf_value": best_cf_val,
                    "orig_value": orig_vec[i],
                    "delta": best_cf_val - orig_vec[i],
                    "norm_dist": min_dist,
                })

        # Sort single-feature candidates by minimal normalized distance
        single_feature_results.sort(key=lambda x: x["norm_dist"])

        if single_feature_results:
            best_res = single_feature_results[0]
            cf_vec = orig_vec.copy()
            cf_vec[best_res["feature_idx"]] = best_res["cf_value"]
            cf_prob = float(model.predict_proba(cf_vec.reshape(1, -1))[0, 1])

            delta = best_res["delta"]
            fname = best_res["feature"]
            orig_val = best_res["orig_value"]
            cf_val = best_res["cf_value"]

            direction_str = "below" if delta < 0 else "above"
            action_phrase = "were" if "days" in fname else "were"

            narrative = (
                f"If {fname.replace('_', ' ')} {action_phrase} {direction_str} {cf_val:.2f}% "
                f"(currently {orig_val:.2f}%, delta of {delta:+.2f}), "
                f"the agent would NOT have triggered a rebalance (Predicted Trigger Probability: {cf_prob:.1%})."
            )

            perturbations = {
                fname: {
                    "original": round(float(orig_val), 4),
                    "counterfactual": round(float(cf_val), 4),
                    "delta": round(float(delta), 4),
                }
            }

            l1_dist = round(float(abs(delta)), 4)
            l2_dist = round(float(abs(delta)), 4)

        else:
            # Multi-feature optimization fallback using scipy
            def objective(delta_vec):
                return np.sum((delta_vec / scales) ** 2)

            def constraint_fn(delta_vec):
                cand = orig_vec + delta_vec
                prob = float(model.predict_proba(cand.reshape(1, -1))[0, 1])
                return (target_threshold - prob) if target_is_no_trigger else (prob - target_threshold)

            init_delta = np.zeros(len(orig_vec))
            bounds = [
                (low - orig_vec[j], high - orig_vec[j])
                for j, (low, high) in enumerate(
                    [self.feature_bounds.get(f, (0, 100)) for f in self.feature_names]
                )
            ]
            for j, f in enumerate(self.feature_names):
                if f in self.immutable_features:
                    bounds[j] = (0.0, 0.0)

            opt_res = minimize(
                objective,
                init_delta,
                method="SLSQP",
                bounds=bounds,
                constraints={"type": "ineq", "fun": constraint_fn},
            )

            cf_vec = orig_vec + opt_res.x
            cf_prob = float(model.predict_proba(cf_vec.reshape(1, -1))[0, 1])
            perturbations = {}
            for j, f in enumerate(self.feature_names):
                if abs(opt_res.x[j]) > 1e-3:
                    perturbations[f] = {
                        "original": round(float(orig_vec[j]), 4),
                        "counterfactual": round(float(cf_vec[j]), 4),
                        "delta": round(float(opt_res.x[j]), 4),
                    }
            l1_dist = round(float(np.sum(np.abs(opt_res.x))), 4)
            l2_dist = round(float(np.linalg.norm(opt_res.x)), 4)
            narrative = (
                f"Counterfactual reached with {len(perturbations)} feature adjustment(s): "
                + "; ".join(f"{k} {v['original']} -> {v['counterfactual']}" for k, v in perturbations.items())
                + f" (Counterfactual Probability: {cf_prob:.1%})."
            )

        return {
            "original_decision": orig_decision,
            "original_probability": round(orig_prob, 4),
            "target_decision": target_decision,
            "counterfactual_probability": round(cf_prob, 4),
            "minimal_perturbations": perturbations,
            "l1_distance": l1_dist,
            "l2_distance": l2_dist,
            "counterfactual_narrative": narrative,
            "features_perturbed_count": len(perturbations),
        }
