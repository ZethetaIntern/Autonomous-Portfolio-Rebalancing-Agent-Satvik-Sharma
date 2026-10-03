"""LIME Local Interpretable Model-agnostic Explanations for WealthPilot AI.

Computes local linear approximations around individual rebalancing decisions,
returning top 5 contributing features with directionality and weight magnitude.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

try:
    from lime.lime_tabular import LimeTabularExplainer
    HAS_LIME = True
except ImportError:
    HAS_LIME = False

from sklearn.linear_model import Ridge
from src.explainability.shap_integration import EXPLAINABILITY_FEATURES, ShapExplainer

logger = logging.getLogger(__name__)


class LimeExplainer:
    """Computes local interpretable linear approximations (LIME) around rebalance decisions."""

    def __init__(
        self,
        surrogate_shap_explainer: Optional[ShapExplainer] = None,
        random_state: int = 42,
    ) -> None:
        self.random_state = random_state
        self.shap_explainer = surrogate_shap_explainer or ShapExplainer(random_state=random_state)
        self.feature_names = EXPLAINABILITY_FEATURES.copy()
        self.tabular_explainer: Optional[Any] = None
        self._init_lime()

    def _init_lime(self) -> None:
        """Initializes LimeTabularExplainer with representative background distribution."""
        X_train, _ = self.shap_explainer._generate_synthetic_training_data(1000)
        if HAS_LIME:
            try:
                self.tabular_explainer = LimeTabularExplainer(
                    training_data=X_train,
                    feature_names=self.feature_names,
                    class_names=["NoTrigger", "Trigger"],
                    mode="classification",
                    random_state=self.random_state,
                    verbose=False,
                )
            except Exception as e:
                logger.warning(f"LIME initialization warning: {e}. Falling back to internal local Ridge.")
                self.tabular_explainer = None

    def explain_instance(
        self,
        features: Union[Dict[str, float], np.ndarray, List[float]],
        num_features: int = 5,
    ) -> Dict[str, Any]:
        """Generates local linear surrogate explanation returning top 5 contributing features.
        
        Returns:
            Dict containing:
                - local_prediction: probability of rebalance trigger
                - top_features: list of dicts with feature, weight, direction, and magnitude
                - intercept: float
                - local_r2: surrogate linear fit quality
                - narrative: human-readable summary
        """
        if isinstance(features, dict):
            feat_vec = np.array([float(features.get(f, 0.0)) for f in self.feature_names], dtype=np.float64)
        else:
            feat_vec = np.asarray(features, dtype=np.float64).flatten()

        model = self.shap_explainer.model

        if HAS_LIME and self.tabular_explainer is not None:
            exp = self.tabular_explainer.explain_instance(
                data_row=feat_vec,
                predict_fn=model.predict_proba,
                num_features=num_features,
            )
            raw_list = exp.as_list()
            local_pred = float(exp.local_pred[0]) if hasattr(exp, "local_pred") else float(model.predict_proba(feat_vec.reshape(1, -1))[0, 1])
            local_score = float(exp.score) if hasattr(exp, "score") else 0.85
            intercept = float(exp.intercept[1]) if hasattr(exp, "intercept") and len(exp.intercept) > 1 else 0.0

            top_features: List[Dict[str, Any]] = []
            for rule_or_name, weight in raw_list[:num_features]:
                direction = "INCREASED_TRIGGER_PROBABILITY" if weight > 0 else "DECREASED_TRIGGER_PROBABILITY"
                top_features.append({
                    "feature_rule": rule_or_name,
                    "weight": round(float(weight), 4),
                    "abs_weight": round(abs(float(weight)), 4),
                    "direction": direction,
                })
        else:
            # Analytical local perturbation Ridge fallback
            top_features, local_pred, local_score, intercept = self._local_ridge_approximation(
                feat_vec, model, num_features
            )

        # Build narrative
        top_driver = top_features[0] if top_features else {"feature_rule": "N/A", "weight": 0.0}
        direction_word = "elevated" if top_driver["weight"] > 0 else "suppressed"
        narrative = (
            f"Local linear approximation indicates rebalancing trigger probability of {local_pred:.1%}. "
            f"The primary local driver was [{top_driver['feature_rule']}], which {direction_word} "
            f"decision likelihood with a weight coefficient of {top_driver['weight']:+.4f}."
        )

        return {
            "local_prediction": round(local_pred, 4),
            "local_r2_score": round(local_score, 4),
            "intercept": round(intercept, 4),
            "top_features": top_features,
            "feature_count": len(top_features),
            "narrative": narrative,
        }

    def _local_ridge_approximation(
        self,
        x: np.ndarray,
        model: Any,
        num_features: int,
        n_samples: int = 200,
    ) -> Tuple[List[Dict[str, Any]], float, float, float]:
        """Custom local perturbation Ridge regression when external LIME package is disabled."""
        rng = np.random.default_rng(self.random_state)
        # Sample gaussian perturbations around x
        pert = rng.normal(0, 0.15, size=(n_samples, len(x)))
        X_local = x + pert
        y_probs = model.predict_proba(X_local)[:, 1]

        # Exponential kernel distance weights
        dists = np.linalg.norm(pert, axis=1)
        weights = np.exp(-dists / 0.5)

        ridge = Ridge(alpha=1.0, random_state=self.random_state)
        ridge.fit(pert, y_probs, sample_weight=weights)

        r2 = float(ridge.score(pert, y_probs, sample_weight=weights))
        coefs = ridge.coef_
        intercept = float(ridge.intercept_)
        pred = float(model.predict_proba(x.reshape(1, -1))[0, 1])

        ranked_idx = np.argsort(np.abs(coefs))[::-1][:num_features]
        top_list = []
        for idx in ranked_idx:
            w = coefs[idx]
            top_list.append({
                "feature_rule": f"{self.feature_names[idx]} = {x[idx]:.2f}",
                "weight": round(float(w), 4),
                "abs_weight": round(abs(float(w)), 4),
                "direction": "INCREASED_TRIGGER_PROBABILITY" if w > 0 else "DECREASED_TRIGGER_PROBABILITY",
            })

        return top_list, pred, max(0.0, r2), intercept
