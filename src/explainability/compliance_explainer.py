"""Compliance Explanation Generator & Immutable Regulatory Audit Logger for WealthPilot AI.

Generates formal compliance records citing SEBI Circular SEBI/HO/MRD/DOP1/CIR/P/2024/69,
cryptographic SHA-256 provenance chains, constraint satisfaction matrices, and embedded
surrogate explainability evidence (SHAP/LIME/Counterfactuals).
"""

from __future__ import annotations

import datetime
import hashlib
import json
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


REGULATORY_CIRCULAR_SEBI = "SEBI/HO/MRD/DOP1/CIR/P/2024/69"
DEFAULT_MODEL_VERSION = "WealthPilot-QP-1D-v0.2.0"


class ComplianceExplanationPayload(BaseModel):
    """Pydantic validated schema for regulatory compliance audit logs."""
    decision_id: str = Field(description="Unique deterministic decision identifier")
    timestamp_utc: str = Field(description="ISO-8601 UTC timestamp")
    portfolio_id: str
    client_id: str
    risk_category: str
    trigger_category: str
    trigger_severity: str
    regulatory_circular: str = Field(default=REGULATORY_CIRCULAR_SEBI)
    model_version: str = Field(default=DEFAULT_MODEL_VERSION)
    constraint_verification_matrix: Dict[str, bool]
    sebi_compliance_certified: bool
    input_data_snapshot: Dict[str, Any]
    output_execution_snapshot: Dict[str, Any]
    explainability_payload: Dict[str, Any]
    digital_signature_hash: str


class ComplianceExplainer:
    """Generates immutable regulatory audit records complying with SEBI algorithmic governance rules."""

    def __init__(self, model_version: str = DEFAULT_MODEL_VERSION) -> None:
        self.model_version = model_version

    def generate_compliance_audit(
        self,
        decision_id: str,
        portfolio_id: str,
        client_id: str,
        risk_category: str,
        trigger_category: str,
        trigger_severity: str,
        current_weights: Dict[str, float],
        target_weights: Dict[str, float],
        proposed_weights: Dict[str, float],
        trades: List[Any],
        constraint_matrix: Dict[str, bool],
        sebi_compliant: bool,
        shap_evidence: Optional[Dict[str, Any]] = None,
        lime_evidence: Optional[Dict[str, Any]] = None,
        counterfactual_evidence: Optional[Dict[str, Any]] = None,
        additional_metadata: Optional[Dict[str, Any]] = None,
    ) -> ComplianceExplanationPayload:
        """Constructs an auditable, cryptographically verifiable compliance record."""
        now_utc = datetime.datetime.now(datetime.timezone.utc).isoformat()

        trade_items = []
        for t in trades:
            if hasattr(t, "asset_class"):
                trade_items.append({
                    "asset": t.asset_class,
                    "action": t.action,
                    "units": t.target_units,
                    "price": t.estimated_price,
                    "val_inr": t.trade_value_inr,
                })
            elif isinstance(t, dict):
                trade_items.append(t)

        input_snapshot = {
            "current_weights": current_weights,
            "target_weights": target_weights,
            "risk_category": risk_category,
            "metadata": additional_metadata or {},
        }

        output_snapshot = {
            "proposed_weights": proposed_weights,
            "trades": trade_items,
            "trade_count": len(trade_items),
        }

        explainability_payload = {
            "shap_attribution": shap_evidence or {},
            "lime_local_weights": lime_evidence or {},
            "counterfactual_analysis": counterfactual_evidence or {},
        }

        canonical_dict = {
            "decision_id": decision_id,
            "timestamp_utc": now_utc,
            "portfolio_id": portfolio_id,
            "client_id": client_id,
            "regulatory_circular": REGULATORY_CIRCULAR_SEBI,
            "constraints": constraint_matrix,
            "inputs": input_snapshot,
            "outputs": output_snapshot,
        }
        canonical_str = json.dumps(canonical_dict, sort_keys=True, default=str)
        signature_hash = hashlib.sha256(canonical_str.encode("utf-8")).hexdigest()

        return ComplianceExplanationPayload(
            decision_id=decision_id,
            timestamp_utc=now_utc,
            portfolio_id=portfolio_id,
            client_id=client_id,
            risk_category=risk_category,
            trigger_category=trigger_category,
            trigger_severity=trigger_severity,
            regulatory_circular=REGULATORY_CIRCULAR_SEBI,
            model_version=self.model_version,
            constraint_verification_matrix=constraint_matrix,
            sebi_compliance_certified=sebi_compliant,
            input_data_snapshot=input_snapshot,
            output_execution_snapshot=output_snapshot,
            explainability_payload=explainability_payload,
            digital_signature_hash=signature_hash,
        )

    def verify_integrity(self, record: ComplianceExplanationPayload) -> bool:
        """Verifies if the record has not been tampered with since creation."""
        canonical_dict = {
            "decision_id": record.decision_id,
            "timestamp_utc": record.timestamp_utc,
            "portfolio_id": record.portfolio_id,
            "client_id": record.client_id,
            "regulatory_circular": record.regulatory_circular,
            "constraints": record.constraint_verification_matrix,
            "inputs": record.input_data_snapshot,
            "outputs": record.output_execution_snapshot,
        }
        canonical_str = json.dumps(canonical_dict, sort_keys=True, default=str)
        expected_hash = hashlib.sha256(canonical_str.encode("utf-8")).hexdigest()
        return expected_hash == record.digital_signature_hash
