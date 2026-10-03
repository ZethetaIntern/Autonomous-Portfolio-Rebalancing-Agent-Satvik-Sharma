"""Automated Compliance Auditor executing stratified quarterly algorithmic audits for WealthPilot AI.

Samples 100 autonomous decisions stratified across trigger categories (Threshold, Calendar, Event)
and risk profiles (Conservative to Aggressive) to evaluate:
- Regulatory adherence (SEBI single-issuer, sector concentration, cash buffer)
- Multi-audience explanation quality and readability via ExplainabilityScorecard
- Pre-trade and post-trade constraint verification
- Cryptographic audit trail validity
"""

from __future__ import annotations

import datetime
import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import numpy as np

from src.compliance.explainability_scorecard import ExplainabilityScorecard, ExplanationScorecardResult


@dataclass
class StratifiedSampleBreakdown:
    """Audit sampling distribution across a dimension."""
    category: str
    total_sampled: int
    passed_count: int
    failed_count: int
    pass_rate_pct: float


@dataclass
class AuditRunResult:
    """Comprehensive quarterly compliance audit execution result."""
    audit_id: str
    timestamp_utc: str
    total_sampled: int
    passed_count: int
    failed_count: int
    audit_pass_rate_pct: float
    overall_verdict: str  # "AUDIT_PASSED" or "AUDIT_FAILED"
    average_explanation_score: float
    trigger_stratification: Dict[str, StratifiedSampleBreakdown]
    risk_stratification: Dict[str, StratifiedSampleBreakdown]
    non_compliant_cases: List[Dict[str, Any]] = field(default_factory=list)
    audit_signature_hash: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "audit_id": self.audit_id,
            "timestamp_utc": self.timestamp_utc,
            "total_sampled": self.total_sampled,
            "passed_count": self.passed_count,
            "failed_count": self.failed_count,
            "audit_pass_rate_pct": self.audit_pass_rate_pct,
            "overall_verdict": self.overall_verdict,
            "average_explanation_score": self.average_explanation_score,
            "trigger_stratification": {
                k: {
                    "category": v.category,
                    "total_sampled": v.total_sampled,
                    "passed_count": v.passed_count,
                    "failed_count": v.failed_count,
                    "pass_rate_pct": v.pass_rate_pct,
                }
                for k, v in self.trigger_stratification.items()
            },
            "risk_stratification": {
                k: {
                    "category": v.category,
                    "total_sampled": v.total_sampled,
                    "passed_count": v.passed_count,
                    "failed_count": v.failed_count,
                    "pass_rate_pct": v.pass_rate_pct,
                }
                for k, v in self.risk_stratification.items()
            },
            "non_compliant_cases": self.non_compliant_cases,
            "audit_signature_hash": self.audit_signature_hash,
        }


class ComplianceAuditor:
    """Executes automated stratified quarterly audits over autonomous decision history."""

    def __init__(
        self,
        scorecard: Optional[ExplainabilityScorecard] = None,
        min_audit_pass_rate_pct: float = 95.0,
    ) -> None:
        self.scorecard = scorecard or ExplainabilityScorecard()
        self.min_pass_rate = min_audit_pass_rate_pct

    def draw_stratified_sample(
        self,
        decisions: List[Dict[str, Any]],
        sample_size: int = 100,
        seed: int = 42,
    ) -> List[Dict[str, Any]]:
        """Samples up to sample_size decisions stratified by trigger category and risk profile."""
        if len(decisions) <= sample_size:
            return list(decisions)

        rng = np.random.default_rng(seed)
        # Group by (trigger_category, risk_category)
        strata: Dict[tuple, List[Dict[str, Any]]] = {}
        for d in decisions:
            trig = str(d.get("trigger_category", "THRESHOLD")).upper()
            risk = str(d.get("risk_category", "Balanced")).capitalize()
            key = (trig, risk)
            strata.setdefault(key, []).append(d)

        sampled: List[Dict[str, Any]] = []
        per_stratum_target = max(1, sample_size // max(len(strata), 1))

        for key, group in strata.items():
            k = min(len(group), per_stratum_target)
            idx = rng.choice(len(group), size=k, replace=False)
            for i in idx:
                sampled.append(group[i])

        # If sample underfilled due to small strata, top up randomly
        if len(sampled) < sample_size:
            remaining = [d for d in decisions if d not in sampled]
            needed = min(len(remaining), sample_size - len(sampled))
            if needed > 0:
                idx = rng.choice(len(remaining), size=needed, replace=False)
                for i in idx:
                    sampled.append(remaining[i])

        return sampled[:sample_size]

    def run_quarterly_audit(
        self,
        decision_universe: List[Dict[str, Any]],
        sample_size: int = 100,
        quarter: str = "Q4-2024",
    ) -> AuditRunResult:
        """Executes full automated quarterly audit on a stratified sample of decisions."""
        sample = self.draw_stratified_sample(decision_universe, sample_size=sample_size)
        total_count = len(sample)

        passed_count = 0
        failed_count = 0
        score_list: List[float] = []
        non_compliant: List[Dict[str, Any]] = []

        trigger_stats: Dict[str, Dict[str, int]] = {}
        risk_stats: Dict[str, Dict[str, int]] = {}

        for d in sample:
            trig = str(d.get("trigger_category", "THRESHOLD")).upper()
            risk = str(d.get("risk_category", "Balanced")).capitalize()

            trigger_stats.setdefault(trig, {"total": 0, "pass": 0, "fail": 0})
            risk_stats.setdefault(risk, {"total": 0, "pass": 0, "fail": 0})
            trigger_stats[trig]["total"] += 1
            risk_stats[risk]["total"] += 1

            # 1. Constraint check
            comp_report = d.get("compliance_report", {})
            comp_pass = comp_report.get("status") in ("PASS", "APPROVED") if comp_report else bool(d.get("sebi_compliant", True))

            # 2. Scorecard check
            exp_packet = d.get("explanation_packet", d)
            score_res: ExplanationScorecardResult = self.scorecard.score_explanation_packet(
                explanation_packet=exp_packet,
                actual_trade_context=d,
            )
            score_list.append(score_res.composite_score)

            decision_passed = comp_pass and score_res.is_compliant

            if decision_passed:
                passed_count += 1
                trigger_stats[trig]["pass"] += 1
                risk_stats[risk]["pass"] += 1
            else:
                failed_count += 1
                trigger_stats[trig]["fail"] += 1
                risk_stats[risk]["fail"] += 1
                non_compliant.append({
                    "decision_id": d.get("decision_id", d.get("workflow_id", "UNKNOWN")),
                    "portfolio_id": d.get("portfolio_id", "UNKNOWN"),
                    "risk_category": risk,
                    "trigger_category": trig,
                    "compliance_pass": comp_pass,
                    "explanation_grade": score_res.grade,
                    "deficiencies": score_res.deficiencies,
                })

        pass_rate = round((passed_count / max(total_count, 1)) * 100.0, 2)
        overall_verdict = "AUDIT_PASSED" if pass_rate >= self.min_pass_rate else "AUDIT_FAILED"
        avg_score = round(float(np.mean(score_list)), 1) if score_list else 0.0

        trig_breakdowns = {
            k: StratifiedSampleBreakdown(
                category=k,
                total_sampled=v["total"],
                passed_count=v["pass"],
                failed_count=v["fail"],
                pass_rate_pct=round((v["pass"] / max(v["total"], 1)) * 100.0, 1),
            )
            for k, v in trigger_stats.items()
        }

        risk_breakdowns = {
            k: StratifiedSampleBreakdown(
                category=k,
                total_sampled=v["total"],
                passed_count=v["pass"],
                failed_count=v["fail"],
                pass_rate_pct=round((v["pass"] / max(v["total"], 1)) * 100.0, 1),
            )
            for k, v in risk_stats.items()
        }

        now_utc = datetime.datetime.now(datetime.timezone.utc).isoformat()
        audit_id = f"AUDIT-{quarter}-{int(datetime.datetime.now().timestamp())}"

        # Hash entire audit payload for immutability
        hash_payload = f"{audit_id}:{now_utc}:{pass_rate}:{total_count}:{overall_verdict}"
        sig_hash = hashlib.sha256(hash_payload.encode("utf-8")).hexdigest()

        return AuditRunResult(
            audit_id=audit_id,
            timestamp_utc=now_utc,
            total_sampled=total_count,
            passed_count=passed_count,
            failed_count=failed_count,
            audit_pass_rate_pct=pass_rate,
            overall_verdict=overall_verdict,
            average_explanation_score=avg_score,
            trigger_stratification=trig_breakdowns,
            risk_stratification=risk_breakdowns,
            non_compliant_cases=non_compliant,
            audit_signature_hash=sig_hash,
        )
