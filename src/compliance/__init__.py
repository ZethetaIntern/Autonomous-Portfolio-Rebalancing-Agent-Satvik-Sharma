"""Compliance Auditing, Fairness Bias Detection, and SEBI Regulatory Reporting Package.

Components:
- ExplainabilityScorecard: Multi-dimensional scoring (Accuracy, Completeness, Readability, Regulatory Sufficiency)
- BiasDetector: Algorithmic fairness and parity audit (Risk Frequency, Cost Underestimation, AUM Disparity, Momentum)
- ComplianceAuditor: Automated stratified quarterly audit engine sampling decisions
- RegulatoryReporter: Exportable SEBI statutory compliance packages and cryptographic audit seals
"""

from src.compliance.explainability_scorecard import (
    ExplainabilityScorecard,
    ExplanationScorecardResult,
)
from src.compliance.bias_detector import (
    BiasDetector,
    BiasReport,
    BiasMetricSummary,
)
from src.compliance.compliance_auditor import (
    ComplianceAuditor,
    AuditRunResult,
    StratifiedSampleBreakdown,
)
from src.compliance.regulatory_reporter import (
    RegulatoryReporter,
)

__all__ = [
    "ExplainabilityScorecard",
    "ExplanationScorecardResult",
    "BiasDetector",
    "BiasReport",
    "BiasMetricSummary",
    "ComplianceAuditor",
    "AuditRunResult",
    "StratifiedSampleBreakdown",
    "RegulatoryReporter",
]
