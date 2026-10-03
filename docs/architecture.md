# WealthPilot AI - Enterprise System Architecture & Engineering Specification

> **Project Code**: 1D  
> **Codename**: WealthPilot AI  
> **Target Audience**: Institutional PMS Providers, Wealth Management Firms, SEBI Compliance Auditors, Quantitative Engineers  
> **Jurisdiction**: Indian Capital Markets (NSE / BSE / SEBI PMS Regulations)

---

## 1. Executive Summary & Architectural Principles

**WealthPilot AI** is an enterprise autonomous multi-agent quantitative framework designed to monitor, optimize, execute, and explain portfolio rebalancing decisions across large investor universes (**50,000+ client accounts**). 

The platform operates on six core architectural principles:
1. **Mathematical Rigor & Convexity**: All allocation solutions are formulated as convex quadratic programs (QP) solved via CVXPY with guaranteed convergence, zero local-minima traps, and hard constraint enforcement.
2. **Sub-Second Vectorized Telemetry**: Real-time universe drift monitoring is vectorized via contiguous NumPy arrays, scanning 50,000 portfolios in <0.85 seconds.
3. **Institutional Tax-Awareness**: Optimized under the Indian Income Tax Act (Sections 111A and 112A), utilizing lot-level HIFO depletion, ₹1.25 Lakh annual LTCG exemptions, and 30-day wash-sale avoidance substitutions.
4. **Surrogate Explainability by Construction**: Rebalancing decisions are grounded in game-theoretic Tree SHAP values, LIME local surrogates, and counterfactuals, eliminating LLM numerical hallucinations.
5. **Human-in-the-Loop Governance**: A 4-tier graduated intervention model routes low-risk trades to automated straight-through processing while preserving explicit advisor sign-off for high-impact allocations.
6. **Regulatory Compliance as Code**: Tamper-evident SHA-256 digital seals, stratified quarterly audits, and algorithmic bias detectors guarantee adherence to SEBI Circular `SEBI/HO/MRD/DOP1/CIR/P/2024/69`.

---

## 2. End-to-End System Topology & Data Flow

```mermaid
flowchart TD
    subgraph DataLayer [1. Data & Universe Ingestion]
        PFS[50,000 Investor Portfolios]
        MKT[Real-Time NSE / BSE Market Ticks]
        TAX[Tax Lot Ledgers & Acquisition Dates]
    end

    subgraph MonitoringLayer [2. Vectorized Drift Monitoring Engine]
        VEC[Vectorized NumPy Drift Scanner<br/>50k accounts in <0.85s]
        SAD[SAD & Maximum Absolute Drift Matrix]
        THRESH[Volatility-Adjusted Dynamic Corridors]
        VEC --> SAD
        SAD --> THRESH
    end

    subgraph TriggerLayer [3. Three-Tier Trigger Consolidation]
        T1[Tier 1: Threshold & Concentration Drift]
        T2[Tier 2: Calendar & FY-End Windows]
        T3[Tier 3: Cash Deposits & Market Shocks]
        PRIO[Dynamic Priority Queue<br/>Severity * Log AUM]
        T1 --> PRIO
        T2 --> PRIO
        T3 --> PRIO
    end

    subgraph AgentSwarm [4. CrewAI Specialist Agent Swarm]
        ORCH[Orchestrator Agent<br/>SharedMemoryStore & Retries]
        ANALYST[Portfolio Analyst<br/>CVXPY QP Optimizer]
        TAX_SPEC[Tax Specialist<br/>HIFO Lots & Wash-Sale]
        RISK_MGR[Risk Manager<br/>VaR & Stress Testing]
        COMP_OFF[Compliance Officer<br/>SEBI Pre-Trade Caps]
        WRITER[Explanation Writer<br/>Multi-Tier Narratives]

        ORCH --> ANALYST
        ANALYST --> TAX_SPEC
        TAX_SPEC --> RISK_MGR
        RISK_MGR --> COMP_OFF
        COMP_OFF --> WRITER
        WRITER --> ORCH
    end

    subgraph GovernanceLayer [5. Governance, Overrides & OMS]
        TIER[4-Tier Intervention Classifier]
        KS[Circuit Breaker Kill-Switch]
        ADV[Advisor Approval Portal]
        OMS[Execution OMS / Broker Routing]

        ORCH --> TIER
        TIER -->|Tier 1 Auto| OMS
        TIER -->|Tier 2-3 Review| ADV
        ADV -->|Approve| OMS
        KS -.->|Halt Signal| OMS
    end

    subgraph ExplainabilityLayer [6. Multi-Audience Explainability & Audit]
        SHAP_ENG[Tree SHAP & LIME Surrogate]
        CLT_EXP[Client Plain-Language Portal]
        ADV_DOS[Advisor Technical Dossier]
        SEBI_AUD[SEBI Compliance Ledger & SHA-256 Seal]

        WRITER --> SHAP_ENG
        SHAP_ENG --> CLT_EXP
        SHAP_ENG --> ADV_DOS
        SHAP_ENG --> SEBI_AUD
    end

    PFS --> VEC
    MKT --> VEC
    TAX --> TAX_SPEC
    THRESH --> T1
```

---

## 3. Mathematical Optimization & Tax Harvesting

### 3.1 Convex Quadratic Programming Formulation
The core rebalancing decision is formulated as a constrained quadratic program:

$$\min_{\mathbf{w}_{\text{trade}}} \quad (\mathbf{w}_{\text{curr}} + \mathbf{w}_{\text{trade}} - \mathbf{w}_{\text{targ}})^T \boldsymbol{\Sigma} (\mathbf{w}_{\text{curr}} + \mathbf{w}_{\text{trade}} - \mathbf{w}_{\text{targ}}) + \lambda \|\mathbf{w}_{\text{trade}}\|_1$$

Subject to hard constraints:
1. **Cash Flow / Budget Balance**: $\sum_{i=1}^n w_{\text{trade}, i} = \Delta \text{cash}$
2. **Long-Only Requirement**: $w_{\text{curr}, i} + w_{\text{trade}, i} \ge 0 \quad \forall i$
3. **Turnover Budget**: $\frac{1}{2} \sum_{i=1}^n |w_{\text{trade}, i}| \le \tau_{\max}$
4. **SEBI Single-Issuer Limit**: $w_{\text{curr}, i} + w_{\text{trade}, i} \le 0.10$
5. **SEBI Sector Concentration**: $\sum_{i \in \text{Sector}} (w_{\text{curr}, i} + w_{\text{trade}, i}) \le 0.30$
6. **Minimum Cash Buffer**: $w_{\text{curr}, \text{cash}} + w_{\text{trade}, \text{cash}} \ge 0.02$

### 3.2 Indian Tax Optimization (Budget 2024 Framework)
- **Section 111A (STCG)**: Equities held $< 365$ days taxed at flat **20.0%**.
- **Section 112A (LTCG)**: Equities held $\ge 365$ days taxed at **12.5%** with ₹1,25,000 annual exemption.
- **Lot Selection (TAX_MINIMIZER)**: Depletes STCL lots first $\to$ LTCL lots $\to$ LTCG exempt lots $\to$ deferring STCG gains.
- **Wash-Sale Proxy Routing**: 30-day avoidance corridors reroute buy orders into highly correlated proxy ETFs (e.g., `NIFTY_50_EQUITY` $\to$ `NIFTY_NEXT_50_EQUITY`).

---

## 4. Multi-Agent Orchestration Topology

```mermaid
sequenceDiagram
    autonumber
    participant Trig as Trigger Consolidator
    participant Orch as Orchestrator Agent
    participant Analyst as Portfolio Analyst
    participant Tax as Tax Specialist
    participant Risk as Risk Manager
    participant Comp as Compliance Officer
    participant Writer as Explanation Writer
    participant Gov as Intervention Classifier

    Trig->>Orch: Submit Prioritized Portfolio Batch
    Orch->>Analyst: Request Target Weight Rebalance (CVXPY QP)
    Analyst-->>Orch: Candidate Trade Weights (w_trade)
    Orch->>Tax: Evaluate Tax Lots & Harvest Opportunities
    Tax-->>Orch: Lot Depletion, STCG/LTCG, Wash-Sale Substitutions
    Orch->>Risk: Stress Test Trade Plan & Compute Pre/Post VaR
    Risk-->>Orch: Parametric 95% VaR, Tracking Error, Shock Profiles
    Orch->>Comp: Verify SEBI Concentration & Mandate Suitability
    Comp-->>Orch: Statutory Compliance Certification (PASS)
    Orch->>Writer: Generate Multi-Audience Dossier & SHAP Waterfall
    Writer-->>Orch: Client, Advisor, and Compliance Explanations
    Orch->>Gov: Classify Intervention Tier (Tier 1 to 4)
    Gov-->>Orch: Route to OMS (Tier 1) or Advisor Queue (Tier 2-4)
```

---

## 5. Governance & Circuit Breakers

The system implements a **4-Tier Graduated Intervention Model**:
- **Tier 1 (Informational)**: Turnover $\le 5\%$, Confidence $\ge 0.90$ $\implies$ Auto-executed.
- **Tier 2 (Advisory)**: Turnover 5–15%, Confidence 0.75–0.90 $\implies$ 4-hour delay window.
- **Tier 3 (Approval Required)**: Turnover $> 15\%$, Value $\ge$ ₹50 Lakh $\implies$ Explicit advisor sign-off.
- **Tier 4 (Escalation)**: Mandate breaches, SAD $> 15\%$ $\implies$ CRO / Investment Committee review.

**Emergency Kill-Switch Circuit Breaker**:
- Trips automatically when **India VIX $\ge 40.0$**, **execution error rate $\ge 1.0\%$**, or **single-day drop $\le -5.0\%$**.
- Freezes autonomous trading, isolates affected portfolios, and transmits incident alerts.

---

## 6. Regulatory Verification & Bias Mitigation

- **SEBI Circular Alignment**: Built to comply with `SEBI/HO/MRD/DOP1/CIR/P/2024/69` (Algorithmic Trading & Pre-Trade Risk Controls).
- **Explainability Scorecard**: Numerically verifies accuracy, completeness, readability (Flesch-Kincaid Grade $\le 8.0$), and regulatory sufficiency.
- **Algorithmic Bias Detector**: Audits decision logs for risk-profile frequency parity, cost underestimation neutrality, AUM tier disparity, and contrarian discipline.
