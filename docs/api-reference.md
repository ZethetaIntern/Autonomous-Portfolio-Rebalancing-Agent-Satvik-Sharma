# WealthPilot AI - Public API & Module Reference

Comprehensive technical contract and signature documentation for core production modules in `src/`.

---

## 1. Monitoring & Drift Detection (`src/monitoring/`)

### `DriftCalculator`
High-performance vectorized NumPy computation engine for portfolio drift metrics.
```python
class DriftCalculator:
    def calculate_drift_matrix(
        self,
        current_weights: np.ndarray,
        target_weights: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """Calculates signed drift, absolute drift, max drift, and SAD across N portfolios.
        
        Args:
            current_weights: (N, K) float64 array of current asset weights.
            target_weights: (N, K) float64 array of target strategic weights.
            
        Returns:
            Dict containing 'signed_drift', 'absolute_drift', 'max_drift', and 'sad'.
        """
```

### `DriftMonitor`
Orchestrates batch-level universe scanning, priority queueing, and drift breach alerts.
```python
class DriftMonitor:
    def scan_universe(
        self,
        portfolios: List[Portfolio],
        chunk_size: int = 10000,
    ) -> List[DriftBreachEvent]:
        """Scans up to 50,000 portfolios in chunks under 1.0 second SLA."""
```

---

## 2. Trigger Evaluation & Consolidation (`src/triggers/`)

### `TriggerConsolidator`
Consolidates multi-source events (drift breaches, calendar milestones, cash deposits, market shocks) into an integrated priority queue.
```python
class TriggerConsolidator:
    def consolidate_triggers(
        self,
        triggers: List[TriggerEvent],
    ) -> List[PrioritizedRebalanceTask]:
        """Ranks rebalancing candidates by urgency score: Severity * log10(AUM)."""
```

---

## 3. Mathematical Optimization & Tax Management (`src/optimisation/`)

### `PortfolioOptimiser`
Convex Quadratic Programming (QP) solver for constrained tracking error minimization.
```python
class PortfolioOptimiser:
    def optimize_allocation(
        self,
        current_weights: Union[np.ndarray, Dict[str, float]],
        target_weights: Union[np.ndarray, Dict[str, float]],
        turnover_budget: Optional[float] = None,
        net_cash_flow: float = 0.0,
        min_cash_buffer: float = 0.02,
        sebi_issuer_limit: Optional[float] = 0.10,
        sebi_sector_limits: Optional[Dict[str, float]] = None,
        solver: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Solves quadratic tracking error variance minimization under hard constraints."""
```

### `TaxOptimiser`
Calculates statutory capital gains tax under Sections 111A/112A and enforces wash-sale avoidance.
```python
class TaxOptimiser:
    def calculate_tax_liability(
        self,
        realized_stcg: float,
        realized_stcl: float = 0.0,
        realized_ltcg: float = 0.0,
        realized_ltcl: float = 0.0,
        claimed_ltcg_exemption: float = 0.0,
    ) -> Dict[str, float]:
        """Computes tax liability under Indian set-off and ₹1.25L exemption rules."""

    def evaluate_trade_plan_taxes(
        self,
        portfolio_id: str,
        orders: List[TradeOrder],
        lot_manager: TaxLotManager,
        strategy: str = "TAX_MINIMIZER",
        apply_wash_sale_substitutes: bool = True,
    ) -> Dict[str, Any]:
        """Generates lot depletion orders and substitutes wash-sale restricted buys."""
```

---

## 4. Multi-Agent Swarm (`src/agents/`)

### `OrchestratorAgent`
```python
class OrchestratorAgent:
    def process_rebalance_trigger(
        self,
        trigger_payload: Dict[str, Any],
        max_retries: int = 3,
    ) -> Dict[str, Any]:
        """Coordinates analyst, tax specialist, risk manager, and compliance officer."""
```

### `PortfolioAnalystAgent`, `TaxSpecialistAgent`, `RiskManagerAgent`, `ComplianceOfficerAgent`, `ExplanationWriterAgent`
All agents support:
- `execute(input_data: Dict[str, Any]) -> Dict[str, Any]`: High-speed deterministic execution (<50ms).
- `as_crewai_agent() -> Agent`: Instantiates a CrewAI `Agent` with role, goal, backstory, and LLM bindings.

---

## 5. Explainable AI Engine (`src/explainability/`)

### `ExplanationGenerator`
```python
class ExplanationGenerator:
    def generate_explanation_packet(
        self,
        portfolio_id: str,
        trade_plan: Dict[str, Any],
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Generates multi-tier explanations for Client, Advisor, and Compliance."""
```

### `ShapExplainer` & `LimeExplainer`
- `explain_portfolio(features: np.ndarray) -> Dict[str, float]`: Computes game-theoretic Shapley feature attributions.
- `explain_local_instance(features: np.ndarray) -> Dict[str, float]`: Fits local linear surrogate perturbation boundary.

---

## 6. Overrides, Governance & Circuit Breakers (`src/override/`)

### `InterventionClassifier`
```python
class InterventionClassifier:
    def classify_intervention(
        self,
        turnover_fraction: float,
        trade_value_inr: float,
        agent_confidence: float,
        sad_drift: float = 0.0,
    ) -> InterventionTier:
        """Categorizes proposed rebalances into Tier 1 (Informational) through Tier 4 (Escalation)."""
```

### `KillSwitch`
```python
class KillSwitch:
    def evaluate_automated_triggers(
        self,
        vix_level: Optional[float] = None,
        failed_executions: int = 0,
        total_executions: int = 0,
        daily_market_return: Optional[float] = None,
    ) -> Tuple[CircuitBreakerState, Optional[str]]:
        """Evaluates automated halt conditions: VIX >= 40, errors >= 1%, or crash <= -5%."""
```

---

## 7. Backtesting & Scenario Simulation (`src/backtesting/`)

### `BacktestEngine`
```python
class BacktestEngine:
    def run_simulation(
        self,
        price_history: pd.DataFrame,
        target_weights: Dict[str, float],
        rebalance_rule: str = "AI_AGENT", # "AI_AGENT", "CALENDAR", "THRESHOLD", "BUY_HOLD"
        drift_threshold: float = 0.05,
    ) -> Dict[str, Any]:
        """Simulates 252-day portfolio evolution tracking round-lot costs and taxes."""
```

### `ScenarioRunner`
```python
class ScenarioRunner:
    def run_all_scenarios(self) -> Dict[str, Any]:
        """Executes all 5 mandatory market stress scenarios: Normal Drift, Crash, Rotation, Regulatory, Tax."""
```

---

## 8. Compliance & Regulatory Auditing (`src/compliance/`)

### `ExplainabilityScorecard`
```python
class ExplainabilityScorecard:
    def score_explanation_packet(
        self,
        explanation_packet: Dict[str, Any],
        actual_trade_context: Optional[Dict[str, Any]] = None,
    ) -> ExplanationScorecardResult:
        """Scores explanations on Accuracy, Completeness, Readability, and Regulatory Sufficiency."""
```

### `BiasDetector`
```python
class BiasDetector:
    def analyze_decisions(
        self,
        decisions: List[Dict[str, Any]],
    ) -> BiasReport:
        """Audits decision patterns for statistical parity across risk profiles, costs, and AUM tiers."""
```

### `RegulatoryReporter`
```python
class RegulatoryReporter:
    def generate_sebi_audit_package(
        self,
        audit_result: Optional[AuditRunResult] = None,
        transaction_logs: Optional[List[Dict[str, Any]]] = None,
        reporting_period: str = "Q4-2024",
    ) -> Dict[str, Any]:
        """Compiles exportable audit packages citing SEBI Circular SEBI/HO/MRD/DOP1/CIR/P/2024/69."""
```
