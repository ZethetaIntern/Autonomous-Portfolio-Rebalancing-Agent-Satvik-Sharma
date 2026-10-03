# WealthPilot AI - Research & Exploration Notebooks

This directory contains research, exploration, and prototyping notebooks for WealthPilot AI:

1. `01_market_data_and_synthetic_portfolios.ipynb`:
   - Exploration of Indian capital markets asset classes (NIFTY 50, G-Sec, Corporate Bonds, Gold, Liquid cash).
   - Covariance estimation, Cholesky simulation, and synthetic portfolio distributions.

2. `02_drift_monitoring_and_sla_benchmarks.ipynb`:
   - NumPy vectorized drift calculation benchmarks across 50,000 portfolios.
   - Priority queue verification and heatmaps of asset drift.

3. `03_trigger_evaluation_and_consolidation.ipynb`:
   - Testing Tier 1 (Threshold), Tier 2 (Calendar), and Tier 3 (Event) rebalancing triggers.
   - Consolidation logic and audit trail generation.

4. `04_convex_optimization_and_tax_harvesting.ipynb`:
   - CVXPY quadratic programming formulation for tracking error vs turnover minimization.
   - Section 112A LTCG tax harvesting experiments.

5. `05_multi_agent_consensus_and_explainability.ipynb`:
   - CrewAI / LangChain multi-agent debate simulation.
   - SHAP and LIME feature attributions and counterfactual generation.
