"""Explainability Centre - Multi-Audience Narratives, SHAP Waterfall & Counterfactuals."""

from __future__ import annotations

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="Explainability Centre | WealthPilot AI", page_icon="🧠", layout="wide")

st.title("🧠 Explainability Centre & Decision Repository")
st.markdown("Searchable repository of AI-generated rebalancing explanations tailored across **Client**, **Advisor**, and **SEBI Compliance** tiers, featuring SHAP feature attributions and counterfactual reasoning.")

# Filter Bar
col_f1, col_f2, col_f3 = st.columns(3)
with col_f1:
    audience_filter = st.selectbox("Target Audience:", ["Client (Plain Language)", "Advisor (Technical & QP)", "Compliance (SEBI Regulatory)"])
with col_f2:
    trigger_filter = st.selectbox("Trigger Type:", ["All Triggers", "Threshold Drift", "Calendar Rebalance", "Event / Cash Flow"])
with col_f3:
    search_query = st.text_input("Search Portfolio / Keyword:", placeholder="e.g. PORT-00104, IT Sector, Capital Gains")

st.divider()

# Sample Explanation Records
explanations = [
    {
        "id": "EXP-2026-881",
        "portfolio_id": "PORT-00104",
        "investor": "Arjun Mehta",
        "risk_profile": "Balanced",
        "trigger": "Threshold Drift",
        "drift_amount": "+11.0% Equity",
        "client_text": """
        **Why We Rebalanced Your Portfolio Today:**  
        Over the past three months, your equity investments have grown rapidly, which means stocks now make up **68%** of your total portfolio instead of your target **50%**.  
        While this market growth is great news, leaving your portfolio at 68% stock exposure would expose you to significantly higher volatility than your agreed Balanced profile.  
        We trimmed a portion of your equity profits and locked them safely into high-grade Government Bonds. Because we selectively sold your longest-held shares, **you incurred ₹0 in short-term capital gains tax**!
        """,
        "advisor_text": """
        **Quantitative Execution & Solver Formulation:**  
        • **Trigger:** Maximum absolute drift threshold breached on Large-Cap Equity (Δw = +0.110 vs policy weight 0.350).  
        • **CVXPY Solver:** Quadratic tracking error minimized subject to Σ|Δw_i| ≤ 0.15 turnover budget. Target Tracking Error reduced from 3.82% to 0.44% p.a.  
        • **Tax Lot Selection:** HIFO + LTCG grandfathering applied across 4 discrete lots. Unrealized gains offset against Section 112A annual ₹1,25,000 threshold.  
        • **Execution Venue:** Routed 2 split limit orders to NSE lit order book at VWAP -0.015% execution slippage.
        """,
        "compliance_text": """
        **SEBI Algorithmic Suitability & Mandate Compliance:**  
        • **Regulatory Circular:** Adheres to SEBI/HO/MRD/DOP1/CIR/P/2024/69 for Automated Portfolio Management.  
        • **Suitability Verification:** Post-trade asset mix aligns with Investor Risk Category 'Balanced' (Equity range: 45%-55%, Actual: 50.0%).  
        • **Algorithmic Integrity:** QP solver converged within 14 iterations. Status: OPTIMAL. Zero wash-sale or circular transaction anomalies detected.  
        • **Digital Audit Seal:** SHA-256: `9f83c18b...a72d4` registered in immutable compliance ledger.
        """,
        "shap_features": {
            "Large-Cap Drift (+11.0%)": +0.48,
            "Days Since Last Rebalance (139d)": +0.22,
            "LTCG Exemption Room Available": +0.15,
            "Portfolio Volatility Spike": +0.08,
            "Bid-Ask Spread Frictions": -0.09,
        },
    },
    {
        "id": "EXP-2026-882",
        "portfolio_id": "PORT-01923",
        "investor": "Rajesh Sharma",
        "risk_profile": "Conservative",
        "trigger": "Event / Cash Flow",
        "drift_amount": "₹10 Lakh Fresh Deposit",
        "client_text": """
        **How Your New Deposit Was Invested:**  
        We received your ₹10,00,000 deposit. Rather than leaving it sitting in low-yield cash, our AI immediately allocated it across safe sovereign debt (₹5 Lakh), AAA corporate bonds (₹3 Lakh), and index equities (₹2 Lakh).  
        This restores your exact target conservative asset mix with zero transaction fee drag and no tax consequences.
        """,
        "advisor_text": """
        **Cash Infusion Optimization:**  
        • **Deposit Allocation:** Constrained QP solved with net inflow constraint (1^T Δw = +₹1,000,000).  
        • **Frictions:** Single-pass buy orders placed without liquidating existing holdings, eliminating capital gains crystallization.  
        • **Duration Matching:** Blended portfolio duration maintained at 4.2 years matching benchmark SBI G-Sec Index.
        """,
        "compliance_text": """
        **SEBI Mandate Adherence:**  
        • **Deposit Tracking:** Inflow validated against AML/PMLA thresholds. Source account verified.  
        • **Suitability:** Conservative mandate strict debt cap (min 70% fixed income) satisfied at 74.0%.
        """,
        "shap_features": {
            "Cash Drag Severity (12.2%)": +0.55,
            "Deposit Size vs AUM": +0.32,
            "Fixed Income Yield Opportunity": +0.18,
            "Execution Slippage Risk": -0.05,
        },
    },
]

# Filter logic
displayed_records = [
    r for r in explanations
    if (trigger_filter == "All Triggers" or r["trigger"] == trigger_filter)
    and (search_query.lower() in r["portfolio_id"].lower() or search_query.lower() in r["investor"].lower() or not search_query)
]

col_repo, col_shap = st.columns([3, 2])

with col_repo:
    st.subheader(f"📑 Explanation Records ({len(displayed_records)} Available)")
    if not displayed_records:
        st.warning("No explanations match your filter criteria.")
    
    for rec in displayed_records:
        with st.container(border=True):
            st.markdown(f"#### {rec['id']} — {rec['portfolio_id']} ({rec['investor']})")
            st.caption(f"Risk: **{rec['risk_profile']}** | Trigger: **{rec['trigger']}** | Drift: **{rec['drift_amount']}**")
            
            if "Client" in audience_filter:
                st.markdown(rec["client_text"])
            elif "Advisor" in audience_filter:
                st.markdown(rec["advisor_text"])
            else:
                st.markdown(rec["compliance_text"])

with col_shap:
    st.subheader("🔍 SHAP Feature Attribution Waterfall")
    st.caption("Visualizing the mathematical factors driving the autonomous rebalancing decision.")
    
    selected_rec = displayed_records[0] if displayed_records else explanations[0]
    shap_data = selected_rec["shap_features"]
    
    factors = list(shap_data.keys())
    impacts = list(shap_data.values())
    
    fig_shap = go.Figure(go.Waterfall(
        name="SHAP Values",
        orientation="h",
        y=factors,
        x=impacts,
        decreasing={"marker": {"color": "#ef4444"}},
        increasing={"marker": {"color": "#10b981"}},
        connector={"line": {"color": "#64748b"}},
        text=[f"{v:+.2f}" for v in impacts],
        textposition="outside",
    ))
    
    fig_shap.update_layout(
        title=f"SHAP Decision Drivers: {selected_rec['portfolio_id']}",
        xaxis_title="Marginal SHAP Contribution (Log-Odds of Rebalance)",
        height=350,
        margin=dict(l=20, r=20, t=40, b=20),
    )
    st.plotly_chart(fig_shap, use_container_width=True)

st.divider()

# Section 3: Interactive Counterfactual Simulator
st.subheader("⚡ What-If Counterfactual Reasoning Engine")
st.caption("Simulate alternative market conditions to see if the AI agent would still trigger a rebalance.")

c_cf1, c_cf2, c_cf3 = st.columns(3)
with c_cf1:
    cf_drift = st.slider("Simulated Equity Drift (%):", min_value=0.0, max_value=15.0, value=7.5, step=0.5)
with c_cf2:
    cf_tax_cost = st.slider("Simulated Tax Drag (₹):", min_value=0, max_value=50000, value=8500, step=1000)
with c_cf3:
    cf_volatility = st.selectbox("Market Volatility Regime:", ["Low (VIX < 15)", "Normal (VIX 15-25)", "High (VIX > 25)"])

# Decision rule counterfactual evaluation
threshold_limit = 5.0
would_trigger = (cf_drift >= threshold_limit) and (cf_tax_cost < 30000)

if would_trigger:
    st.success(f"✅ **Rebalance Triggered**: At {cf_drift:.1f}% drift, tracking error penalty exceeds tax drag (₹{cf_tax_cost:,}). The agent executes cost-optimized rebalancing.")
else:
    st.info(f"⏸️ **Rebalance Suppressed (Opportunistic Hold)**: At {cf_drift:.1f}% drift, transaction/tax costs (₹{cf_tax_cost:,}) outweigh the diversification benefits. Portfolio remains in tolerance.")
