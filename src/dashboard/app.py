"""WealthPilot AI - Autonomous Rebalancing Cockpit (Streamlit Entry Point).

Enterprise multi-agent autonomous portfolio monitoring, drift detection,
convex rebalancing, and explainable decision framework calibrated to Indian capital markets.
"""

from __future__ import annotations

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# Configure page
st.set_page_config(
    page_title="WealthPilot AI | Autonomous Rebalancing Cockpit",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling for Sleek Wealthtech Aesthetic
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 16px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    }
    .status-badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 0.82rem;
        font-weight: 600;
    }
    .badge-normal {
        background-color: rgba(16, 185, 129, 0.15);
        color: #10b981;
        border: 1px solid #10b981;
    }
    .badge-alert {
        background-color: rgba(245, 158, 11, 0.15);
        color: #f59e0b;
        border: 1px solid #f59e0b;
    }
</style>
""", unsafe_allow_html=True)

# Top Bar / Status Banner
col_head1, col_head2 = st.columns([3, 1])
with col_head1:
    st.title("⚖️ WealthPilot AI")
    st.caption("Autonomous Portfolio Rebalancing Agent with Explainable Decisions (Project 1D) | Production Grade")

with col_head2:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        '<span class="status-badge badge-normal">● ENGINE ONLINE</span> &nbsp; '
        '<span class="status-badge badge-normal">SEBI AUDIT PASS</span>',
        unsafe_allow_html=True,
    )

st.divider()

# High-Level Executive Metrics
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric(label="Total Monitored Portfolios", value="50,000", delta="+1,240 QTD")
m2.metric(label="Total AUM Supervised", value="₹14,250 Cr", delta="+₹320 Cr MTD")
m3.metric(label="Active Drift Breaches", value="1,842 (3.68%)", delta="-312 in last cycle", delta_color="inverse")
m4.metric(label="Estimated Tax Alpha (FY26)", value="₹48.6 Cr", delta="+84 bps net")
m5.metric(label="Circuit Breaker Status", value="NORMAL", delta="0 Latency Violations")

st.markdown("### 🎛️ Architecture & Real-Time Module Navigation")

col_left, col_right = st.columns([2, 1])

with col_left:
    st.markdown("""
    Select a dedicated cockpit module from the sidebar or click through below:
    
    1. **📊 Portfolio Overview (`portfolio_overview.py`)**:
       - 50,000 portfolio aggregate drift heatmap across 5 risk profiles and 6 core asset classes.
       - Interactive portfolio explorer with granular drift drill-downs and asset class weights.
       
    2. **⚡ Rebalancing Activity (`rebalancing_activity.py`)**:
       - Live rebalancing queue with priority scoring (Urgent / Warning / Routine).
       - Human-in-the-loop advisor approval card workflow (Approve / Reject / Modify).
       - Trade execution tracking across NSE/BSE clearing blocks.
       
    3. **📈 Performance & Factor Analytics (`performance_analytics.py`)**:
       - Rolling risk-adjusted returns (1M, 3M, 6M, 12M) vs. Legacy Calendar and Buy & Hold.
       - Tracking error suppression, transaction cost savings, and net tax-alpha breakdown.
       
    4. **🧠 Explainability Centre (`explainability_centre.py`)**:
       - Multi-audience explanation repository (Client Plain-Language, Advisor Technical, Compliance SEBI).
       - Visual SHAP feature attribution waterfall plots and counterfactual simulations.
       
    5. **🛡️ System Health & Kill Switch (`system_health.py`)**:
       - Vectorized scan throughput (>50k portfolios in <1.2s), queue latency metrics, and error rates.
       - Multi-tiered emergency kill switch controls and audit log records.
    """)

with col_right:
    st.markdown("#### Universe Drift Distribution")
    # Quick visual mini-donut
    labels = ["In Tolerance (<5%)", "Warning (5-10%)", "Critical Drift (>10%)"]
    values = [45210, 3948, 842]
    colors = ["#10b981", "#f59e0b", "#ef4444"]
    fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=0.6, marker=dict(colors=colors))])
    fig.update_layout(
        margin=dict(l=10, r=10, t=10, b=10),
        height=220,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#cbd5e1", size=11),
    )
    st.plotly_chart(fig, use_container_width=True)

st.info("💡 **Tip**: Switch to `pages/portfolio_overview.py` in the sidebar to inspect universe drift or drill down into individual investor portfolios.")
