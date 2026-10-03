"""Portfolio Overview Page - 50,000 Portfolios Drift Heatmap and Granular Drill-Downs."""

from __future__ import annotations

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(page_title="Portfolio Overview | WealthPilot AI", page_icon="📊", layout="wide")

st.title("📊 Portfolio Universe & Drift Heatmap")
st.markdown("Real-time monitoring of **50,000 active portfolios** across risk tiers, asset classes, and individual holding deviations.")

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Supervised Portfolios", "50,000", "100% Vector Monitored")
col2.metric("Total AUM", "₹14,250.80 Cr", "Avg ₹28.50 L")
col3.metric("Portfolios In-Tolerance", "45,210 (90.42%)", "Target >90%")
col4.metric("Moderate Drift (5-10%)", "3,948 (7.90%)", "Queued for Review")
col5.metric("Critical Drift (>10%)", "842 (1.68%)", "Priority 1 Rebalance", delta_color="inverse")

st.divider()

st.subheader("🔥 Aggregate Asset Class Drift Heatmap across Risk Bands")
st.caption("Average percentage point drift [Current Weight - Target Weight] across all 50,000 portfolios.")

risk_bands = ["Conservative", "Moderately Conservative", "Balanced", "Growth", "Aggressive"]
asset_classes = [
    "Large-Cap Equity",
    "Mid/Small-Cap Equity",
    "Govt Securities (G-Sec)",
    "Corporate Debt",
    "Gold ETF",
    "Liquid Cash",
]

drift_matrix = np.array([
    [+1.8, +0.6, -1.4, -0.7, +0.2, -0.5],
    [+3.1, +1.2, -2.1, -1.5, -0.3, -0.4],
    [+5.4, +2.8, -3.8, -2.6, -0.6, -1.2],
    [+6.9, +4.1, -4.5, -3.2, -1.1, -2.2],
    [+8.4, +5.5, -5.2, -3.9, -1.4, -3.4],
])

heatmap_fig = go.Figure(data=go.Heatmap(
    z=drift_matrix,
    x=asset_classes,
    y=risk_bands,
    colorscale="RdBu_r",
    zmid=0,
    text=np.char.add(np.char.mod('%+.2f', drift_matrix), '%'),
    texttemplate="%{text}",
    textfont={"size": 13, "color": "white"},
    colorbar=dict(title="Drift (% pts)", ticksuffix="%"),
))

heatmap_fig.update_layout(
    title="Aggregate Allocation Drift (% Points from Target Allocation)",
    xaxis_title="Asset Class",
    yaxis_title="Investor Risk Profile",
    height=380,
    margin=dict(l=40, r=40, t=50, b=40),
)
st.plotly_chart(heatmap_fig, use_container_width=True)

st.divider()

st.subheader("🔍 Individual Portfolio Holdings Drill-Down")

c_filter1, c_filter2 = st.columns([1, 2])
with c_filter1:
    sample_ids = [
        "PORT-00104 (Balanced - High Equity Drift)",
        "PORT-00428 (Aggressive - Critical Tech Drift)",
        "PORT-01923 (Conservative - Cash Infusion)",
        "PORT-08472 (Growth - Sector Imbalance)",
        "PORT-23910 (Balanced - In-Tolerance)",
    ]
    selected_item = st.selectbox("Select Portfolio to Inspect:", sample_ids)
    selected_id = selected_item.split()[0]

    portfolio_metadata = {
        "PORT-00104": {
            "name": "Arjun Mehta",
            "risk": "Balanced",
            "aum": 4500000.0,
            "tax_status": "Individual Resident",
            "last_rebalance": "2025-11-15 (139 days ago)",
            "drift_score": 6.8,
            "current": [0.46, 0.22, 0.16, 0.08, 0.04, 0.04],
            "target": [0.35, 0.15, 0.25, 0.15, 0.05, 0.05],
        },
        "PORT-00428": {
            "name": "Pooja Hegde",
            "risk": "Aggressive",
            "aum": 12800000.0,
            "tax_status": "HNI / Family Office",
            "last_rebalance": "2025-09-10 (205 days ago)",
            "drift_score": 11.4,
            "current": [0.58, 0.31, 0.04, 0.03, 0.02, 0.02],
            "target": [0.45, 0.25, 0.12, 0.08, 0.05, 0.05],
        },
        "PORT-01923": {
            "name": "Rajesh Sharma",
            "risk": "Conservative",
            "aum": 8200000.0,
            "tax_status": "Senior Citizen",
            "last_rebalance": "2026-01-20 (73 days ago)",
            "drift_score": 5.2,
            "current": [0.18, 0.04, 0.44, 0.20, 0.04, 0.10],
            "target": [0.15, 0.05, 0.50, 0.20, 0.05, 0.05],
        },
        "PORT-08472": {
            "name": "Siddharth Nair",
            "risk": "Growth",
            "aum": 3100000.0,
            "tax_status": "Individual Resident",
            "last_rebalance": "2025-12-05 (119 days ago)",
            "drift_score": 7.3,
            "current": [0.48, 0.28, 0.11, 0.06, 0.03, 0.04],
            "target": [0.40, 0.20, 0.20, 0.10, 0.05, 0.05],
        },
        "PORT-23910": {
            "name": "Kavita Rao",
            "risk": "Balanced",
            "aum": 5400000.0,
            "tax_status": "NRI",
            "last_rebalance": "2026-02-18 (44 days ago)",
            "drift_score": 1.9,
            "current": [0.36, 0.16, 0.24, 0.14, 0.05, 0.05],
            "target": [0.35, 0.15, 0.25, 0.15, 0.05, 0.05],
        },
    }

    meta = portfolio_metadata[selected_id]
    st.markdown(f"""
    **Investor:** {meta['name']}  
    **Risk Profile:** `{meta['risk']}`  
    **Portfolio AUM:** ₹{meta['aum']:,.2f}  
    **Tax Category:** {meta['tax_status']}  
    **Last Rebalanced:** {meta['last_rebalance']}  
    **Max Absolute Drift:** **{meta['drift_score']:.1f}%**
    """)

with c_filter2:
    current_w = meta["current"]
    target_w = meta["target"]
    drift_w = [c - t for c, t in zip(current_w, target_w)]

    bar_df = pd.DataFrame({
        "Asset Class": asset_classes,
        "Current Weight (%)": [c * 100 for c in current_w],
        "Target Weight (%)": [t * 100 for t in target_w],
        "Drift (% pts)": [d * 100 for d in drift_w],
    })

    fig_bars = go.Figure()
    fig_bars.add_trace(go.Bar(
        x=bar_df["Asset Class"],
        y=bar_df["Current Weight (%)"],
        name="Current Weight",
        marker_color="#38bdf8",
    ))
    fig_bars.add_trace(go.Bar(
        x=bar_df["Asset Class"],
        y=bar_df["Target Weight (%)"],
        name="Target Policy Weight",
        marker_color="#94a3b8",
    ))

    fig_bars.update_layout(
        title=f"Holdings Allocation Comparison: {selected_id}",
        barmode="group",
        yaxis_title="Allocation (%)",
        height=320,
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig_bars, use_container_width=True)

st.markdown("#### Granular Asset Drift & Proposed Action")
table_data = []
for i, asset in enumerate(asset_classes):
    c_val = meta["aum"] * current_w[i]
    t_val = meta["aum"] * target_w[i]
    diff_val = t_val - c_val
    action = "BUY" if diff_val > 0 else "SELL" if diff_val < 0 else "HOLD"
    table_data.append({
        "Asset Class": asset,
        "Current Weight": f"{current_w[i]*100:.2f}%",
        "Target Weight": f"{target_w[i]*100:.2f}%",
        "Drift": f"{drift_w[i]*100:+.2f}%",
        "Current Value (INR)": f"₹{c_val:,.0f}",
        "Target Value (INR)": f"₹{t_val:,.0f}",
        "Trade Adjustment": f"{action} ₹{abs(diff_val):,.0f}",
        "Status": "⚠️ Out of Tolerance" if abs(drift_w[i]) >= 0.05 else "✅ In Tolerance",
    })

st.dataframe(pd.DataFrame(table_data), use_container_width=True)
