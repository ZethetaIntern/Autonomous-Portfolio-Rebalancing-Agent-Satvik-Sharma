"""Rebalancing Activity Cockpit - Live Decision Queue, Advisor Workflow & Trade Tracking."""

from __future__ import annotations

import streamlit as st
import pandas as pd
import numpy as np
import datetime
import plotly.express as px

st.set_page_config(page_title="Rebalancing Activity | WealthPilot AI", page_icon="⚡", layout="wide")

st.title("⚡ Rebalancing Activity & Execution Queue")
st.markdown("Live agent decision queue, human-in-the-loop advisor approval portal, and trade execution tracking.")

# Session state initialization for interactive approvals
if "decision_status" not in st.session_state:
    st.session_state.decision_status = {
        "DEC-2026-0091": "Pending Advisor Approval",
        "DEC-2026-0092": "Pending Advisor Approval",
        "DEC-2026-0093": "Approved (Queued)",
        "DEC-2026-0094": "Pending Advisor Approval",
        "DEC-2026-0095": "Auto-Executed (Direct Tier)",
    }

# Top KPI ribbon
q1, q2, q3, q4 = st.columns(4)
pending_count = sum(1 for s in st.session_state.decision_status.values() if "Pending" in s)
q1.metric("Live Queue Size", "28 Decisions", f"{pending_count} Require Review")
q2.metric("Executed Today", "142 Portfolios", "₹41.8 Cr Traded")
q3.metric("Avg Trade Slippage", "1.8 bps", "-0.4 bps vs NSE VWAP", delta_color="inverse")
q4.metric("Advisor Approval SLA", "4.2 mins", "SEBI Benchmark <15m")

st.divider()

# Tab layout: 1. Live Decision Queue & Approval, 2. Trade Execution Tracking
tab_queue, tab_execution = st.tabs(["📋 Live Decision Queue & Advisor Approvals", "🚀 Trade Execution Tracking"])

with tab_queue:
    st.subheader("1. Pending Advisor Approval Cards")
    st.caption("High-AUM or critical threshold breaches requiring explicit human-in-the-loop sign-off.")

    sample_decisions = [
        {
            "id": "DEC-2026-0091",
            "portfolio": "PORT-00428 (Pooja Hegde)",
            "aum": "₹1.28 Cr",
            "risk": "Aggressive",
            "trigger": "Threshold Breach (+11.4% Mid-Cap)",
            "priority": "🔴 Urgent",
            "proposed_trades": [
                {"Symbol": "NIFTY_MIDCAP_ETF", "Action": "SELL", "Units": 4200, "Est_Price": 142.50, "Val_INR": 598500, "Tax": "LTCG (₹0 Net)"},
                {"Symbol": "GSEC_10Y_ETF", "Action": "BUY", "Units": 3800, "Est_Price": 105.20, "Val_INR": 399760, "Tax": "N/A"},
                {"Symbol": "LIQUID_BEES", "Action": "BUY", "Units": 1980, "Est_Price": 1000.00, "Val_INR": 198740, "Tax": "N/A"},
            ],
            "rationale": "Mid-cap allocation grew to 31.0% against 25.0% mandate due to outperformance. Trim mid-caps, re-allocate to G-Sec and liquid reserves while avoiding STCG via grandfathered lots.",
        },
        {
            "id": "DEC-2026-0092",
            "portfolio": "PORT-01923 (Rajesh Sharma)",
            "aum": "₹82.0 L",
            "risk": "Conservative",
            "trigger": "Cash Infusion (+₹10.0 L Deposit)",
            "priority": "🟡 Warning",
            "proposed_trades": [
                {"Symbol": "GSEC_10Y_ETF", "Action": "BUY", "Units": 4800, "Est_Price": 105.20, "Val_INR": 504960, "Tax": "N/A"},
                {"Symbol": "CORP_BOND_AAA", "Action": "BUY", "Units": 2800, "Est_Price": 108.50, "Val_INR": 303800, "Tax": "N/A"},
                {"Symbol": "NIFTY_BEES", "Action": "BUY", "Units": 750, "Est_Price": 255.40, "Val_INR": 191550, "Tax": "N/A"},
            ],
            "rationale": "Fresh cash deposit diluted target bond allocation. Rebalance without selling existing equity to preserve long-term compounding.",
        },
        {
            "id": "DEC-2026-0094",
            "portfolio": "PORT-08472 (Siddharth Nair)",
            "aum": "₹31.0 L",
            "risk": "Growth",
            "trigger": "Quarterly Calendar & Tax Harvesting",
            "priority": "🟢 Routine",
            "proposed_trades": [
                {"Symbol": "BANKBEES", "Action": "SELL", "Units": 600, "Est_Price": 485.00, "Val_INR": 291000, "Tax": "Loss Harvest (-₹24,500)"},
                {"Symbol": "NIFTY_BEES", "Action": "BUY", "Units": 1140, "Est_Price": 255.40, "Val_INR": 291156, "Tax": "N/A"},
            ],
            "rationale": "Harvest unrealized banking sector short-term losses to offset net capital gains before March 31, rotating into broad Nifty 50 basket.",
        },
    ]

    for item in sample_decisions:
        dec_id = item["id"]
        status = st.session_state.decision_status.get(dec_id, "Pending")
        
        with st.expander(f"{item['priority']} | **{dec_id}** — {item['portfolio']} | {item['trigger']} | Status: `{status}`", expanded=(status == "Pending Advisor Approval")):
            c_meta1, c_meta2 = st.columns([2, 1])
            with c_meta1:
                st.markdown(f"**AUM:** {item['aum']} &nbsp;|&nbsp; **Risk Profile:** `{item['risk']}`")
                st.markdown(f"**AI Strategy Rationale:** {item['rationale']}")
            with c_meta2:
                st.markdown(f"**Decision Status:** `{status}`")

            st.markdown("##### Proposed Execution Package")
            st.dataframe(pd.DataFrame(item["proposed_trades"]), use_container_width=True)

            if status == "Pending Advisor Approval":
                b1, b2, b3 = st.columns([1, 1, 2])
                with b1:
                    if st.button("✅ Approve Rebalance", key=f"app_{dec_id}", type="primary"):
                        st.session_state.decision_status[dec_id] = "Approved (Sent to OMS)"
                        st.success(f"Decision {dec_id} approved and transmitted to Execution OMS!")
                        st.rerun()
                with b2:
                    if st.button("❌ Reject / Cancel", key=f"rej_{dec_id}"):
                        st.session_state.decision_status[dec_id] = "Rejected by Advisor"
                        st.warning(f"Decision {dec_id} rejected. Reason logged to compliance journal.")
                        st.rerun()
                with b3:
                    override_note = st.text_input("Override Notes / Custom Constraints:", key=f"note_{dec_id}", placeholder="Optional notes for compliance audit")
            else:
                st.info(f"Decision state locked: **{status}**")

with tab_execution:
    st.subheader("2. Real-Time Trade Execution Tracking (NSE / BSE)")
    st.caption("Active orders clearing through primary broker and Indian custodian clearing corporations (NCL/ICCL).")

    exec_records = [
        {"Trade ID": "TRD-88412", "Portfolio": "PORT-00104", "Symbol": "NIFTY_BEES", "Side": "SELL", "Qty": 1250, "Limit (INR)": 255.50, "Executed (INR)": 255.45, "Slippage": "-0.02%", "Venue": "NSE", "Status": "COMPLETED (T+1)"},
        {"Trade ID": "TRD-88413", "Portfolio": "PORT-00104", "Symbol": "GSEC_10Y_ETF", "Side": "BUY", "Qty": 3000, "Limit (INR)": 105.30, "Executed (INR)": 105.28, "Slippage": "-0.01%", "Venue": "NSE", "Status": "COMPLETED (T+1)"},
        {"Trade ID": "TRD-88414", "Portfolio": "PORT-00219", "Symbol": "JUNIORBEES", "Side": "SELL", "Qty": 850, "Limit (INR)": 642.00, "Executed (INR)": 641.80, "Slippage": "-0.03%", "Venue": "BSE", "Status": "COMPLETED (T+1)"},
        {"Trade ID": "TRD-88415", "Portfolio": "PORT-00219", "Symbol": "LIQUID_BEES", "Side": "BUY", "Qty": 540, "Limit (INR)": 1000.00, "Executed (INR)": 1000.00, "Slippage": "0.00%", "Venue": "NSE", "Status": "COMPLETED (T+1)"},
        {"Trade ID": "TRD-88416", "Portfolio": "PORT-00384", "Symbol": "GOLDBEES", "Side": "BUY", "Qty": 2100, "Limit (INR)": 62.40, "Executed (INR)": 62.38, "Slippage": "-0.03%", "Venue": "NSE", "Status": "ROUTING"},
    ]

    st.dataframe(pd.DataFrame(exec_records), use_container_width=True)

    col_slippage, col_venues = st.columns(2)
    with col_slippage:
        st.markdown("#### Execution Slippage vs NSE VWAP Benchmark")
        slippages = np.random.normal(loc=-0.018, scale=0.01, size=200)
        fig_slip = px.histogram(
            x=slippages,
            nbins=30,
            title="Distribution of Order Slippage (Negative = Price Improvement)",
            labels={"x": "Slippage (% Points)"},
            color_discrete_sequence=["#10b981"],
        )
        fig_slip.update_layout(height=260, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_slip, use_container_width=True)

    with col_venues:
        st.markdown("#### Routing Venue Fill Breakdown")
        fig_venue = px.pie(
            values=[72, 28],
            names=["NSE (National Stock Exchange)", "BSE (Bombay Stock Exchange)"],
            title="Exchange Volume Share",
            color_discrete_sequence=["#38bdf8", "#818cf8"],
            hole=0.5,
        )
        fig_venue.update_layout(height=260, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_venue, use_container_width=True)
