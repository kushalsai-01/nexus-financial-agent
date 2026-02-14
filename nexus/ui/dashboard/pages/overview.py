from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


def _generate_demo_portfolio() -> dict[str, Any]:
    np.random.seed(42)
    dates = pd.date_range(end=datetime.now(), periods=252, freq="B")
    values = [1_000_000.0]
    for _ in range(251):
        ret = np.random.normal(0.0004, 0.012)
        values.append(values[-1] * (1 + ret))

    positions = [
        {"ticker": "AAPL", "quantity": 150, "entry_price": 178.50, "current_price": 195.20, "pnl": 2505.0, "pnl_pct": 9.36, "weight": 12.5},
        {"ticker": "MSFT", "quantity": 80, "entry_price": 380.00, "current_price": 415.30, "pnl": 2824.0, "pnl_pct": 9.29, "weight": 14.2},
        {"ticker": "GOOGL", "quantity": 60, "entry_price": 140.25, "current_price": 155.80, "pnl": 933.0, "pnl_pct": 11.09, "weight": 4.0},
        {"ticker": "TSLA", "quantity": 100, "entry_price": 245.00, "current_price": 232.50, "pnl": -1250.0, "pnl_pct": -5.10, "weight": 9.9},
        {"ticker": "NVDA", "quantity": 40, "entry_price": 680.00, "current_price": 780.00, "pnl": 4000.0, "pnl_pct": 14.71, "weight": 13.3},
        {"ticker": "JPM", "quantity": 120, "entry_price": 175.00, "current_price": 188.50, "pnl": 1620.0, "pnl_pct": 7.71, "weight": 9.7},
        {"ticker": "SPY", "quantity": 200, "entry_price": 470.00, "current_price": 485.30, "pnl": 3060.0, "pnl_pct": 3.26, "weight": 41.5},
    ]

    return {
        "dates": dates,
        "values": values,
        "positions": positions,
        "total_value": values[-1],
        "daily_pnl": values[-1] - values[-2],
        "daily_pnl_pct": (values[-1] - values[-2]) / values[-2] * 100,
        "total_pnl": values[-1] - 1_000_000,
        "total_pnl_pct": (values[-1] - 1_000_000) / 1_000_000 * 100,
        "cash": values[-1] * 0.05,
    }


def render() -> None:
    st.title("Portfolio Overview")
    st.markdown("---")

    data = _generate_demo_portfolio()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Portfolio Value", f"${data['total_value']:,.0f}", f"${data['daily_pnl']:,.0f}")
    with col2:
        st.metric("Today's P&L", f"${data['daily_pnl']:,.0f}", f"{data['daily_pnl_pct']:.2f}%")
    with col3:
        st.metric("Total P&L", f"${data['total_pnl']:,.0f}", f"{data['total_pnl_pct']:.2f}%")
    with col4:
        st.metric("Cash", f"${data['cash']:,.0f}", f"{data['cash']/data['total_value']*100:.1f}%")

    st.markdown("### Portfolio Value")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=data["dates"],
        y=data["values"],
        mode="lines",
        fill="tozeroy",
        line={"color": "#00d4ff", "width": 2},
        fillcolor="rgba(0, 212, 255, 0.1)",
    ))
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0f172a",
        plot_bgcolor="#1e293b",
        margin={"l": 40, "r": 20, "t": 20, "b": 40},
        height=350,
        xaxis={"gridcolor": "#334155"},
        yaxis={"gridcolor": "#334155", "tickformat": "$,.0f"},
    )
    st.plotly_chart(fig, use_container_width=True)

    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.markdown("### Current Positions")
        df = pd.DataFrame(data["positions"])
        df["pnl_display"] = df["pnl"].apply(lambda x: f"${x:,.0f}")
        df["pnl_pct_display"] = df["pnl_pct"].apply(lambda x: f"{x:+.2f}%")
        df["weight_display"] = df["weight"].apply(lambda x: f"{x:.1f}%")

        st.dataframe(
            df[["ticker", "quantity", "entry_price", "current_price", "pnl_display", "pnl_pct_display", "weight_display"]].rename(
                columns={
                    "ticker": "Ticker",
                    "quantity": "Qty",
                    "entry_price": "Entry",
                    "current_price": "Current",
                    "pnl_display": "P&L",
                    "pnl_pct_display": "P&L %",
                    "weight_display": "Weight",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

    with col_right:
        st.markdown("### Top Movers")
        sorted_pos = sorted(data["positions"], key=lambda x: abs(x["pnl"]), reverse=True)
        for pos in sorted_pos[:5]:
            color = "🟢" if pos["pnl"] > 0 else "🔴"
            st.markdown(f"{color} **{pos['ticker']}** — ${pos['pnl']:+,.0f} ({pos['pnl_pct']:+.2f}%)")

        st.markdown("### Allocation")
        labels = [p["ticker"] for p in data["positions"]]
        values = [p["weight"] for p in data["positions"]]
        fig_pie = go.Figure(data=[go.Pie(
            labels=labels,
            values=values,
            hole=0.4,
            marker={"colors": px.colors.qualitative.Set3},
        )])
        fig_pie.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0f172a",
            margin={"l": 20, "r": 20, "t": 20, "b": 20},
            height=250,
            showlegend=True,
            legend={"font": {"size": 10}},
        )
        st.plotly_chart(fig_pie, use_container_width=True)
