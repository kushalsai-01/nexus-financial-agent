from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


def _generate_demo_risk() -> dict[str, Any]:
    np.random.seed(77)
    tickers = ["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA", "SPY", "JPM"]
    n = len(tickers)

    returns = np.random.multivariate_normal(
        mean=[0.0005] * n,
        cov=np.diag([0.02] * n) + 0.005,
        size=252,
    )
    corr = np.corrcoef(returns.T)

    sectors = {
        "Technology": 45.2, "Financial": 12.3, "Consumer": 8.5,
        "Healthcare": 10.1, "Energy": 5.8, "Industrial": 7.3,
        "Other": 10.8,
    }

    weights = np.array([12.5, 14.2, 4.0, 9.9, 13.3, 41.5, 4.6]) / 100
    position_sizes = weights * 1_050_000

    return {
        "tickers": tickers,
        "correlation": corr,
        "sectors": sectors,
        "position_sizes": dict(zip(tickers, position_sizes)),
        "var_1d": -15230.0,
        "var_10d": -48150.0,
        "var_1d_pct": -1.45,
        "var_10d_pct": -4.59,
        "cvar_1d": -22840.0,
        "portfolio_beta": 0.87,
        "sharpe": 1.42,
        "volatility": 0.237,
        "max_dd": -0.142,
        "current_dd": -0.038,
        "weights": dict(zip(tickers, weights)),
    }


def render() -> None:
    st.title("Risk Dashboard")
    st.markdown("---")

    data = _generate_demo_risk()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("1-Day VaR (95%)", f"${data['var_1d']:,.0f}", f"{data['var_1d_pct']:.2f}%")
    with col2:
        st.metric("10-Day VaR (95%)", f"${data['var_10d']:,.0f}", f"{data['var_10d_pct']:.2f}%")
    with col3:
        st.metric("Portfolio Beta", f"{data['portfolio_beta']:.2f}")
    with col4:
        st.metric("Current Drawdown", f"{data['current_dd']*100:.2f}%", f"Max: {data['max_dd']*100:.2f}%")

    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("### Sector Exposure")
        fig_sector = go.Figure(data=[go.Pie(
            labels=list(data["sectors"].keys()),
            values=list(data["sectors"].values()),
            hole=0.4,
            marker={"colors": px.colors.qualitative.Set3},
        )])
        fig_sector.update_layout(
            template="plotly_dark", paper_bgcolor="#0f172a",
            margin={"l": 20, "r": 20, "t": 20, "b": 20}, height=350,
        )
        st.plotly_chart(fig_sector, use_container_width=True)

    with col_right:
        st.markdown("### Correlation Heatmap")
        fig_corr = go.Figure(data=go.Heatmap(
            z=data["correlation"],
            x=data["tickers"],
            y=data["tickers"],
            colorscale=[[0, "#ef4444"], [0.5, "#1e293b"], [1, "#22c55e"]],
            zmid=0,
            text=np.round(data["correlation"], 2),
            texttemplate="%{text}",
            textfont={"size": 10},
        ))
        fig_corr.update_layout(
            template="plotly_dark", paper_bgcolor="#0f172a", plot_bgcolor="#1e293b",
            margin={"l": 60, "r": 20, "t": 20, "b": 60}, height=350,
        )
        st.plotly_chart(fig_corr, use_container_width=True)

    st.markdown("### Position Size Distribution")
    pos_df = pd.DataFrame({
        "Ticker": data["position_sizes"].keys(),
        "Value": data["position_sizes"].values(),
    }).sort_values("Value", ascending=True)
    fig_pos = go.Figure(data=[go.Bar(
        y=pos_df["Ticker"], x=pos_df["Value"], orientation="h",
        marker_color="#00d4ff",
    )])
    fig_pos.update_layout(
        template="plotly_dark", paper_bgcolor="#0f172a", plot_bgcolor="#1e293b",
        margin={"l": 60, "r": 20, "t": 20, "b": 40}, height=300,
        xaxis={"gridcolor": "#334155", "tickformat": "$,.0f"},
    )
    st.plotly_chart(fig_pos, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        st.markdown("### Risk Metrics")
        metrics_data = {
            "VaR 1-Day (95%)": f"${data['var_1d']:,.0f}",
            "VaR 10-Day (95%)": f"${data['var_10d']:,.0f}",
            "CVaR 1-Day": f"${data['cvar_1d']:,.0f}",
            "Portfolio Beta": f"{data['portfolio_beta']:.2f}",
            "Volatility (Ann.)": f"{data['volatility']*100:.1f}%",
            "Sharpe Ratio": f"{data['sharpe']:.2f}",
            "Max Drawdown": f"{data['max_dd']*100:.2f}%",
            "Current Drawdown": f"{data['current_dd']*100:.2f}%",
        }
        for k, v in metrics_data.items():
            st.markdown(f"**{k}:** {v}")

    with col4:
        st.markdown("### Risk Limits Status")
        limits = [
            ("Position Size", 12.5, 15.0, True),
            ("Sector Exposure", 45.2, 50.0, True),
            ("Leverage", 1.0, 1.0, True),
            ("Drawdown", 3.8, 20.0, True),
            ("Daily Loss", 0.5, 3.0, True),
            ("Cash Reserve", 5.0, 5.0, True),
        ]
        for name, current, limit, ok in limits:
            icon = "✅" if ok else "❌"
            st.markdown(f"{icon} **{name}**: {current:.1f}% / {limit:.1f}%")

        st.markdown("---")
        st.markdown("**Overall Risk Status:** 🟢 All Limits OK")
