from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


def _generate_demo_backtest() -> dict[str, Any]:
    np.random.seed(123)
    dates = pd.date_range("2020-01-01", "2024-01-01", freq="B")
    equity = [1_000_000.0]
    for _ in range(len(dates) - 1):
        ret = np.random.normal(0.0005, 0.015)
        equity.append(equity[-1] * (1 + ret))

    peak = pd.Series(equity).expanding().max()
    drawdown = ((pd.Series(equity) - peak) / peak).tolist()

    monthly = pd.DataFrame({"date": dates[:len(equity)], "equity": equity})
    monthly["return"] = monthly["equity"].pct_change()
    monthly["month"] = monthly["date"].dt.month
    monthly["year"] = monthly["date"].dt.year
    monthly_returns = monthly.groupby(["year", "month"])["return"].sum().reset_index()
    heatmap_data = monthly_returns.pivot(index="year", columns="month", values="return")

    trades = []
    for i in range(50):
        entry = np.random.uniform(100, 500)
        pnl_pct = np.random.normal(0.02, 0.08)
        exit_price = entry * (1 + pnl_pct)
        trades.append({
            "id": i + 1,
            "ticker": np.random.choice(["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA", "SPY"]),
            "side": np.random.choice(["buy", "sell"]),
            "entry_price": round(entry, 2),
            "exit_price": round(exit_price, 2),
            "quantity": np.random.randint(10, 200),
            "pnl": round(pnl_pct * entry * np.random.randint(10, 200), 2),
            "pnl_pct": round(pnl_pct * 100, 2),
            "holding_days": np.random.randint(1, 60),
        })

    return {
        "dates": dates[:len(equity)],
        "equity": equity,
        "drawdown": drawdown,
        "heatmap": heatmap_data,
        "trades": trades,
        "metrics": {
            "Total Return": f"{(equity[-1] / equity[0] - 1) * 100:.2f}%",
            "CAGR": f"{((equity[-1] / equity[0]) ** (1 / 4) - 1) * 100:.2f}%",
            "Sharpe Ratio": "1.42",
            "Sortino Ratio": "2.18",
            "Max Drawdown": f"{min(drawdown) * 100:.2f}%",
            "Calmar Ratio": "1.85",
            "Win Rate": "58.0%",
            "Profit Factor": "1.67",
            "Total Trades": str(len(trades)),
            "Avg Trade P&L": f"${np.mean([t['pnl'] for t in trades]):,.0f}",
            "Volatility": "23.7%",
            "Beta": "0.85",
        },
    }


def render() -> None:
    st.title("Backtest Results")
    st.markdown("---")

    backtest_options = ["Momentum Strategy (2020-2024)", "Mean Reversion (2021-2024)", "Agent Ensemble (2022-2024)"]
    selected = st.selectbox("Select Backtest", backtest_options)

    data = _generate_demo_backtest()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Return", data["metrics"]["Total Return"])
    with col2:
        st.metric("Sharpe Ratio", data["metrics"]["Sharpe Ratio"])
    with col3:
        st.metric("Max Drawdown", data["metrics"]["Max Drawdown"])
    with col4:
        st.metric("Win Rate", data["metrics"]["Win Rate"])

    st.markdown("### Equity Curve")
    fig_eq = go.Figure()
    fig_eq.add_trace(go.Scatter(
        x=data["dates"], y=data["equity"],
        mode="lines", fill="tozeroy",
        line={"color": "#00d4ff", "width": 2},
        fillcolor="rgba(0, 212, 255, 0.1)",
    ))
    fig_eq.update_layout(
        template="plotly_dark", paper_bgcolor="#0f172a", plot_bgcolor="#1e293b",
        margin={"l": 60, "r": 20, "t": 20, "b": 40}, height=350,
        yaxis={"gridcolor": "#334155", "tickformat": "$,.0f"},
        xaxis={"gridcolor": "#334155"},
    )
    st.plotly_chart(fig_eq, use_container_width=True)

    st.markdown("### Drawdown")
    fig_dd = go.Figure()
    fig_dd.add_trace(go.Scatter(
        x=data["dates"], y=data["drawdown"],
        mode="lines", fill="tozeroy",
        line={"color": "#ef4444", "width": 1.5},
        fillcolor="rgba(239, 68, 68, 0.2)",
    ))
    fig_dd.update_layout(
        template="plotly_dark", paper_bgcolor="#0f172a", plot_bgcolor="#1e293b",
        margin={"l": 60, "r": 20, "t": 20, "b": 40}, height=200,
        yaxis={"gridcolor": "#334155", "tickformat": ".0%"},
        xaxis={"gridcolor": "#334155"},
    )
    st.plotly_chart(fig_dd, use_container_width=True)

    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.markdown("### Monthly Returns Heatmap")
        if data["heatmap"] is not None and not data["heatmap"].empty:
            month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
            fig_hm = go.Figure(data=go.Heatmap(
                z=data["heatmap"].values * 100,
                x=month_names[:data["heatmap"].shape[1]],
                y=[str(y) for y in data["heatmap"].index],
                colorscale=[[0, "#ef4444"], [0.5, "#1e293b"], [1, "#22c55e"]],
                zmid=0,
                text=np.round(data["heatmap"].values * 100, 1),
                texttemplate="%{text}%",
                textfont={"size": 10},
            ))
            fig_hm.update_layout(
                template="plotly_dark", paper_bgcolor="#0f172a", plot_bgcolor="#1e293b",
                margin={"l": 60, "r": 20, "t": 20, "b": 40}, height=250,
            )
            st.plotly_chart(fig_hm, use_container_width=True)

    with col_right:
        st.markdown("### Metrics")
        for k, v in data["metrics"].items():
            st.markdown(f"**{k}:** {v}")

    st.markdown("### Trade List")
    trade_df = pd.DataFrame(data["trades"])
    st.dataframe(
        trade_df[["id", "ticker", "side", "entry_price", "exit_price", "quantity", "pnl", "pnl_pct", "holding_days"]].rename(
            columns={
                "id": "#", "ticker": "Ticker", "side": "Side",
                "entry_price": "Entry", "exit_price": "Exit", "quantity": "Qty",
                "pnl": "P&L ($)", "pnl_pct": "P&L (%)", "holding_days": "Days",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )
