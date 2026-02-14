from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


def _generate_demo_trades() -> list[dict[str, Any]]:
    np.random.seed(55)
    tickers = ["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA", "SPY", "META", "AMZN", "JPM", "V"]
    agents = ["technical", "fundamental", "sentiment", "quantitative", "bull", "bear", "coordinator"]
    trades = []

    for i in range(150):
        ticker = np.random.choice(tickers)
        agent = np.random.choice(agents)
        side = np.random.choice(["buy", "sell"])
        entry = np.random.uniform(80, 800)
        pnl_pct = np.random.normal(0.015, 0.06)
        exit_price = entry * (1 + pnl_pct)
        qty = np.random.randint(5, 300)
        holding = np.random.randint(1, 90)
        slippage = np.random.uniform(0, 0.3)
        entry_date = datetime.now() - timedelta(days=np.random.randint(1, 365))

        trades.append({
            "id": i + 1,
            "ticker": ticker,
            "agent": agent,
            "side": side,
            "entry_date": entry_date.strftime("%Y-%m-%d"),
            "exit_date": (entry_date + timedelta(days=holding)).strftime("%Y-%m-%d"),
            "entry_price": round(entry, 2),
            "exit_price": round(exit_price, 2),
            "quantity": qty,
            "pnl": round(pnl_pct * entry * qty, 2),
            "pnl_pct": round(pnl_pct * 100, 2),
            "holding_days": holding,
            "slippage_bps": round(slippage, 2),
            "commission": round(entry * qty * 0.0001, 2),
        })

    return sorted(trades, key=lambda t: t["entry_date"], reverse=True)


def render() -> None:
    st.title("Trade History")
    st.markdown("---")

    trades = _generate_demo_trades()
    df = pd.DataFrame(trades)

    col1, col2, col3 = st.columns(3)
    with col1:
        date_range = st.date_input(
            "Date Range",
            value=(datetime.now() - timedelta(days=365), datetime.now()),
        )
    with col2:
        ticker_filter = st.multiselect("Ticker", options=sorted(df["ticker"].unique()))
    with col3:
        agent_filter = st.multiselect("Agent", options=sorted(df["agent"].unique()))

    filtered = df.copy()
    if ticker_filter:
        filtered = filtered[filtered["ticker"].isin(ticker_filter)]
    if agent_filter:
        filtered = filtered[filtered["agent"].isin(agent_filter)]
    if date_range and len(date_range) == 2:
        start_str = date_range[0].strftime("%Y-%m-%d")
        end_str = date_range[1].strftime("%Y-%m-%d")
        filtered = filtered[(filtered["entry_date"] >= start_str) & (filtered["entry_date"] <= end_str)]

    total_pnl = filtered["pnl"].sum()
    win_count = (filtered["pnl"] > 0).sum()
    total_count = len(filtered)
    win_rate = (win_count / total_count * 100) if total_count > 0 else 0
    avg_holding = filtered["holding_days"].mean() if total_count > 0 else 0
    avg_slippage = filtered["slippage_bps"].mean() if total_count > 0 else 0

    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        st.metric("Total Trades", total_count)
    with m2:
        st.metric("Total P&L", f"${total_pnl:,.0f}")
    with m3:
        st.metric("Win Rate", f"{win_rate:.1f}%")
    with m4:
        st.metric("Avg Holding", f"{avg_holding:.0f} days")
    with m5:
        st.metric("Avg Slippage", f"{avg_slippage:.2f} bps")

    st.markdown("### P&L Distribution")
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=filtered["pnl"],
        nbinsx=40,
        marker_color=["#22c55e" if x > 0 else "#ef4444" for x in filtered["pnl"]],
    ))
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="#0f172a", plot_bgcolor="#1e293b",
        margin={"l": 40, "r": 20, "t": 20, "b": 40}, height=250,
        xaxis={"gridcolor": "#334155", "tickprefix": "$"},
        yaxis={"gridcolor": "#334155"},
    )
    st.plotly_chart(fig, use_container_width=True)

    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("### P&L by Ticker")
        ticker_pnl = filtered.groupby("ticker")["pnl"].sum().sort_values()
        colors = ["#22c55e" if v > 0 else "#ef4444" for v in ticker_pnl.values]
        fig_t = go.Figure(data=[go.Bar(
            y=ticker_pnl.index, x=ticker_pnl.values, orientation="h",
            marker_color=colors,
        )])
        fig_t.update_layout(
            template="plotly_dark", paper_bgcolor="#0f172a", plot_bgcolor="#1e293b",
            margin={"l": 60, "r": 20, "t": 20, "b": 40}, height=300,
            xaxis={"gridcolor": "#334155", "tickprefix": "$"},
        )
        st.plotly_chart(fig_t, use_container_width=True)

    with col_right:
        st.markdown("### Execution Quality")
        fig_slip = go.Figure()
        fig_slip.add_trace(go.Scatter(
            x=filtered["pnl"], y=filtered["slippage_bps"],
            mode="markers",
            marker={"color": "#7c3aed", "size": 5, "opacity": 0.6},
        ))
        fig_slip.update_layout(
            template="plotly_dark", paper_bgcolor="#0f172a", plot_bgcolor="#1e293b",
            margin={"l": 40, "r": 20, "t": 20, "b": 40}, height=300,
            xaxis={"title": "P&L ($)", "gridcolor": "#334155"},
            yaxis={"title": "Slippage (bps)", "gridcolor": "#334155"},
        )
        st.plotly_chart(fig_slip, use_container_width=True)

    st.markdown("### All Trades")
    display_cols = ["id", "ticker", "agent", "side", "entry_date", "exit_date", "entry_price", "exit_price", "quantity", "pnl", "pnl_pct", "holding_days", "slippage_bps"]
    rename_map = {
        "id": "#", "ticker": "Ticker", "agent": "Agent", "side": "Side",
        "entry_date": "Entry Date", "exit_date": "Exit Date",
        "entry_price": "Entry $", "exit_price": "Exit $", "quantity": "Qty",
        "pnl": "P&L ($)", "pnl_pct": "P&L (%)", "holding_days": "Days",
        "slippage_bps": "Slip (bps)",
    }
    st.dataframe(
        filtered[display_cols].rename(columns=rename_map),
        use_container_width=True,
        hide_index=True,
        height=400,
    )
