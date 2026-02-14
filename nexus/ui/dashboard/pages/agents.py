from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


def _generate_demo_agent_data() -> dict[str, Any]:
    np.random.seed(42)
    agents = [
        "market_data", "technical", "fundamental", "quantitative",
        "sentiment", "macro", "event", "rl_agent",
        "bull", "bear", "risk", "portfolio", "execution", "coordinator",
    ]

    accuracy: dict[str, list[float]] = {}
    decisions: dict[str, dict[str, int]] = {}
    confidence: dict[str, float] = {}
    cost: dict[str, float] = {}
    attribution: dict[str, float] = {}

    for agent in agents:
        base_acc = 0.55 + np.random.random() * 0.25
        accuracy[agent] = [max(0.3, min(0.95, base_acc + np.random.normal(0, 0.05))) for _ in range(30)]
        buy_pct = np.random.random()
        hold_pct = np.random.random() * (1 - buy_pct)
        sell_pct = 1 - buy_pct - hold_pct
        total = np.random.randint(50, 200)
        decisions[agent] = {
            "buy": int(total * buy_pct),
            "sell": int(total * sell_pct),
            "hold": int(total * hold_pct),
        }
        confidence[agent] = 0.5 + np.random.random() * 0.35
        cost[agent] = np.random.random() * 5
        attribution[agent] = np.random.normal(0, 2)

    return {
        "agents": agents,
        "accuracy": accuracy,
        "decisions": decisions,
        "confidence": confidence,
        "cost": cost,
        "attribution": attribution,
    }


def render() -> None:
    st.title("Agent Performance")
    st.markdown("---")

    data = _generate_demo_agent_data()

    selected = st.selectbox("Select Agent", ["All"] + data["agents"])

    if selected == "All":
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Accuracy by Agent")
            avg_acc = {a: np.mean(v) for a, v in data["accuracy"].items()}
            fig = go.Figure(data=[go.Bar(
                x=list(avg_acc.keys()),
                y=list(avg_acc.values()),
                marker_color=["#22c55e" if v > 0.6 else "#f59e0b" if v > 0.5 else "#ef4444" for v in avg_acc.values()],
            )])
            fig.update_layout(
                template="plotly_dark", paper_bgcolor="#0f172a", plot_bgcolor="#1e293b",
                margin={"l": 40, "r": 20, "t": 20, "b": 80}, height=350,
                yaxis={"gridcolor": "#334155", "tickformat": ".0%"},
                xaxis={"tickangle": -45},
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("### LLM Cost per Agent ($)")
            fig_cost = go.Figure(data=[go.Bar(
                x=list(data["cost"].keys()),
                y=list(data["cost"].values()),
                marker_color="#f59e0b",
            )])
            fig_cost.update_layout(
                template="plotly_dark", paper_bgcolor="#0f172a", plot_bgcolor="#1e293b",
                margin={"l": 40, "r": 20, "t": 20, "b": 80}, height=350,
                yaxis={"gridcolor": "#334155", "tickprefix": "$"},
                xaxis={"tickangle": -45},
            )
            st.plotly_chart(fig_cost, use_container_width=True)

        st.markdown("### Decision Distribution")
        decision_df = pd.DataFrame(data["decisions"]).T
        fig_dec = go.Figure()
        for col, color in [("buy", "#22c55e"), ("sell", "#ef4444"), ("hold", "#f59e0b")]:
            fig_dec.add_trace(go.Bar(name=col.capitalize(), x=decision_df.index, y=decision_df[col], marker_color=color))
        fig_dec.update_layout(
            barmode="stack", template="plotly_dark", paper_bgcolor="#0f172a", plot_bgcolor="#1e293b",
            margin={"l": 40, "r": 20, "t": 20, "b": 80}, height=350,
            xaxis={"tickangle": -45}, yaxis={"gridcolor": "#334155"},
        )
        st.plotly_chart(fig_dec, use_container_width=True)

        col3, col4 = st.columns(2)
        with col3:
            st.markdown("### Average Confidence")
            conf_df = pd.DataFrame({"Agent": data["confidence"].keys(), "Confidence": data["confidence"].values()})
            conf_df = conf_df.sort_values("Confidence", ascending=True)
            fig_conf = go.Figure(data=[go.Bar(
                y=conf_df["Agent"], x=conf_df["Confidence"], orientation="h",
                marker_color="#7c3aed",
            )])
            fig_conf.update_layout(
                template="plotly_dark", paper_bgcolor="#0f172a", plot_bgcolor="#1e293b",
                margin={"l": 100, "r": 20, "t": 20, "b": 40}, height=400,
                xaxis={"gridcolor": "#334155", "tickformat": ".0%"},
            )
            st.plotly_chart(fig_conf, use_container_width=True)

        with col4:
            st.markdown("### Return Attribution (%)")
            attr_df = pd.DataFrame({"Agent": data["attribution"].keys(), "Attribution": data["attribution"].values()})
            attr_df = attr_df.sort_values("Attribution")
            colors = ["#22c55e" if v > 0 else "#ef4444" for v in attr_df["Attribution"]]
            fig_attr = go.Figure(data=[go.Bar(
                y=attr_df["Agent"], x=attr_df["Attribution"], orientation="h",
                marker_color=colors,
            )])
            fig_attr.update_layout(
                template="plotly_dark", paper_bgcolor="#0f172a", plot_bgcolor="#1e293b",
                margin={"l": 100, "r": 20, "t": 20, "b": 40}, height=400,
                xaxis={"gridcolor": "#334155", "ticksuffix": "%"},
            )
            st.plotly_chart(fig_attr, use_container_width=True)
    else:
        st.markdown(f"### {selected.replace('_', ' ').title()} Agent Details")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Avg Accuracy", f"{np.mean(data['accuracy'][selected]):.1%}")
        with col2:
            st.metric("Confidence", f"{data['confidence'][selected]:.1%}")
        with col3:
            st.metric("LLM Cost", f"${data['cost'][selected]:.2f}")
        with col4:
            st.metric("Attribution", f"{data['attribution'][selected]:+.2f}%")

        st.markdown("### Accuracy Over Time")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            y=data["accuracy"][selected], mode="lines+markers",
            line={"color": "#00d4ff", "width": 2},
        ))
        fig.update_layout(
            template="plotly_dark", paper_bgcolor="#0f172a", plot_bgcolor="#1e293b",
            margin={"l": 40, "r": 20, "t": 20, "b": 40}, height=300,
            yaxis={"gridcolor": "#334155", "tickformat": ".0%"},
            xaxis={"title": "Decision #", "gridcolor": "#334155"},
        )
        st.plotly_chart(fig, use_container_width=True)

        decisions = data["decisions"][selected]
        fig_pie = go.Figure(data=[go.Pie(
            labels=["Buy", "Sell", "Hold"],
            values=[decisions["buy"], decisions["sell"], decisions["hold"]],
            marker={"colors": ["#22c55e", "#ef4444", "#f59e0b"]},
            hole=0.4,
        )])
        fig_pie.update_layout(
            template="plotly_dark", paper_bgcolor="#0f172a",
            margin={"l": 20, "r": 20, "t": 20, "b": 20}, height=300,
        )
        st.plotly_chart(fig_pie, use_container_width=True)
