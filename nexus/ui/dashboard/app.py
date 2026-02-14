from __future__ import annotations

import streamlit as st


def configure_page() -> None:
    st.set_page_config(
        page_title="NEXUS Dashboard",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown(
        """
        <style>
        .stApp { background-color: #0f172a; }
        .css-1d391kg { background-color: #1e293b; }
        [data-testid="stSidebar"] { background-color: #1e293b; }
        .metric-card {
            background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
            border: 1px solid #475569;
            border-radius: 12px;
            padding: 20px;
            margin: 8px 0;
        }
        .metric-value { font-size: 2rem; font-weight: bold; color: #f8fafc; }
        .metric-label { font-size: 0.85rem; color: #94a3b8; text-transform: uppercase; }
        .positive { color: #22c55e; }
        .negative { color: #ef4444; }
        h1, h2, h3 { color: #f8fafc !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    configure_page()

    st.sidebar.title("📊 NEXUS")
    st.sidebar.markdown("---")

    page = st.sidebar.radio(
        "Navigation",
        ["Overview", "Agents", "Backtests", "Trades", "Risk"],
        index=0,
    )

    st.sidebar.markdown("---")
    st.sidebar.caption("NEXUS v1.0.0")
    st.sidebar.caption("Multi-Agent Trading System")

    if page == "Overview":
        from nexus.ui.dashboard.pages.overview import render
        render()
    elif page == "Agents":
        from nexus.ui.dashboard.pages.agents import render
        render()
    elif page == "Backtests":
        from nexus.ui.dashboard.pages.backtests import render
        render()
    elif page == "Trades":
        from nexus.ui.dashboard.pages.trades import render
        render()
    elif page == "Risk":
        from nexus.ui.dashboard.pages.risk import render
        render()


if __name__ == "__main__":
    main()
