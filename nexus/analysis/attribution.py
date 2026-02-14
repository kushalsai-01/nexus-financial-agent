from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from nexus.core.logging import get_logger
from nexus.core.types import BacktestResult, Trade

logger = get_logger("analysis.attribution")


class PerformanceAttributor:
    def __init__(self, risk_free_rate: float = 0.04) -> None:
        self.risk_free_rate = risk_free_rate

    def attribute_by_agent(
        self,
        trades: list[Trade],
        initial_capital: float = 100_000,
    ) -> dict[str, dict[str, Any]]:
        agent_trades: dict[str, list[Trade]] = {}
        for trade in trades:
            agent = trade.agent_name or "unknown"
            if agent not in agent_trades:
                agent_trades[agent] = []
            agent_trades[agent].append(trade)

        results: dict[str, dict[str, Any]] = {}
        for agent, atrades in agent_trades.items():
            pnls = [t.pnl for t in atrades if t.pnl is not None]
            total_pnl = sum(pnls)
            wins = [p for p in pnls if p > 0]
            losses = [p for p in pnls if p < 0]

            results[agent] = {
                "total_trades": len(atrades),
                "total_pnl": round(total_pnl, 2),
                "pnl_contribution_pct": round(total_pnl / initial_capital * 100, 2),
                "win_rate": round(len(wins) / len(pnls) * 100, 2) if pnls else 0,
                "avg_win": round(float(np.mean(wins)), 2) if wins else 0,
                "avg_loss": round(float(np.mean(losses)), 2) if losses else 0,
                "profit_factor": round(sum(wins) / abs(sum(losses)), 2) if losses else float("inf") if wins else 0,
                "sharpe_contribution": round(
                    float(np.mean(pnls) / np.std(pnls)) if len(pnls) > 1 and np.std(pnls) > 0 else 0, 3
                ),
                "total_commission": round(sum(t.commission for t in atrades), 2),
            }

        return results

    def attribute_by_ticker(
        self,
        trades: list[Trade],
    ) -> dict[str, dict[str, Any]]:
        ticker_trades: dict[str, list[Trade]] = {}
        for trade in trades:
            if trade.ticker not in ticker_trades:
                ticker_trades[trade.ticker] = []
            ticker_trades[trade.ticker].append(trade)

        results: dict[str, dict[str, Any]] = {}
        for ticker, ttrades in ticker_trades.items():
            pnls = [t.pnl for t in ttrades if t.pnl is not None]
            total_pnl = sum(pnls)

            results[ticker] = {
                "total_trades": len(ttrades),
                "total_pnl": round(total_pnl, 2),
                "avg_pnl": round(float(np.mean(pnls)), 2) if pnls else 0,
                "win_rate": round(sum(1 for p in pnls if p > 0) / len(pnls) * 100, 2) if pnls else 0,
                "best_trade": round(max(pnls), 2) if pnls else 0,
                "worst_trade": round(min(pnls), 2) if pnls else 0,
            }

        return dict(sorted(results.items(), key=lambda x: x[1]["total_pnl"], reverse=True))

    def attribute_by_time(
        self,
        trades: list[Trade],
        period: str = "month",
    ) -> dict[str, dict[str, Any]]:
        if not trades:
            return {}

        df = pd.DataFrame([
            {"timestamp": t.timestamp, "pnl": t.pnl or 0, "ticker": t.ticker}
            for t in trades
        ])
        df["timestamp"] = pd.to_datetime(df["timestamp"])

        if period == "month":
            df["period"] = df["timestamp"].dt.to_period("M").astype(str)
        elif period == "week":
            df["period"] = df["timestamp"].dt.to_period("W").astype(str)
        elif period == "quarter":
            df["period"] = df["timestamp"].dt.to_period("Q").astype(str)
        else:
            df["period"] = df["timestamp"].dt.to_period("Y").astype(str)

        results: dict[str, dict[str, Any]] = {}
        for period_key, group in df.groupby("period"):
            pnls = group["pnl"].tolist()
            results[str(period_key)] = {
                "total_trades": len(group),
                "total_pnl": round(sum(pnls), 2),
                "avg_pnl": round(float(np.mean(pnls)), 2),
                "win_rate": round(sum(1 for p in pnls if p > 0) / len(pnls) * 100, 2) if pnls else 0,
                "unique_tickers": group["ticker"].nunique(),
            }

        return results

    def compute_risk_adjusted_attribution(
        self,
        trades: list[Trade],
        equity_curve: pd.Series,
    ) -> dict[str, Any]:
        agent_attr = self.attribute_by_agent(trades)
        ticker_attr = self.attribute_by_ticker(trades)

        total_pnl = sum(a["total_pnl"] for a in agent_attr.values())
        returns = equity_curve.pct_change().dropna()
        vol = float(returns.std() * np.sqrt(252)) if len(returns) > 1 else 0

        return {
            "total_pnl": round(total_pnl, 2),
            "portfolio_volatility": round(vol, 4),
            "by_agent": agent_attr,
            "by_ticker": ticker_attr,
            "top_performer": max(agent_attr.items(), key=lambda x: x[1]["total_pnl"])[0] if agent_attr else "none",
            "worst_performer": min(agent_attr.items(), key=lambda x: x[1]["total_pnl"])[0] if agent_attr else "none",
        }
