from __future__ import annotations

import json
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from nexus.backtest.metrics import (
    compute_all_metrics,
    compute_max_drawdown,
    compute_monthly_returns,
    compute_rolling_sharpe,
    compute_yearly_returns,
)
from nexus.backtest.regime import RegimeClassifier
from nexus.core.logging import get_logger
from nexus.core.types import BacktestResult

logger = get_logger("backtest.reports")


class BacktestReport:
    def __init__(self, result: BacktestResult) -> None:
        self.result = result
        self.equity_series = pd.Series(result.equity_curve)
        self.regime_classifier = RegimeClassifier()

    def summary(self) -> dict[str, Any]:
        return {
            "strategy": self.result.strategy_name,
            "period": f"{self.result.start_date} to {self.result.end_date}",
            "initial_capital": self.result.initial_capital,
            "final_value": round(self.result.final_value, 2),
            "total_return_pct": round(self.result.total_return_pct, 2),
            "cagr": round(self.result.cagr * 100, 2),
            "sharpe_ratio": round(self.result.sharpe_ratio, 3),
            "sortino_ratio": round(self.result.sortino_ratio, 3),
            "calmar_ratio": round(self.result.calmar_ratio, 3),
            "max_drawdown": round(self.result.max_drawdown * 100, 2),
            "max_dd_duration_days": self.result.max_drawdown_duration_days,
            "volatility": round(self.result.volatility * 100, 2),
            "win_rate": round(self.result.win_rate * 100, 2),
            "profit_factor": round(self.result.profit_factor, 3),
            "total_trades": self.result.total_trades,
            "avg_trade_pnl": round(self.result.avg_trade_pnl, 2),
            "best_trade": round(self.result.best_trade, 2),
            "worst_trade": round(self.result.worst_trade, 2),
            "beta": round(self.result.beta, 3),
            "alpha": round(self.result.alpha * 100, 2) if self.result.alpha else 0,
        }

    def trade_analysis(self) -> dict[str, Any]:
        if not self.result.trades:
            return {"total_trades": 0}

        buy_trades = [t for t in self.result.trades if t.side.value == "buy"]
        sell_trades = [t for t in self.result.trades if t.side.value == "sell"]
        pnls = [t.pnl for t in self.result.trades if t.pnl is not None]

        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]

        total_commission = sum(t.commission for t in self.result.trades)
        total_slippage = sum(t.slippage * t.quantity for t in self.result.trades)

        return {
            "total_trades": len(self.result.trades),
            "buy_trades": len(buy_trades),
            "sell_trades": len(sell_trades),
            "winning_trades": len(wins),
            "losing_trades": len(losses),
            "avg_win": round(float(np.mean(wins)), 2) if wins else 0,
            "avg_loss": round(float(np.mean(losses)), 2) if losses else 0,
            "largest_win": round(max(pnls), 2) if pnls else 0,
            "largest_loss": round(min(pnls), 2) if pnls else 0,
            "total_commission": round(total_commission, 2),
            "total_slippage": round(total_slippage, 2),
            "total_transaction_costs": round(total_commission + total_slippage, 2),
        }

    def drawdown_analysis(self) -> dict[str, Any]:
        if len(self.equity_series) < 2:
            return {}

        peak = self.equity_series.expanding().max()
        drawdown = (self.equity_series - peak) / peak

        dd_periods: list[dict[str, Any]] = []
        in_dd = False
        dd_start = 0
        max_dd_in_period = 0.0

        for i in range(len(drawdown)):
            if drawdown.iloc[i] < 0:
                if not in_dd:
                    in_dd = True
                    dd_start = i
                    max_dd_in_period = drawdown.iloc[i]
                else:
                    max_dd_in_period = min(max_dd_in_period, drawdown.iloc[i])
            else:
                if in_dd:
                    dd_periods.append({
                        "start_idx": dd_start,
                        "end_idx": i,
                        "duration": i - dd_start,
                        "max_drawdown": round(float(max_dd_in_period) * 100, 2),
                    })
                    in_dd = False
                    max_dd_in_period = 0.0

        if in_dd:
            dd_periods.append({
                "start_idx": dd_start,
                "end_idx": len(drawdown) - 1,
                "duration": len(drawdown) - dd_start,
                "max_drawdown": round(float(max_dd_in_period) * 100, 2),
            })

        dd_periods.sort(key=lambda x: x["max_drawdown"])
        top_5 = dd_periods[:5]

        return {
            "total_drawdown_periods": len(dd_periods),
            "longest_drawdown_days": max((d["duration"] for d in dd_periods), default=0),
            "deepest_drawdown_pct": round(float(drawdown.min()) * 100, 2),
            "avg_drawdown_pct": round(float(drawdown[drawdown < 0].mean()) * 100, 2) if (drawdown < 0).any() else 0,
            "top_5_drawdowns": top_5,
        }

    def to_json(self, filepath: str | None = None) -> str:
        report = {
            "generated_at": datetime.now().isoformat(),
            "summary": self.summary(),
            "trade_analysis": self.trade_analysis(),
            "drawdown_analysis": self.drawdown_analysis(),
        }

        output = json.dumps(report, indent=2, default=str)

        if filepath:
            with open(filepath, "w") as f:
                f.write(output)
            logger.info(f"Report saved to {filepath}")

        return output

    def to_html(self, filepath: str | None = None) -> str:
        summary = self.summary()
        trade_info = self.trade_analysis()
        dd_info = self.drawdown_analysis()

        html_parts = [
            "<!DOCTYPE html>",
            "<html><head><title>NEXUS Backtest Report</title>",
            "<style>",
            "body{font-family:Arial,sans-serif;margin:20px;background:#1a1a2e;color:#e0e0e0}",
            "h1{color:#00d4ff}h2{color:#7b68ee;border-bottom:1px solid #333;padding-bottom:5px}",
            "table{border-collapse:collapse;width:100%;margin:15px 0}",
            "th,td{padding:8px 12px;text-align:left;border:1px solid #333}",
            "th{background:#16213e;color:#00d4ff}td{background:#0f3460}",
            ".positive{color:#00ff88}.negative{color:#ff4757}",
            ".metric-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin:15px 0}",
            ".metric-card{background:#16213e;padding:15px;border-radius:8px;text-align:center}",
            ".metric-value{font-size:24px;font-weight:bold;color:#00d4ff}",
            ".metric-label{font-size:12px;color:#888}",
            "</style></head><body>",
            f"<h1>NEXUS Backtest Report: {summary['strategy']}</h1>",
            f"<p>Period: {summary['period']}</p>",
            '<div class="metric-grid">',
        ]

        metrics_cards = [
            ("Total Return", f"{summary['total_return_pct']}%", summary['total_return_pct'] >= 0),
            ("CAGR", f"{summary['cagr']}%", summary['cagr'] >= 0),
            ("Sharpe Ratio", f"{summary['sharpe_ratio']}", summary['sharpe_ratio'] >= 0),
            ("Max Drawdown", f"{summary['max_drawdown']}%", False),
            ("Win Rate", f"{summary['win_rate']}%", summary['win_rate'] >= 50),
            ("Profit Factor", f"{summary['profit_factor']}", summary['profit_factor'] >= 1),
            ("Volatility", f"{summary['volatility']}%", True),
            ("Total Trades", f"{summary['total_trades']}", True),
        ]

        for label, value, is_positive in metrics_cards:
            css_class = "positive" if is_positive else "negative"
            html_parts.append(
                f'<div class="metric-card">'
                f'<div class="metric-value {css_class}">{value}</div>'
                f'<div class="metric-label">{label}</div></div>'
            )

        html_parts.append("</div>")

        html_parts.append("<h2>Trade Analysis</h2><table>")
        for key, value in trade_info.items():
            label = key.replace("_", " ").title()
            html_parts.append(f"<tr><th>{label}</th><td>{value}</td></tr>")
        html_parts.append("</table>")

        html_parts.append("<h2>Drawdown Analysis</h2><table>")
        for key, value in dd_info.items():
            if key != "top_5_drawdowns":
                label = key.replace("_", " ").title()
                html_parts.append(f"<tr><th>{label}</th><td>{value}</td></tr>")
        html_parts.append("</table>")

        html_parts.append("</body></html>")
        html = "\n".join(html_parts)

        if filepath:
            with open(filepath, "w") as f:
                f.write(html)
            logger.info(f"HTML report saved to {filepath}")

        return html
