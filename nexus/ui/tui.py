from __future__ import annotations

import asyncio
import signal
import sys
import time
from datetime import datetime
from typing import Any

from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.text import Text

from nexus.core.logging import get_logger
from nexus.monitoring.metrics import NexusMetrics
from nexus.ui.components import make_banner, make_report_panel
from nexus.ui.layouts import CompactLayout, NexusLayout
from nexus.ui.themes import TEAM_LABELS, get_status_style

logger = get_logger("ui.tui")


class AgentMessage:
    __slots__ = ("timestamp", "msg_type", "agent", "content")

    def __init__(self, msg_type: str, agent: str, content: str) -> None:
        self.timestamp = datetime.now()
        self.msg_type = msg_type
        self.agent = agent
        self.content = content

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "type": self.msg_type,
            "agent": self.agent,
            "content": self.content,
        }


class TradingTUI:
    def __init__(self, refresh_rate: int = 4) -> None:
        self._console = Console()
        self._layout = NexusLayout()
        self._refresh_rate = refresh_rate
        self._running = False
        self._agent_states: dict[str, dict[str, Any]] = {}
        self._messages: list[dict[str, Any]] = []
        self._report: dict[str, Any] | None = None
        self._ticker: str = ""
        self._phase: str = "Initializing"
        self._start_time: float = 0.0
        self._metrics = NexusMetrics.get_instance()
        self._max_messages = 200

    def update_agent_status(
        self,
        agent_name: str,
        status: str,
        signal: str = "",
        confidence: float = 0.0,
        latency_ms: float = 0.0,
    ) -> None:
        self._agent_states[agent_name] = {
            "status": status,
            "signal": signal,
            "confidence": confidence,
            "latency_ms": latency_ms,
        }

    def add_message(self, msg_type: str, agent: str, content: str) -> None:
        msg = AgentMessage(msg_type, agent, content)
        self._messages.append(msg.to_dict())
        if len(self._messages) > self._max_messages:
            self._messages = self._messages[-self._max_messages:]

    def set_report(self, report: dict[str, Any]) -> None:
        self._report = report

    def set_ticker(self, ticker: str) -> None:
        self._ticker = ticker

    def set_phase(self, phase: str) -> None:
        self._phase = phase

    def _get_uptime(self) -> float:
        if self._start_time == 0:
            return 0.0
        return time.monotonic() - self._start_time

    def show_banner(self) -> None:
        self._console.print(make_banner())

    def show_completion(self, report: dict[str, Any]) -> None:
        self._console.print()
        self._console.print(make_report_panel(report))
        self._console.print()

    async def run_with_graph(self, ticker: str, graph: Any, capital: float | None = None) -> dict[str, Any]:
        self._ticker = ticker
        self._start_time = time.monotonic()
        self._running = True

        for team_agents in TEAM_LABELS.values():
            for agent in team_agents:
                self._agent_states[agent] = {"status": "pending"}

        result: dict[str, Any] = {}

        async def execute_graph() -> None:
            nonlocal result
            try:
                from nexus.orchestration.graph import TradingGraph
                if isinstance(graph, TradingGraph):
                    state = await graph.run(ticker, initial_data=None, capital=capital)
                    result = dict(state)
                else:
                    result = {"error": "Invalid graph type"}
            except Exception as e:
                logger.error(f"Graph execution failed: {e}")
                result = {"error": str(e)}

        with Live(
            self._layout.update(
                agent_states=self._agent_states,
                messages=self._messages,
                report=self._report,
                uptime=self._get_uptime(),
                ticker=self._ticker,
                phase=self._phase,
            ),
            console=self._console,
            refresh_per_second=self._refresh_rate,
            screen=True,
        ) as live:
            task = asyncio.create_task(execute_graph())

            while not task.done():
                live.update(
                    self._layout.update(
                        agent_states=self._agent_states,
                        messages=self._messages,
                        report=self._report,
                        uptime=self._get_uptime(),
                        ticker=self._ticker,
                        phase=self._phase,
                    )
                )
                await asyncio.sleep(1.0 / self._refresh_rate)

            await task

        self._running = False
        return result

    async def run_demo(self, ticker: str = "TSLA") -> None:
        self._ticker = ticker
        self._start_time = time.monotonic()
        self._running = True

        for team_agents in TEAM_LABELS.values():
            for agent in team_agents:
                self._agent_states[agent] = {"status": "pending"}

        demo_sequence = [
            ("market_data", "running", "tool", "Fetching price data..."),
            ("market_data", "running", "tool", "get_stock_price(TSLA)"),
            ("sentiment", "running", "tool", "get_stock_news(TSLA)"),
            ("fundamental", "running", "tool", "get_financials(TSLA)"),
            ("market_data", "completed", "signal", "Data collected"),
            ("sentiment", "completed", "reason", "Analyzing sentiment patterns..."),
            ("fundamental", "completed", "reason", "Revenue growth +25% YoY"),
            ("technical", "running", "tool", "compute_indicators()"),
            ("quantitative", "running", "tool", "run_factor_model()"),
            ("macro", "running", "tool", "get_economic_data()"),
            ("technical", "completed", "signal", "RSI=45, MACD bullish crossover"),
            ("quantitative", "completed", "signal", "Momentum factor: +0.3σ"),
            ("macro", "completed", "reason", "Rates stable, GDP growth solid"),
            ("bull", "running", "reason", "Building bullish case..."),
            ("bear", "running", "reason", "Analyzing risks..."),
            ("event", "running", "tool", "check_upcoming_events()"),
            ("bull", "completed", "signal", "BUY - Strong momentum + fundamentals"),
            ("bear", "completed", "signal", "HOLD - Valuation stretched"),
            ("event", "completed", "info", "Earnings in 15 days"),
            ("risk", "running", "risk", "Running risk checks..."),
            ("coordinator", "running", "reason", "Aggregating signals..."),
            ("risk", "completed", "risk", "All risk checks passed"),
            ("rl_agent", "running", "tool", "Evaluating policy..."),
            ("rl_agent", "completed", "signal", "Policy suggests small position"),
            ("coordinator", "completed", "reason", "Consensus reached"),
            ("portfolio", "running", "reason", "Calculating optimal position..."),
            ("portfolio", "completed", "signal", "Size: 2.5% of portfolio"),
            ("execution", "running", "tool", "Preparing VWAP order..."),
            ("execution", "completed", "info", "Order ready for execution"),
        ]

        signal_map = {
            "market_data": ("", 0.0),
            "sentiment": ("buy", 0.72),
            "fundamental": ("buy", 0.68),
            "technical": ("buy", 0.65),
            "quantitative": ("buy", 0.61),
            "macro": ("hold", 0.55),
            "bull": ("strong_buy", 0.82),
            "bear": ("hold", 0.45),
            "event": ("hold", 0.50),
            "risk": ("buy", 0.70),
            "coordinator": ("buy", 0.73),
            "rl_agent": ("buy", 0.58),
            "portfolio": ("buy", 0.75),
            "execution": ("buy", 0.78),
        }

        with Live(
            self._layout.update(
                agent_states=self._agent_states,
                messages=self._messages,
                uptime=self._get_uptime(),
                ticker=self._ticker,
                phase="Starting",
            ),
            console=self._console,
            refresh_per_second=self._refresh_rate,
            screen=True,
        ) as live:
            for agent, status, msg_type, content in demo_sequence:
                if not self._running:
                    break

                if status == "running":
                    self._agent_states[agent] = {"status": "running"}
                    self._phase = f"Running {agent.replace('_', ' ').title()}"
                elif status == "completed":
                    sig, conf = signal_map.get(agent, ("", 0.0))
                    self._agent_states[agent] = {
                        "status": "completed",
                        "signal": sig,
                        "confidence": conf,
                        "latency_ms": 150 + hash(agent) % 300,
                    }

                self.add_message(msg_type, agent, content)

                live.update(
                    self._layout.update(
                        agent_states=self._agent_states,
                        messages=self._messages,
                        report=self._report,
                        uptime=self._get_uptime(),
                        ticker=self._ticker,
                        phase=self._phase,
                    )
                )
                await asyncio.sleep(0.4)

            self._phase = "Complete"
            self._report = {
                "recommendation": "BUY",
                "ticker": ticker,
                "exposure": 2.5,
                "confidence": 0.73,
                "reasoning": (
                    f"Multi-agent consensus: 9/14 agents recommend BUY for {ticker}. "
                    f"Strong technical momentum (RSI=45, MACD bullish crossover), "
                    f"solid fundamentals (revenue +25% YoY), positive sentiment. "
                    f"Risk checks passed. Recommended position size: 2.5% of portfolio via VWAP."
                ),
                "total_cost": 0.0847,
            }

            live.update(
                self._layout.update(
                    agent_states=self._agent_states,
                    messages=self._messages,
                    report=self._report,
                    uptime=self._get_uptime(),
                    ticker=self._ticker,
                    phase=self._phase,
                )
            )
            await asyncio.sleep(5)

        self._running = False

    def run_sync(self, ticker: str = "TSLA") -> None:
        try:
            asyncio.run(self.run_demo(ticker))
        except KeyboardInterrupt:
            self._running = False
            self._console.print("\n[dim]NEXUS terminated.[/]")

    def stop(self) -> None:
        self._running = False


def main() -> None:
    import sys
    ticker = sys.argv[1] if len(sys.argv) > 1 else "TSLA"
    tui = TradingTUI()
    tui.run_sync(ticker)


if __name__ == "__main__":
    main()
