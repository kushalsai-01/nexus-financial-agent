from __future__ import annotations

from typing import Any

from rich.layout import Layout
from rich.panel import Panel
from rich.text import Text

from nexus.ui.components import (
    make_banner,
    make_messages_table,
    make_metrics_panel,
    make_portfolio_summary,
    make_progress_table,
    make_report_panel,
    make_status_bar,
)


class NexusLayout:
    def __init__(self) -> None:
        self._layout = Layout()
        self._setup_layout()

    def _setup_layout(self) -> None:
        self._layout.split_column(
            Layout(name="header", size=14),
            Layout(name="body", ratio=3),
            Layout(name="report", size=12),
            Layout(name="footer", size=1),
        )

        self._layout["body"].split_row(
            Layout(name="progress", ratio=2),
            Layout(name="messages", ratio=3),
        )

    def update(
        self,
        agent_states: dict[str, dict[str, Any]] | None = None,
        messages: list[dict[str, Any]] | None = None,
        report: dict[str, Any] | None = None,
        uptime: float = 0.0,
        ticker: str = "",
        phase: str = "",
    ) -> Layout:
        self._layout["header"].update(make_banner())

        progress_table = make_progress_table(agent_states or {})
        self._layout["progress"].update(
            Panel(progress_table, title="[bold]Progress[/]", border_style="bright_cyan")
        )

        messages_table = make_messages_table(messages or [])
        self._layout["messages"].update(
            Panel(messages_table, title="[bold]Messages & Tools[/]", border_style="bright_cyan")
        )

        if report:
            self._layout["report"].update(make_report_panel(report))
        else:
            self._layout["report"].update(
                Panel(
                    Text("Awaiting analysis...", style="dim"),
                    title="[bold]Current Report[/]",
                    border_style="dim",
                )
            )

        self._layout["footer"].update(make_status_bar(uptime, ticker, phase))
        return self._layout

    @property
    def layout(self) -> Layout:
        return self._layout


class DashboardLayout:
    def __init__(self) -> None:
        self._layout = Layout()
        self._setup_layout()

    def _setup_layout(self) -> None:
        self._layout.split_column(
            Layout(name="top_bar", size=1),
            Layout(name="main", ratio=1),
            Layout(name="bottom_bar", size=1),
        )

        self._layout["main"].split_row(
            Layout(name="left", ratio=1),
            Layout(name="right", ratio=2),
        )

        self._layout["left"].split_column(
            Layout(name="portfolio", ratio=1),
            Layout(name="metrics", ratio=1),
        )

    def update(
        self,
        portfolio: dict[str, Any] | None = None,
        metrics: dict[str, Any] | None = None,
        content: Any = None,
        uptime: float = 0.0,
    ) -> Layout:
        self._layout["top_bar"].update(make_status_bar(uptime))
        self._layout["portfolio"].update(make_portfolio_summary(portfolio or {}))
        self._layout["metrics"].update(make_metrics_panel(metrics or {}))

        if content:
            self._layout["right"].update(content)
        else:
            self._layout["right"].update(Panel("No data", border_style="dim"))

        self._layout["bottom_bar"].update(
            Text(" Press 'q' to quit | 'r' to refresh | 'h' for help ", style="dim")
        )
        return self._layout

    @property
    def layout(self) -> Layout:
        return self._layout


class CompactLayout:
    def __init__(self) -> None:
        self._layout = Layout()
        self._setup_layout()

    def _setup_layout(self) -> None:
        self._layout.split_column(
            Layout(name="header", size=3),
            Layout(name="body", ratio=1),
            Layout(name="footer", size=1),
        )

    def update(
        self,
        content: Any = None,
        title: str = "NEXUS",
        uptime: float = 0.0,
    ) -> Layout:
        header = Text()
        header.append(" NEXUS ", style="bold bright_white on blue")
        header.append(f"  {title}", style="bright_cyan")
        self._layout["header"].update(Panel(header, border_style="bright_cyan"))

        if content:
            self._layout["body"].update(content)
        else:
            self._layout["body"].update(Panel("Ready", border_style="dim"))

        self._layout["footer"].update(make_status_bar(uptime))
        return self._layout
