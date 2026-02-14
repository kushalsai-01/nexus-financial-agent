from __future__ import annotations

from datetime import datetime
from typing import Any

from rich.align import Align
from rich.columns import Columns
from rich.console import Group, RenderableType
from rich.layout import Layout
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table
from rich.text import Text

from nexus.ui.themes import (
    AGENT_COLORS,
    MESSAGE_TYPE_STYLES,
    NEXUS_BANNER,
    SIGNAL_COLORS,
    STATUS_COLORS,
    TEAM_LABELS,
    format_currency,
    format_duration,
    format_pct,
    get_agent_style,
    get_signal_style,
    get_status_style,
)


def make_banner() -> Panel:
    banner_text = Text(NEXUS_BANNER, style="bright_cyan")
    welcome = Text("Welcome to NEXUS", style="bold bright_white")
    subtitle = Text("Multi-Agent Autonomous Trading Framework", style="dim white")
    workflow = Text("Workflow: Analyst → Research → Trader → Risk → Portfolio", style="bright_yellow")
    content = Group(
        Align.center(welcome),
        Text(""),
        Align.center(banner_text),
        Text(""),
        Align.center(subtitle),
        Text(""),
        Align.center(workflow),
    )
    return Panel(content, border_style="bright_cyan", padding=(1, 2))


def make_progress_table(
    agent_states: dict[str, dict[str, Any]],
) -> Table:
    table = Table(
        show_header=True,
        header_style="bold bright_white",
        border_style="bright_cyan",
        expand=True,
        pad_edge=True,
    )
    table.add_column("Team", style="bold", width=12)
    table.add_column("Agent", width=14)
    table.add_column("Status", width=10)
    table.add_column("Signal", width=8)
    table.add_column("Conf", width=6, justify="right")
    table.add_column("Time", width=8, justify="right")

    for team_name, agents in TEAM_LABELS.items():
        first = True
        for agent_name in agents:
            state = agent_states.get(agent_name, {})
            status = state.get("status", "pending")
            signal = state.get("signal", "")
            confidence = state.get("confidence", 0.0)
            latency = state.get("latency_ms", 0.0)

            team_cell = team_name if first else ""
            status_style = get_status_style(status)
            agent_style = get_agent_style(agent_name)
            display_name = agent_name.replace("_", " ").title()

            signal_text = Text(signal, style=get_signal_style(signal)) if signal else Text("-", style="dim")
            conf_text = f"{confidence:.0%}" if confidence > 0 else "-"
            time_text = f"{latency:.0f}ms" if latency > 0 else "-"

            table.add_row(
                Text(team_cell, style="bold bright_yellow"),
                Text(display_name, style=agent_style),
                Text(status.capitalize(), style=status_style),
                signal_text,
                conf_text,
                time_text,
            )
            first = False
        table.add_row("", "", "", "", "", "")

    return table


def make_messages_table(
    messages: list[dict[str, Any]],
    max_rows: int = 20,
) -> Table:
    table = Table(
        show_header=True,
        header_style="bold bright_white",
        border_style="bright_cyan",
        expand=True,
        pad_edge=True,
    )
    table.add_column("Time", width=10, style="dim")
    table.add_column("Type", width=8)
    table.add_column("Agent", width=12)
    table.add_column("Content", ratio=1)

    recent = messages[-max_rows:] if len(messages) > max_rows else messages
    for msg in recent:
        timestamp = msg.get("timestamp", "")
        if isinstance(timestamp, datetime):
            timestamp = timestamp.strftime("%H:%M:%S")
        elif isinstance(timestamp, str) and len(timestamp) > 8:
            timestamp = timestamp[11:19] if "T" in timestamp else timestamp[:8]

        msg_type = msg.get("type", "info")
        style, label = MESSAGE_TYPE_STYLES.get(msg_type, ("white", "Info"))
        agent = msg.get("agent", "")
        content = msg.get("content", "")

        if len(content) > 60:
            content = content[:57] + "..."

        table.add_row(
            timestamp,
            Text(label, style=style),
            Text(agent, style=get_agent_style(agent)),
            content,
        )

    return table


def make_report_panel(report: dict[str, Any]) -> Panel:
    recommendation = report.get("recommendation", "HOLD")
    ticker = report.get("ticker", "")
    exposure = report.get("exposure", 0.0)
    confidence = report.get("confidence", 0.0)
    reasoning = report.get("reasoning", "")
    cost = report.get("total_cost", 0.0)

    rec_color = "bright_green" if "BUY" in recommendation.upper() else (
        "bright_red" if "SELL" in recommendation.upper() else "bright_yellow"
    )

    header = Text(f"Recommendation: {recommendation}", style=f"bold {rec_color}")
    details = Text()
    details.append(f"\nTicker: ", style="dim")
    details.append(ticker, style="bold bright_white")
    details.append(f"  |  Exposure: ", style="dim")
    details.append(format_pct(exposure), style="bright_cyan")
    details.append(f"  |  Confidence: ", style="dim")
    details.append(f"{confidence:.0%}", style="bright_cyan")
    details.append(f"  |  Cost: ", style="dim")
    details.append(f"${cost:.4f}", style="bright_yellow")

    if reasoning:
        details.append(f"\n\n{reasoning}", style="white")

    content = Group(Align.center(header), details)
    return Panel(
        content,
        title="[bold bright_white]Current Report[/]",
        border_style="bright_cyan",
        padding=(1, 2),
    )


def make_portfolio_summary(portfolio: dict[str, Any]) -> Panel:
    table = Table(show_header=False, expand=True, pad_edge=True, box=None)
    table.add_column("Label", style="dim", width=18)
    table.add_column("Value", style="bright_white")

    total_value = portfolio.get("total_value", 0)
    daily_pnl = portfolio.get("daily_pnl", 0)
    total_pnl = portfolio.get("total_pnl", 0)
    positions = portfolio.get("positions_count", 0)
    cash = portfolio.get("cash", 0)

    pnl_style = "bright_green" if daily_pnl >= 0 else "bright_red"
    total_pnl_style = "bright_green" if total_pnl >= 0 else "bright_red"

    table.add_row("Portfolio Value", Text(format_currency(total_value), style="bold bright_white"))
    table.add_row("Today's P&L", Text(format_currency(daily_pnl), style=pnl_style))
    table.add_row("Total P&L", Text(format_currency(total_pnl), style=total_pnl_style))
    table.add_row("Cash", format_currency(cash))
    table.add_row("Positions", str(positions))

    return Panel(table, title="[bold]Portfolio[/]", border_style="bright_cyan")


def make_metrics_panel(metrics: dict[str, Any]) -> Panel:
    table = Table(show_header=False, expand=True, pad_edge=True, box=None)
    table.add_column("Metric", style="dim", width=18)
    table.add_column("Value", style="bright_white")

    table.add_row("Decisions", str(int(metrics.get("decisions_total", 0))))
    table.add_row("Dec/Min", f"{metrics.get('decisions_per_minute', 0):.1f}")
    table.add_row("LLM Cost", f"${metrics.get('llm_cost_total_usd', 0):.4f}")
    table.add_row("Cost/Decision", f"${metrics.get('cost_per_decision', 0):.6f}")
    table.add_row("Avg Latency", f"{metrics.get('agent_latency_avg_ms', 0):.0f}ms")
    table.add_row("API Success", f"{metrics.get('api_success_rate', 1) * 100:.1f}%")

    return Panel(table, title="[bold]System Metrics[/]", border_style="bright_yellow")


def make_status_bar(
    uptime: float,
    ticker: str = "",
    phase: str = "",
) -> Text:
    bar = Text()
    bar.append(" NEXUS ", style="bold bright_white on blue")
    bar.append(" ")
    if ticker:
        bar.append(f" {ticker} ", style="bold bright_white on bright_cyan")
        bar.append(" ")
    if phase:
        bar.append(f" {phase} ", style="bold black on bright_yellow")
        bar.append(" ")
    bar.append(f" Uptime: {format_duration(uptime)} ", style="dim")
    bar.append(" [q]Quit  [r]Restart ", style="dim")
    return bar
