from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Optional

import click

from nexus.core.logging import get_logger, setup_logging

logger = get_logger("cli")

CONTEXT_SETTINGS = {"help_option_names": ["-h", "--help"]}


class NexusContext:
    def __init__(self) -> None:
        self.config_path: str | None = None
        self.verbose: bool = False
        self.debug: bool = False


pass_context = click.make_pass_decorator(NexusContext, ensure=True)


@click.group(context_settings=CONTEXT_SETTINGS)
@click.version_option(version="0.1.0", prog_name="nexus")
@click.option("-c", "--config", "config_path", type=click.Path(exists=True), help="Path to configuration file.")
@click.option("-v", "--verbose", is_flag=True, help="Enable verbose output.")
@click.option("--debug", is_flag=True, help="Enable debug mode.")
@pass_context
def main(ctx: NexusContext, config_path: str | None, verbose: bool, debug: bool) -> None:
    """NEXUS — Autonomous Multi-Agent Hedge Fund System."""
    ctx.config_path = config_path
    ctx.verbose = verbose
    ctx.debug = debug
    level = "DEBUG" if debug else ("INFO" if verbose else "WARNING")
    setup_logging(log_level=level)


@main.command()
@click.option("--ticker", "-t", multiple=True, required=True, help="Ticker symbol(s) to analyze.")
@click.option("--capital", "-k", type=float, default=100_000, show_default=True, help="Starting capital.")
@click.option("--max-position", type=float, default=0.1, show_default=True, help="Max position pct (0-1).")
@click.option("--paper", is_flag=True, help="Paper trading mode (no real execution).")
@click.option("--ui/--no-ui", "show_ui", default=True, show_default=True, help="Show terminal UI.")
@click.option("--demo", is_flag=True, help="Run demo mode with simulated data.")
@pass_context
def run(
    ctx: NexusContext,
    ticker: tuple[str, ...],
    capital: float,
    max_position: float,
    paper: bool,
    show_ui: bool,
    demo: bool,
) -> None:
    """Run the NEXUS trading agent system."""
    from nexus.ui.tui import TradingTUI

    tui = TradingTUI()

    if demo:
        click.echo(click.style("Starting NEXUS demo mode...", fg="cyan", bold=True))
        tui.run_sync(demo=True)
        return

    click.echo(click.style("Starting NEXUS...", fg="cyan", bold=True))
    click.echo(f"  Tickers:      {', '.join(ticker)}")
    click.echo(f"  Capital:      ${capital:,.0f}")
    click.echo(f"  Max Position: {max_position:.0%}")
    click.echo(f"  Paper Mode:   {paper}")

    from nexus.core.config import get_config
    config = get_config()

    if ctx.config_path:
        click.echo(f"  Config:       {ctx.config_path}")

    async def _run() -> None:
        from nexus.orchestration.graph import TradingGraph

        graph = TradingGraph(config=config)
        if show_ui:
            await tui.run_with_graph(
                graph=graph,
                tickers=list(ticker),
                capital=capital,
            )
        else:
            for tk in ticker:
                result = await graph.run(ticker=tk, capital=capital)
                _print_result(tk, result)

    asyncio.run(_run())


def _print_result(ticker: str, result: dict) -> None:
    click.echo("")
    click.echo(click.style(f"═══ {ticker} ═══", fg="cyan", bold=True))
    recommendation = result.get("recommendation", "HOLD")
    colors = {"BUY": "green", "SELL": "red", "HOLD": "yellow", "SHORT": "magenta"}
    click.echo(f"  Recommendation: {click.style(recommendation, fg=colors.get(recommendation, 'white'), bold=True)}")
    click.echo(f"  Confidence:     {result.get('confidence', 0):.1%}")
    click.echo(f"  Exposure:       {result.get('exposure', 0):.1%}")
    click.echo(f"  Reasoning:      {result.get('reasoning', 'N/A')}")
    if result.get("cost_usd"):
        click.echo(f"  LLM Cost:       ${result['cost_usd']:.4f}")


@main.command()
@click.option("--ticker", "-t", multiple=True, required=True, help="Ticker(s) to backtest.")
@click.option("--start", "-s", type=click.DateTime(formats=["%Y-%m-%d"]), required=True, help="Start date (YYYY-MM-DD).")
@click.option("--end", "-e", type=click.DateTime(formats=["%Y-%m-%d"]), help="End date (YYYY-MM-DD).")
@click.option("--capital", "-k", type=float, default=100_000, show_default=True, help="Starting capital.")
@click.option("--output", "-o", type=click.Path(), help="Output report file path.")
@click.option("--format", "output_format", type=click.Choice(["text", "html", "json"]), default="text", show_default=True)
@pass_context
def backtest(
    ctx: NexusContext,
    ticker: tuple[str, ...],
    start: object,
    end: object | None,
    capital: float,
    output: str | None,
    output_format: str,
) -> None:
    """Run a backtest over historical data."""
    from datetime import datetime

    start_date = start if isinstance(start, datetime) else datetime.now()
    end_date = end if isinstance(end, datetime) else datetime.now()

    click.echo(click.style("Starting NEXUS Backtest...", fg="cyan", bold=True))
    click.echo(f"  Tickers:  {', '.join(ticker)}")
    click.echo(f"  Period:   {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    click.echo(f"  Capital:  ${capital:,.0f}")
    click.echo("")

    from nexus.core.config import get_config

    config = get_config()

    async def _backtest() -> None:
        from nexus.backtest.engine import BacktestEngine

        engine = BacktestEngine(config=config)
        with click.progressbar(length=100, label="Running backtest") as bar:
            result = await engine.run(
                tickers=list(ticker),
                start_date=start_date.strftime("%Y-%m-%d"),
                end_date=end_date.strftime("%Y-%m-%d"),
                initial_capital=capital,
            )
            bar.update(100)

        from nexus.reports.backtest import BacktestReport

        report = BacktestReport(result=result)

        if output_format == "html":
            content = report.to_html()
        elif output_format == "json":
            import json
            content = json.dumps(report.generate(), indent=2, default=str)
        else:
            content = report.to_text()

        if output:
            Path(output).write_text(content, encoding="utf-8")
            click.echo(f"\nReport saved to: {output}")
        else:
            click.echo(content)

    asyncio.run(_backtest())


@main.command()
@click.option("--ticker", "-t", multiple=True, required=True, help="Ticker(s) to analyze.")
@click.option("--depth", type=click.Choice(["quick", "standard", "deep"]), default="standard", show_default=True)
@click.option("--output", "-o", type=click.Path(), help="Output file path.")
@pass_context
def analyze(ctx: NexusContext, ticker: tuple[str, ...], depth: str, output: str | None) -> None:
    """Run analysis on specified tickers without trading."""
    click.echo(click.style("Running NEXUS Analysis...", fg="cyan", bold=True))
    click.echo(f"  Tickers:  {', '.join(ticker)}")
    click.echo(f"  Depth:    {depth}")

    from nexus.core.config import get_config

    config = get_config()

    async def _analyze() -> None:
        from nexus.orchestration.graph import TradingGraph

        graph = TradingGraph(config=config)

        for tk in ticker:
            click.echo(f"\nAnalyzing {click.style(tk, fg='cyan', bold=True)}...")
            result = await graph.run(ticker=tk, capital=0)
            _print_result(tk, result)

    asyncio.run(_analyze())


@main.command()
@click.option("--type", "report_type", type=click.Choice(["daily", "weekly"]), default="daily", show_default=True)
@click.option("--format", "output_format", type=click.Choice(["text", "html"]), default="text", show_default=True)
@click.option("--output", "-o", type=click.Path(), help="Output file path.")
def report(report_type: str, output_format: str, output: str | None) -> None:
    """Generate a performance report."""
    click.echo(click.style(f"Generating {report_type} report...", fg="cyan", bold=True))

    if report_type == "daily":
        from nexus.reports.daily import DailyReport
        r = DailyReport()
        content = r.to_html() if output_format == "html" else r.to_text()
    else:
        from nexus.reports.weekly import WeeklyReport
        r = WeeklyReport()
        content = r.to_html() if output_format == "html" else r.to_text()

    if output:
        Path(output).write_text(content, encoding="utf-8")
        click.echo(f"Report saved to: {output}")
    else:
        click.echo(content)


@main.command()
def status() -> None:
    """Show system status and health checks."""
    click.echo(click.style("NEXUS System Status", fg="cyan", bold=True))
    click.echo("")

    async def _status() -> None:
        from nexus.monitoring.health import HealthChecker

        checker = HealthChecker()
        report = await checker.run_all_checks()

        status_icons = {"healthy": "●", "degraded": "◐", "unhealthy": "○"}
        status_colors = {"healthy": "green", "degraded": "yellow", "unhealthy": "red"}

        for component in report.get("components", []):
            comp_status = component.get("status", "unhealthy")
            icon = status_icons.get(comp_status, "?")
            color = status_colors.get(comp_status, "white")
            latency_ms = component.get("latency_ms", 0)
            latency = f" ({latency_ms:.0f}ms)" if latency_ms else ""
            styled = click.style(f"{icon} {component['name']}", fg=color)
            click.echo(f"  {styled}{latency}")
            msg = component.get("message", "")
            if msg:
                click.echo(f"    └─ {click.style(msg, fg='red')}")

        overall = report.get("status", "unhealthy")
        click.echo("")
        click.echo(f"  Overall: {click.style(overall.upper(), fg=status_colors.get(overall, 'red'), bold=True)}")

    asyncio.run(_status())


@main.command()
@click.option("--host", default="0.0.0.0", show_default=True, help="Dashboard host.")
@click.option("--port", type=int, default=8501, show_default=True, help="Dashboard port.")
def dashboard(host: str, port: int) -> None:
    """Launch the Streamlit web dashboard."""
    import subprocess

    dashboard_path = Path(__file__).parent.parent / "ui" / "dashboard" / "app.py"
    if not dashboard_path.exists():
        click.echo(click.style("Dashboard app not found!", fg="red"))
        sys.exit(1)

    click.echo(click.style(f"Launching dashboard at http://{host}:{port}", fg="cyan", bold=True))
    subprocess.run(
        [
            sys.executable, "-m", "streamlit", "run",
            str(dashboard_path),
            "--server.address", host,
            "--server.port", str(port),
            "--server.headless", "true",
        ],
        check=True,
    )


@main.command()
@click.option("--format", "output_format", type=click.Choice(["text", "json"]), default="text", show_default=True)
def costs(output_format: str) -> None:
    """Show LLM cost tracking summary."""
    from nexus.monitoring.cost import CostTracker

    tracker = CostTracker()
    summary = tracker.get_summary()

    if output_format == "json":
        import json
        click.echo(json.dumps(summary, indent=2, default=str))
        return

    click.echo(click.style("NEXUS LLM Cost Summary", fg="cyan", bold=True))
    click.echo("")
    click.echo(f"  Total Cost:        ${summary.get('total_cost_usd', 0):.4f}")
    click.echo(f"  Today's Cost:      ${summary.get('today_cost_usd', 0):.4f}")
    click.echo(f"  Budget Remaining:  ${summary.get('budget_remaining_usd', 0):.2f}")
    click.echo(f"  Budget Used:       {summary.get('budget_utilization_pct', 0):.1f}%")
    click.echo(f"  Total Records:     {summary.get('total_records', 0)}")

    by_agent = tracker.cost_by_agent()
    if by_agent:
        click.echo("")
        click.echo(click.style("  Cost by Agent:", fg="yellow"))
        for agent, cost in sorted(by_agent.items(), key=lambda x: x[1], reverse=True):
            click.echo(f"    {agent:<20} ${cost:.4f}")


@main.command()
def init() -> None:
    """Initialize a new NEXUS project with default configuration."""
    config_dir = Path("config")
    config_dir.mkdir(exist_ok=True)

    default_config = {
        "logging": {"level": "INFO", "log_dir": "logs", "format": "json"},
        "llm": {
            "primary": {"provider": "anthropic", "model": "claude-sonnet-4-20250514", "temperature": 0.1},
            "fallback": {"provider": "openai", "model": "gpt-4o", "temperature": 0.1},
        },
        "risk": {
            "max_position_pct": 0.10,
            "max_portfolio_risk": 0.25,
            "max_drawdown_pct": 0.15,
            "daily_loss_limit_pct": 0.05,
        },
        "execution": {"mode": "paper", "broker": "alpaca"},
        "monitoring": {
            "prometheus": {"enabled": False, "port": 9090},
            "langfuse": {"enabled": False},
        },
    }

    import yaml

    config_path = config_dir / "nexus.yaml"
    if config_path.exists():
        click.confirm(f"{config_path} already exists. Overwrite?", abort=True)

    config_path.write_text(yaml.dump(default_config, default_flow_style=False, sort_keys=False), encoding="utf-8")

    env_example = Path(".env.example")
    if not env_example.exists():
        env_example.write_text(
            "ANTHROPIC_API_KEY=\nOPENAI_API_KEY=\nALPACA_API_KEY=\nALPACA_SECRET_KEY=\n",
            encoding="utf-8",
        )

    Path("logs").mkdir(exist_ok=True)
    Path("data").mkdir(exist_ok=True)
    Path("reports").mkdir(exist_ok=True)

    click.echo(click.style("NEXUS project initialized!", fg="green", bold=True))
    click.echo(f"  Config:  {config_path}")
    click.echo(f"  Next:    Copy .env.example to .env and add your API keys")


if __name__ == "__main__":
    main()
