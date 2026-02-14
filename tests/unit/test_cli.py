from __future__ import annotations

import pytest
from click.testing import CliRunner

from nexus.cli.main import main


class TestCLI:
    def setup_method(self) -> None:
        self.runner = CliRunner()

    def test_help(self) -> None:
        result = self.runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "NEXUS" in result.output

    def test_version(self) -> None:
        result = self.runner.invoke(main, ["--version"])
        assert result.exit_code == 0
        assert "0.1.0" in result.output

    def test_run_help(self) -> None:
        result = self.runner.invoke(main, ["run", "--help"])
        assert result.exit_code == 0
        assert "--ticker" in result.output
        assert "--capital" in result.output
        assert "--demo" in result.output

    def test_backtest_help(self) -> None:
        result = self.runner.invoke(main, ["backtest", "--help"])
        assert result.exit_code == 0
        assert "--ticker" in result.output
        assert "--start" in result.output

    def test_analyze_help(self) -> None:
        result = self.runner.invoke(main, ["analyze", "--help"])
        assert result.exit_code == 0
        assert "--depth" in result.output

    def test_report_help(self) -> None:
        result = self.runner.invoke(main, ["report", "--help"])
        assert result.exit_code == 0
        assert "--type" in result.output

    def test_status_help(self) -> None:
        result = self.runner.invoke(main, ["status", "--help"])
        assert result.exit_code == 0

    def test_dashboard_help(self) -> None:
        result = self.runner.invoke(main, ["dashboard", "--help"])
        assert result.exit_code == 0
        assert "--port" in result.output

    def test_costs_help(self) -> None:
        result = self.runner.invoke(main, ["costs", "--help"])
        assert result.exit_code == 0
        assert "--format" in result.output

    def test_init_help(self) -> None:
        result = self.runner.invoke(main, ["init", "--help"])
        assert result.exit_code == 0

    def test_report_daily_text(self) -> None:
        result = self.runner.invoke(main, ["report", "--type", "daily"])
        assert result.exit_code == 0
        assert "NEXUS Daily Report" in result.output

    def test_costs_text(self) -> None:
        result = self.runner.invoke(main, ["costs"])
        assert result.exit_code == 0
        assert "Cost Summary" in result.output
