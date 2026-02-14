from __future__ import annotations

from datetime import date, datetime
from unittest.mock import MagicMock

import pytest

from nexus.reports.daily import DailyReport
from nexus.reports.weekly import WeeklyReport


class TestDailyReport:
    def _make_portfolio(self) -> MagicMock:
        portfolio = MagicMock()
        portfolio.total_value = 105_000.0
        portfolio.daily_pnl = 2_500.0
        portfolio.cash = 20_000.0
        portfolio.positions = {
            "AAPL": MagicMock(
                quantity=100, entry_price=150.0, current_price=175.0,
                unrealized_pnl=2500.0, weight=0.15,
            ),
        }
        return portfolio

    def test_generate_empty(self) -> None:
        report = DailyReport()
        data = report.generate()
        assert data["report_type"] == "daily"
        assert data["portfolio"]["total_value"] == 0.0
        assert data["trading"]["total_trades"] == 0

    def test_generate_with_portfolio(self) -> None:
        portfolio = self._make_portfolio()
        report = DailyReport(portfolio=portfolio)
        data = report.generate()
        assert data["portfolio"]["total_value"] == 105_000.0
        assert data["portfolio"]["daily_pnl"] == 2_500.0
        assert len(data["positions"]) == 1
        assert data["positions"][0]["ticker"] == "AAPL"

    def test_to_text(self) -> None:
        portfolio = self._make_portfolio()
        report = DailyReport(portfolio=portfolio)
        text = report.to_text()
        assert "NEXUS Daily Report" in text
        assert "PORTFOLIO SUMMARY" in text
        assert "AAPL" in text

    def test_to_html(self) -> None:
        portfolio = self._make_portfolio()
        report = DailyReport(portfolio=portfolio)
        html = report.to_html()
        assert "<!DOCTYPE html>" in html
        assert "NEXUS Daily Report" in html
        assert "$105,000" in html

    def test_add_alerts(self) -> None:
        report = DailyReport()
        report.add_alerts([{"severity": "warning", "message": "test alert"}])
        data = report.generate()
        assert len(data["alerts"]) == 1


class TestWeeklyReport:
    def test_generate_empty(self) -> None:
        report = WeeklyReport()
        data = report.generate()
        assert data["report_type"] == "weekly"
        assert data["trading"]["total_trades"] == 0

    def test_to_text(self) -> None:
        report = WeeklyReport()
        text = report.to_text()
        assert "NEXUS Weekly Report" in text
        assert "PERFORMANCE" in text

    def test_to_html(self) -> None:
        report = WeeklyReport()
        html = report.to_html()
        assert "<!DOCTYPE html>" in html
        assert "NEXUS Weekly Report" in html

    def test_daily_values(self) -> None:
        report = WeeklyReport()
        report.add_daily_values([
            {"date": "2024-01-15", "value": 100000},
            {"date": "2024-01-16", "value": 101000},
            {"date": "2024-01-17", "value": 102500},
        ])
        data = report.generate()
        assert data["performance"]["weekly_return_pct"] == pytest.approx(2.5, abs=0.1)
