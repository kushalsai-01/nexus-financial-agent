from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nexus.core.types import Action, OrderSide, OrderType, Portfolio, Position
from nexus.risk.limits import RiskCheckResult, RiskLimitChecker, RiskLimits
from nexus.risk.monitor import RiskAlert, RiskMonitor
from nexus.risk.portfolio import PortfolioRiskAnalyzer
from nexus.risk.scenarios import MonteCarloSimulator
from nexus.risk.stress import HISTORICAL_SCENARIOS, StressTester


def _make_returns(n: int = 500, seed: int = 42) -> pd.Series:
    rng = np.random.RandomState(seed)
    return pd.Series(rng.normal(0.0003, 0.015, n), index=pd.date_range("2020-01-01", periods=n, freq="B"))


def _make_portfolio(cash: float = 50_000, positions: dict | None = None) -> Portfolio:
    p = Portfolio(cash=cash, total_value=100_000)
    if positions:
        for ticker, data in positions.items():
            pos = Position(
                ticker=ticker,
                quantity=data["qty"],
                entry_price=data["price"],
                current_price=data["price"],
            )
            pos.update_price(data["price"])
            p.positions[ticker] = pos
    p.update()
    return p


def _make_action(ticker: str = "AAPL", qty: float = 100, price: float = 150, side: OrderSide = OrderSide.BUY) -> Action:
    return Action(
        ticker=ticker,
        side=side,
        quantity=qty,
        order_type=OrderType.MARKET,
        limit_price=price,
        confidence=0.8,
    )


class TestPortfolioRiskAnalyzer:
    def test_var_historical(self) -> None:
        analyzer = PortfolioRiskAnalyzer()
        returns = _make_returns()
        var = analyzer.compute_var_historical(returns)
        assert var > 0

    def test_var_parametric(self) -> None:
        analyzer = PortfolioRiskAnalyzer()
        returns = _make_returns()
        var = analyzer.compute_var_parametric(returns)
        assert var > 0

    def test_var_monte_carlo(self) -> None:
        analyzer = PortfolioRiskAnalyzer()
        returns = _make_returns()
        var = analyzer.compute_var_monte_carlo(returns)
        assert var > 0

    def test_cvar(self) -> None:
        analyzer = PortfolioRiskAnalyzer()
        returns = _make_returns()
        cvar = analyzer.compute_cvar(returns)
        assert cvar > 0

    def test_cvar_greater_than_var(self) -> None:
        analyzer = PortfolioRiskAnalyzer()
        returns = _make_returns()
        var = analyzer.compute_var_historical(returns)
        cvar = analyzer.compute_cvar(returns)
        assert cvar >= var

    def test_var_all_horizons(self) -> None:
        analyzer = PortfolioRiskAnalyzer()
        returns = _make_returns()
        result = analyzer.compute_var_all_horizons(returns)
        assert "daily" in result
        assert "10_day" in result
        assert "30_day" in result
        assert result["30_day"]["historical"] > result["daily"]["historical"]

    def test_var_short_series(self) -> None:
        analyzer = PortfolioRiskAnalyzer()
        returns = pd.Series([0.01, -0.02])
        var = analyzer.compute_var_historical(returns)
        assert var == 0.0

    def test_correlation_matrix(self) -> None:
        analyzer = PortfolioRiskAnalyzer()
        rng = np.random.RandomState(42)
        r = {
            "AAPL": pd.Series(rng.normal(0, 0.02, 100)),
            "MSFT": pd.Series(rng.normal(0, 0.02, 100)),
        }
        corr = analyzer.compute_correlation_matrix(r)
        assert corr.shape == (2, 2)
        assert abs(corr.iloc[0, 0] - 1.0) < 0.01

    def test_portfolio_beta(self) -> None:
        analyzer = PortfolioRiskAnalyzer()
        rng = np.random.RandomState(42)
        bench = pd.Series(rng.normal(0, 0.01, 200))
        port = bench * 1.2 + rng.normal(0, 0.005, 200)
        beta = analyzer.compute_portfolio_beta(pd.Series(port), bench)
        assert 0.5 < beta < 2.0

    def test_portfolio_risk_summary(self) -> None:
        analyzer = PortfolioRiskAnalyzer()
        returns = _make_returns()
        summary = analyzer.compute_portfolio_risk_summary(returns)
        assert "annualized_volatility" in summary
        assert "var" in summary
        assert "skewness" in summary


class TestRiskLimits:
    def test_defaults(self) -> None:
        limits = RiskLimits()
        assert limits.max_position_pct == 0.15
        assert limits.max_leverage == 1.0
        assert limits.max_drawdown_pct == 0.20

    def test_check_all_pass(self) -> None:
        checker = RiskLimitChecker()
        portfolio = _make_portfolio(cash=90_000, positions={"AAPL": {"qty": 50, "price": 150}})
        action = _make_action(qty=10, price=150)
        results = checker.run_all_checks(action, portfolio)
        assert all(r.passed for r in results)

    def test_position_size_violation(self) -> None:
        checker = RiskLimitChecker(RiskLimits(max_position_pct=0.05))
        portfolio = _make_portfolio(cash=50_000, positions={"AAPL": {"qty": 100, "price": 150}})
        action = _make_action(qty=200, price=150)
        results = checker.run_all_checks(action, portfolio)
        size_check = next(r for r in results if r.check_name == "position_size")
        assert not size_check.passed

    def test_drawdown_violation(self) -> None:
        checker = RiskLimitChecker()
        portfolio = _make_portfolio()
        portfolio.max_drawdown = -0.25
        action = _make_action()
        results = checker.run_all_checks(action, portfolio)
        dd_check = next(r for r in results if r.check_name == "drawdown")
        assert not dd_check.passed

    def test_should_veto_critical(self) -> None:
        checker = RiskLimitChecker()
        results = [
            RiskCheckResult(passed=False, check_name="test", severity="critical"),
        ]
        assert checker.should_veto(results) is True

    def test_should_not_veto_warning(self) -> None:
        checker = RiskLimitChecker()
        results = [
            RiskCheckResult(passed=False, check_name="test", severity="warning"),
        ]
        assert checker.should_veto(results) is False


class TestMonteCarloSimulator:
    def test_parametric(self) -> None:
        sim = MonteCarloSimulator(n_simulations=100)
        returns = _make_returns()
        paths = sim.simulate_portfolio_paths(returns, horizon_days=50)
        assert paths.shape == (100, 50)

    def test_bootstrap(self) -> None:
        sim = MonteCarloSimulator(n_simulations=100)
        returns = _make_returns()
        result = sim.simulate_returns(returns, horizon_days=50, method="bootstrap")
        assert result.shape == (100, 50)

    def test_block_bootstrap(self) -> None:
        sim = MonteCarloSimulator(n_simulations=100)
        returns = _make_returns()
        result = sim.simulate_returns(returns, horizon_days=50, method="block_bootstrap")
        assert result.shape == (100, 50)

    def test_distribution_stats(self) -> None:
        sim = MonteCarloSimulator(n_simulations=500)
        returns = _make_returns()
        paths = sim.simulate_portfolio_paths(returns, horizon_days=50)
        stats = sim.compute_distribution_stats(paths)
        assert "mean_return" in stats
        assert "prob_loss" in stats
        assert "var_95" in stats
        assert 0 <= stats["prob_loss"] <= 1

    def test_invalid_method(self) -> None:
        sim = MonteCarloSimulator()
        with pytest.raises(ValueError):
            sim.simulate_returns(_make_returns(), method="invalid")


class TestStressTester:
    def test_apply_scenario(self) -> None:
        tester = StressTester()
        returns = _make_returns()
        result = tester.apply_scenario(returns, HISTORICAL_SCENARIOS["2008_financial_crisis"])
        assert "portfolio_impact_pct" in result
        assert "max_drawdown" in result

    def test_run_all_scenarios(self) -> None:
        tester = StressTester()
        returns = _make_returns()
        results = tester.run_all_scenarios(returns)
        assert len(results) == len(HISTORICAL_SCENARIOS)
        assert "2020_covid_crash" in results

    def test_custom_stress(self) -> None:
        tester = StressTester()
        returns = _make_returns()
        results = tester.custom_stress_test(returns, [-0.10, -0.20, -0.30])
        assert len(results) == 3

    def test_stress_summary(self) -> None:
        tester = StressTester()
        returns = _make_returns()
        results = tester.run_all_scenarios(returns)
        summary = tester.compute_stress_summary(results)
        assert "worst_scenario" in summary
        assert "avg_impact_pct" in summary


class TestRiskMonitor:
    def test_update_portfolio(self) -> None:
        monitor = RiskMonitor()
        portfolio = _make_portfolio()
        alerts = monitor.update_portfolio_state(portfolio)
        assert isinstance(alerts, list)

    def test_daily_loss_halt(self) -> None:
        monitor = RiskMonitor()
        portfolio = _make_portfolio()
        monitor._daily_pnl_start = 100_000
        portfolio.total_value = 96_000
        alerts = monitor.update_portfolio_state(portfolio)
        critical = [a for a in alerts if a.severity == "critical"]
        assert len(critical) > 0
        assert monitor.is_halted

    def test_vix_reduction(self) -> None:
        monitor = RiskMonitor()
        result = monitor.check_vix(35.0)
        assert result["action"] == "reduce_size"
        assert result["position_size_multiplier"] < 1.0

    def test_vix_halt(self) -> None:
        monitor = RiskMonitor()
        result = monitor.check_vix(50.0)
        assert result["action"] == "halt_trading"
        assert monitor.is_halted

    def test_pre_trade_check(self) -> None:
        monitor = RiskMonitor()
        portfolio = _make_portfolio(cash=90_000)
        action = _make_action(qty=10, price=150)
        result = monitor.pre_trade_check(action, portfolio)
        assert "approved" in result

    def test_pre_trade_halted(self) -> None:
        monitor = RiskMonitor()
        monitor._trading_halted = True
        result = monitor.pre_trade_check(_make_action(), _make_portfolio())
        assert result["approved"] is False

    def test_reset_daily(self) -> None:
        monitor = RiskMonitor()
        monitor._trading_halted = True
        monitor.reset_daily(100_000)
        assert not monitor.is_halted
