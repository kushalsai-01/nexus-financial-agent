from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from nexus.backtest.costs import TransactionCostModel
from nexus.backtest.engine import BacktestConfig, BacktestEngine
from nexus.backtest.metrics import (
    compute_all_metrics,
    compute_cagr,
    compute_calmar,
    compute_max_drawdown,
    compute_profit_factor,
    compute_sharpe,
    compute_sortino,
    compute_total_return,
    compute_volatility,
    compute_win_rate,
)
from nexus.backtest.regime import MarketRegime, RegimeClassifier
from nexus.backtest.reports import BacktestReport
from nexus.backtest.strategy import MeanReversionStrategy, MomentumStrategy
from nexus.backtest.validation import LookaheadBiasChecker, WalkForwardValidator


def _make_equity_curve(n: int = 500, drift: float = 0.0003, vol: float = 0.015, seed: int = 42) -> pd.Series:
    rng = np.random.RandomState(seed)
    returns = rng.normal(drift, vol, n)
    prices = 100_000 * np.cumprod(1 + returns)
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    return pd.Series(prices, index=dates)


def _make_ohlcv(n: int = 500, seed: int = 42) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    close = 100 * np.cumprod(1 + rng.normal(0.0003, 0.015, n))
    high = close * (1 + rng.uniform(0, 0.02, n))
    low = close * (1 - rng.uniform(0, 0.02, n))
    opn = close * (1 + rng.normal(0, 0.005, n))
    volume = rng.randint(100_000, 5_000_000, n)
    return pd.DataFrame({"open": opn, "high": high, "low": low, "close": close, "volume": volume}, index=dates)


class TestTransactionCostModel:
    def test_commission(self) -> None:
        model = TransactionCostModel()
        assert model.compute_commission(100_000) == max(100_000 * 0.0001, 1.0)

    def test_min_commission(self) -> None:
        model = TransactionCostModel()
        assert model.compute_commission(100) == 1.0

    def test_spread_cost(self) -> None:
        model = TransactionCostModel()
        cost = model.compute_spread_cost(100, 100)
        assert cost > 0

    def test_market_impact_zero_volume(self) -> None:
        model = TransactionCostModel()
        assert model.compute_market_impact(100, 0, 0.02) == 0.0

    def test_slippage_liquid(self) -> None:
        model = TransactionCostModel()
        slip = model.compute_slippage(100, 100, 5_000_000)
        assert slip > 0
        assert slip < 1.0

    def test_slippage_illiquid(self) -> None:
        model = TransactionCostModel()
        slip_illiquid = model.compute_slippage(100, 100, 10_000)
        slip_liquid = model.compute_slippage(100, 100, 5_000_000)
        assert slip_illiquid > slip_liquid

    def test_total_cost(self) -> None:
        model = TransactionCostModel()
        costs = model.total_cost(100, 100)
        assert "commission" in costs
        assert "spread_cost" in costs
        assert "total_cost" in costs
        assert costs["total_cost"] > 0
        assert costs["cost_pct"] > 0


class TestMetrics:
    def test_total_return(self) -> None:
        equity = _make_equity_curve()
        ret = compute_total_return(equity)
        assert isinstance(ret, float)

    def test_total_return_empty(self) -> None:
        assert compute_total_return(pd.Series([100])) == 0.0

    def test_cagr(self) -> None:
        equity = _make_equity_curve()
        cagr = compute_cagr(equity)
        assert cagr > 0

    def test_cagr_negative(self) -> None:
        equity = _make_equity_curve(drift=-0.001)
        cagr = compute_cagr(equity)
        assert cagr < 0.1

    def test_volatility(self) -> None:
        returns = _make_equity_curve().pct_change().dropna()
        vol = compute_volatility(returns)
        assert vol > 0
        assert vol < 1.0

    def test_sharpe(self) -> None:
        returns = _make_equity_curve().pct_change().dropna()
        sharpe = compute_sharpe(returns)
        assert isinstance(sharpe, float)

    def test_sharpe_zero_std(self) -> None:
        returns = pd.Series([0.0, 0.0, 0.0])
        assert compute_sharpe(returns) == 0.0

    def test_sortino(self) -> None:
        returns = _make_equity_curve().pct_change().dropna()
        sortino = compute_sortino(returns)
        assert isinstance(sortino, float)

    def test_max_drawdown(self) -> None:
        equity = _make_equity_curve()
        max_dd, duration = compute_max_drawdown(equity)
        assert max_dd <= 0
        assert duration >= 0

    def test_calmar(self) -> None:
        equity = _make_equity_curve()
        calmar = compute_calmar(equity)
        assert isinstance(calmar, float)

    def test_win_rate(self) -> None:
        pnls = [100, -50, 200, -30, 150]
        wr = compute_win_rate(pnls)
        assert wr == 0.6

    def test_win_rate_empty(self) -> None:
        assert compute_win_rate([]) == 0.0

    def test_profit_factor(self) -> None:
        pnls = [100, -50, 200, -30]
        pf = compute_profit_factor(pnls)
        assert pf == (300 / 80)

    def test_profit_factor_no_losses(self) -> None:
        assert compute_profit_factor([100, 200]) == float("inf")

    def test_compute_all_metrics(self) -> None:
        equity = _make_equity_curve()
        pnls = [100, -50, 200, -30, 150, -20]
        metrics = compute_all_metrics(equity, pnls)
        assert "total_return" in metrics
        assert "sharpe_ratio" in metrics
        assert "max_drawdown" in metrics
        assert "win_rate" in metrics


class TestStrategies:
    def test_momentum_signals(self) -> None:
        data = _make_ohlcv()
        strategy = MomentumStrategy()
        signals = strategy.generate_signals(data)
        assert "signal" in signals.columns
        assert len(signals) == len(data)
        assert set(signals["signal"].unique()).issubset({-1, 0, 1})

    def test_mean_reversion_signals(self) -> None:
        data = _make_ohlcv()
        strategy = MeanReversionStrategy()
        signals = strategy.generate_signals(data)
        assert "signal" in signals.columns
        assert len(signals) == len(data)

    def test_validate_data(self) -> None:
        data = _make_ohlcv()
        strategy = MomentumStrategy()
        assert strategy.validate_data(data) is True

    def test_validate_data_missing_columns(self) -> None:
        data = pd.DataFrame({"close": [1, 2, 3]})
        strategy = MomentumStrategy()
        assert strategy.validate_data(data) is False


class TestRegime:
    def test_classify(self) -> None:
        prices = _make_ohlcv()["close"]
        classifier = RegimeClassifier()
        regimes = classifier.classify(prices)
        assert len(regimes) == len(prices)

    def test_classify_period_bull(self) -> None:
        prices = pd.Series([100, 105, 110, 115, 120])
        classifier = RegimeClassifier()
        regime = classifier.classify_period(prices)
        assert isinstance(regime, MarketRegime)

    def test_classify_period_crisis(self) -> None:
        prices = pd.Series([100, 90, 75, 70, 65])
        classifier = RegimeClassifier()
        regime = classifier.classify_period(prices)
        assert regime in (MarketRegime.CRISIS, MarketRegime.BEAR, MarketRegime.HIGH_VOL)

    def test_regime_periods(self) -> None:
        prices = _make_ohlcv()["close"]
        classifier = RegimeClassifier()
        periods = classifier.get_regime_periods(prices)
        assert len(periods) > 0
        assert "regime" in periods[0]
        assert "duration_days" in periods[0]


class TestBacktestEngine:
    def test_run_momentum(self) -> None:
        data = _make_ohlcv(n=300)
        strategy = MomentumStrategy()
        engine = BacktestEngine()
        result = engine.run(data, strategy)
        assert result.initial_capital == 100_000
        assert result.strategy_name == "momentum"
        assert len(result.equity_curve) == len(data)

    def test_run_mean_reversion(self) -> None:
        data = _make_ohlcv(n=300)
        strategy = MeanReversionStrategy()
        engine = BacktestEngine()
        result = engine.run(data, strategy)
        assert len(result.equity_curve) == len(data)

    def test_custom_config(self) -> None:
        config = BacktestConfig(initial_capital=500_000, stop_loss_pct=0.05)
        engine = BacktestEngine(config=config)
        data = _make_ohlcv(n=200)
        result = engine.run(data, MomentumStrategy())
        assert result.initial_capital == 500_000

    def test_invalid_data(self) -> None:
        data = pd.DataFrame({"close": [1, 2, 3]}, index=pd.date_range("2020-01-01", periods=3))
        engine = BacktestEngine()
        with pytest.raises(Exception):
            engine.run(data, MomentumStrategy())

    def test_non_datetime_index(self) -> None:
        data = _make_ohlcv()
        data.index = range(len(data))
        engine = BacktestEngine()
        with pytest.raises(Exception):
            engine.run(data, MomentumStrategy())

    def test_reproducibility(self) -> None:
        data = _make_ohlcv(n=200)
        strategy = MomentumStrategy()
        engine = BacktestEngine(config=BacktestConfig(seed=42))
        r1 = engine.run(data, strategy)
        r2 = engine.run(data, strategy)
        assert r1.final_value == r2.final_value


class TestWalkForward:
    def test_generate_windows(self) -> None:
        data = _make_ohlcv(n=800)
        validator = WalkForwardValidator(train_days=252, test_days=63, step_days=21)
        windows = validator.generate_windows(data)
        assert len(windows) > 0

    def test_expanding_windows(self) -> None:
        data = _make_ohlcv(n=800)
        validator = WalkForwardValidator(expanding=True)
        windows = validator.generate_windows(data)
        assert len(windows) > 0
        for w in windows:
            assert w["train_start_idx"] == 0

    def test_validate(self) -> None:
        data = _make_ohlcv(n=800)
        validator = WalkForwardValidator(train_days=252, test_days=63, step_days=63)
        strategy = MomentumStrategy()
        result = validator.validate(data, strategy, None)
        assert "total_folds" in result
        assert result["total_folds"] > 0


class TestLookaheadBias:
    def test_timestamp_ordering(self) -> None:
        data = _make_ohlcv()
        assert LookaheadBiasChecker.check_timestamp_ordering(data) is True

    def test_timestamp_ordering_unsorted(self) -> None:
        data = _make_ohlcv()
        data = data.iloc[::-1]
        assert LookaheadBiasChecker.check_timestamp_ordering(data) is False


class TestReports:
    def test_summary(self) -> None:
        data = _make_ohlcv(n=200)
        engine = BacktestEngine()
        result = engine.run(data, MomentumStrategy())
        report = BacktestReport(result)
        summary = report.summary()
        assert "strategy" in summary
        assert "total_return_pct" in summary

    def test_to_json(self) -> None:
        data = _make_ohlcv(n=200)
        engine = BacktestEngine()
        result = engine.run(data, MomentumStrategy())
        report = BacktestReport(result)
        json_str = report.to_json()
        assert len(json_str) > 100

    def test_to_html(self) -> None:
        data = _make_ohlcv(n=200)
        engine = BacktestEngine()
        result = engine.run(data, MomentumStrategy())
        report = BacktestReport(result)
        html = report.to_html()
        assert "<html>" in html
        assert "NEXUS" in html
