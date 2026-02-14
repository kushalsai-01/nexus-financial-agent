from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nexus.data.processors.technical import TechnicalAnalyzer
from nexus.data.processors.features import FeatureEngineer


@pytest.fixture
def market_df() -> pd.DataFrame:
    np.random.seed(42)
    n = 250
    dates = pd.date_range("2023-01-01", periods=n, freq="B")
    close = 150 + np.cumsum(np.random.randn(n) * 2)
    high = close + np.abs(np.random.randn(n)) * 3
    low = close - np.abs(np.random.randn(n)) * 3
    open_ = close + np.random.randn(n) * 1.5
    volume = np.random.randint(500000, 2000000, n)

    return pd.DataFrame(
        {
            "timestamp": dates,
            "ticker": "AAPL",
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )


class TestTechnicalAnalyzer:
    def test_compute_all_adds_indicators(self, market_df: pd.DataFrame) -> None:
        result = TechnicalAnalyzer.compute_all(market_df)
        assert "sma_20" in result.columns
        assert "rsi_14" in result.columns
        assert "macd" in result.columns
        assert "bb_upper" in result.columns
        assert "atr" in result.columns
        assert "obv" in result.columns
        assert len(result) == len(market_df)

    def test_trend_indicators(self, market_df: pd.DataFrame) -> None:
        result = TechnicalAnalyzer.compute_all(market_df)
        assert "sma_10" in result.columns
        assert "sma_50" in result.columns
        assert "sma_200" in result.columns
        assert "ema_9" in result.columns
        assert "ema_21" in result.columns
        assert "macd_signal" in result.columns
        assert "macd_histogram" in result.columns
        assert "adx" in result.columns

    def test_momentum_indicators(self, market_df: pd.DataFrame) -> None:
        result = TechnicalAnalyzer.compute_all(market_df)
        assert "rsi_14" in result.columns
        assert "rsi_7" in result.columns
        assert "stoch_k" in result.columns
        assert "stoch_d" in result.columns
        assert "williams_r" in result.columns
        assert "cci" in result.columns
        assert "mfi" in result.columns

    def test_volatility_indicators(self, market_df: pd.DataFrame) -> None:
        result = TechnicalAnalyzer.compute_all(market_df)
        assert "bb_upper" in result.columns
        assert "bb_lower" in result.columns
        assert "bb_width" in result.columns
        assert "atr" in result.columns
        assert "volatility_20" in result.columns

    def test_volume_indicators(self, market_df: pd.DataFrame) -> None:
        result = TechnicalAnalyzer.compute_all(market_df)
        assert "obv" in result.columns
        assert "cmf" in result.columns
        assert "volume_ratio" in result.columns

    def test_custom_indicators(self, market_df: pd.DataFrame) -> None:
        result = TechnicalAnalyzer.compute_all(market_df)
        assert "golden_cross" in result.columns
        assert "death_cross" in result.columns
        assert "rsi_oversold" in result.columns
        assert "rsi_overbought" in result.columns

    def test_get_summary(self, market_df: pd.DataFrame) -> None:
        result = TechnicalAnalyzer.compute_all(market_df)
        summary = TechnicalAnalyzer.get_summary(result)
        assert "close" in summary
        assert "rsi_14" in summary

    def test_empty_dataframe(self) -> None:
        df = pd.DataFrame()
        result = TechnicalAnalyzer.compute_all(df)
        assert result.empty

    def test_get_summary_empty(self) -> None:
        summary = TechnicalAnalyzer.get_summary(pd.DataFrame())
        assert summary == {}


class TestFeatureEngineer:
    def test_build_features_adds_columns(self, market_df: pd.DataFrame) -> None:
        result = FeatureEngineer.build_features(market_df)
        assert "return_1d" in result.columns
        assert "return_5d" in result.columns
        assert "log_return_1d" in result.columns
        assert "high_low_range" in result.columns
        assert "volume_change" in result.columns
        assert "volatility_20d" in result.columns

    def test_returns_features(self, market_df: pd.DataFrame) -> None:
        result = FeatureEngineer.build_features(market_df)
        assert "return_1d" in result.columns
        assert "return_10d" in result.columns
        assert "return_20d" in result.columns
        assert "log_return_1d" in result.columns

    def test_statistical_features(self, market_df: pd.DataFrame) -> None:
        result = FeatureEngineer.build_features(market_df)
        assert "volatility_20d" in result.columns
        assert "skew_20d" in result.columns
        assert "z_score_20d" in result.columns

    def test_lag_features(self, market_df: pd.DataFrame) -> None:
        result = FeatureEngineer.build_features(market_df)
        assert "return_lag_1" in result.columns
        assert "return_lag_5" in result.columns

    def test_time_features(self, market_df: pd.DataFrame) -> None:
        result = FeatureEngineer.build_features(market_df)
        assert "day_of_week" in result.columns
        assert "month" in result.columns
        assert "quarter" in result.columns

    def test_empty_dataframe(self) -> None:
        df = pd.DataFrame()
        result = FeatureEngineer.build_features(df)
        assert result.empty

    def test_normalize(self, market_df: pd.DataFrame) -> None:
        result = FeatureEngineer.normalize(market_df, columns=["close", "volume"])
        assert abs(result["close"].mean()) < 0.01
        assert abs(result["close"].std() - 1.0) < 0.01

    def test_select_features(self, market_df: pd.DataFrame) -> None:
        enriched = FeatureEngineer.build_features(market_df)
        enriched = enriched.dropna()
        if len(enriched) > 10:
            selected = FeatureEngineer.select_features(enriched)
            assert len(selected) > 0
