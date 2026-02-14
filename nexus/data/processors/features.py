from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from nexus.core.logging import get_logger

logger = get_logger("data.processors.features")


class FeatureEngineer:
    @staticmethod
    def build_features(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty or "close" not in df.columns:
            return df

        result = df.copy()
        result = FeatureEngineer._add_returns(result)
        result = FeatureEngineer._add_price_features(result)
        result = FeatureEngineer._add_volume_features(result)
        result = FeatureEngineer._add_statistical_features(result)
        result = FeatureEngineer._add_lag_features(result)
        result = FeatureEngineer._add_time_features(result)
        return result

    @staticmethod
    def _add_returns(df: pd.DataFrame) -> pd.DataFrame:
        close = df["close"]

        df["return_1d"] = close.pct_change(1)
        df["return_5d"] = close.pct_change(5)
        df["return_10d"] = close.pct_change(10)
        df["return_20d"] = close.pct_change(20)
        df["return_60d"] = close.pct_change(60)

        df["log_return_1d"] = np.log(close / close.shift(1))
        df["log_return_5d"] = np.log(close / close.shift(5))

        df["cum_return_5d"] = (1 + df["return_1d"]).rolling(5).apply(np.prod, raw=True) - 1
        df["cum_return_20d"] = (1 + df["return_1d"]).rolling(20).apply(np.prod, raw=True) - 1

        return df

    @staticmethod
    def _add_price_features(df: pd.DataFrame) -> pd.DataFrame:
        close = df["close"]
        high = df["high"]
        low = df["low"]

        df["high_low_range"] = (high - low) / close
        df["close_open_range"] = (close - df["open"]) / df["open"]

        df["high_5d"] = high.rolling(5).max()
        df["low_5d"] = low.rolling(5).min()
        df["high_20d"] = high.rolling(20).max()
        df["low_20d"] = low.rolling(20).min()

        df["dist_from_high_20d"] = (close - df["high_20d"]) / df["high_20d"]
        df["dist_from_low_20d"] = (close - df["low_20d"]) / df["low_20d"]

        df["price_velocity"] = close.diff(1) / close.shift(1)
        df["price_acceleration"] = df["price_velocity"].diff(1)

        df["gap"] = df["open"] / close.shift(1) - 1

        for period in [10, 20, 50]:
            sma_col = f"sma_{period}"
            if sma_col in df.columns:
                df[f"dist_from_sma_{period}"] = (close - df[sma_col]) / df[sma_col]

        return df

    @staticmethod
    def _add_volume_features(df: pd.DataFrame) -> pd.DataFrame:
        volume = df["volume"].astype(float)

        df["volume_change"] = volume.pct_change()
        df["volume_ma5"] = volume.rolling(5).mean()
        df["volume_ma20"] = volume.rolling(20).mean()
        df["rel_volume_5"] = volume / df["volume_ma5"]
        df["rel_volume_20"] = volume / df["volume_ma20"]

        df["dollar_volume"] = df["close"] * volume
        df["dollar_volume_ma20"] = df["dollar_volume"].rolling(20).mean()

        returns = df.get("return_1d", df["close"].pct_change())
        df["up_volume"] = np.where(returns > 0, volume, 0)
        df["down_volume"] = np.where(returns < 0, volume, 0)
        df["up_down_volume_ratio"] = (
            pd.Series(df["up_volume"]).rolling(20).sum()
            / pd.Series(df["down_volume"]).rolling(20).sum().replace(0, np.nan)
        )

        return df

    @staticmethod
    def _add_statistical_features(df: pd.DataFrame) -> pd.DataFrame:
        close = df["close"]
        returns = df.get("return_1d", close.pct_change())

        for window in [5, 10, 20, 60]:
            df[f"volatility_{window}d"] = returns.rolling(window).std() * np.sqrt(252)
            df[f"skew_{window}d"] = returns.rolling(window).skew()
            df[f"kurtosis_{window}d"] = returns.rolling(window).kurt()
            df[f"mean_return_{window}d"] = returns.rolling(window).mean()

        df["sharpe_20d"] = (
            df.get("mean_return_20d", returns.rolling(20).mean())
            / df.get("volatility_20d", returns.rolling(20).std())
        ) * np.sqrt(252)

        df["downside_vol_20d"] = (
            returns.clip(upper=0).rolling(20).std() * np.sqrt(252)
        )
        df["sortino_20d"] = (
            df.get("mean_return_20d", returns.rolling(20).mean())
            / df["downside_vol_20d"].replace(0, np.nan)
        ) * np.sqrt(252)

        df["z_score_20d"] = (
            (close - close.rolling(20).mean()) / close.rolling(20).std()
        )
        df["z_score_50d"] = (
            (close - close.rolling(50).mean()) / close.rolling(50).std()
        )

        return df

    @staticmethod
    def _add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
        returns = df.get("return_1d", df["close"].pct_change())
        volume = df["volume"].astype(float)

        for lag in [1, 2, 3, 5]:
            df[f"return_lag_{lag}"] = returns.shift(lag)
            df[f"volume_lag_{lag}"] = volume.shift(lag)

        df["return_momentum_5_20"] = (
            df.get("return_5d", df["close"].pct_change(5))
            - df.get("return_20d", df["close"].pct_change(20))
        )

        return df

    @staticmethod
    def _add_time_features(df: pd.DataFrame) -> pd.DataFrame:
        if "timestamp" in df.columns:
            ts = pd.to_datetime(df["timestamp"])
        elif isinstance(df.index, pd.DatetimeIndex):
            ts = df.index
        else:
            return df

        df["day_of_week"] = ts.dayofweek
        df["day_of_month"] = ts.day
        df["month"] = ts.month
        df["quarter"] = ts.quarter
        df["is_month_start"] = ts.is_month_start.astype(int)
        df["is_month_end"] = ts.is_month_end.astype(int)
        df["is_quarter_start"] = ts.is_quarter_start.astype(int)
        df["is_quarter_end"] = ts.is_quarter_end.astype(int)

        return df

    @staticmethod
    def select_features(
        df: pd.DataFrame,
        target_col: str = "return_1d",
        correlation_threshold: float = 0.95,
        min_variance: float = 0.001,
    ) -> list[str]:
        numeric_df = df.select_dtypes(include=[np.number]).drop(
            columns=[target_col], errors="ignore"
        )
        numeric_df = numeric_df.dropna(axis=1, how="all")

        low_var = numeric_df.columns[numeric_df.var() < min_variance].tolist()
        numeric_df = numeric_df.drop(columns=low_var)

        corr_matrix = numeric_df.corr().abs()
        upper = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        to_drop = [
            col for col in upper.columns if any(upper[col] > correlation_threshold)
        ]
        selected = [c for c in numeric_df.columns if c not in to_drop]

        logger.info(
            f"Selected {len(selected)} features from {len(numeric_df.columns)} "
            f"(dropped {len(low_var)} low-var, {len(to_drop)} high-corr)"
        )
        return selected

    @staticmethod
    def normalize(df: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
        result = df.copy()
        cols = columns or result.select_dtypes(include=[np.number]).columns.tolist()
        for col in cols:
            if col in result.columns:
                mean = result[col].mean()
                std = result[col].std()
                if std > 0:
                    result[col] = (result[col] - mean) / std
        return result
