from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import ta

from nexus.core.logging import get_logger

logger = get_logger("data.processors.technical")


class TechnicalAnalyzer:
    @staticmethod
    def compute_all(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty or "close" not in df.columns:
            return df

        result = df.copy()
        result = TechnicalAnalyzer._add_trend_indicators(result)
        result = TechnicalAnalyzer._add_momentum_indicators(result)
        result = TechnicalAnalyzer._add_volatility_indicators(result)
        result = TechnicalAnalyzer._add_volume_indicators(result)
        result = TechnicalAnalyzer._add_custom_indicators(result)
        return result

    @staticmethod
    def _add_trend_indicators(df: pd.DataFrame) -> pd.DataFrame:
        close = df["close"]
        high = df["high"]
        low = df["low"]

        df["sma_10"] = ta.trend.sma_indicator(close, window=10)
        df["sma_20"] = ta.trend.sma_indicator(close, window=20)
        df["sma_50"] = ta.trend.sma_indicator(close, window=50)
        df["sma_200"] = ta.trend.sma_indicator(close, window=200)

        df["ema_9"] = ta.trend.ema_indicator(close, window=9)
        df["ema_21"] = ta.trend.ema_indicator(close, window=21)
        df["ema_55"] = ta.trend.ema_indicator(close, window=55)

        macd = ta.trend.MACD(close)
        df["macd"] = macd.macd()
        df["macd_signal"] = macd.macd_signal()
        df["macd_histogram"] = macd.macd_diff()

        adx = ta.trend.ADXIndicator(high, low, close)
        df["adx"] = adx.adx()
        df["adx_pos"] = adx.adx_pos()
        df["adx_neg"] = adx.adx_neg()

        ichimoku = ta.trend.IchimokuIndicator(high, low)
        df["ichimoku_a"] = ichimoku.ichimoku_a()
        df["ichimoku_b"] = ichimoku.ichimoku_b()
        df["ichimoku_base"] = ichimoku.ichimoku_base_line()
        df["ichimoku_conv"] = ichimoku.ichimoku_conversion_line()

        df["aroon_up"] = ta.trend.aroon_up(high, low)
        df["aroon_down"] = ta.trend.aroon_down(high, low)

        return df

    @staticmethod
    def _add_momentum_indicators(df: pd.DataFrame) -> pd.DataFrame:
        close = df["close"]
        high = df["high"]
        low = df["low"]
        volume = df["volume"].astype(float)

        df["rsi_14"] = ta.momentum.rsi(close, window=14)
        df["rsi_7"] = ta.momentum.rsi(close, window=7)

        stoch = ta.momentum.StochasticOscillator(high, low, close)
        df["stoch_k"] = stoch.stoch()
        df["stoch_d"] = stoch.stoch_signal()

        df["williams_r"] = ta.momentum.williams_r(high, low, close)

        df["cci"] = ta.trend.cci(high, low, close)

        df["roc_10"] = ta.momentum.roc(close, window=10)
        df["roc_20"] = ta.momentum.roc(close, window=20)

        df["mfi"] = ta.volume.money_flow_index(high, low, close, volume)

        df["tsi"] = ta.momentum.tsi(close)

        df["uo"] = ta.momentum.ultimate_oscillator(high, low, close)

        return df

    @staticmethod
    def _add_volatility_indicators(df: pd.DataFrame) -> pd.DataFrame:
        close = df["close"]
        high = df["high"]
        low = df["low"]

        bb = ta.volatility.BollingerBands(close)
        df["bb_upper"] = bb.bollinger_hband()
        df["bb_middle"] = bb.bollinger_mavg()
        df["bb_lower"] = bb.bollinger_lband()
        df["bb_width"] = bb.bollinger_wband()
        df["bb_pct"] = bb.bollinger_pband()

        atr = ta.volatility.AverageTrueRange(high, low, close)
        df["atr"] = atr.average_true_range()

        kc = ta.volatility.KeltnerChannel(high, low, close)
        df["kc_upper"] = kc.keltner_channel_hband()
        df["kc_lower"] = kc.keltner_channel_lband()
        df["kc_middle"] = kc.keltner_channel_mband()

        dc = ta.volatility.DonchianChannel(high, low, close)
        df["dc_upper"] = dc.donchian_channel_hband()
        df["dc_lower"] = dc.donchian_channel_lband()
        df["dc_middle"] = dc.donchian_channel_mband()

        df["volatility_20"] = close.pct_change().rolling(window=20).std() * np.sqrt(252)

        return df

    @staticmethod
    def _add_volume_indicators(df: pd.DataFrame) -> pd.DataFrame:
        close = df["close"]
        high = df["high"]
        low = df["low"]
        volume = df["volume"].astype(float)

        df["obv"] = ta.volume.on_balance_volume(close, volume)

        df["vwap_calc"] = (
            (close * volume).cumsum() / volume.cumsum()
        )

        cmf = ta.volume.ChaikinMoneyFlowIndicator(high, low, close, volume)
        df["cmf"] = cmf.chaikin_money_flow()

        fi = ta.volume.ForceIndexIndicator(close, volume)
        df["force_index"] = fi.force_index()

        eom = ta.volume.EaseOfMovementIndicator(high, low, volume)
        df["eom"] = eom.ease_of_movement()

        df["volume_sma_20"] = volume.rolling(window=20).mean()
        df["volume_ratio"] = volume / df["volume_sma_20"]

        return df

    @staticmethod
    def _add_custom_indicators(df: pd.DataFrame) -> pd.DataFrame:
        close = df["close"]

        df["golden_cross"] = (
            (df.get("sma_50", pd.Series(dtype=float)) > df.get("sma_200", pd.Series(dtype=float)))
            & (df.get("sma_50", pd.Series(dtype=float)).shift(1) <= df.get("sma_200", pd.Series(dtype=float)).shift(1))
        ).astype(int)

        df["death_cross"] = (
            (df.get("sma_50", pd.Series(dtype=float)) < df.get("sma_200", pd.Series(dtype=float)))
            & (df.get("sma_50", pd.Series(dtype=float)).shift(1) >= df.get("sma_200", pd.Series(dtype=float)).shift(1))
        ).astype(int)

        rsi = df.get("rsi_14", pd.Series(dtype=float))
        df["rsi_oversold"] = (rsi < 30).astype(int)
        df["rsi_overbought"] = (rsi > 70).astype(int)

        bb_pct = df.get("bb_pct", pd.Series(dtype=float))
        df["bb_squeeze"] = (df.get("bb_width", pd.Series(dtype=float)) < df.get("bb_width", pd.Series(dtype=float)).rolling(120).quantile(0.1)).astype(int)

        df["price_above_sma200"] = (close > df.get("sma_200", pd.Series(dtype=float))).astype(int)

        df["macd_crossover"] = (
            (df.get("macd", pd.Series(dtype=float)) > df.get("macd_signal", pd.Series(dtype=float)))
            & (df.get("macd", pd.Series(dtype=float)).shift(1) <= df.get("macd_signal", pd.Series(dtype=float)).shift(1))
        ).astype(int)

        return df

    @staticmethod
    def get_summary(df: pd.DataFrame) -> dict[str, Any]:
        if df.empty:
            return {}

        latest = df.iloc[-1]
        summary: dict[str, Any] = {
            "close": latest.get("close"),
            "rsi_14": latest.get("rsi_14"),
            "macd": latest.get("macd"),
            "macd_signal": latest.get("macd_signal"),
            "adx": latest.get("adx"),
            "bb_pct": latest.get("bb_pct"),
            "atr": latest.get("atr"),
            "obv": latest.get("obv"),
            "volume_ratio": latest.get("volume_ratio"),
            "volatility_20": latest.get("volatility_20"),
        }

        rsi = latest.get("rsi_14")
        if rsi is not None and not np.isnan(rsi):
            if rsi > 70:
                summary["rsi_signal"] = "overbought"
            elif rsi < 30:
                summary["rsi_signal"] = "oversold"
            else:
                summary["rsi_signal"] = "neutral"

        macd_val = latest.get("macd")
        macd_sig = latest.get("macd_signal")
        if macd_val is not None and macd_sig is not None:
            if not np.isnan(macd_val) and not np.isnan(macd_sig):
                summary["macd_trend"] = "bullish" if macd_val > macd_sig else "bearish"

        adx_val = latest.get("adx")
        if adx_val is not None and not np.isnan(adx_val):
            if adx_val > 25:
                summary["trend_strength"] = "strong"
            elif adx_val > 20:
                summary["trend_strength"] = "moderate"
            else:
                summary["trend_strength"] = "weak"

        return {k: v for k, v in summary.items() if v is not None and (not isinstance(v, float) or not np.isnan(v))}
