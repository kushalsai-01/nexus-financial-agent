from __future__ import annotations

from enum import Enum
from typing import Any

import numpy as np
import pandas as pd

from nexus.core.logging import get_logger

logger = get_logger("backtest.regime")


class MarketRegime(str, Enum):
    BULL = "bull"
    BEAR = "bear"
    SIDEWAYS = "sideways"
    HIGH_VOL = "high_volatility"
    CRISIS = "crisis"


class RegimeClassifier:
    def __init__(
        self,
        bull_threshold: float = 0.10,
        bear_threshold: float = -0.10,
        volatility_lookback: int = 21,
        trend_lookback: int = 63,
        high_vol_threshold: float = 0.25,
        crisis_drawdown: float = -0.20,
    ) -> None:
        self.bull_threshold = bull_threshold
        self.bear_threshold = bear_threshold
        self.volatility_lookback = volatility_lookback
        self.trend_lookback = trend_lookback
        self.high_vol_threshold = high_vol_threshold
        self.crisis_drawdown = crisis_drawdown

    def classify(self, prices: pd.Series, window: int = 252) -> pd.Series:
        regimes = pd.Series(MarketRegime.SIDEWAYS, index=prices.index)

        rolling_return = prices.pct_change(window).fillna(0)
        rolling_vol = prices.pct_change().rolling(self.volatility_lookback).std() * np.sqrt(252)
        peak = prices.expanding().max()
        drawdown = (prices - peak) / peak

        bull_mask = rolling_return > self.bull_threshold
        bear_mask = rolling_return < self.bear_threshold
        crisis_mask = drawdown < self.crisis_drawdown
        high_vol_mask = rolling_vol > self.high_vol_threshold

        regimes[bull_mask] = MarketRegime.BULL
        regimes[bear_mask] = MarketRegime.BEAR
        regimes[high_vol_mask & ~crisis_mask] = MarketRegime.HIGH_VOL
        regimes[crisis_mask] = MarketRegime.CRISIS

        return regimes

    def classify_period(self, prices: pd.Series) -> MarketRegime:
        if len(prices) < 2:
            return MarketRegime.SIDEWAYS

        total_return = (prices.iloc[-1] / prices.iloc[0]) - 1
        annualized = total_return * (252 / len(prices))

        peak = prices.expanding().max()
        max_dd = float(((prices - peak) / peak).min())
        vol = float(prices.pct_change().std() * np.sqrt(252))

        if max_dd < self.crisis_drawdown:
            return MarketRegime.CRISIS
        if vol > self.high_vol_threshold:
            return MarketRegime.HIGH_VOL
        if annualized > self.bull_threshold:
            return MarketRegime.BULL
        if annualized < self.bear_threshold:
            return MarketRegime.BEAR
        return MarketRegime.SIDEWAYS

    def get_regime_periods(self, prices: pd.Series) -> list[dict[str, Any]]:
        regimes = self.classify(prices)
        periods: list[dict[str, Any]] = []
        current_regime = regimes.iloc[0]
        start_idx = 0

        for i in range(1, len(regimes)):
            if regimes.iloc[i] != current_regime:
                periods.append({
                    "regime": current_regime,
                    "start": regimes.index[start_idx],
                    "end": regimes.index[i - 1],
                    "duration_days": i - start_idx,
                })
                current_regime = regimes.iloc[i]
                start_idx = i

        periods.append({
            "regime": current_regime,
            "start": regimes.index[start_idx],
            "end": regimes.index[-1],
            "duration_days": len(regimes) - start_idx,
        })

        return periods

    def compute_regime_performance(
        self,
        equity_curve: pd.Series,
        prices: pd.Series,
    ) -> dict[str, dict[str, float]]:
        regimes = self.classify(prices)
        results: dict[str, dict[str, float]] = {}

        for regime in MarketRegime:
            mask = regimes == regime
            if mask.sum() < 5:
                continue

            regime_equity = equity_curve[mask]
            if len(regime_equity) < 2:
                continue

            returns = regime_equity.pct_change().dropna()
            total_ret = (regime_equity.iloc[-1] / regime_equity.iloc[0]) - 1 if regime_equity.iloc[0] != 0 else 0
            vol = float(returns.std() * np.sqrt(252)) if len(returns) > 1 else 0
            sharpe = float(returns.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else 0

            results[regime.value] = {
                "total_return": total_ret,
                "volatility": vol,
                "sharpe_ratio": sharpe,
                "days": int(mask.sum()),
                "pct_of_total": float(mask.sum() / len(regimes)),
            }

        return results
