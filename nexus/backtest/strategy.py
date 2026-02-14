from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pandas as pd

from nexus.core.types import OrderSide, SignalType


class BaseStrategy(ABC):
    def __init__(self, name: str, params: dict[str, Any] | None = None) -> None:
        self.name = name
        self.params = params or {}

    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        ...

    def validate_data(self, data: pd.DataFrame) -> bool:
        required = {"open", "high", "low", "close", "volume"}
        return required.issubset(set(data.columns))


class AgentStrategy(BaseStrategy):
    def __init__(
        self,
        name: str = "agent_ensemble",
        agent_weights: dict[str, float] | None = None,
        confidence_threshold: float = 0.6,
        params: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(name, params)
        self.agent_weights = agent_weights or {}
        self.confidence_threshold = confidence_threshold

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        signals = pd.DataFrame(index=data.index)
        signals["signal"] = 0
        signals["confidence"] = 0.0
        signals["position_size"] = 0.0

        if "agent_signals" in data.columns:
            for i, row in data.iterrows():
                agent_data = row.get("agent_signals", {})
                if isinstance(agent_data, dict):
                    weighted_signal, total_weight = self._aggregate_signals(agent_data)
                    if abs(weighted_signal) > self.confidence_threshold:
                        signals.loc[i, "signal"] = 1 if weighted_signal > 0 else -1
                        signals.loc[i, "confidence"] = min(abs(weighted_signal), 1.0)
                        signals.loc[i, "position_size"] = min(abs(weighted_signal), 1.0) * 0.5
        return signals

    def _aggregate_signals(self, agent_data: dict[str, Any]) -> tuple[float, float]:
        total_weighted = 0.0
        total_weight = 0.0

        for agent_name, signal_data in agent_data.items():
            weight = self.agent_weights.get(agent_name, 1.0)
            signal_val = signal_data.get("signal_value", 0)
            confidence = signal_data.get("confidence", 0.5)
            total_weighted += signal_val * confidence * weight
            total_weight += weight

        if total_weight == 0:
            return 0.0, 0.0
        return total_weighted / total_weight, total_weight


class MomentumStrategy(BaseStrategy):
    def __init__(
        self,
        fast_period: int = 10,
        slow_period: int = 50,
        rsi_period: int = 14,
        rsi_overbought: float = 70,
        rsi_oversold: float = 30,
        params: dict[str, Any] | None = None,
    ) -> None:
        super().__init__("momentum", params)
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.rsi_period = rsi_period
        self.rsi_overbought = rsi_overbought
        self.rsi_oversold = rsi_oversold

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        signals = pd.DataFrame(index=data.index)
        signals["signal"] = 0
        signals["confidence"] = 0.0
        signals["position_size"] = 0.0

        close = data["close"]
        sma_fast = close.rolling(self.fast_period).mean()
        sma_slow = close.rolling(self.slow_period).mean()

        delta = close.diff()
        gain = delta.clip(lower=0).rolling(self.rsi_period).mean()
        loss = (-delta.clip(upper=0)).rolling(self.rsi_period).mean()
        rs = gain / loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))

        bullish_cross = (sma_fast > sma_slow) & (sma_fast.shift(1) <= sma_slow.shift(1))
        bearish_cross = (sma_fast < sma_slow) & (sma_fast.shift(1) >= sma_slow.shift(1))

        buy_signal = bullish_cross & (rsi < self.rsi_overbought)
        sell_signal = bearish_cross | (rsi > self.rsi_overbought)

        signals.loc[buy_signal, "signal"] = 1
        signals.loc[sell_signal, "signal"] = -1
        signals["confidence"] = (1 - abs(rsi - 50) / 50).clip(0, 1).fillna(0)
        signals["position_size"] = signals["confidence"] * 0.5

        return signals


class MeanReversionStrategy(BaseStrategy):
    def __init__(
        self,
        lookback: int = 20,
        entry_z: float = -2.0,
        exit_z: float = 0.0,
        stop_z: float = -3.5,
        params: dict[str, Any] | None = None,
    ) -> None:
        super().__init__("mean_reversion", params)
        self.lookback = lookback
        self.entry_z = entry_z
        self.exit_z = exit_z
        self.stop_z = stop_z

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        signals = pd.DataFrame(index=data.index)
        signals["signal"] = 0
        signals["confidence"] = 0.0
        signals["position_size"] = 0.0

        close = data["close"]
        mean = close.rolling(self.lookback).mean()
        std = close.rolling(self.lookback).std()
        z_score = (close - mean) / std.replace(0, np.nan)

        signals.loc[z_score < self.entry_z, "signal"] = 1
        signals.loc[z_score > abs(self.entry_z), "signal"] = -1
        signals.loc[(z_score > self.exit_z) & (z_score.shift(1) < self.exit_z), "signal"] = 0
        signals.loc[z_score < self.stop_z, "signal"] = 0

        signals["confidence"] = (abs(z_score) / 3.0).clip(0, 1).fillna(0)
        signals["position_size"] = signals["confidence"] * 0.3

        return signals


STRATEGY_REGISTRY: dict[str, type[BaseStrategy]] = {
    "agent_ensemble": AgentStrategy,
    "momentum": MomentumStrategy,
    "mean_reversion": MeanReversionStrategy,
}
