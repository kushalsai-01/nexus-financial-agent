from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from uuid import uuid4

import numpy as np

from nexus.core.logging import get_logger
from nexus.core.types import Action, OrderSide, Trade

logger = get_logger("execution.algorithms")


@dataclass
class AlgoConfig:
    max_participation_rate: float = 0.05
    urgency: float = 0.5
    start_time: str = "09:30"
    end_time: str = "16:00"
    min_fill_size: int = 100
    randomize: bool = True
    seed: int = 42


@dataclass
class SliceResult:
    slice_id: int
    quantity: int
    price: float
    timestamp: datetime
    slippage_bps: float = 0.0


class ExecutionAlgorithm(ABC):
    def __init__(self, config: AlgoConfig | None = None) -> None:
        self.config = config or AlgoConfig()
        self._rng = np.random.RandomState(self.config.seed)

    @abstractmethod
    def generate_schedule(
        self,
        total_quantity: int,
        market_data: dict[str, Any],
    ) -> list[dict[str, Any]]:
        ...

    def compute_slippage(
        self,
        scheduled_price: float,
        executed_price: float,
        side: OrderSide,
    ) -> float:
        if scheduled_price == 0:
            return 0.0
        if side == OrderSide.BUY:
            return (executed_price - scheduled_price) / scheduled_price * 10000
        return (scheduled_price - executed_price) / scheduled_price * 10000


class VWAPAlgorithm(ExecutionAlgorithm):
    def generate_schedule(
        self,
        total_quantity: int,
        market_data: dict[str, Any],
    ) -> list[dict[str, Any]]:
        volume_profile = market_data.get("volume_profile")
        if volume_profile is None:
            volume_profile = self._default_volume_profile()

        n_slices = min(int(total_quantity / self.config.min_fill_size), 10)
        n_slices = max(n_slices, 1)

        profile_sum = sum(volume_profile[:n_slices]) if n_slices <= len(volume_profile) else sum(volume_profile)
        if profile_sum == 0:
            profile_sum = 1

        schedule: list[dict[str, Any]] = []
        remaining = total_quantity

        for i in range(n_slices):
            weight = volume_profile[i % len(volume_profile)] / profile_sum
            slice_qty = int(total_quantity * weight)

            if i == n_slices - 1:
                slice_qty = remaining
            else:
                slice_qty = min(slice_qty, remaining)

            if slice_qty <= 0:
                continue

            if self.config.randomize:
                jitter = self._rng.uniform(-0.1, 0.1) * slice_qty
                slice_qty = max(1, int(slice_qty + jitter))
                slice_qty = min(slice_qty, remaining)

            schedule.append({
                "slice_id": i,
                "quantity": slice_qty,
                "target_pct": weight,
                "cumulative_pct": 1 - remaining / total_quantity + slice_qty / total_quantity,
            })
            remaining -= slice_qty

        return schedule

    @staticmethod
    def _default_volume_profile() -> list[float]:
        return [0.08, 0.06, 0.05, 0.04, 0.04, 0.05, 0.06, 0.07, 0.09, 0.11, 0.12, 0.10, 0.13]


class TWAPAlgorithm(ExecutionAlgorithm):
    def generate_schedule(
        self,
        total_quantity: int,
        market_data: dict[str, Any],
    ) -> list[dict[str, Any]]:
        n_slices = min(int(total_quantity / self.config.min_fill_size), 10)
        n_slices = max(n_slices, 1)

        base_qty = total_quantity // n_slices
        remainder = total_quantity % n_slices

        schedule: list[dict[str, Any]] = []
        for i in range(n_slices):
            qty = base_qty + (1 if i < remainder else 0)

            if self.config.randomize:
                jitter = self._rng.uniform(-0.05, 0.05) * qty
                qty = max(1, int(qty + jitter))

            schedule.append({
                "slice_id": i,
                "quantity": qty,
                "target_pct": 1.0 / n_slices,
                "cumulative_pct": (i + 1) / n_slices,
            })

        return schedule


class POVAlgorithm(ExecutionAlgorithm):
    def __init__(
        self,
        target_participation: float = 0.05,
        config: AlgoConfig | None = None,
    ) -> None:
        super().__init__(config)
        self.target_participation = min(target_participation, self.config.max_participation_rate)

    def generate_schedule(
        self,
        total_quantity: int,
        market_data: dict[str, Any],
    ) -> list[dict[str, Any]]:
        avg_daily_volume = market_data.get("avg_daily_volume", 1_000_000)
        max_per_interval = int(avg_daily_volume * self.target_participation / 13)
        max_per_interval = max(max_per_interval, self.config.min_fill_size)

        schedule: list[dict[str, Any]] = []
        remaining = total_quantity
        slice_id = 0

        while remaining > 0:
            qty = min(max_per_interval, remaining)

            if self.config.randomize:
                jitter = self._rng.uniform(-0.1, 0.1) * qty
                qty = max(1, int(qty + jitter))
                qty = min(qty, remaining)

            schedule.append({
                "slice_id": slice_id,
                "quantity": qty,
                "participation_rate": self.target_participation,
                "cumulative_pct": 1 - (remaining - qty) / total_quantity,
            })

            remaining -= qty
            slice_id += 1

            if slice_id > 100:
                if remaining > 0:
                    schedule.append({
                        "slice_id": slice_id,
                        "quantity": remaining,
                        "participation_rate": self.target_participation,
                        "cumulative_pct": 1.0,
                    })
                break

        return schedule


class ImplementationShortfall(ExecutionAlgorithm):
    def __init__(
        self,
        risk_aversion: float = 0.5,
        config: AlgoConfig | None = None,
    ) -> None:
        super().__init__(config)
        self.risk_aversion = risk_aversion

    def generate_schedule(
        self,
        total_quantity: int,
        market_data: dict[str, Any],
    ) -> list[dict[str, Any]]:
        volatility = market_data.get("volatility", 0.02)
        avg_volume = market_data.get("avg_daily_volume", 1_000_000)

        urgency = self.config.urgency
        participation = total_quantity / avg_volume if avg_volume > 0 else 0.01

        front_load = urgency * 0.5 + participation * 0.3 + self.risk_aversion * 0.2
        front_load = max(0.2, min(0.8, front_load))

        n_slices = min(int(total_quantity / self.config.min_fill_size), 10)
        n_slices = max(n_slices, 2)

        weights: list[float] = []
        for i in range(n_slices):
            pct = i / (n_slices - 1) if n_slices > 1 else 0
            w = front_load * (1 - pct) + (1 - front_load) * pct
            weights.append(w)

        total_w = sum(weights)
        weights = [w / total_w for w in weights]

        schedule: list[dict[str, Any]] = []
        remaining = total_quantity

        for i in range(n_slices):
            if i == n_slices - 1:
                qty = remaining
            else:
                qty = int(total_quantity * weights[i])
                qty = min(qty, remaining)

            schedule.append({
                "slice_id": i,
                "quantity": qty,
                "weight": weights[i],
                "cumulative_pct": 1 - (remaining - qty) / total_quantity,
                "urgency": urgency,
            })
            remaining -= qty

        return schedule


ALGO_REGISTRY: dict[str, type[ExecutionAlgorithm]] = {
    "vwap": VWAPAlgorithm,
    "twap": TWAPAlgorithm,
    "pov": POVAlgorithm,
    "implementation_shortfall": ImplementationShortfall,
}
