from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from nexus.core.logging import get_logger

logger = get_logger("execution.slippage")


@dataclass
class SlippageModel:
    base_bps: float = 1.0
    volatility_multiplier: float = 0.5
    size_impact_exponent: float = 0.5
    urgency_multiplier: float = 1.0

    def estimate(
        self,
        order_size: float,
        daily_volume: float,
        volatility: float = 0.02,
        spread_bps: float = 1.0,
        urgency: float = 0.5,
    ) -> dict[str, float]:
        spread_cost = spread_bps / 2

        if daily_volume > 0:
            participation = abs(order_size) / daily_volume
        else:
            participation = 0.01

        market_impact = (
            self.volatility_multiplier
            * volatility
            * math.pow(participation, self.size_impact_exponent)
            * 10000
        )

        urgency_cost = self.base_bps * (1 + urgency * self.urgency_multiplier)

        total_bps = spread_cost + market_impact + urgency_cost

        return {
            "spread_cost_bps": round(spread_cost, 2),
            "market_impact_bps": round(market_impact, 2),
            "urgency_cost_bps": round(urgency_cost, 2),
            "total_slippage_bps": round(total_bps, 2),
            "participation_rate": round(participation, 6),
        }


@dataclass
class TieredSlippageModel:
    tiers: list[dict[str, Any]] | None = None

    def __post_init__(self) -> None:
        if self.tiers is None:
            self.tiers = [
                {"max_participation": 0.005, "base_bps": 0.5, "impact_mult": 0.3},
                {"max_participation": 0.01, "base_bps": 1.0, "impact_mult": 0.5},
                {"max_participation": 0.05, "base_bps": 2.0, "impact_mult": 1.0},
                {"max_participation": 0.10, "base_bps": 5.0, "impact_mult": 2.0},
                {"max_participation": 1.00, "base_bps": 15.0, "impact_mult": 5.0},
            ]

    def estimate(
        self,
        order_size: float,
        daily_volume: float,
        volatility: float = 0.02,
    ) -> dict[str, float]:
        participation = abs(order_size) / daily_volume if daily_volume > 0 else 0.01

        tier = self.tiers[-1]
        for t in self.tiers:
            if participation <= t["max_participation"]:
                tier = t
                break

        base = tier["base_bps"]
        impact = tier["impact_mult"] * volatility * math.sqrt(participation) * 10000

        total = base + impact

        return {
            "tier_base_bps": round(base, 2),
            "impact_bps": round(impact, 2),
            "total_bps": round(total, 2),
            "participation_rate": round(participation, 6),
            "selected_tier": tier["max_participation"],
        }


def estimate_execution_cost(
    price: float,
    quantity: float,
    daily_volume: float = 1_000_000,
    volatility: float = 0.02,
    urgency: float = 0.5,
    spread_bps: float = 1.0,
) -> dict[str, float]:
    model = SlippageModel()
    estimate = model.estimate(
        order_size=quantity,
        daily_volume=daily_volume,
        volatility=volatility,
        spread_bps=spread_bps,
        urgency=urgency,
    )

    notional = price * abs(quantity)
    cost_dollars = notional * estimate["total_slippage_bps"] / 10000

    return {
        **estimate,
        "notional": round(notional, 2),
        "estimated_cost_dollars": round(cost_dollars, 2),
    }
