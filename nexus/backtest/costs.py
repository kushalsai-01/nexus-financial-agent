from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


@dataclass
class TransactionCostModel:
    commission_pct: float = 0.0001
    min_commission: float = 1.0
    spread_bps: float = 1.0
    market_impact_factor: float = 0.5
    overnight_cost_pct: float = 0.0001

    def compute_commission(self, notional: float) -> float:
        return max(notional * self.commission_pct, self.min_commission)

    def compute_spread_cost(self, price: float, quantity: float) -> float:
        half_spread = price * (self.spread_bps / 10000) / 2
        return half_spread * abs(quantity)

    def compute_market_impact(
        self,
        order_size: float,
        daily_volume: float,
        volatility: float,
    ) -> float:
        if daily_volume <= 0:
            return 0.0
        participation = abs(order_size) / daily_volume
        return math.sqrt(participation) * volatility * self.market_impact_factor

    def compute_slippage(
        self,
        price: float,
        quantity: float,
        daily_volume: float,
        volatility: float = 0.02,
    ) -> float:
        if daily_volume <= 0:
            return price * 0.001

        if daily_volume > 1_000_000:
            base_slippage_pct = 0.0002
        elif daily_volume > 100_000:
            base_slippage_pct = 0.0005
        else:
            base_slippage_pct = 0.0015

        impact = self.compute_market_impact(quantity, daily_volume, volatility)
        return price * (base_slippage_pct + impact)

    def total_cost(
        self,
        price: float,
        quantity: float,
        daily_volume: float = 1_000_000,
        volatility: float = 0.02,
    ) -> dict[str, float]:
        notional = price * abs(quantity)
        commission = self.compute_commission(notional)
        spread = self.compute_spread_cost(price, quantity)
        slippage = self.compute_slippage(price, quantity, daily_volume, volatility)
        total_slippage = slippage * abs(quantity)

        return {
            "commission": commission,
            "spread_cost": spread,
            "slippage_per_share": slippage,
            "total_slippage": total_slippage,
            "total_cost": commission + spread + total_slippage,
            "cost_pct": (commission + spread + total_slippage) / notional if notional > 0 else 0.0,
        }
