from __future__ import annotations

from datetime import datetime, time, timezone
from typing import Any
from uuid import uuid4

import numpy as np

from nexus.backtest.costs import TransactionCostModel
from nexus.core.exceptions import InsufficientFundsError, OrderError
from nexus.core.logging import get_logger
from nexus.core.types import (
    Action,
    OrderSide,
    OrderStatus,
    OrderType,
    Position,
    Portfolio,
    Trade,
)

logger = get_logger("execution.simulator")


class PaperTradingSimulator:
    def __init__(
        self,
        initial_capital: float = 100_000,
        cost_model: TransactionCostModel | None = None,
        fill_probability: float = 0.98,
        partial_fill_probability: float = 0.20,
        reject_probability: float = 0.02,
        enforce_market_hours: bool = True,
        seed: int = 42,
    ) -> None:
        self.cost_model = cost_model or TransactionCostModel()
        self.fill_probability = fill_probability
        self.partial_fill_probability = partial_fill_probability
        self.reject_probability = reject_probability
        self.enforce_market_hours = enforce_market_hours
        self._rng = np.random.RandomState(seed)

        self.portfolio = Portfolio(
            cash=initial_capital,
            total_value=initial_capital,
        )
        self.trades: list[Trade] = []
        self.pending_orders: dict[str, dict[str, Any]] = {}
        self._market_open = time(9, 30)
        self._market_close = time(16, 0)

    def submit_order(
        self,
        action: Action,
        current_price: float,
        daily_volume: float = 1_000_000,
        current_time: datetime | None = None,
    ) -> dict[str, Any]:
        now = current_time or datetime.now(timezone.utc)

        if self.enforce_market_hours:
            local_time = now.time()
            if local_time < self._market_open or local_time >= self._market_close:
                return {
                    "order_id": str(uuid4()),
                    "status": OrderStatus.REJECTED.value,
                    "reason": "Market closed",
                }

        roll = self._rng.random()
        if roll < self.reject_probability:
            return {
                "order_id": str(uuid4()),
                "status": OrderStatus.REJECTED.value,
                "reason": "Random rejection (simulated broker reject)",
            }

        order_id = str(uuid4())

        if action.order_type == OrderType.LIMIT:
            if action.side == OrderSide.BUY and (action.limit_price or 0) < current_price:
                self.pending_orders[order_id] = {
                    "action": action,
                    "submitted_at": now,
                    "limit_price": action.limit_price,
                }
                return {"order_id": order_id, "status": OrderStatus.PENDING.value}
            elif action.side == OrderSide.SELL and (action.limit_price or float("inf")) > current_price:
                self.pending_orders[order_id] = {
                    "action": action,
                    "submitted_at": now,
                    "limit_price": action.limit_price,
                }
                return {"order_id": order_id, "status": OrderStatus.PENDING.value}

        if action.order_type == OrderType.STOP:
            if action.side == OrderSide.SELL and current_price > (action.stop_price or 0):
                self.pending_orders[order_id] = {
                    "action": action,
                    "submitted_at": now,
                    "stop_price": action.stop_price,
                }
                return {"order_id": order_id, "status": OrderStatus.PENDING.value}

        return self._execute_fill(action, current_price, daily_volume, now, order_id)

    def _execute_fill(
        self,
        action: Action,
        current_price: float,
        daily_volume: float,
        timestamp: datetime,
        order_id: str,
    ) -> dict[str, Any]:
        costs = self.cost_model.total_cost(current_price, action.quantity, daily_volume)
        fill_price = current_price + costs["slippage_per_share"] * (1 if action.side == OrderSide.BUY else -1)

        is_partial = self._rng.random() < self.partial_fill_probability
        fill_qty = action.quantity
        status = OrderStatus.FILLED

        if is_partial:
            fill_pct = self._rng.uniform(0.5, 0.95)
            fill_qty = max(1, int(action.quantity * fill_pct))
            status = OrderStatus.PARTIAL

        if action.side == OrderSide.BUY:
            total_cost = fill_price * fill_qty + costs["commission"]
            if total_cost > self.portfolio.cash:
                affordable = int((self.portfolio.cash - costs["commission"]) / fill_price)
                if affordable <= 0:
                    return {
                        "order_id": order_id,
                        "status": OrderStatus.REJECTED.value,
                        "reason": "Insufficient funds",
                    }
                fill_qty = affordable
                total_cost = fill_price * fill_qty + costs["commission"]

            self.portfolio.cash -= total_cost

            existing = self.portfolio.positions.get(action.ticker)
            if existing:
                total_qty = existing.quantity + fill_qty
                avg_price = (existing.entry_price * existing.quantity + fill_price * fill_qty) / total_qty
                existing.quantity = total_qty
                existing.entry_price = avg_price
                existing.update_price(current_price)
            else:
                pos = Position(
                    ticker=action.ticker,
                    quantity=fill_qty,
                    entry_price=fill_price,
                    current_price=current_price,
                    side=OrderSide.BUY,
                )
                pos.update_price(current_price)
                self.portfolio.positions[action.ticker] = pos

        else:
            existing = self.portfolio.positions.get(action.ticker)
            if existing is None:
                return {
                    "order_id": order_id,
                    "status": OrderStatus.REJECTED.value,
                    "reason": f"No position in {action.ticker}",
                }

            fill_qty = min(fill_qty, int(existing.quantity))
            pnl = (fill_price - existing.entry_price) * fill_qty
            self.portfolio.cash += fill_price * fill_qty - costs["commission"]

            if fill_qty >= existing.quantity:
                del self.portfolio.positions[action.ticker]
            else:
                existing.quantity -= fill_qty
                existing.update_price(current_price)

        trade = Trade(
            trade_id=str(uuid4()),
            ticker=action.ticker,
            side=action.side,
            quantity=fill_qty,
            price=fill_price,
            commission=costs["commission"],
            slippage=costs["slippage_per_share"],
            timestamp=timestamp,
            order_id=order_id,
            agent_name=action.reasoning[:50] if action.reasoning else "",
            pnl=pnl if action.side == OrderSide.SELL else None,
        )
        self.trades.append(trade)
        self.portfolio.update()

        return {
            "order_id": order_id,
            "status": status.value,
            "filled_qty": fill_qty,
            "fill_price": fill_price,
            "commission": costs["commission"],
            "slippage_bps": costs["cost_pct"] * 10000,
            "trade_id": trade.trade_id,
        }

    def check_pending_orders(
        self,
        current_price: float,
        daily_volume: float = 1_000_000,
        current_time: datetime | None = None,
    ) -> list[dict[str, Any]]:
        now = current_time or datetime.now(timezone.utc)
        filled: list[dict[str, Any]] = []
        to_remove: list[str] = []

        for order_id, order_data in self.pending_orders.items():
            action = order_data["action"]
            should_fill = False

            if action.order_type == OrderType.LIMIT:
                if action.side == OrderSide.BUY and current_price <= (action.limit_price or 0):
                    should_fill = True
                elif action.side == OrderSide.SELL and current_price >= (action.limit_price or 0):
                    should_fill = True
            elif action.order_type == OrderType.STOP:
                if action.side == OrderSide.SELL and current_price <= (action.stop_price or 0):
                    should_fill = True

            if should_fill:
                result = self._execute_fill(action, current_price, daily_volume, now, order_id)
                filled.append(result)
                to_remove.append(order_id)

        for order_id in to_remove:
            del self.pending_orders[order_id]

        return filled

    def get_portfolio_snapshot(self) -> dict[str, Any]:
        self.portfolio.update()
        return {
            "cash": self.portfolio.cash,
            "total_value": self.portfolio.total_value,
            "positions": {
                ticker: {
                    "quantity": pos.quantity,
                    "entry_price": pos.entry_price,
                    "current_price": pos.current_price,
                    "unrealized_pnl": pos.unrealized_pnl,
                    "weight": pos.weight,
                }
                for ticker, pos in self.portfolio.positions.items()
            },
            "total_trades": len(self.trades),
            "pending_orders": len(self.pending_orders),
        }

    def reset(self, initial_capital: float = 100_000) -> None:
        self.portfolio = Portfolio(cash=initial_capital, total_value=initial_capital)
        self.trades.clear()
        self.pending_orders.clear()
