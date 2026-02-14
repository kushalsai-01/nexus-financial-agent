from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any
from uuid import uuid4

from nexus.core.exceptions import OrderError
from nexus.core.logging import get_logger
from nexus.core.types import Action, OrderSide, OrderStatus, OrderType

logger = get_logger("execution.orders")


class OrderPriority(str, Enum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


@dataclass
class Order:
    order_id: str
    ticker: str
    side: OrderSide
    quantity: float
    order_type: OrderType
    status: OrderStatus = OrderStatus.PENDING
    limit_price: float | None = None
    stop_price: float | None = None
    filled_quantity: float = 0.0
    filled_avg_price: float = 0.0
    priority: OrderPriority = OrderPriority.NORMAL
    parent_order_id: str | None = None
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    expires_at: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def remaining_quantity(self) -> float:
        return self.quantity - self.filled_quantity

    @property
    def is_complete(self) -> bool:
        return self.status in (OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED, OrderStatus.EXPIRED)

    @property
    def fill_pct(self) -> float:
        return (self.filled_quantity / self.quantity * 100) if self.quantity > 0 else 0


class OrderManager:
    def __init__(self, max_orders: int = 10000) -> None:
        self.orders: dict[str, Order] = {}
        self.max_orders = max_orders

    def create_order(
        self,
        action: Action,
        priority: OrderPriority = OrderPriority.NORMAL,
        expires_at: datetime | None = None,
    ) -> Order:
        order = Order(
            order_id=str(uuid4()),
            ticker=action.ticker,
            side=action.side,
            quantity=action.quantity,
            order_type=action.order_type,
            limit_price=action.limit_price,
            stop_price=action.stop_price,
            priority=priority,
            expires_at=expires_at,
        )

        self.orders[order.order_id] = order
        self._cleanup_old_orders()

        logger.info(
            f"Order created: {order.order_id} "
            f"{order.side.value} {order.quantity} {order.ticker} "
            f"@ {order.order_type.value}"
        )
        return order

    def create_bracket_order(
        self,
        action: Action,
        take_profit_price: float,
        stop_loss_price: float,
    ) -> list[Order]:
        entry = self.create_order(action, priority=OrderPriority.HIGH)

        exit_side = OrderSide.SELL if action.side == OrderSide.BUY else OrderSide.BUY

        tp_action = Action(
            ticker=action.ticker,
            side=exit_side,
            quantity=action.quantity,
            order_type=OrderType.LIMIT,
            limit_price=take_profit_price,
            confidence=action.confidence,
        )
        tp_order = self.create_order(tp_action)
        tp_order.parent_order_id = entry.order_id
        tp_order.status = OrderStatus.PENDING

        sl_action = Action(
            ticker=action.ticker,
            side=exit_side,
            quantity=action.quantity,
            order_type=OrderType.STOP,
            stop_price=stop_loss_price,
            confidence=action.confidence,
        )
        sl_order = self.create_order(sl_action)
        sl_order.parent_order_id = entry.order_id
        sl_order.status = OrderStatus.PENDING

        return [entry, tp_order, sl_order]

    def create_iceberg_order(
        self,
        action: Action,
        visible_quantity: float,
    ) -> list[Order]:
        orders: list[Order] = []
        remaining = action.quantity
        slice_num = 0

        while remaining > 0:
            slice_qty = min(visible_quantity, remaining)
            slice_action = Action(
                ticker=action.ticker,
                side=action.side,
                quantity=slice_qty,
                order_type=action.order_type,
                limit_price=action.limit_price,
                confidence=action.confidence,
            )
            order = self.create_order(slice_action)
            order.metadata["iceberg_slice"] = slice_num
            order.metadata["total_iceberg_qty"] = action.quantity

            if slice_num > 0:
                order.status = OrderStatus.PENDING

            orders.append(order)
            remaining -= slice_qty
            slice_num += 1

        return orders

    def update_fill(
        self,
        order_id: str,
        filled_quantity: float,
        fill_price: float,
    ) -> Order:
        order = self.orders.get(order_id)
        if order is None:
            raise OrderError(f"Order {order_id} not found")

        prev_filled = order.filled_quantity
        total_cost = order.filled_avg_price * prev_filled + fill_price * filled_quantity
        new_filled = prev_filled + filled_quantity

        order.filled_quantity = new_filled
        order.filled_avg_price = total_cost / new_filled if new_filled > 0 else 0
        order.updated_at = datetime.now()

        if order.filled_quantity >= order.quantity:
            order.status = OrderStatus.FILLED
        elif order.filled_quantity > 0:
            order.status = OrderStatus.PARTIAL

        return order

    def cancel_order(self, order_id: str) -> bool:
        order = self.orders.get(order_id)
        if order is None:
            return False
        if order.is_complete:
            return False
        order.status = OrderStatus.CANCELLED
        order.updated_at = datetime.now()
        return True

    def cancel_all(self, ticker: str | None = None) -> int:
        cancelled = 0
        for order in self.orders.values():
            if order.is_complete:
                continue
            if ticker and order.ticker != ticker:
                continue
            order.status = OrderStatus.CANCELLED
            order.updated_at = datetime.now()
            cancelled += 1
        return cancelled

    def get_open_orders(self, ticker: str | None = None) -> list[Order]:
        result: list[Order] = []
        for order in self.orders.values():
            if order.is_complete:
                continue
            if ticker and order.ticker != ticker:
                continue
            result.append(order)
        return sorted(result, key=lambda o: o.created_at)

    def get_order(self, order_id: str) -> Order | None:
        return self.orders.get(order_id)

    def expire_old_orders(self, current_time: datetime | None = None) -> int:
        now = current_time or datetime.now()
        expired = 0
        for order in self.orders.values():
            if order.is_complete:
                continue
            if order.expires_at and order.expires_at <= now:
                order.status = OrderStatus.EXPIRED
                order.updated_at = now
                expired += 1
        return expired

    def get_summary(self) -> dict[str, Any]:
        total = len(self.orders)
        by_status: dict[str, int] = {}
        for order in self.orders.values():
            status = order.status.value
            by_status[status] = by_status.get(status, 0) + 1

        return {
            "total_orders": total,
            "by_status": by_status,
            "open_orders": sum(1 for o in self.orders.values() if not o.is_complete),
        }

    def _cleanup_old_orders(self) -> None:
        if len(self.orders) <= self.max_orders:
            return
        completed = [
            (oid, o) for oid, o in self.orders.items()
            if o.is_complete
        ]
        completed.sort(key=lambda x: x[1].updated_at)
        to_remove = len(self.orders) - self.max_orders
        for oid, _ in completed[:to_remove]:
            del self.orders[oid]
