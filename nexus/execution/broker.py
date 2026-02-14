from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any

from nexus.core.exceptions import BrokerConnectionError, OrderError
from nexus.core.logging import get_logger
from nexus.core.types import (
    Action,
    MarketData,
    OrderSide,
    OrderStatus,
    OrderType,
    Position,
    Trade,
)

logger = get_logger("execution.broker")


class BaseBroker(ABC):
    def __init__(self, name: str, paper: bool = True) -> None:
        self.name = name
        self.paper = paper
        self._connected = False

    @abstractmethod
    async def connect(self) -> None:
        ...

    @abstractmethod
    async def disconnect(self) -> None:
        ...

    @abstractmethod
    async def submit_order(self, action: Action) -> str:
        ...

    @abstractmethod
    async def cancel_order(self, order_id: str) -> bool:
        ...

    @abstractmethod
    async def get_order_status(self, order_id: str) -> dict[str, Any]:
        ...

    @abstractmethod
    async def get_positions(self) -> dict[str, Position]:
        ...

    @abstractmethod
    async def get_account_info(self) -> dict[str, Any]:
        ...

    @abstractmethod
    async def get_quote(self, ticker: str) -> dict[str, float]:
        ...

    @property
    def is_connected(self) -> bool:
        return self._connected


class AlpacaBroker(BaseBroker):
    def __init__(
        self,
        api_key: str = "",
        secret_key: str = "",
        paper: bool = True,
    ) -> None:
        super().__init__("alpaca", paper)
        self.api_key = api_key
        self.secret_key = secret_key
        self._base_url = (
            "https://paper-api.alpaca.markets" if paper
            else "https://api.alpaca.markets"
        )
        self._client: Any = None

    async def connect(self) -> None:
        try:
            from alpaca.trading.client import TradingClient
            self._client = TradingClient(
                self.api_key,
                self.secret_key,
                paper=self.paper,
            )
            self._connected = True
            logger.info(f"Connected to Alpaca ({'paper' if self.paper else 'live'})")
        except ImportError:
            logger.warning("alpaca-py not installed, using stub mode")
            self._connected = True
        except Exception as e:
            raise BrokerConnectionError(f"Alpaca connection failed: {e}")

    async def disconnect(self) -> None:
        self._client = None
        self._connected = False
        logger.info("Disconnected from Alpaca")

    async def submit_order(self, action: Action) -> str:
        if not self._connected:
            raise BrokerConnectionError("Not connected to Alpaca")

        if self._client is None:
            logger.warning("Alpaca stub mode — order simulated")
            return f"stub-{datetime.now().timestamp()}"

        try:
            from alpaca.trading.requests import (
                LimitOrderRequest,
                MarketOrderRequest,
                StopLimitOrderRequest,
                StopOrderRequest,
            )
            from alpaca.trading.enums import OrderSide as AlpacaSide, TimeInForce

            side = AlpacaSide.BUY if action.side == OrderSide.BUY else AlpacaSide.SELL

            if action.order_type == OrderType.MARKET:
                req = MarketOrderRequest(
                    symbol=action.ticker,
                    qty=action.quantity,
                    side=side,
                    time_in_force=TimeInForce.DAY,
                )
            elif action.order_type == OrderType.LIMIT:
                req = LimitOrderRequest(
                    symbol=action.ticker,
                    qty=action.quantity,
                    side=side,
                    time_in_force=TimeInForce.DAY,
                    limit_price=action.limit_price,
                )
            elif action.order_type == OrderType.STOP:
                req = StopOrderRequest(
                    symbol=action.ticker,
                    qty=action.quantity,
                    side=side,
                    time_in_force=TimeInForce.DAY,
                    stop_price=action.stop_price,
                )
            else:
                req = StopLimitOrderRequest(
                    symbol=action.ticker,
                    qty=action.quantity,
                    side=side,
                    time_in_force=TimeInForce.DAY,
                    limit_price=action.limit_price,
                    stop_price=action.stop_price,
                )

            order = self._client.submit_order(req)
            logger.info(f"Alpaca order submitted: {order.id}")
            return str(order.id)

        except Exception as e:
            raise OrderError(f"Alpaca order failed: {e}")

    async def cancel_order(self, order_id: str) -> bool:
        if self._client is None:
            return True
        try:
            self._client.cancel_order_by_id(order_id)
            return True
        except Exception as e:
            logger.error(f"Cancel failed: {e}")
            return False

    async def get_order_status(self, order_id: str) -> dict[str, Any]:
        if self._client is None:
            return {"order_id": order_id, "status": "filled"}
        try:
            order = self._client.get_order_by_id(order_id)
            return {
                "order_id": str(order.id),
                "status": str(order.status),
                "filled_qty": float(order.filled_qty or 0),
                "filled_avg_price": float(order.filled_avg_price or 0),
            }
        except Exception as e:
            raise OrderError(f"Failed to get order status: {e}")

    async def get_positions(self) -> dict[str, Position]:
        if self._client is None:
            return {}
        try:
            positions = self._client.get_all_positions()
            result: dict[str, Position] = {}
            for pos in positions:
                result[pos.symbol] = Position(
                    ticker=pos.symbol,
                    quantity=float(pos.qty),
                    entry_price=float(pos.avg_entry_price),
                    current_price=float(pos.current_price),
                    market_value=float(pos.market_value),
                    unrealized_pnl=float(pos.unrealized_pl),
                    unrealized_pnl_pct=float(pos.unrealized_plpc) * 100,
                    side=OrderSide.BUY if float(pos.qty) > 0 else OrderSide.SELL,
                )
            return result
        except Exception as e:
            raise BrokerConnectionError(f"Failed to get positions: {e}")

    async def get_account_info(self) -> dict[str, Any]:
        if self._client is None:
            return {"cash": 100000, "portfolio_value": 100000, "buying_power": 200000}
        try:
            account = self._client.get_account()
            return {
                "cash": float(account.cash),
                "portfolio_value": float(account.portfolio_value),
                "buying_power": float(account.buying_power),
                "equity": float(account.equity),
                "status": str(account.status),
            }
        except Exception as e:
            raise BrokerConnectionError(f"Failed to get account info: {e}")

    async def get_quote(self, ticker: str) -> dict[str, float]:
        if self._client is None:
            return {"bid": 0, "ask": 0, "last": 0}
        return {"bid": 0, "ask": 0, "last": 0}


class IBKRBroker(BaseBroker):
    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 7497,
        client_id: int = 1,
        paper: bool = True,
    ) -> None:
        super().__init__("ibkr", paper)
        self.host = host
        self.port = port
        self.client_id = client_id
        self._ib: Any = None

    async def connect(self) -> None:
        try:
            from ib_insync import IB
            self._ib = IB()
            await self._ib.connectAsync(self.host, self.port, clientId=self.client_id)
            self._connected = True
            logger.info(f"Connected to IBKR at {self.host}:{self.port}")
        except ImportError:
            logger.warning("ib_insync not installed, using stub mode")
            self._connected = True
        except Exception as e:
            raise BrokerConnectionError(f"IBKR connection failed: {e}")

    async def disconnect(self) -> None:
        if self._ib is not None:
            self._ib.disconnect()
        self._connected = False

    async def submit_order(self, action: Action) -> str:
        if not self._connected:
            raise BrokerConnectionError("Not connected to IBKR")

        if self._ib is None:
            return f"ibkr-stub-{datetime.now().timestamp()}"

        try:
            from ib_insync import MarketOrder, LimitOrder, StopOrder, Stock

            contract = Stock(action.ticker, "SMART", "USD")

            ib_side = "BUY" if action.side == OrderSide.BUY else "SELL"

            if action.order_type == OrderType.MARKET:
                order = MarketOrder(ib_side, action.quantity)
            elif action.order_type == OrderType.LIMIT:
                order = LimitOrder(ib_side, action.quantity, action.limit_price or 0)
            elif action.order_type == OrderType.STOP:
                order = StopOrder(ib_side, action.quantity, action.stop_price or 0)
            else:
                order = LimitOrder(ib_side, action.quantity, action.limit_price or 0)

            trade = self._ib.placeOrder(contract, order)
            return str(trade.order.orderId)

        except Exception as e:
            raise OrderError(f"IBKR order failed: {e}")

    async def cancel_order(self, order_id: str) -> bool:
        return True

    async def get_order_status(self, order_id: str) -> dict[str, Any]:
        return {"order_id": order_id, "status": "unknown"}

    async def get_positions(self) -> dict[str, Position]:
        if self._ib is None:
            return {}

        try:
            positions = self._ib.positions()
            result: dict[str, Position] = {}
            for pos in positions:
                ticker = pos.contract.symbol
                result[ticker] = Position(
                    ticker=ticker,
                    quantity=float(pos.position),
                    entry_price=float(pos.avgCost),
                    current_price=float(pos.avgCost),
                    side=OrderSide.BUY if pos.position > 0 else OrderSide.SELL,
                )
            return result
        except Exception as e:
            raise BrokerConnectionError(f"IBKR positions failed: {e}")

    async def get_account_info(self) -> dict[str, Any]:
        if self._ib is None:
            return {"cash": 100000}

        try:
            summary = self._ib.accountSummary()
            info: dict[str, Any] = {}
            for item in summary:
                info[item.tag] = item.value
            return info
        except Exception as e:
            raise BrokerConnectionError(f"IBKR account info failed: {e}")

    async def get_quote(self, ticker: str) -> dict[str, float]:
        return {"bid": 0, "ask": 0, "last": 0}


BROKER_REGISTRY: dict[str, type[BaseBroker]] = {
    "alpaca": AlpacaBroker,
    "ibkr": IBKRBroker,
}


def create_broker(name: str, **kwargs: Any) -> BaseBroker:
    broker_cls = BROKER_REGISTRY.get(name)
    if broker_cls is None:
        raise ValueError(f"Unknown broker: {name}. Available: {list(BROKER_REGISTRY.keys())}")
    return broker_cls(**kwargs)
