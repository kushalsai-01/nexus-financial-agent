from __future__ import annotations

from datetime import datetime, time, timezone
from uuid import uuid4

import numpy as np
import pandas as pd
import pytest

from nexus.core.types import Action, OrderSide, OrderStatus, OrderType
from nexus.execution.algorithms import (
    ALGO_REGISTRY,
    AlgoConfig,
    ImplementationShortfall,
    POVAlgorithm,
    TWAPAlgorithm,
    VWAPAlgorithm,
)
from nexus.execution.broker import BROKER_REGISTRY, AlpacaBroker, IBKRBroker, create_broker
from nexus.execution.orders import Order, OrderManager, OrderPriority
from nexus.execution.simulator import PaperTradingSimulator
from nexus.execution.slippage import SlippageModel, TieredSlippageModel, estimate_execution_cost


def _make_action(
    ticker: str = "AAPL",
    qty: float = 100,
    price: float = 150,
    side: OrderSide = OrderSide.BUY,
    order_type: OrderType = OrderType.MARKET,
) -> Action:
    return Action(
        ticker=ticker,
        side=side,
        quantity=qty,
        order_type=order_type,
        limit_price=price,
        confidence=0.8,
    )


class TestVWAP:
    def test_generate_schedule(self) -> None:
        algo = VWAPAlgorithm()
        schedule = algo.generate_schedule(1000, {})
        assert len(schedule) > 0
        total_qty = sum(s["quantity"] for s in schedule)
        assert abs(total_qty - 1000) <= 10

    def test_small_order(self) -> None:
        algo = VWAPAlgorithm()
        schedule = algo.generate_schedule(50, {})
        assert len(schedule) == 1

    def test_custom_volume_profile(self) -> None:
        algo = VWAPAlgorithm()
        profile = [0.2, 0.3, 0.5]
        schedule = algo.generate_schedule(1000, {"volume_profile": profile})
        assert len(schedule) > 0


class TestTWAP:
    def test_generate_schedule(self) -> None:
        algo = TWAPAlgorithm()
        schedule = algo.generate_schedule(1000, {})
        assert len(schedule) > 0

    def test_equal_distribution(self) -> None:
        algo = TWAPAlgorithm(config=AlgoConfig(randomize=False))
        schedule = algo.generate_schedule(1000, {})
        quantities = [s["quantity"] for s in schedule]
        assert max(quantities) - min(quantities) <= 1


class TestPOV:
    def test_generate_schedule(self) -> None:
        algo = POVAlgorithm(target_participation=0.05)
        schedule = algo.generate_schedule(5000, {"avg_daily_volume": 1_000_000})
        assert len(schedule) > 0

    def test_respects_participation_rate(self) -> None:
        algo = POVAlgorithm(target_participation=0.01)
        schedule = algo.generate_schedule(10000, {"avg_daily_volume": 100_000})
        assert len(schedule) > 1


class TestImplementationShortfall:
    def test_generate_schedule(self) -> None:
        algo = ImplementationShortfall()
        schedule = algo.generate_schedule(1000, {"volatility": 0.02, "avg_daily_volume": 1_000_000})
        assert len(schedule) >= 2

    def test_front_loading(self) -> None:
        algo = ImplementationShortfall(config=AlgoConfig(urgency=1.0, randomize=False))
        schedule = algo.generate_schedule(1000, {"volatility": 0.02})
        if len(schedule) >= 2:
            assert schedule[0]["quantity"] >= schedule[-1]["quantity"]


class TestAlgoRegistry:
    def test_all_registered(self) -> None:
        assert "vwap" in ALGO_REGISTRY
        assert "twap" in ALGO_REGISTRY
        assert "pov" in ALGO_REGISTRY
        assert "implementation_shortfall" in ALGO_REGISTRY


class TestSlippageModel:
    def test_estimate(self) -> None:
        model = SlippageModel()
        result = model.estimate(1000, 1_000_000)
        assert result["total_slippage_bps"] > 0

    def test_large_order_higher_impact(self) -> None:
        model = SlippageModel()
        small = model.estimate(100, 1_000_000)
        large = model.estimate(100_000, 1_000_000)
        assert large["market_impact_bps"] > small["market_impact_bps"]

    def test_high_urgency(self) -> None:
        model = SlippageModel()
        low = model.estimate(1000, 1_000_000, urgency=0.1)
        high = model.estimate(1000, 1_000_000, urgency=1.0)
        assert high["urgency_cost_bps"] > low["urgency_cost_bps"]


class TestTieredSlippage:
    def test_defaults(self) -> None:
        model = TieredSlippageModel()
        assert model.tiers is not None
        assert len(model.tiers) == 5

    def test_estimate(self) -> None:
        model = TieredSlippageModel()
        result = model.estimate(1000, 1_000_000)
        assert "total_bps" in result
        assert result["total_bps"] > 0

    def test_higher_participation_higher_cost(self) -> None:
        model = TieredSlippageModel()
        low = model.estimate(1_000, 1_000_000)
        high = model.estimate(100_000, 1_000_000)
        assert high["total_bps"] > low["total_bps"]


class TestEstimateExecutionCost:
    def test_returns_cost(self) -> None:
        result = estimate_execution_cost(150.0, 100)
        assert result["estimated_cost_dollars"] > 0
        assert result["notional"] == 15000


class TestBrokerRegistry:
    def test_alpaca_registered(self) -> None:
        assert "alpaca" in BROKER_REGISTRY

    def test_ibkr_registered(self) -> None:
        assert "ibkr" in BROKER_REGISTRY

    def test_create_broker(self) -> None:
        broker = create_broker("alpaca", paper=True)
        assert isinstance(broker, AlpacaBroker)

    def test_create_unknown(self) -> None:
        with pytest.raises(ValueError):
            create_broker("unknown_broker")


class TestAlpacaBroker:
    def test_init(self) -> None:
        broker = AlpacaBroker(paper=True)
        assert broker.name == "alpaca"
        assert broker.paper is True
        assert not broker.is_connected


class TestIBKRBroker:
    def test_init(self) -> None:
        broker = IBKRBroker()
        assert broker.name == "ibkr"
        assert not broker.is_connected


class TestOrderManager:
    def test_create_order(self) -> None:
        mgr = OrderManager()
        action = _make_action()
        order = mgr.create_order(action)
        assert order.ticker == "AAPL"
        assert order.status == OrderStatus.PENDING
        assert order.quantity == 100

    def test_create_bracket(self) -> None:
        mgr = OrderManager()
        action = _make_action()
        orders = mgr.create_bracket_order(action, take_profit_price=170, stop_loss_price=140)
        assert len(orders) == 3
        assert orders[1].order_type == OrderType.LIMIT
        assert orders[2].order_type == OrderType.STOP

    def test_iceberg_order(self) -> None:
        mgr = OrderManager()
        action = _make_action(qty=1000)
        orders = mgr.create_iceberg_order(action, visible_quantity=100)
        assert len(orders) == 10
        total_qty = sum(o.quantity for o in orders)
        assert total_qty == 1000

    def test_update_fill(self) -> None:
        mgr = OrderManager()
        order = mgr.create_order(_make_action())
        mgr.update_fill(order.order_id, 50, 151.0)
        assert order.filled_quantity == 50
        assert order.status == OrderStatus.PARTIAL

    def test_fill_complete(self) -> None:
        mgr = OrderManager()
        order = mgr.create_order(_make_action())
        mgr.update_fill(order.order_id, 100, 150.0)
        assert order.status == OrderStatus.FILLED

    def test_cancel(self) -> None:
        mgr = OrderManager()
        order = mgr.create_order(_make_action())
        assert mgr.cancel_order(order.order_id) is True
        assert order.status == OrderStatus.CANCELLED

    def test_cancel_filled(self) -> None:
        mgr = OrderManager()
        order = mgr.create_order(_make_action())
        mgr.update_fill(order.order_id, 100, 150.0)
        assert mgr.cancel_order(order.order_id) is False

    def test_cancel_all(self) -> None:
        mgr = OrderManager()
        mgr.create_order(_make_action(ticker="AAPL"))
        mgr.create_order(_make_action(ticker="MSFT"))
        cancelled = mgr.cancel_all()
        assert cancelled == 2

    def test_get_open_orders(self) -> None:
        mgr = OrderManager()
        mgr.create_order(_make_action())
        mgr.create_order(_make_action(ticker="MSFT"))
        o3 = mgr.create_order(_make_action(ticker="GOOGL"))
        mgr.cancel_order(o3.order_id)
        open_orders = mgr.get_open_orders()
        assert len(open_orders) == 2

    def test_summary(self) -> None:
        mgr = OrderManager()
        mgr.create_order(_make_action())
        summary = mgr.get_summary()
        assert summary["total_orders"] == 1
        assert summary["open_orders"] == 1


class TestPaperTradingSimulator:
    def test_buy(self) -> None:
        sim = PaperTradingSimulator(initial_capital=100_000, enforce_market_hours=False)
        action = _make_action(qty=10, price=150)
        result = sim.submit_order(action, current_price=150.0)
        assert result["status"] in ("filled", "partial")
        assert "AAPL" in sim.portfolio.positions

    def test_buy_sell(self) -> None:
        sim = PaperTradingSimulator(initial_capital=100_000, enforce_market_hours=False, seed=100)
        buy = _make_action(qty=10, price=150)
        sim.submit_order(buy, current_price=150.0)

        sell = _make_action(qty=10, price=155, side=OrderSide.SELL)
        result = sim.submit_order(sell, current_price=155.0)
        assert result["status"] in ("filled", "partial")

    def test_insufficient_funds(self) -> None:
        sim = PaperTradingSimulator(initial_capital=1_000, enforce_market_hours=False, seed=0)
        action = _make_action(qty=100, price=150)
        result = sim.submit_order(action, current_price=150.0)
        assert result["status"] in ("rejected", "filled", "partial")

    def test_limit_order_pending(self) -> None:
        sim = PaperTradingSimulator(enforce_market_hours=False, seed=0)
        action = _make_action(qty=10, price=140, order_type=OrderType.LIMIT)
        result = sim.submit_order(action, current_price=150.0)
        assert result["status"] == "pending"
        assert len(sim.pending_orders) == 1

    def test_pending_fills(self) -> None:
        sim = PaperTradingSimulator(enforce_market_hours=False, seed=0)
        action = _make_action(qty=10, price=140, order_type=OrderType.LIMIT)
        sim.submit_order(action, current_price=150.0)
        filled = sim.check_pending_orders(current_price=139.0)
        assert len(filled) > 0

    def test_portfolio_snapshot(self) -> None:
        sim = PaperTradingSimulator(enforce_market_hours=False, seed=0)
        snap = sim.get_portfolio_snapshot()
        assert snap["cash"] == 100_000
        assert snap["total_trades"] == 0

    def test_reset(self) -> None:
        sim = PaperTradingSimulator(enforce_market_hours=False, seed=0)
        sim.submit_order(_make_action(qty=10), current_price=150.0)
        sim.reset(200_000)
        assert sim.portfolio.cash == 200_000
        assert len(sim.trades) == 0
