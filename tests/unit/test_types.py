from __future__ import annotations

from datetime import datetime

import pytest
from pydantic import ValidationError

from nexus.core.types import (
    Action,
    AgentState,
    BacktestResult,
    Fundamental,
    MarketData,
    NewsEvent,
    OHLCV,
    OrderSide,
    OrderStatus,
    OrderType,
    Portfolio,
    Position,
    Sentiment,
    Signal,
    SignalType,
    TimeFrame,
    Trade,
)


class TestEnums:
    def test_signal_type_values(self) -> None:
        assert SignalType.BUY == "buy"
        assert SignalType.SELL == "sell"
        assert SignalType.HOLD == "hold"
        assert SignalType.STRONG_BUY == "strong_buy"
        assert SignalType.STRONG_SELL == "strong_sell"

    def test_order_side_values(self) -> None:
        assert OrderSide.BUY == "buy"
        assert OrderSide.SELL == "sell"

    def test_order_type_values(self) -> None:
        assert OrderType.MARKET == "market"
        assert OrderType.LIMIT == "limit"
        assert OrderType.STOP == "stop"
        assert OrderType.STOP_LIMIT == "stop_limit"

    def test_order_status_values(self) -> None:
        assert len(OrderStatus) == 7

    def test_timeframe_values(self) -> None:
        assert TimeFrame.DAILY == "1Day"
        assert TimeFrame.WEEKLY == "1Week"
        assert TimeFrame.MINUTE_1 == "1Min"

    def test_sentiment_values(self) -> None:
        assert Sentiment.VERY_BULLISH == "very_bullish"
        assert Sentiment.VERY_BEARISH == "very_bearish"


class TestSignal:
    def test_create_signal(self) -> None:
        signal = Signal(
            ticker="AAPL",
            signal_type=SignalType.BUY,
            confidence=0.85,
            agent_name="test_agent",
        )
        assert signal.ticker == "AAPL"
        assert signal.signal_type == SignalType.BUY
        assert signal.confidence == 0.85
        assert signal.timeframe == TimeFrame.DAILY

    def test_signal_confidence_bounds(self) -> None:
        with pytest.raises(ValidationError):
            Signal(
                ticker="AAPL",
                signal_type=SignalType.BUY,
                confidence=1.5,
                agent_name="test",
            )

        with pytest.raises(ValidationError):
            Signal(
                ticker="AAPL",
                signal_type=SignalType.BUY,
                confidence=-0.1,
                agent_name="test",
            )

    def test_signal_with_metadata(self) -> None:
        signal = Signal(
            ticker="TSLA",
            signal_type=SignalType.STRONG_SELL,
            confidence=0.92,
            agent_name="risk_agent",
            metadata={"reason": "earnings_miss", "impact": "high"},
        )
        assert signal.metadata["reason"] == "earnings_miss"


class TestAction:
    def test_create_action(self) -> None:
        action = Action(
            ticker="MSFT",
            side=OrderSide.BUY,
            quantity=100,
            confidence=0.75,
        )
        assert action.ticker == "MSFT"
        assert action.order_type == OrderType.MARKET
        assert action.quantity == 100

    def test_action_quantity_positive(self) -> None:
        with pytest.raises(ValidationError):
            Action(
                ticker="MSFT",
                side=OrderSide.BUY,
                quantity=0,
                confidence=0.5,
            )


class TestMarketData:
    def test_create_market_data(self) -> None:
        md = MarketData(
            ticker="AAPL",
            timestamp=datetime.now(),
            open=150.0,
            high=155.0,
            low=149.0,
            close=153.0,
            volume=1000000,
        )
        assert md.close == 153.0
        assert md.vwap is None


class TestPosition:
    def test_update_price(self) -> None:
        pos = Position(
            ticker="AAPL",
            quantity=100,
            entry_price=150.0,
        )
        pos.update_price(160.0)
        assert pos.current_price == 160.0
        assert pos.market_value == 16000.0
        assert pos.unrealized_pnl == 1000.0
        assert pos.cost_basis == 15000.0

    def test_negative_pnl(self) -> None:
        pos = Position(
            ticker="TSLA",
            quantity=50,
            entry_price=200.0,
        )
        pos.update_price(180.0)
        assert pos.unrealized_pnl == -1000.0


class TestPortfolio:
    def test_portfolio_update(self) -> None:
        portfolio = Portfolio(cash=100000.0)
        pos = Position(ticker="AAPL", quantity=100, entry_price=150.0)
        pos.update_price(160.0)
        portfolio.positions["AAPL"] = pos
        portfolio.update()
        assert portfolio.total_value == 116000.0
        assert portfolio.positions["AAPL"].weight > 0

    def test_empty_portfolio(self) -> None:
        portfolio = Portfolio(cash=50000.0)
        portfolio.update()
        assert portfolio.total_value == 50000.0


class TestTrade:
    def test_create_trade(self) -> None:
        trade = Trade(
            trade_id="T001",
            ticker="GOOGL",
            side=OrderSide.BUY,
            quantity=10,
            price=2800.0,
        )
        assert trade.trade_id == "T001"
        assert trade.commission == 0.0


class TestBacktestResult:
    def test_create_result(self) -> None:
        result = BacktestResult(
            strategy_name="test_strategy",
            initial_capital=1_000_000.0,
            final_value=1_150_000.0,
            total_return=150_000.0,
            total_return_pct=15.0,
            sharpe_ratio=1.8,
            max_drawdown=-5.2,
        )
        assert result.total_return_pct == 15.0
        assert result.sharpe_ratio == 1.8
