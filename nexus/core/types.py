from __future__ import annotations

from datetime import date, datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class Environment(str, Enum):
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


class SignalType(str, Enum):
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    STRONG_BUY = "strong_buy"
    STRONG_SELL = "strong_sell"


class OrderSide(str, Enum):
    BUY = "buy"
    SELL = "sell"


class OrderType(str, Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderStatus(str, Enum):
    PENDING = "pending"
    SUBMITTED = "submitted"
    PARTIAL = "partial"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


class TimeFrame(str, Enum):
    MINUTE_1 = "1Min"
    MINUTE_5 = "5Min"
    MINUTE_15 = "15Min"
    MINUTE_30 = "30Min"
    HOUR_1 = "1Hour"
    HOUR_4 = "4Hour"
    DAILY = "1Day"
    WEEKLY = "1Week"
    MONTHLY = "1Month"


class AssetClass(str, Enum):
    EQUITY = "equity"
    OPTION = "option"
    CRYPTO = "crypto"
    FOREX = "forex"


class Sentiment(str, Enum):
    VERY_BULLISH = "very_bullish"
    BULLISH = "bullish"
    NEUTRAL = "neutral"
    BEARISH = "bearish"
    VERY_BEARISH = "very_bearish"


class Signal(BaseModel):
    ticker: str
    signal_type: SignalType
    confidence: float = Field(ge=0.0, le=1.0)
    agent_name: str
    reasoning: str = ""
    target_price: float | None = None
    stop_loss: float | None = None
    timeframe: TimeFrame = TimeFrame.DAILY
    timestamp: datetime = Field(default_factory=lambda: datetime.now())
    metadata: dict[str, Any] = Field(default_factory=dict)


class Action(BaseModel):
    ticker: str
    side: OrderSide
    quantity: float = Field(gt=0)
    order_type: OrderType = OrderType.MARKET
    limit_price: float | None = None
    stop_price: float | None = None
    confidence: float = Field(ge=0.0, le=1.0)
    signals: list[Signal] = Field(default_factory=list)
    reasoning: str = ""
    risk_score: float = Field(default=0.0, ge=0.0, le=1.0)


class MarketData(BaseModel):
    ticker: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    vwap: float | None = None
    trade_count: int | None = None


class OHLCV(BaseModel):
    ticker: str
    dates: list[datetime] = Field(default_factory=list)
    open: list[float] = Field(default_factory=list)
    high: list[float] = Field(default_factory=list)
    low: list[float] = Field(default_factory=list)
    close: list[float] = Field(default_factory=list)
    volume: list[int] = Field(default_factory=list)


class NewsEvent(BaseModel):
    title: str
    source: str
    url: str
    published_at: datetime
    content: str = ""
    summary: str = ""
    tickers: list[str] = Field(default_factory=list)
    sentiment_score: float | None = None
    relevance_score: float | None = None


class Fundamental(BaseModel):
    ticker: str
    period: str
    fiscal_date: date
    revenue: float | None = None
    net_income: float | None = None
    eps: float | None = None
    pe_ratio: float | None = None
    pb_ratio: float | None = None
    debt_to_equity: float | None = None
    roe: float | None = None
    roa: float | None = None
    current_ratio: float | None = None
    free_cash_flow: float | None = None
    gross_margin: float | None = None
    operating_margin: float | None = None
    market_cap: float | None = None
    dividend_yield: float | None = None


class Trade(BaseModel):
    trade_id: str
    ticker: str
    side: OrderSide
    quantity: float
    price: float
    commission: float = 0.0
    slippage: float = 0.0
    timestamp: datetime = Field(default_factory=lambda: datetime.now())
    order_id: str | None = None
    agent_name: str = ""
    pnl: float | None = None
    pnl_pct: float | None = None


class Position(BaseModel):
    ticker: str
    quantity: float
    entry_price: float
    current_price: float = 0.0
    market_value: float = 0.0
    unrealized_pnl: float = 0.0
    unrealized_pnl_pct: float = 0.0
    weight: float = 0.0
    cost_basis: float = 0.0
    side: OrderSide = OrderSide.BUY
    opened_at: datetime = Field(default_factory=lambda: datetime.now())

    def update_price(self, price: float) -> None:
        self.current_price = price
        self.market_value = self.quantity * price
        self.cost_basis = self.quantity * self.entry_price
        self.unrealized_pnl = self.market_value - self.cost_basis
        self.unrealized_pnl_pct = (
            (self.unrealized_pnl / self.cost_basis * 100) if self.cost_basis else 0.0
        )


class Portfolio(BaseModel):
    cash: float = 0.0
    total_value: float = 0.0
    positions: dict[str, Position] = Field(default_factory=dict)
    daily_pnl: float = 0.0
    total_pnl: float = 0.0
    total_pnl_pct: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    win_rate: float = 0.0
    total_trades: int = 0

    def update(self) -> None:
        positions_value = sum(p.market_value for p in self.positions.values())
        self.total_value = self.cash + positions_value
        for pos in self.positions.values():
            pos.weight = pos.market_value / self.total_value if self.total_value else 0.0


class AgentState(BaseModel):
    agent_name: str
    status: str = "idle"
    last_signal: Signal | None = None
    last_run: datetime | None = None
    decisions_count: int = 0
    accuracy: float = 0.0
    avg_confidence: float = 0.0
    total_cost_usd: float = 0.0
    error_count: int = 0


class BacktestResult(BaseModel):
    strategy_name: str = ""
    start_date: date | None = None
    end_date: date | None = None
    initial_capital: float = 0.0
    final_value: float = 0.0
    total_return: float = 0.0
    total_return_pct: float = 0.0
    cagr: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_duration_days: int = 0
    volatility: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    total_trades: int = 0
    avg_trade_pnl: float = 0.0
    best_trade: float = 0.0
    worst_trade: float = 0.0
    avg_holding_days: float = 0.0
    beta: float = 0.0
    alpha: float = 0.0
    trades: list[Trade] = Field(default_factory=list)
    equity_curve: list[float] = Field(default_factory=list)
    drawdown_series: list[float] = Field(default_factory=list)
