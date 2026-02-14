from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    Index,
    Integer,
    String,
    Text,
    create_engine,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

from nexus.core.config import get_config
from nexus.core.logging import get_logger

logger = get_logger("data.storage.postgres")


class Base(DeclarativeBase):
    pass


class PriceRecord(Base):
    __tablename__ = "prices"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ticker: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    open: Mapped[float] = mapped_column(Float, nullable=False)
    high: Mapped[float] = mapped_column(Float, nullable=False)
    low: Mapped[float] = mapped_column(Float, nullable=False)
    close: Mapped[float] = mapped_column(Float, nullable=False)
    volume: Mapped[int] = mapped_column(Integer, nullable=False)
    vwap: Mapped[float | None] = mapped_column(Float, nullable=True)
    timeframe: Mapped[str] = mapped_column(String(10), default="1Day")

    __table_args__ = (
        Index("ix_prices_ticker_timestamp", "ticker", "timestamp"),
    )


class FundamentalRecord(Base):
    __tablename__ = "fundamentals"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ticker: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    period: Mapped[str] = mapped_column(String(20), nullable=False)
    fiscal_date: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    revenue: Mapped[float | None] = mapped_column(Float, nullable=True)
    net_income: Mapped[float | None] = mapped_column(Float, nullable=True)
    eps: Mapped[float | None] = mapped_column(Float, nullable=True)
    pe_ratio: Mapped[float | None] = mapped_column(Float, nullable=True)
    pb_ratio: Mapped[float | None] = mapped_column(Float, nullable=True)
    debt_to_equity: Mapped[float | None] = mapped_column(Float, nullable=True)
    roe: Mapped[float | None] = mapped_column(Float, nullable=True)
    roa: Mapped[float | None] = mapped_column(Float, nullable=True)
    current_ratio: Mapped[float | None] = mapped_column(Float, nullable=True)
    free_cash_flow: Mapped[float | None] = mapped_column(Float, nullable=True)
    gross_margin: Mapped[float | None] = mapped_column(Float, nullable=True)
    operating_margin: Mapped[float | None] = mapped_column(Float, nullable=True)
    market_cap: Mapped[float | None] = mapped_column(Float, nullable=True)
    data: Mapped[dict | None] = mapped_column(JSONB, nullable=True)

    __table_args__ = (
        Index("ix_fundamentals_ticker_date", "ticker", "fiscal_date"),
    )


class TradeRecord(Base):
    __tablename__ = "trades"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    trade_id: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    ticker: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    side: Mapped[str] = mapped_column(String(10), nullable=False)
    quantity: Mapped[float] = mapped_column(Float, nullable=False)
    price: Mapped[float] = mapped_column(Float, nullable=False)
    commission: Mapped[float] = mapped_column(Float, default=0.0)
    slippage: Mapped[float] = mapped_column(Float, default=0.0)
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=func.now())
    order_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    agent_name: Mapped[str] = mapped_column(String(64), default="")
    pnl: Mapped[float | None] = mapped_column(Float, nullable=True)
    pnl_pct: Mapped[float | None] = mapped_column(Float, nullable=True)
    metadata_: Mapped[dict | None] = mapped_column("metadata", JSONB, nullable=True)

    __table_args__ = (
        Index("ix_trades_ticker_timestamp", "ticker", "timestamp"),
    )


class PositionRecord(Base):
    __tablename__ = "positions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ticker: Mapped[str] = mapped_column(String(20), nullable=False, unique=True)
    quantity: Mapped[float] = mapped_column(Float, nullable=False)
    entry_price: Mapped[float] = mapped_column(Float, nullable=False)
    current_price: Mapped[float] = mapped_column(Float, default=0.0)
    market_value: Mapped[float] = mapped_column(Float, default=0.0)
    unrealized_pnl: Mapped[float] = mapped_column(Float, default=0.0)
    side: Mapped[str] = mapped_column(String(10), default="buy")
    opened_at: Mapped[datetime] = mapped_column(DateTime, default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=func.now(), onupdate=func.now())


class AgentDecisionRecord(Base):
    __tablename__ = "agent_decisions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    agent_name: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    ticker: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    signal_type: Mapped[str] = mapped_column(String(20), nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    reasoning: Mapped[str] = mapped_column(Text, default="")
    target_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    stop_loss: Mapped[float | None] = mapped_column(Float, nullable=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=func.now())
    cost_usd: Mapped[float] = mapped_column(Float, default=0.0)
    latency_ms: Mapped[float] = mapped_column(Float, default=0.0)
    metadata_: Mapped[dict | None] = mapped_column("metadata", JSONB, nullable=True)

    __table_args__ = (
        Index("ix_decisions_agent_timestamp", "agent_name", "timestamp"),
    )


class NewsRecord(Base):
    __tablename__ = "news"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    source: Mapped[str] = mapped_column(String(128), nullable=False)
    url: Mapped[str] = mapped_column(Text, nullable=False)
    published_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    content: Mapped[str] = mapped_column(Text, default="")
    tickers: Mapped[list | None] = mapped_column(JSONB, nullable=True)
    sentiment_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    content_hash: Mapped[str | None] = mapped_column(String(64), unique=True, nullable=True)


class PostgresStorage:
    def __init__(self) -> None:
        config = get_config()
        self._url = config.storage.postgres.url
        self._pool_size = config.storage.postgres.pool_size
        self._max_overflow = config.storage.postgres.max_overflow
        self._echo = config.storage.postgres.echo
        self._engine = create_engine(
            self._url,
            pool_size=self._pool_size,
            max_overflow=self._max_overflow,
            echo=self._echo,
        )
        self._session_factory = sessionmaker(bind=self._engine)

    def create_tables(self) -> None:
        Base.metadata.create_all(self._engine)
        logger.info("Database tables created")

    def drop_tables(self) -> None:
        Base.metadata.drop_all(self._engine)
        logger.info("Database tables dropped")

    def get_session(self) -> Session:
        return self._session_factory()

    def save_prices(self, records: list[dict[str, Any]]) -> int:
        with self.get_session() as session:
            objs = [PriceRecord(**r) for r in records]
            session.add_all(objs)
            session.commit()
            logger.info(f"Saved {len(objs)} price records")
            return len(objs)

    def get_prices(
        self,
        ticker: str,
        start: datetime | None = None,
        end: datetime | None = None,
        limit: int = 1000,
    ) -> list[dict[str, Any]]:
        with self.get_session() as session:
            query = session.query(PriceRecord).filter(PriceRecord.ticker == ticker)
            if start:
                query = query.filter(PriceRecord.timestamp >= start)
            if end:
                query = query.filter(PriceRecord.timestamp <= end)
            query = query.order_by(PriceRecord.timestamp.desc()).limit(limit)
            return [
                {
                    "ticker": r.ticker,
                    "timestamp": r.timestamp,
                    "open": r.open,
                    "high": r.high,
                    "low": r.low,
                    "close": r.close,
                    "volume": r.volume,
                    "vwap": r.vwap,
                }
                for r in query.all()
            ]

    def save_trade(self, trade_data: dict[str, Any]) -> str:
        with self.get_session() as session:
            record = TradeRecord(**trade_data)
            session.add(record)
            session.commit()
            logger.info(f"Saved trade: {record.trade_id}")
            return record.trade_id

    def get_trades(
        self,
        ticker: str | None = None,
        start: datetime | None = None,
        end: datetime | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        with self.get_session() as session:
            query = session.query(TradeRecord)
            if ticker:
                query = query.filter(TradeRecord.ticker == ticker)
            if start:
                query = query.filter(TradeRecord.timestamp >= start)
            if end:
                query = query.filter(TradeRecord.timestamp <= end)
            query = query.order_by(TradeRecord.timestamp.desc()).limit(limit)
            return [
                {
                    "trade_id": r.trade_id,
                    "ticker": r.ticker,
                    "side": r.side,
                    "quantity": r.quantity,
                    "price": r.price,
                    "commission": r.commission,
                    "timestamp": r.timestamp,
                    "agent_name": r.agent_name,
                    "pnl": r.pnl,
                }
                for r in query.all()
            ]

    def save_decision(self, decision_data: dict[str, Any]) -> int:
        with self.get_session() as session:
            record = AgentDecisionRecord(**decision_data)
            session.add(record)
            session.commit()
            return record.id

    def get_decisions(
        self,
        agent_name: str | None = None,
        ticker: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        with self.get_session() as session:
            query = session.query(AgentDecisionRecord)
            if agent_name:
                query = query.filter(AgentDecisionRecord.agent_name == agent_name)
            if ticker:
                query = query.filter(AgentDecisionRecord.ticker == ticker)
            query = query.order_by(AgentDecisionRecord.timestamp.desc()).limit(limit)
            return [
                {
                    "agent_name": r.agent_name,
                    "ticker": r.ticker,
                    "signal_type": r.signal_type,
                    "confidence": r.confidence,
                    "reasoning": r.reasoning,
                    "timestamp": r.timestamp,
                    "cost_usd": r.cost_usd,
                }
                for r in query.all()
            ]

    def health_check(self) -> bool:
        try:
            with self.get_session() as session:
                session.execute(func.now())
                return True
        except Exception:
            return False
