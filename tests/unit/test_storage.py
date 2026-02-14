from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest


class TestPostgresStorage:
    def test_price_record_model(self) -> None:
        from nexus.data.storage.postgres import PriceRecord

        record = PriceRecord(
            ticker="AAPL",
            timestamp=datetime.now(),
            open=150.0,
            high=155.0,
            low=149.0,
            close=153.0,
            volume=1000000,
        )
        assert record.ticker == "AAPL"
        assert record.close == 153.0

    def test_trade_record_model(self) -> None:
        from nexus.data.storage.postgres import TradeRecord

        record = TradeRecord(
            trade_id="T001",
            ticker="AAPL",
            side="buy",
            quantity=100.0,
            price=150.0,
        )
        assert record.trade_id == "T001"
        assert record.quantity == 100.0

    def test_agent_decision_record_model(self) -> None:
        from nexus.data.storage.postgres import AgentDecisionRecord

        record = AgentDecisionRecord(
            agent_name="tech_analyst",
            ticker="MSFT",
            signal_type="buy",
            confidence=0.85,
        )
        assert record.agent_name == "tech_analyst"
        assert record.confidence == 0.85

    def test_base_metadata(self) -> None:
        from nexus.data.storage.postgres import Base

        tables = Base.metadata.tables
        assert "prices" in tables
        assert "trades" in tables
        assert "positions" in tables
        assert "agent_decisions" in tables
        assert "fundamentals" in tables
        assert "news" in tables
