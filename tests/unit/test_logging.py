from __future__ import annotations

import io
import json
import logging

import pytest

from nexus.core.logging import JSONFormatter, NexusLogger, get_logger, setup_logging


class TestJSONFormatter:
    def test_format_log_record(self) -> None:
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=42,
            msg="Test message",
            args=(),
            exc_info=None,
        )
        output = formatter.format(record)
        data = json.loads(output)
        assert data["message"] == "Test message"
        assert data["level"] == "INFO"
        assert data["logger"] == "test"

    def test_format_with_exception(self) -> None:
        formatter = JSONFormatter()
        try:
            raise ValueError("test error")
        except ValueError:
            import sys
            record = logging.LogRecord(
                name="test",
                level=logging.ERROR,
                pathname="test.py",
                lineno=42,
                msg="Error occurred",
                args=(),
                exc_info=sys.exc_info(),
            )
        output = formatter.format(record)
        data = json.loads(output)
        assert "exception" in data
        assert "ValueError" in data["exception"]


class TestNexusLogger:
    def test_logger_creation(self) -> None:
        logger = NexusLogger("test_module")
        assert logger._name == "test_module"

    def test_logger_with_data(self) -> None:
        logger = NexusLogger("test_module")
        child = logger.with_data(request_id="abc123", ticker="AAPL")
        assert child._extra_data["request_id"] == "abc123"
        assert child._extra_data["ticker"] == "AAPL"


class TestGetLogger:
    def test_get_logger_returns_nexus_logger(self) -> None:
        logger = get_logger("test_component")
        assert isinstance(logger, NexusLogger)

    def test_get_logger_singleton(self) -> None:
        logger1 = get_logger("singleton_test")
        logger2 = get_logger("singleton_test")
        assert logger1 is logger2
