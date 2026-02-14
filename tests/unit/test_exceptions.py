from __future__ import annotations

import pytest

from nexus.core.exceptions import (
    AgentConsensusError,
    AgentError,
    AgentTimeoutError,
    BrokerConnectionError,
    ConfigError,
    DataError,
    DataFetchError,
    DataStorageError,
    DataValidationError,
    DrawdownLimitError,
    ErrorCode,
    ExecutionError,
    ExposureLimitError,
    InsufficientFundsError,
    LLMError,
    LLMRateLimitError,
    NexusError,
    OrderError,
    PositionLimitError,
    RiskViolation,
)


class TestExceptionHierarchy:
    def test_nexus_error_base(self) -> None:
        err = NexusError("test error")
        assert str(err) == "test error"
        assert err.code is None

    def test_nexus_error_with_code(self) -> None:
        err = NexusError("test", code=ErrorCode.DATA_FETCH_FAILED)
        assert err.code == ErrorCode.DATA_FETCH_FAILED

    def test_data_errors_inherit(self) -> None:
        assert issubclass(DataFetchError, DataError)
        assert issubclass(DataValidationError, DataError)
        assert issubclass(DataStorageError, DataError)
        assert issubclass(DataError, NexusError)

    def test_agent_errors_inherit(self) -> None:
        assert issubclass(AgentTimeoutError, AgentError)
        assert issubclass(AgentConsensusError, AgentError)
        assert issubclass(LLMError, AgentError)
        assert issubclass(LLMRateLimitError, LLMError)

    def test_execution_errors_inherit(self) -> None:
        assert issubclass(OrderError, ExecutionError)
        assert issubclass(BrokerConnectionError, ExecutionError)
        assert issubclass(InsufficientFundsError, ExecutionError)

    def test_risk_errors_inherit(self) -> None:
        assert issubclass(PositionLimitError, RiskViolation)
        assert issubclass(DrawdownLimitError, RiskViolation)
        assert issubclass(ExposureLimitError, RiskViolation)

    def test_catch_by_base_class(self) -> None:
        with pytest.raises(NexusError):
            raise DataFetchError("API timeout")

        with pytest.raises(DataError):
            raise DataValidationError("Invalid data")

    def test_error_details(self) -> None:
        err = DataFetchError(
            "Failed to fetch AAPL",
            code=ErrorCode.DATA_FETCH_FAILED,
            details={"ticker": "AAPL", "provider": "yfinance"},
        )
        assert err.details["ticker"] == "AAPL"
        assert err.code == ErrorCode.DATA_FETCH_FAILED

    def test_config_error(self) -> None:
        err = ConfigError("Missing API key")
        assert isinstance(err, NexusError)

    def test_error_code_values(self) -> None:
        assert ErrorCode.DATA_FETCH_FAILED.value == "DATA_FETCH_FAILED"
        assert len(ErrorCode) >= 10
