from __future__ import annotations

from enum import Enum


class NexusError(Exception):
    def __init__(self, message: str, code: str | None = None, details: dict | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.code = code
        self.details = details or {}


class DataError(NexusError):
    pass


class DataFetchError(DataError):
    pass


class DataValidationError(DataError):
    pass


class DataStorageError(DataError):
    pass


class AgentError(NexusError):
    pass


class AgentTimeoutError(AgentError):
    pass


class AgentConsensusError(AgentError):
    pass


class LLMError(AgentError):
    pass


class LLMRateLimitError(LLMError):
    pass


class ExecutionError(NexusError):
    pass


class OrderError(ExecutionError):
    pass


class BrokerConnectionError(ExecutionError):
    pass


class InsufficientFundsError(ExecutionError):
    pass


class RiskViolation(NexusError):
    pass


class PositionLimitError(RiskViolation):
    pass


class DrawdownLimitError(RiskViolation):
    pass


class ExposureLimitError(RiskViolation):
    pass


class ConfigError(NexusError):
    pass


class ErrorCode(str, Enum):
    DATA_FETCH_FAILED = "DATA_FETCH_FAILED"
    DATA_VALIDATION_FAILED = "DATA_VALIDATION_FAILED"
    DATA_STORAGE_FAILED = "DATA_STORAGE_FAILED"
    AGENT_TIMEOUT = "AGENT_TIMEOUT"
    AGENT_CONSENSUS_FAILED = "AGENT_CONSENSUS_FAILED"
    LLM_RATE_LIMIT = "LLM_RATE_LIMIT"
    LLM_ERROR = "LLM_ERROR"
    ORDER_FAILED = "ORDER_FAILED"
    BROKER_DISCONNECTED = "BROKER_DISCONNECTED"
    INSUFFICIENT_FUNDS = "INSUFFICIENT_FUNDS"
    RISK_POSITION_LIMIT = "RISK_POSITION_LIMIT"
    RISK_DRAWDOWN_LIMIT = "RISK_DRAWDOWN_LIMIT"
    RISK_EXPOSURE_LIMIT = "RISK_EXPOSURE_LIMIT"
    CONFIG_INVALID = "CONFIG_INVALID"
