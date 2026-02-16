from __future__ import annotations

import asyncio
import json
import time
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field

from nexus.core.exceptions import AgentError, AgentTimeoutError, LLMError
from nexus.core.logging import get_logger
from nexus.core.types import Signal, SignalType, TimeFrame
from nexus.llm.providers import BaseLLMProvider, LLMResponse, create_provider


class AgentOutput(BaseModel):
    agent_name: str
    agent_type: str
    ticker: str
    timestamp: datetime = Field(default_factory=datetime.now)
    execution_id: str = Field(default_factory=lambda: str(uuid4()))
    raw_response: str = ""
    parsed_output: dict[str, Any] = Field(default_factory=dict)
    signal: Signal | None = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    latency_ms: float = 0.0
    llm_cost_usd: float = 0.0
    model_used: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class BaseAgent(ABC):
    def __init__(
        self,
        name: str,
        agent_type: str,
        provider: BaseLLMProvider | None = None,
        timeout: int = 120,
    ) -> None:
        self.name = name
        self.agent_type = agent_type
        self.provider = provider
        self.timeout = timeout
        self.logger = get_logger(f"agent.{name}")
        self._total_cost = 0.0
        self._total_calls = 0
        self._total_errors = 0

    @abstractmethod
    async def analyze(self, ticker: str, data: dict[str, Any]) -> AgentOutput:
        ...

    async def execute(self, ticker: str, data: dict[str, Any]) -> AgentOutput:
        start = time.monotonic()
        try:
            result = await asyncio.wait_for(
                self.analyze(ticker, data),
                timeout=self.timeout,
            )
            result.latency_ms = (time.monotonic() - start) * 1000
            self._total_calls += 1
            self._total_cost += result.llm_cost_usd
            self.logger.info(
                f"Agent {self.name} completed for {ticker}",
                extra={"latency_ms": result.latency_ms, "cost": result.llm_cost_usd},
            )

            # Record cost in global tracker for "nexus costs" visibility
            if not result.error and result.llm_cost_usd > 0 and result.model_used:
                try:
                    from nexus.monitoring.cost import CostTracker
                    CostTracker.get_instance().record(
                        agent_name=self.name,
                        model=result.model_used,
                        input_tokens=result.input_tokens,
                        output_tokens=result.output_tokens,
                        cost_usd=result.llm_cost_usd,
                        ticker=ticker,
                    )
                except Exception:
                    pass  # never let cost tracking crash the agent

            return result
        except asyncio.TimeoutError:
            self._total_errors += 1
            self.logger.error(f"Agent {self.name} timed out after {self.timeout}s for {ticker}")
            return AgentOutput(
                agent_name=self.name,
                agent_type=self.agent_type,
                ticker=ticker,
                error=f"Timeout after {self.timeout}s",
                latency_ms=(time.monotonic() - start) * 1000,
            )
        except Exception as e:
            self._total_errors += 1
            self.logger.error(f"Agent {self.name} failed for {ticker}: {e}")
            return AgentOutput(
                agent_name=self.name,
                agent_type=self.agent_type,
                ticker=ticker,
                error=str(e),
                latency_ms=(time.monotonic() - start) * 1000,
            )

    async def call_llm(
        self,
        prompt: str,
        system: str = "",
        temperature: float | None = None,
    ) -> LLMResponse:
        if not self.provider:
            raise AgentError(f"No LLM provider for agent {self.name}")
        return await self.provider.generate_with_retry(prompt, system, temperature)

    def parse_json_response(self, raw: str) -> dict[str, Any]:
        text = raw.strip()
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                try:
                    return json.loads(text[start:end])
                except json.JSONDecodeError:
                    pass
            self.logger.warning(f"Failed to parse JSON from {self.name}")
            return {}

    def build_signal(
        self,
        ticker: str,
        signal_type: SignalType,
        confidence: float,
        reasoning: str = "",
        target_price: float | None = None,
        stop_loss: float | None = None,
        timeframe: TimeFrame = TimeFrame.DAILY,
        metadata: dict[str, Any] | None = None,
    ) -> Signal:
        return Signal(
            ticker=ticker,
            signal_type=signal_type,
            confidence=confidence,
            agent_name=self.name,
            reasoning=reasoning,
            target_price=target_price,
            stop_loss=stop_loss,
            timeframe=timeframe,
            metadata=metadata or {},
        )

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "type": self.agent_type,
            "total_calls": self._total_calls,
            "total_errors": self._total_errors,
            "total_cost_usd": self._total_cost,
            "error_rate": self._total_errors / max(self._total_calls, 1),
        }
