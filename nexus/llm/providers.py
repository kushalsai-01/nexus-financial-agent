from __future__ import annotations

import asyncio
import hashlib
import time
from abc import ABC, abstractmethod
from typing import Any

import httpx

from nexus.core.config import get_config
from nexus.core.exceptions import LLMError, LLMRateLimitError
from nexus.core.logging import get_logger

logger = get_logger("llm.providers")


class LLMResponse:
    def __init__(
        self,
        content: str,
        model: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
        latency_ms: float = 0.0,
        cost_usd: float = 0.0,
        cached: bool = False,
    ) -> None:
        self.content = content
        self.model = model
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.latency_ms = latency_ms
        self.cost_usd = cost_usd
        self.cached = cached

    def to_dict(self) -> dict[str, Any]:
        return {
            "content": self.content,
            "model": self.model,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "latency_ms": self.latency_ms,
            "cost_usd": self.cost_usd,
            "cached": self.cached,
        }


_PRICING: dict[str, tuple[float, float]] = {
    "claude-sonnet-4-20250514": (3.0 / 1_000_000, 15.0 / 1_000_000),
    "claude-opus-4-20250514": (15.0 / 1_000_000, 75.0 / 1_000_000),
    "claude-3-5-haiku-20241022": (0.80 / 1_000_000, 4.0 / 1_000_000),
    "gpt-4o": (2.50 / 1_000_000, 10.0 / 1_000_000),
    "gpt-4o-mini": (0.15 / 1_000_000, 0.60 / 1_000_000),
    "llama-3.3-70b-versatile": (0.59 / 1_000_000, 0.79 / 1_000_000),
    "grok-3": (3.0 / 1_000_000, 15.0 / 1_000_000),
    "grok-2": (2.0 / 1_000_000, 10.0 / 1_000_000),
}


def _estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    if model in _PRICING:
        inp_rate, out_rate = _PRICING[model]
        return input_tokens * inp_rate + output_tokens * out_rate
    return 0.0


class BaseLLMProvider(ABC):
    def __init__(self, model: str, max_tokens: int, temperature: float, timeout: int, max_retries: int) -> None:
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout
        self.max_retries = max_retries

    @abstractmethod
    async def generate(self, prompt: str, system: str = "", temperature: float | None = None) -> LLMResponse:
        ...

    async def generate_with_retry(self, prompt: str, system: str = "", temperature: float | None = None) -> LLMResponse:
        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                return await self.generate(prompt, system, temperature)
            except LLMRateLimitError:
                wait = min(2 ** attempt * 2, 60)
                logger.warning(f"Rate limited on {self.model}, attempt {attempt + 1}, waiting {wait}s")
                await asyncio.sleep(wait)
                last_error = LLMRateLimitError(f"Rate limited after {self.max_retries} attempts")
            except LLMError as e:
                wait = 2 ** attempt
                logger.error(f"LLM error on {self.model}, attempt {attempt + 1}: {e}")
                await asyncio.sleep(wait)
                last_error = e
            except Exception as e:
                logger.error(f"Unexpected error on {self.model}: {e}")
                last_error = LLMError(f"Provider error: {e}")
                break
        raise last_error or LLMError("Unknown LLM error")


class AnthropicProvider(BaseLLMProvider):
    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        max_tokens: int = 4096,
        temperature: float = 0.1,
        timeout: int = 60,
        max_retries: int = 3,
        api_key: str = "",
    ) -> None:
        super().__init__(model, max_tokens, temperature, timeout, max_retries)
        self._api_key = api_key or get_config().anthropic_api_key
        self._base_url = "https://api.anthropic.com/v1"

    async def generate(self, prompt: str, system: str = "", temperature: float | None = None) -> LLMResponse:
        start = time.monotonic()
        headers = {
            "x-api-key": self._api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        body: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": temperature if temperature is not None else self.temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            body["system"] = system

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.post(f"{self._base_url}/messages", json=body, headers=headers)
                if resp.status_code == 429:
                    raise LLMRateLimitError(f"Anthropic rate limit: {resp.text}")
                if resp.status_code != 200:
                    raise LLMError(f"Anthropic API error {resp.status_code}: {resp.text}")
                data = resp.json()
        except httpx.TimeoutException as e:
            raise LLMError(f"Anthropic timeout after {self.timeout}s") from e
        except (LLMError, LLMRateLimitError):
            raise
        except Exception as e:
            raise LLMError(f"Anthropic request failed: {e}") from e

        content = ""
        for block in data.get("content", []):
            if block.get("type") == "text":
                content += block.get("text", "")

        usage = data.get("usage", {})
        input_tokens = usage.get("input_tokens", 0)
        output_tokens = usage.get("output_tokens", 0)
        latency = (time.monotonic() - start) * 1000

        return LLMResponse(
            content=content,
            model=self.model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency,
            cost_usd=_estimate_cost(self.model, input_tokens, output_tokens),
        )


class OpenAIProvider(BaseLLMProvider):
    def __init__(
        self,
        model: str = "gpt-4o",
        max_tokens: int = 4096,
        temperature: float = 0.1,
        timeout: int = 60,
        max_retries: int = 3,
        api_key: str = "",
    ) -> None:
        super().__init__(model, max_tokens, temperature, timeout, max_retries)
        self._api_key = api_key or get_config().openai_api_key
        self._base_url = "https://api.openai.com/v1"

    async def generate(self, prompt: str, system: str = "", temperature: float | None = None) -> LLMResponse:
        start = time.monotonic()
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        body: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": temperature if temperature is not None else self.temperature,
            "messages": messages,
        }

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.post(f"{self._base_url}/chat/completions", json=body, headers=headers)
                if resp.status_code == 429:
                    raise LLMRateLimitError(f"OpenAI rate limit: {resp.text}")
                if resp.status_code != 200:
                    raise LLMError(f"OpenAI API error {resp.status_code}: {resp.text}")
                data = resp.json()
        except httpx.TimeoutException as e:
            raise LLMError(f"OpenAI timeout after {self.timeout}s") from e
        except (LLMError, LLMRateLimitError):
            raise
        except Exception as e:
            raise LLMError(f"OpenAI request failed: {e}") from e

        choices = data.get("choices")
        if not choices:
            raise LLMError("OpenAI returned empty choices – check model name and API key")
        content = choices[0].get("message", {}).get("content", "") or ""
        usage = data.get("usage", {})
        input_tokens = usage.get("prompt_tokens", 0)
        output_tokens = usage.get("completion_tokens", 0)
        latency = (time.monotonic() - start) * 1000

        return LLMResponse(
            content=content,
            model=self.model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency,
            cost_usd=_estimate_cost(self.model, input_tokens, output_tokens),
        )


class GroqProvider(BaseLLMProvider):
    def __init__(
        self,
        model: str = "llama-3.3-70b-versatile",
        max_tokens: int = 4096,
        temperature: float = 0.1,
        timeout: int = 30,
        max_retries: int = 3,
        api_key: str = "",
    ) -> None:
        super().__init__(model, max_tokens, temperature, timeout, max_retries)
        config = get_config()
        self._api_key = api_key or getattr(config, "groq_api_key", "")
        self._base_url = "https://api.groq.com/openai/v1"

    async def generate(self, prompt: str, system: str = "", temperature: float | None = None) -> LLMResponse:
        start = time.monotonic()
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        body: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": temperature if temperature is not None else self.temperature,
            "messages": messages,
        }

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.post(f"{self._base_url}/chat/completions", json=body, headers=headers)
                if resp.status_code == 429:
                    raise LLMRateLimitError(f"Groq rate limit: {resp.text}")
                if resp.status_code != 200:
                    raise LLMError(f"Groq API error {resp.status_code}: {resp.text}")
                data = resp.json()
        except httpx.TimeoutException as e:
            raise LLMError(f"Groq timeout after {self.timeout}s") from e
        except (LLMError, LLMRateLimitError):
            raise
        except Exception as e:
            raise LLMError(f"Groq request failed: {e}") from e

        choices = data.get("choices")
        if not choices:
            raise LLMError("Groq returned empty choices – check model name and API key")
        content = choices[0].get("message", {}).get("content", "") or ""
        usage = data.get("usage", {})
        input_tokens = usage.get("prompt_tokens", 0)
        output_tokens = usage.get("completion_tokens", 0)
        latency = (time.monotonic() - start) * 1000

        return LLMResponse(
            content=content,
            model=self.model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency,
            cost_usd=_estimate_cost(self.model, input_tokens, output_tokens),
        )


class GrokProvider(BaseLLMProvider):
    """xAI Grok provider — OpenAI-compatible endpoint at https://api.x.ai/v1."""

    def __init__(
        self,
        model: str = "grok-3",
        max_tokens: int = 4096,
        temperature: float = 0.1,
        timeout: int = 60,
        max_retries: int = 3,
        api_key: str = "",
    ) -> None:
        super().__init__(model, max_tokens, temperature, timeout, max_retries)
        config = get_config()
        self._api_key = api_key or getattr(config, "grok_api_key", "")
        self._base_url = "https://api.x.ai/v1"

    async def generate(self, prompt: str, system: str = "", temperature: float | None = None) -> LLMResponse:
        start = time.monotonic()
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        body: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": temperature if temperature is not None else self.temperature,
            "messages": messages,
        }

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.post(f"{self._base_url}/chat/completions", json=body, headers=headers)
                if resp.status_code == 429:
                    raise LLMRateLimitError(f"Grok rate limit: {resp.text}")
                if resp.status_code != 200:
                    raise LLMError(f"Grok API error {resp.status_code}: {resp.text}")
                data = resp.json()
        except httpx.TimeoutException as e:
            raise LLMError(f"Grok timeout after {self.timeout}s") from e
        except (LLMError, LLMRateLimitError):
            raise
        except Exception as e:
            raise LLMError(f"Grok request failed: {e}") from e

        choices = data.get("choices")
        if not choices:
            raise LLMError("Grok returned empty choices – check model name and API key")
        content = choices[0].get("message", {}).get("content", "") or ""
        usage = data.get("usage", {})
        input_tokens = usage.get("prompt_tokens", 0)
        output_tokens = usage.get("completion_tokens", 0)
        latency = (time.monotonic() - start) * 1000

        return LLMResponse(
            content=content,
            model=self.model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency,
            cost_usd=_estimate_cost(self.model, input_tokens, output_tokens),
        )


def create_provider(
    provider: str = "anthropic",
    model: str | None = None,
    **kwargs: Any,
) -> BaseLLMProvider:
    if provider == "anthropic":
        return AnthropicProvider(model=model or "claude-sonnet-4-20250514", **kwargs)
    elif provider == "openai":
        return OpenAIProvider(model=model or "gpt-4o", **kwargs)
    elif provider == "groq":
        return GroqProvider(model=model or "llama-3.3-70b-versatile", **kwargs)
    elif provider == "grok":
        return GrokProvider(model=model or "grok-3", **kwargs)
    else:
        raise ValueError(f"Unknown LLM provider: {provider}")


def get_primary_provider() -> BaseLLMProvider:
    config = get_config()
    return create_provider(
        provider=config.llm.primary.provider,
        model=config.llm.primary.model,
        max_tokens=config.llm.primary.max_tokens,
        temperature=config.llm.primary.temperature,
        timeout=config.llm.primary.timeout,
        max_retries=config.llm.primary.max_retries,
    )


def get_fallback_provider() -> BaseLLMProvider:
    config = get_config()
    return create_provider(
        provider=config.llm.fallback.provider,
        model=config.llm.fallback.model,
        max_tokens=config.llm.fallback.max_tokens,
        temperature=config.llm.fallback.temperature,
        timeout=config.llm.fallback.timeout,
        max_retries=config.llm.fallback.max_retries,
    )
