from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from nexus.llm.providers import (
    AnthropicProvider,
    BaseLLMProvider,
    GroqProvider,
    LLMResponse,
    OpenAIProvider,
    create_provider,
    _estimate_cost,
)
from nexus.llm.cache import LLMCache, SemanticCache
from nexus.llm.prompts import PROMPT_REGISTRY, get_system_prompt, render_prompt
from nexus.llm.tools import ToolRegistry


class TestLLMResponse:
    def test_to_dict(self):
        resp = LLMResponse(
            content="test",
            model="test-model",
            input_tokens=100,
            output_tokens=200,
            latency_ms=50.0,
            cost_usd=0.001,
        )
        d = resp.to_dict()
        assert d["content"] == "test"
        assert d["model"] == "test-model"
        assert d["input_tokens"] == 100
        assert d["cost_usd"] == 0.001


class TestCostEstimation:
    def test_known_model(self):
        cost = _estimate_cost("gpt-4o", 1000, 500)
        assert cost > 0

    def test_unknown_model(self):
        cost = _estimate_cost("unknown-model", 1000, 500)
        assert cost == 0.0


class TestCreateProvider:
    def test_create_anthropic(self):
        provider = create_provider("anthropic")
        assert isinstance(provider, AnthropicProvider)

    def test_create_openai(self):
        provider = create_provider("openai")
        assert isinstance(provider, OpenAIProvider)

    def test_create_groq(self):
        provider = create_provider("groq")
        assert isinstance(provider, GroqProvider)

    def test_create_unknown(self):
        with pytest.raises(ValueError, match="Unknown LLM provider"):
            create_provider("unknown")

    def test_custom_model(self):
        provider = create_provider("openai", model="gpt-4o-mini")
        assert provider.model == "gpt-4o-mini"


class TestLLMCache:
    def test_put_and_get(self):
        cache = LLMCache(ttl_seconds=60)
        cache.put("prompt", "system", "model", 0.1, {"content": "response"})
        result = cache.get("prompt", "system", "model", 0.1)
        assert result == {"content": "response"}

    def test_cache_miss(self):
        cache = LLMCache(ttl_seconds=60)
        result = cache.get("prompt", "system", "model", 0.1)
        assert result is None

    def test_different_params_miss(self):
        cache = LLMCache(ttl_seconds=60)
        cache.put("prompt", "system", "model", 0.1, {"content": "response"})
        result = cache.get("prompt", "system", "model", 0.5)
        assert result is None

    def test_eviction(self):
        cache = LLMCache(ttl_seconds=60, max_size=2)
        cache.put("p1", "s", "m", 0.1, {"v": 1})
        cache.put("p2", "s", "m", 0.1, {"v": 2})
        cache.put("p3", "s", "m", 0.1, {"v": 3})
        assert cache.get("p1", "s", "m", 0.1) is None
        assert cache.get("p3", "s", "m", 0.1) == {"v": 3}

    def test_stats(self):
        cache = LLMCache()
        cache.put("p", "s", "m", 0.1, {"v": 1})
        cache.get("p", "s", "m", 0.1)
        cache.get("miss", "s", "m", 0.1)
        stats = cache.stats
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["hit_rate"] == 0.5


class TestSemanticCache:
    def test_exact_match(self):
        cache = SemanticCache(similarity_threshold=0.9)
        cache.put("analyze AAPL stock price", "model", {"v": 1})
        result = cache.get("analyze AAPL stock price", "model")
        assert result == {"v": 1}

    def test_no_match_different_model(self):
        cache = SemanticCache()
        cache.put("test prompt", "model_a", {"v": 1})
        result = cache.get("test prompt", "model_b")
        assert result is None


class TestPromptRegistry:
    def test_all_agent_types_registered(self):
        expected = [
            "technical", "fundamental", "sentiment", "quantitative",
            "macro", "event", "bull", "bear", "risk", "portfolio",
            "execution", "rl_agent", "coordinator",
        ]
        for agent_type in expected:
            assert agent_type in PROMPT_REGISTRY

    def test_render_prompt(self):
        result = render_prompt(
            "technical",
            ticker="AAPL",
            price_data="{}",
            indicators="{}",
        )
        assert "AAPL" in result

    def test_render_unknown_type(self):
        with pytest.raises(ValueError, match="Unknown agent type"):
            render_prompt("nonexistent", ticker="AAPL")

    def test_get_system_prompt(self):
        prompt = get_system_prompt("technical")
        assert "technical analysis" in prompt

    def test_system_prompt_base(self):
        prompt = get_system_prompt("unknown")
        assert "quantitative analyst" in prompt


class TestToolRegistry:
    def test_register_and_list(self):
        registry = ToolRegistry()
        registry.register("test_tool", "A test tool", lambda: "result")
        tools = registry.list_tools()
        assert len(tools) == 1
        assert tools[0]["name"] == "test_tool"

    @pytest.mark.asyncio
    async def test_execute_sync(self):
        registry = ToolRegistry()
        registry.register("add", "Add numbers", lambda a, b: a + b)
        result = await registry.execute("add", a=2, b=3)
        assert result == 5

    @pytest.mark.asyncio
    async def test_execute_async(self):
        async def async_fn(x):
            return x * 2

        registry = ToolRegistry()
        registry.register("double", "Double a number", async_fn)
        result = await registry.execute("double", x=5)
        assert result == 10

    @pytest.mark.asyncio
    async def test_execute_missing_tool(self):
        registry = ToolRegistry()
        with pytest.raises(ValueError, match="Tool not found"):
            await registry.execute("nonexistent")
