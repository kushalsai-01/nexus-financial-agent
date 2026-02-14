from __future__ import annotations

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock

from nexus.orchestration.state import (
    TradingState,
    create_initial_state,
    get_state_summary,
    update_state_with_output,
)
from nexus.orchestration.router import LLMRouter
from nexus.orchestration.protocol import A2ABroker, A2AMessage, MCPServer
from nexus.orchestration.memory import DecisionMemory, VectorMemory
from nexus.agents.base import AgentOutput
from nexus.core.types import Signal, SignalType


class TestTradingState:
    def test_create_initial_state(self):
        state = create_initial_state("AAPL")
        assert state["ticker"] == "AAPL"
        assert state["current_price"] == 0.0
        assert state["errors"] == []
        assert state["agent_outputs"] == {}

    def test_update_state_with_output(self):
        state = create_initial_state("AAPL")
        output = AgentOutput(
            agent_name="technical",
            agent_type="research",
            ticker="AAPL",
            confidence=0.8,
            llm_cost_usd=0.001,
            latency_ms=50.0,
        )
        state = update_state_with_output(state, output)
        assert "technical" in state["agent_outputs"]
        assert state["total_llm_cost"] == 0.001

    def test_update_state_with_error(self):
        state = create_initial_state("AAPL")
        output = AgentOutput(
            agent_name="failed_agent",
            agent_type="research",
            ticker="AAPL",
            error="Something went wrong",
        )
        state = update_state_with_output(state, output)
        assert len(state["errors"]) == 1

    def test_get_state_summary(self):
        state = create_initial_state("AAPL")
        output = AgentOutput(
            agent_name="technical",
            agent_type="research",
            ticker="AAPL",
            confidence=0.8,
            signal=Signal(
                ticker="AAPL",
                signal_type=SignalType.BUY,
                confidence=0.8,
                agent_name="technical",
            ),
        )
        state = update_state_with_output(state, output)
        summary = get_state_summary(state)
        assert summary["ticker"] == "AAPL"
        assert summary["agents_completed"] == 1
        assert len(summary["signals"]) == 1


class TestLLMRouter:
    def test_tier_configs_exist(self):
        assert "cheap" in LLMRouter.TIER_CONFIGS
        assert "medium" in LLMRouter.TIER_CONFIGS
        assert "expensive" in LLMRouter.TIER_CONFIGS

    def test_agent_tier_mapping(self):
        assert LLMRouter.AGENT_TIERS["bull"] == "expensive"
        assert LLMRouter.AGENT_TIERS["bear"] == "expensive"
        assert LLMRouter.AGENT_TIERS["technical"] == "medium"
        assert LLMRouter.AGENT_TIERS["event"] == "cheap"

    def test_get_tier_for_agent(self):
        router = LLMRouter()
        assert router.get_tier_for_agent("bull") == "expensive"
        assert router.get_tier_for_agent("unknown") == "cheap"

    def test_stats(self):
        router = LLMRouter()
        stats = router.stats
        assert "total_cost_usd" in stats
        assert "calls_by_tier" in stats


class TestMCPServer:
    def test_register_tool(self):
        server = MCPServer()
        server.register_tool("test", "A test tool", lambda: None)
        tools = server.list_tools()
        assert len(tools) == 1
        assert tools[0]["name"] == "test"

    @pytest.mark.asyncio
    async def test_call_tool(self):
        server = MCPServer()
        server.register_tool("add", "Add numbers", lambda a, b: a + b)
        result = await server.call_tool("add", {"a": 2, "b": 3})
        assert result == 5

    @pytest.mark.asyncio
    async def test_call_missing_tool(self):
        server = MCPServer()
        with pytest.raises(ValueError, match="MCP tool not found"):
            await server.call_tool("nonexistent")

    def test_register_resource(self):
        server = MCPServer()
        server.register_resource("nexus://data", "Market Data", "Real-time data", lambda: {})
        resources = server.list_resources()
        assert len(resources) == 1


class TestA2ABroker:
    @pytest.mark.asyncio
    async def test_send_and_receive(self):
        broker = A2ABroker()
        broker.register_agent("agent_a")
        broker.register_agent("agent_b")

        msg = A2AMessage(
            sender="agent_a",
            receiver="agent_b",
            content={"signal": "BUY"},
        )
        await broker.send(msg)

        received = await broker.receive("agent_b", timeout=1.0)
        assert received is not None
        assert received.content == {"signal": "BUY"}

    @pytest.mark.asyncio
    async def test_broadcast(self):
        broker = A2ABroker()
        broker.register_agent("sender")
        broker.register_agent("recv_1")
        broker.register_agent("recv_2")

        msg = A2AMessage(
            sender="sender",
            receiver="*",
            content={"alert": "risk"},
        )
        await broker.send(msg)

        r1 = await broker.receive("recv_1", timeout=1.0)
        r2 = await broker.receive("recv_2", timeout=1.0)
        assert r1 is not None
        assert r2 is not None

    @pytest.mark.asyncio
    async def test_receive_timeout(self):
        broker = A2ABroker()
        broker.register_agent("lonely")
        result = await broker.receive("lonely", timeout=0.1)
        assert result is None

    def test_stats(self):
        broker = A2ABroker()
        broker.register_agent("test")
        stats = broker.stats
        assert "test" in stats["registered_agents"]


class TestDecisionMemory:
    def test_store_and_query(self):
        memory = DecisionMemory()
        entry_id = memory.store("AAPL", {"signal": "BUY"})
        results = memory.query(ticker="AAPL")
        assert len(results) == 1
        assert results[0]["decision"]["signal"] == "BUY"

    def test_update_outcome(self):
        memory = DecisionMemory()
        entry_id = memory.store("AAPL", {"signal": "BUY"})
        updated = memory.update_outcome(entry_id, {"profitable": True, "return_pct": 5.0})
        assert updated is True

    def test_accuracy_tracking(self):
        memory = DecisionMemory()
        memory.store("AAPL", {"signal": "BUY"})
        memory.update_outcome(
            memory._entries[-1]["id"],
            {"profitable": True},
        )
        memory.store("AAPL", {"signal": "SELL"})
        memory.update_outcome(
            memory._entries[-1]["id"],
            {"profitable": False},
        )
        accuracy = memory.get_accuracy("AAPL")
        assert accuracy["total"] == 2
        assert accuracy["correct"] == 1
        assert accuracy["accuracy"] == 0.5

    def test_max_entries(self):
        memory = DecisionMemory(max_entries=5)
        for i in range(10):
            memory.store("AAPL", {"i": i})
        assert len(memory._entries) == 5

    def test_query_by_signal(self):
        memory = DecisionMemory()
        memory.store("AAPL", {"signal": "BUY"})
        memory.store("AAPL", {"signal": "SELL"})
        results = memory.query(ticker="AAPL", signal_type="BUY")
        assert len(results) == 1


class TestVectorMemory:
    @pytest.mark.asyncio
    async def test_store_and_query_fallback(self):
        memory = VectorMemory()
        await memory.store("AAPL buy signal based on technical analysis", {"ticker": "AAPL"})
        results = await memory.query("AAPL technical analysis", n_results=1)
        assert len(results) >= 1

    def test_stats_uninitialized(self):
        memory = VectorMemory()
        stats = memory.stats
        assert stats["initialized"] is False
