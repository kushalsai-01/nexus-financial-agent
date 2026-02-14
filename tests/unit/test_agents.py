from __future__ import annotations

import asyncio
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from nexus.agents.base import AgentOutput, BaseAgent
from nexus.agents.technical import TechnicalAgent
from nexus.agents.fundamental import FundamentalAgent
from nexus.agents.sentiment import SentimentAgent
from nexus.agents.quantitative import QuantitativeAgent
from nexus.agents.macro import MacroAgent
from nexus.agents.event import EventAgent
from nexus.agents.rl_agent import RLAgent
from nexus.agents.bull import BullAgent
from nexus.agents.bear import BearAgent
from nexus.agents.risk import RiskAgent
from nexus.agents.portfolio import PortfolioAgent
from nexus.agents.execution import ExecutionAgent
from nexus.agents.coordinator import CoordinatorAgent
from nexus.agents.market_data import MarketDataAgent
from nexus.agents import AGENT_REGISTRY
from nexus.core.types import SignalType
from nexus.llm.providers import LLMResponse


def _mock_provider(response_json: dict) -> MagicMock:
    provider = MagicMock()
    provider.generate_with_retry = AsyncMock(
        return_value=LLMResponse(
            content=json.dumps(response_json),
            model="test-model",
            input_tokens=100,
            output_tokens=200,
            latency_ms=50.0,
            cost_usd=0.001,
        )
    )
    return provider


class TestAgentRegistry:
    def test_all_agents_registered(self):
        expected = [
            "market_data", "technical", "fundamental", "quantitative",
            "sentiment", "macro", "event", "rl_agent", "bull", "bear",
            "risk", "portfolio", "execution", "coordinator",
        ]
        for name in expected:
            assert name in AGENT_REGISTRY

    def test_registry_count(self):
        assert len(AGENT_REGISTRY) == 14


class TestMarketDataAgent:
    @pytest.mark.asyncio
    async def test_analyze(self):
        agent = MarketDataAgent()
        data = {
            "price_data": {
                "close": 150.0,
                "change_1d": 2.5,
                "high_52w": 180.0,
                "low_52w": 120.0,
            },
            "volume_profile": {"avg_20d": 1000000, "relative": 1.2},
        }
        result = await agent.execute("AAPL", data)
        assert isinstance(result, AgentOutput)
        assert result.agent_name == "market_data"
        assert result.ticker == "AAPL"
        assert result.confidence == 1.0
        assert result.parsed_output["current_price"] == 150.0


class TestTechnicalAgent:
    @pytest.mark.asyncio
    async def test_analyze_buy_signal(self):
        mock = _mock_provider({
            "signal": "BUY",
            "strength": 0.8,
            "support_level": 145.0,
            "resistance_level": 160.0,
            "trend": "bullish",
            "key_patterns": ["golden_cross"],
            "reasoning": "Strong uptrend confirmed",
        })
        agent = TechnicalAgent(provider=mock)
        result = await agent.execute("AAPL", {"price_data": {}, "technical_indicators": {}})
        assert result.signal is not None
        assert result.signal.signal_type == SignalType.BUY
        assert result.confidence == 0.8

    @pytest.mark.asyncio
    async def test_analyze_hold_signal(self):
        mock = _mock_provider({
            "signal": "HOLD",
            "strength": 0.5,
            "support_level": 145.0,
            "resistance_level": 155.0,
            "trend": "neutral",
            "key_patterns": [],
            "reasoning": "No clear direction",
        })
        agent = TechnicalAgent(provider=mock)
        result = await agent.execute("AAPL", {})
        assert result.signal.signal_type == SignalType.HOLD


class TestFundamentalAgent:
    @pytest.mark.asyncio
    async def test_analyze_undervalued(self):
        mock = _mock_provider({
            "fair_value": 200.0,
            "upside_pct": 25.0,
            "downside_pct": -5.0,
            "quality_score": 0.85,
            "moat_rating": "wide",
            "growth_outlook": "strong",
            "reasoning": "Significantly undervalued",
        })
        agent = FundamentalAgent(provider=mock)
        result = await agent.execute("AAPL", {"current_price": 160})
        assert result.signal.signal_type == SignalType.STRONG_BUY
        assert result.confidence == 0.85


class TestSentimentAgent:
    @pytest.mark.asyncio
    async def test_analyze_bullish(self):
        mock = _mock_provider({
            "sentiment": "bullish",
            "score": 0.7,
            "volume_indicator": "high",
            "key_topics": ["earnings beat"],
            "catalyst_events": ["product launch"],
            "reasoning": "Positive sentiment",
        })
        agent = SentimentAgent(provider=mock)
        result = await agent.execute("AAPL", {})
        assert result.signal.signal_type == SignalType.BUY


class TestQuantitativeAgent:
    @pytest.mark.asyncio
    async def test_analyze_high_alpha(self):
        mock = _mock_provider({
            "alpha_score": 2.0,
            "factor_breakdown": {
                "momentum": 0.8,
                "value": 0.3,
                "quality": 0.7,
                "volatility": -0.2,
                "size": 0.1,
            },
            "sharpe_estimate": 1.5,
            "var_95": 0.03,
            "reasoning": "Strong alpha",
        })
        agent = QuantitativeAgent(provider=mock)
        result = await agent.execute("AAPL", {})
        assert result.signal.signal_type == SignalType.STRONG_BUY


class TestBullAgent:
    @pytest.mark.asyncio
    async def test_analyze_high_conviction(self):
        mock = _mock_provider({
            "conviction": 9,
            "position_action": "buy",
            "position_size_pct": 3.0,
            "entry_price": 150.0,
            "price_target": 200.0,
            "catalyst_timeline_days": 60,
            "upside_scenarios": [],
            "key_arguments": ["Strong growth"],
            "reasoning": "Compelling bull case",
        })
        agent = BullAgent(provider=mock)
        result = await agent.execute("AAPL", {"current_price": 150})
        assert result.signal.signal_type == SignalType.STRONG_BUY
        assert result.confidence == 0.9


class TestBearAgent:
    @pytest.mark.asyncio
    async def test_analyze_sell(self):
        mock = _mock_provider({
            "conviction": 8,
            "action": "sell",
            "downside_target": 120.0,
            "stop_loss": 155.0,
            "downside_scenarios": [],
            "key_arguments": ["Overvalued"],
            "reasoning": "Strong bear case",
        })
        agent = BearAgent(provider=mock)
        result = await agent.execute("AAPL", {"current_price": 150})
        assert result.signal.signal_type == SignalType.STRONG_SELL
        assert result.confidence == 0.8


class TestRiskAgent:
    @pytest.mark.asyncio
    async def test_analyze_approve(self):
        mock = _mock_provider({
            "decision": "APPROVE",
            "adjusted_size_pct": 2.5,
            "adjusted_stop_loss": 142.0,
            "risk_score": 0.3,
            "position_var": 0.02,
            "portfolio_impact": {
                "new_exposure_pct": 5.0,
                "sector_concentration": 15.0,
                "correlation_risk": 0.3,
            },
            "veto_reasons": [],
            "conditions": ["Use stop loss at 142"],
            "reasoning": "Acceptable risk",
        })
        agent = RiskAgent(provider=mock)
        result = await agent.execute("AAPL", {})
        assert result.parsed_output["decision"] == "APPROVE"

    @pytest.mark.asyncio
    async def test_analyze_veto(self):
        mock = _mock_provider({
            "decision": "VETO",
            "adjusted_size_pct": 0,
            "adjusted_stop_loss": 0,
            "risk_score": 0.9,
            "position_var": 0.08,
            "portfolio_impact": {},
            "veto_reasons": ["Exceeds position limit"],
            "conditions": [],
            "reasoning": "Too risky",
        })
        agent = RiskAgent(provider=mock)
        result = await agent.execute("AAPL", {})
        assert result.parsed_output["decision"] == "VETO"


class TestPortfolioAgent:
    @pytest.mark.asyncio
    async def test_analyze_buy_decision(self):
        mock = _mock_provider({
            "action": "buy",
            "size_pct": 2.5,
            "limit_price": 149.0,
            "stop_loss": 142.0,
            "take_profit": 175.0,
            "order_type": "limit",
            "urgency": "patient",
            "kelly_fraction": 0.15,
            "expected_return": 0.12,
            "reasoning": "Buy with limit order",
        })
        agent = PortfolioAgent(provider=mock)
        result = await agent.execute("AAPL", {})
        assert result.signal.signal_type == SignalType.BUY

    def test_kelly_computation(self):
        data = {
            "historical_win_rate": 0.6,
            "avg_win_pct": 3.0,
            "avg_loss_pct": 2.0,
        }
        kelly = PortfolioAgent._compute_kelly(data)
        assert kelly["kelly_fraction"] > 0
        assert kelly["half_kelly"] > 0
        assert kelly["recommended_size_pct"] > 0


class TestExecutionAgent:
    @pytest.mark.asyncio
    async def test_analyze_vwap(self):
        mock = _mock_provider({
            "strategy": "VWAP",
            "urgency": "medium",
            "num_slices": 5,
            "time_horizon_minutes": 30,
            "max_participation_rate": 0.05,
            "expected_slippage_bps": 3.0,
            "reasoning": "VWAP optimal for this volume",
        })
        agent = ExecutionAgent(provider=mock)
        result = await agent.execute("AAPL", {})
        assert result.parsed_output["strategy"] == "VWAP"


class TestCoordinatorAgent:
    @pytest.mark.asyncio
    async def test_analyze_consensus(self):
        mock = _mock_provider({
            "consensus_reached": True,
            "consensus_signal": "BUY",
            "consensus_confidence": 0.75,
            "agent_agreement_pct": 80.0,
            "dissenting_agents": ["bear"],
            "recommended_action": {
                "side": "buy",
                "size_pct": 2.0,
                "confidence": 0.75,
            },
            "reasoning": "Strong consensus to buy",
        })
        agent = CoordinatorAgent(provider=mock)
        result = await agent.execute("AAPL", {"current_price": 150})
        assert result.signal.signal_type == SignalType.BUY
        assert result.parsed_output["consensus_reached"] is True


class TestBaseAgentTimeout:
    @pytest.mark.asyncio
    async def test_timeout_returns_error_output(self):
        class SlowAgent(BaseAgent):
            async def analyze(self, ticker, data):
                await asyncio.sleep(10)
                return AgentOutput(agent_name=self.name, agent_type=self.agent_type, ticker=ticker)

        agent = SlowAgent(name="slow", agent_type="test", timeout=1)
        result = await agent.execute("AAPL", {})
        assert result.error is not None
        assert "Timeout" in result.error


class TestBaseAgentJsonParsing:
    def test_parse_clean_json(self):
        agent = MarketDataAgent()
        result = agent.parse_json_response('{"key": "value"}')
        assert result == {"key": "value"}

    def test_parse_json_with_markdown(self):
        agent = MarketDataAgent()
        result = agent.parse_json_response('```json\n{"key": "value"}\n```')
        assert result == {"key": "value"}

    def test_parse_json_with_surrounding_text(self):
        agent = MarketDataAgent()
        result = agent.parse_json_response('Here is the result: {"key": "value"} end.')
        assert result == {"key": "value"}

    def test_parse_invalid_json(self):
        agent = MarketDataAgent()
        result = agent.parse_json_response("not json at all")
        assert result == {}
