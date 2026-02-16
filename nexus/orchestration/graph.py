from __future__ import annotations

import asyncio
import json
import time
from typing import Any

from nexus.agents.base import AgentOutput, BaseAgent
from nexus.agents import (
    AGENT_REGISTRY,
    BearAgent,
    BullAgent,
    CoordinatorAgent,
    EventAgent,
    ExecutionAgent,
    FundamentalAgent,
    MacroAgent,
    MarketDataAgent,
    PortfolioAgent,
    QuantitativeAgent,
    RiskAgent,
    SentimentAgent,
    TechnicalAgent,
)
from nexus.core.config import get_config
from nexus.core.logging import get_logger
from nexus.llm.providers import create_provider
from nexus.orchestration.router import LLMRouter
from nexus.orchestration.state import (
    TradingState,
    create_initial_state,
    get_state_summary,
    update_state_with_output,
)

logger = get_logger("orchestration.graph")


class TradingGraph:
    def __init__(self, router: LLMRouter | None = None, config: Any = None) -> None:
        self._router = router or LLMRouter()
        self._config = config
        self._agents: dict[str, BaseAgent] = {}
        self._initialize_agents()

    def _initialize_agents(self) -> None:
        cheap = self._router.get_provider("cheap")
        medium = self._router.get_provider("medium")
        expensive = self._router.get_provider("expensive")

        self._agents["market_data"] = MarketDataAgent()
        self._agents["technical"] = TechnicalAgent(provider=medium)
        self._agents["fundamental"] = FundamentalAgent(provider=medium)
        self._agents["quantitative"] = QuantitativeAgent(provider=medium)
        self._agents["sentiment"] = SentimentAgent(provider=medium)
        self._agents["macro"] = MacroAgent(provider=medium)
        self._agents["event"] = EventAgent(provider=cheap)
        self._agents["rl_agent"] = AGENT_REGISTRY["rl_agent"](provider=cheap)
        self._agents["bull"] = BullAgent(provider=expensive)
        self._agents["bear"] = BearAgent(provider=expensive)
        self._agents["risk"] = RiskAgent(provider=medium)
        self._agents["portfolio"] = PortfolioAgent(provider=expensive)
        self._agents["execution"] = ExecutionAgent(provider=cheap)
        self._agents["coordinator"] = CoordinatorAgent(provider=medium)

    async def run(self, ticker: str, initial_data: dict[str, Any] | None = None, capital: float | None = None) -> TradingState:
        state = create_initial_state(ticker)
        if initial_data is None:
            initial_data = {}
        if capital is not None:
            initial_data["capital"] = capital
        if initial_data:
            state.update(initial_data)

        start = time.monotonic()
        logger.info(f"Starting trading graph for {ticker}")

        # Fetch real data via DataPipeline if state has no market data yet
        try:
            if not state.get("market_data") or not state["market_data"].get("latest_close"):
                from nexus.data.pipeline import DataPipeline
                pipeline = DataPipeline()
                full = await pipeline.get_full_analysis(ticker, lookback_days=90)
                state["market_data"] = full.get("market_data", state.get("market_data", {}))
                state["news"] = full.get("news", state.get("news", []))
                state["fundamentals"] = full.get("fundamentals", state.get("fundamentals", {}))
                state["social_data"] = full.get("social", state.get("social_data", {}))
                state["technical_indicators"] = full.get("market_data", {}).get("technical_summary", {})
                state["current_price"] = full.get("market_data", {}).get("latest_close", 0) or state.get("current_price", 0)
        except Exception as e:
            logger.warning(f"Data pipeline failed for {ticker}, continuing with existing state: {e}")

        try:
            state = await self._node_fetch_data(state)
            state = await self._node_process_parallel(state)
            state = await self._node_debate_parallel(state)
            state = await self._node_coordinate(state)
            state = await self._node_risk_check(state)

            risk_decision = state.get("risk_assessment", {}).get("decision", "VETO")
            if risk_decision == "APPROVE":
                state = await self._node_execute(state)

            state = await self._node_log_decision(state)

            # Derive top-level recommendation fields from consensus/final_action
            consensus = state.get("consensus", {})
            final = state.get("final_action", {})
            rec_action = consensus.get("recommended_action", {})

            signal_str = (
                rec_action.get("signal", "")
                or rec_action.get("action", "")
                or consensus.get("consensus_signal", "HOLD")
            ).upper()
            if signal_str not in ("BUY", "SELL", "SHORT", "HOLD"):
                signal_str = "HOLD"
            state["recommendation"] = signal_str

            state["confidence"] = float(
                rec_action.get("confidence", 0)
                or consensus.get("confidence", 0)
                or 0
            )

            state["reasoning"] = str(
                consensus.get("reasoning", "")
                or consensus.get("summary", "")
                or "No reasoning provided."
            )

            portfolio_decision = state.get("portfolio_decision", {})
            state["exposure"] = float(
                portfolio_decision.get("position_size_pct", 0)
                or portfolio_decision.get("exposure", 0)
                or 0
            )

            state["cost_usd"] = state.get("total_llm_cost", 0)
        except Exception as e:
            logger.error(f"Graph execution failed for {ticker}: {e}")
            errors = state.get("errors", [])
            errors.append(f"Graph error: {str(e)}")
            state["errors"] = errors

        total_time = (time.monotonic() - start) * 1000
        state["total_latency_ms"] = total_time
        logger.info(f"Graph completed for {ticker} in {total_time:.0f}ms, cost=${state.get('total_llm_cost', 0):.4f}")
        return state

    async def _node_fetch_data(self, state: TradingState) -> TradingState:
        logger.info(f"Node: fetch_data for {state['ticker']}")
        agent = self._agents["market_data"]
        output = await agent.execute(state["ticker"], {
            "price_data": state.get("market_data", {}),
            "volume_profile": state.get("market_data", {}).get("volume_profile", {}),
        })
        state = update_state_with_output(state, output)

        if output.parsed_output:
            state["current_price"] = output.parsed_output.get("current_price", state.get("current_price", 0))
            state["market_data"].update(output.parsed_output)

        return state

    async def _node_process_parallel(self, state: TradingState) -> TradingState:
        logger.info(f"Node: process_parallel for {state['ticker']}")

        research_agents = ["technical", "fundamental", "sentiment", "macro", "event", "quantitative"]
        tasks = []
        for name in research_agents:
            agent = self._agents[name]
            agent_data = self._prepare_agent_data(name, state)
            tasks.append(agent.execute(state["ticker"], agent_data))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Agent {research_agents[i]} failed: {result}")
                errors = state.get("errors", [])
                errors.append(f"{research_agents[i]}: {str(result)}")
                state["errors"] = errors
            elif isinstance(result, AgentOutput):
                state = update_state_with_output(state, result)
                if result.signal:
                    signals = state.get("research_signals", [])
                    signals.append(result.signal.model_dump())
                    state["research_signals"] = signals

        return state

    async def _node_debate_parallel(self, state: TradingState) -> TradingState:
        logger.info(f"Node: debate_parallel for {state['ticker']}")

        agent_analyses = {}
        for name, output in state.get("agent_outputs", {}).items():
            if isinstance(output, AgentOutput):
                agent_analyses[name] = output.parsed_output

        debate_data = {
            "agent_analyses": agent_analyses,
            "current_price": state.get("current_price", 0),
            "position_context": state.get("portfolio", {}),
        }

        bull_task = self._agents["bull"].execute(state["ticker"], debate_data)
        bear_task = self._agents["bear"].execute(state["ticker"], debate_data)

        bull_result, bear_result = await asyncio.gather(bull_task, bear_task, return_exceptions=True)

        if isinstance(bull_result, AgentOutput):
            state = update_state_with_output(state, bull_result)
            state["bull_case"] = bull_result.parsed_output
        elif isinstance(bull_result, Exception):
            logger.error(f"Bull agent failed: {bull_result}")

        if isinstance(bear_result, AgentOutput):
            state = update_state_with_output(state, bear_result)
            state["bear_case"] = bear_result.parsed_output
        elif isinstance(bear_result, Exception):
            logger.error(f"Bear agent failed: {bear_result}")

        return state

    async def _node_coordinate(self, state: TradingState) -> TradingState:
        logger.info(f"Node: coordinate for {state['ticker']}")

        agent_data = {}
        for name, output in state.get("agent_outputs", {}).items():
            if isinstance(output, AgentOutput):
                try:
                    sig = (
                        output.signal.signal_type.value
                        if output.signal and getattr(output.signal, "signal_type", None)
                        else "HOLD"
                    )
                except Exception:
                    sig = "HOLD"
                agent_data[name] = {
                    "signal": sig,
                    "confidence": output.confidence or 0.0,
                    "parsed_output": output.parsed_output or {},
                }

        coordinator = self._agents["coordinator"]
        output = await coordinator.execute(state["ticker"], {
            "agent_outputs": agent_data,
            "current_price": state.get("current_price", 0),
            "consensus_threshold": get_config().agents.consensus_threshold,
        })

        if isinstance(output, AgentOutput):
            state = update_state_with_output(state, output)
            state["consensus"] = output.parsed_output

        return state

    async def _node_risk_check(self, state: TradingState) -> TradingState:
        logger.info(f"Node: risk_check for {state['ticker']}")

        consensus = state.get("consensus", {})
        recommended = consensus.get("recommended_action", {})

        risk_agent = self._agents["risk"]
        output = await risk_agent.execute(state["ticker"], {
            "proposed_action": recommended,
            "portfolio_state": state.get("portfolio", {}),
            "risk_limits": state.get("risk_limits", {}),
            "bull_bear_debate": {
                "bull": state.get("bull_case", {}),
                "bear": state.get("bear_case", {}),
            },
        })

        if isinstance(output, AgentOutput):
            state = update_state_with_output(state, output)
            state["risk_assessment"] = output.parsed_output

        return state

    async def _node_execute(self, state: TradingState) -> TradingState:
        logger.info(f"Node: execute for {state['ticker']}")

        portfolio_agent = self._agents["portfolio"]
        p_output = await portfolio_agent.execute(state["ticker"], {
            "all_analyses": {
                name: out.parsed_output
                for name, out in state.get("agent_outputs", {}).items()
                if isinstance(out, AgentOutput)
            },
            "bull_case": state.get("bull_case", {}),
            "bear_case": state.get("bear_case", {}),
            "risk_assessment": state.get("risk_assessment", {}),
            "portfolio_state": state.get("portfolio", {}),
        })

        if isinstance(p_output, AgentOutput):
            state = update_state_with_output(state, p_output)
            state["portfolio_decision"] = p_output.parsed_output

        exec_agent = self._agents["execution"]
        e_output = await exec_agent.execute(state["ticker"], {
            "trade_order": state.get("portfolio_decision", {}),
            "market_conditions": state.get("market_data", {}),
            "liquidity_data": state.get("market_data", {}).get("volume_profile", {}),
        })

        if isinstance(e_output, AgentOutput):
            state = update_state_with_output(state, e_output)
            state["execution_plan"] = e_output.parsed_output

        return state

    async def _node_log_decision(self, state: TradingState) -> TradingState:
        summary = get_state_summary(state)
        logger.info(f"Decision summary for {state['ticker']}: {json.dumps(summary, default=str)}")
        state["final_action"] = {
            "ticker": state["ticker"],
            "consensus": state.get("consensus", {}),
            "risk_decision": state.get("risk_assessment", {}).get("decision", "VETO"),
            "portfolio_decision": state.get("portfolio_decision", {}),
            "execution_plan": state.get("execution_plan", {}),
            "total_cost_usd": state.get("total_llm_cost", 0),
            "total_latency_ms": state.get("total_latency_ms", 0),
        }
        return state

    def _prepare_agent_data(self, agent_name: str, state: TradingState) -> dict[str, Any]:
        data_map: dict[str, dict[str, Any]] = {
            "technical": {
                "price_data": state.get("market_data", {}),
                "technical_indicators": state.get("technical_indicators", {}),
            },
            "fundamental": {
                "financials": state.get("fundamentals", {}),
                "key_metrics": state.get("fundamentals", {}).get("metrics", {}),
                "industry_comparison": state.get("fundamentals", {}).get("industry", {}),
                "current_price": state.get("current_price", 0),
            },
            "sentiment": {
                "news": state.get("news", []),
                "social_data": state.get("social_data", {}),
                "analyst_ratings": state.get("fundamentals", {}).get("analyst_ratings", {}),
            },
            "macro": {
                "economic_indicators": state.get("economic_indicators", {}),
                "fed_policy": state.get("economic_indicators", {}).get("fed_policy", {}),
                "global_events": state.get("economic_indicators", {}).get("global_events", []),
            },
            "event": {
                "calendar_events": state.get("news", []),
                "event_history": state.get("fundamentals", {}).get("events", {}),
            },
            "quantitative": {
                "factor_data": state.get("technical_indicators", {}).get("factors", {}),
                "statistics": state.get("technical_indicators", {}).get("statistics", {}),
                "correlations": state.get("technical_indicators", {}).get("correlations", {}),
            },
        }
        return data_map.get(agent_name, {})
