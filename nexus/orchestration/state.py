from __future__ import annotations

from datetime import datetime
from typing import Any, TypedDict

from nexus.agents.base import AgentOutput
from nexus.core.types import Portfolio, Signal


class TradingState(TypedDict, total=False):
    ticker: str
    timestamp: str
    current_price: float
    portfolio: dict[str, Any]
    risk_limits: dict[str, Any]
    market_data: dict[str, Any]
    technical_indicators: dict[str, Any]
    fundamentals: dict[str, Any]
    news: list[dict[str, Any]]
    social_data: dict[str, Any]
    economic_indicators: dict[str, Any]
    agent_outputs: dict[str, AgentOutput]
    research_signals: list[dict[str, Any]]
    strategy_signals: list[dict[str, Any]]
    bull_case: dict[str, Any]
    bear_case: dict[str, Any]
    risk_assessment: dict[str, Any]
    portfolio_decision: dict[str, Any]
    execution_plan: dict[str, Any]
    consensus: dict[str, Any]
    final_action: dict[str, Any]
    total_llm_cost: float
    total_latency_ms: float
    errors: list[str]
    decision_log: list[dict[str, Any]]


def create_initial_state(
    ticker: str,
    portfolio: Portfolio | None = None,
    risk_limits: dict[str, Any] | None = None,
) -> TradingState:
    return TradingState(
        ticker=ticker,
        timestamp=datetime.now().isoformat(),
        current_price=0.0,
        portfolio=portfolio.model_dump() if portfolio else {},
        risk_limits=risk_limits or _default_risk_limits(),
        market_data={},
        technical_indicators={},
        fundamentals={},
        news=[],
        social_data={},
        economic_indicators={},
        agent_outputs={},
        research_signals=[],
        strategy_signals=[],
        bull_case={},
        bear_case={},
        risk_assessment={},
        portfolio_decision={},
        execution_plan={},
        consensus={},
        final_action={},
        total_llm_cost=0.0,
        total_latency_ms=0.0,
        errors=[],
        decision_log=[],
    )


def _default_risk_limits() -> dict[str, Any]:
    return {
        "max_position_size_pct": 5.0,
        "max_portfolio_risk_pct": 15.0,
        "max_sector_exposure_pct": 30.0,
        "max_drawdown_pct": 10.0,
        "daily_loss_limit_pct": 3.0,
        "max_leverage": 1.0,
        "min_cash_pct": 10.0,
        "position_limit": 20,
        "var_confidence": 0.95,
        "stop_loss_pct": 5.0,
    }


def update_state_with_output(state: TradingState, output: AgentOutput) -> TradingState:
    state["agent_outputs"][output.agent_name] = output
    state["total_llm_cost"] = state.get("total_llm_cost", 0.0) + output.llm_cost_usd
    state["total_latency_ms"] = state.get("total_latency_ms", 0.0) + output.latency_ms

    if output.error:
        errors = state.get("errors", [])
        errors.append(f"{output.agent_name}: {output.error}")
        state["errors"] = errors

    log_entry = {
        "agent": output.agent_name,
        "type": output.agent_type,
        "ticker": output.ticker,
        "timestamp": output.timestamp.isoformat(),
        "confidence": output.confidence,
        "signal": output.signal.signal_type.value if output.signal else None,
        "cost_usd": output.llm_cost_usd,
        "latency_ms": output.latency_ms,
        "model": output.model_used,
        "error": output.error,
    }
    decision_log = state.get("decision_log", [])
    decision_log.append(log_entry)
    state["decision_log"] = decision_log

    return state


def get_state_summary(state: TradingState) -> dict[str, Any]:
    outputs = state.get("agent_outputs", {})
    signals = []
    for name, output in outputs.items():
        if isinstance(output, AgentOutput) and output.signal:
            signals.append({
                "agent": name,
                "signal": output.signal.signal_type.value,
                "confidence": output.confidence,
            })

    return {
        "ticker": state.get("ticker", ""),
        "current_price": state.get("current_price", 0),
        "agents_completed": len(outputs),
        "signals": signals,
        "total_cost": state.get("total_llm_cost", 0),
        "total_latency_ms": state.get("total_latency_ms", 0),
        "errors": state.get("errors", []),
    }
