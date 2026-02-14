from __future__ import annotations

import json
import math
from typing import Any

from nexus.agents.base import AgentOutput, BaseAgent
from nexus.core.types import OrderSide, OrderType, SignalType
from nexus.llm.prompts import get_system_prompt, render_prompt


class PortfolioAgent(BaseAgent):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(name="portfolio", agent_type="execution", **kwargs)

    async def analyze(self, ticker: str, data: dict[str, Any]) -> AgentOutput:
        analyses = json.dumps(data.get("all_analyses", {}), default=str)
        bull_case = json.dumps(data.get("bull_case", {}), default=str)
        bear_case = json.dumps(data.get("bear_case", {}), default=str)
        risk = json.dumps(data.get("risk_assessment", {}), default=str)
        portfolio = json.dumps(data.get("portfolio_state", {}), default=str)
        kelly = json.dumps(data.get("kelly_data", self._compute_kelly(data)), default=str)

        prompt = render_prompt(
            "portfolio",
            ticker=ticker,
            analyses=analyses,
            bull_case=bull_case,
            bear_case=bear_case,
            risk=risk,
            portfolio=portfolio,
            kelly=kelly,
        )
        system = get_system_prompt("portfolio")
        response = await self.call_llm(prompt, system)
        parsed = self.parse_json_response(response.content)

        action = parsed.get("action", "hold")
        action_signal_map = {
            "buy": SignalType.BUY,
            "sell": SignalType.SELL,
            "hold": SignalType.HOLD,
            "close": SignalType.SELL,
        }
        signal_type = action_signal_map.get(action, SignalType.HOLD)

        order_type_map = {
            "market": OrderType.MARKET,
            "limit": OrderType.LIMIT,
            "stop_limit": OrderType.STOP_LIMIT,
        }

        confidence = float(parsed.get("kelly_fraction", 0.5))

        signal = self.build_signal(
            ticker=ticker,
            signal_type=signal_type,
            confidence=min(confidence, 1.0),
            reasoning=parsed.get("reasoning", ""),
            target_price=parsed.get("take_profit"),
            stop_loss=parsed.get("stop_loss"),
            metadata={
                "action": action,
                "size_pct": parsed.get("size_pct", 0),
                "limit_price": parsed.get("limit_price"),
                "order_type": parsed.get("order_type", "market"),
                "urgency": parsed.get("urgency", "patient"),
                "kelly_fraction": parsed.get("kelly_fraction", 0),
                "expected_return": parsed.get("expected_return", 0),
            },
        )

        return AgentOutput(
            agent_name=self.name,
            agent_type=self.agent_type,
            ticker=ticker,
            raw_response=response.content,
            parsed_output=parsed,
            signal=signal,
            confidence=min(confidence, 1.0),
            llm_cost_usd=response.cost_usd,
            model_used=response.model,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
        )

    @staticmethod
    def _compute_kelly(data: dict[str, Any]) -> dict[str, Any]:
        win_rate = data.get("historical_win_rate", 0.55)
        avg_win = data.get("avg_win_pct", 2.0)
        avg_loss = data.get("avg_loss_pct", 1.5)

        if avg_loss == 0:
            return {"kelly_fraction": 0, "half_kelly": 0}

        win_loss_ratio = avg_win / avg_loss
        kelly = win_rate - (1 - win_rate) / win_loss_ratio
        half_kelly = kelly / 2

        return {
            "win_rate": win_rate,
            "win_loss_ratio": round(win_loss_ratio, 4),
            "kelly_fraction": round(max(kelly, 0), 4),
            "half_kelly": round(max(half_kelly, 0), 4),
            "recommended_size_pct": round(max(half_kelly * 100, 0), 2),
        }
