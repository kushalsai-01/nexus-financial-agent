from __future__ import annotations

import json
from typing import Any

from nexus.agents.base import AgentOutput, BaseAgent
from nexus.core.types import SignalType, TimeFrame
from nexus.llm.prompts import get_system_prompt, render_prompt


class TechnicalAgent(BaseAgent):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(name="technical", agent_type="research", **kwargs)

    async def analyze(self, ticker: str, data: dict[str, Any]) -> AgentOutput:
        price_data = json.dumps(data.get("price_data", {}), default=str)
        indicators = json.dumps(data.get("technical_indicators", {}), default=str)

        prompt = render_prompt(
            "technical",
            ticker=ticker,
            price_data=price_data,
            indicators=indicators,
        )
        system = get_system_prompt("technical")
        response = await self.call_llm(prompt, system)
        parsed = self.parse_json_response(response.content)

        signal_map = {
            "BUY": SignalType.BUY,
            "SELL": SignalType.SELL,
            "HOLD": SignalType.HOLD,
            "STRONG_BUY": SignalType.STRONG_BUY,
            "STRONG_SELL": SignalType.STRONG_SELL,
        }
        raw_signal = parsed.get("signal", "HOLD")
        signal_type = signal_map.get(raw_signal, SignalType.HOLD)
        strength = float(parsed.get("strength", 0.5))

        signal = self.build_signal(
            ticker=ticker,
            signal_type=signal_type,
            confidence=strength,
            reasoning=parsed.get("reasoning", ""),
            target_price=parsed.get("resistance_level"),
            stop_loss=parsed.get("support_level"),
        )

        return AgentOutput(
            agent_name=self.name,
            agent_type=self.agent_type,
            ticker=ticker,
            raw_response=response.content,
            parsed_output=parsed,
            signal=signal,
            confidence=strength,
            llm_cost_usd=response.cost_usd,
            model_used=response.model,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
        )
