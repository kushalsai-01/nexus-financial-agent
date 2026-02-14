from __future__ import annotations

import json
from typing import Any

from nexus.agents.base import AgentOutput, BaseAgent
from nexus.core.types import SignalType
from nexus.llm.prompts import get_system_prompt, render_prompt


class MacroAgent(BaseAgent):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(name="macro", agent_type="research", **kwargs)

    async def analyze(self, ticker: str, data: dict[str, Any]) -> AgentOutput:
        indicators = json.dumps(data.get("economic_indicators", {}), default=str)
        fed_policy = json.dumps(data.get("fed_policy", {}), default=str)
        global_events = json.dumps(data.get("global_events", []), default=str)

        prompt = render_prompt(
            "macro",
            ticker=ticker,
            indicators=indicators,
            fed_policy=fed_policy,
            global_events=global_events,
        )
        system = get_system_prompt("macro")
        response = await self.call_llm(prompt, system)
        parsed = self.parse_json_response(response.content)

        outlook = parsed.get("macro_outlook", "neutral")
        outlook_map = {
            "favorable": SignalType.BUY,
            "neutral": SignalType.HOLD,
            "unfavorable": SignalType.SELL,
        }
        signal_type = outlook_map.get(outlook, SignalType.HOLD)

        rotation = parsed.get("sector_rotation_signal", "neutral")
        rotation_boost = {"overweight": 0.2, "neutral": 0.0, "underweight": -0.2}
        base_confidence = 0.5
        confidence = max(0.0, min(1.0, base_confidence + rotation_boost.get(rotation, 0.0)))

        signal = self.build_signal(
            ticker=ticker,
            signal_type=signal_type,
            confidence=confidence,
            reasoning=parsed.get("reasoning", ""),
            metadata={
                "macro_outlook": outlook,
                "rate_sensitivity": parsed.get("rate_sensitivity", 0),
                "tail_risks": parsed.get("tail_risks", []),
            },
        )

        return AgentOutput(
            agent_name=self.name,
            agent_type=self.agent_type,
            ticker=ticker,
            raw_response=response.content,
            parsed_output=parsed,
            signal=signal,
            confidence=confidence,
            llm_cost_usd=response.cost_usd,
            model_used=response.model,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
        )
