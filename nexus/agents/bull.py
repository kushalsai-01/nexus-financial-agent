from __future__ import annotations

import json
from typing import Any

from nexus.agents.base import AgentOutput, BaseAgent
from nexus.core.types import SignalType
from nexus.llm.providers import create_provider
from nexus.llm.prompts import get_system_prompt, render_prompt


class BullAgent(BaseAgent):
    def __init__(self, **kwargs: Any) -> None:
        if "provider" not in kwargs or kwargs.get("provider") is None:
            kwargs["provider"] = create_provider("anthropic", "claude-sonnet-4-20250514")
        super().__init__(name="bull", agent_type="debate", **kwargs)

    async def analyze(self, ticker: str, data: dict[str, Any]) -> AgentOutput:
        analyses = json.dumps(data.get("agent_analyses", {}), default=str)
        current_price = data.get("current_price", 0)
        context = json.dumps(data.get("position_context", {}), default=str)

        prompt = render_prompt(
            "bull",
            ticker=ticker,
            analyses=analyses,
            current_price=str(current_price),
            context=context,
        )
        system = get_system_prompt("bull")
        response = await self.call_llm(prompt, system, temperature=0.7)
        parsed = self.parse_json_response(response.content)

        conviction = int(parsed.get("conviction", 5))
        position_action = parsed.get("position_action", "hold")

        action_signal_map = {
            "buy": SignalType.STRONG_BUY,
            "add": SignalType.BUY,
            "hold": SignalType.HOLD,
        }
        signal_type = action_signal_map.get(position_action, SignalType.HOLD)
        confidence = conviction / 10.0

        signal = self.build_signal(
            ticker=ticker,
            signal_type=signal_type,
            confidence=confidence,
            reasoning=parsed.get("reasoning", ""),
            target_price=parsed.get("price_target"),
            metadata={
                "conviction": conviction,
                "position_size_pct": parsed.get("position_size_pct", 0),
                "catalyst_timeline_days": parsed.get("catalyst_timeline_days", 0),
                "upside_scenarios": parsed.get("upside_scenarios", []),
                "key_arguments": parsed.get("key_arguments", []),
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
