from __future__ import annotations

import json
from typing import Any

from nexus.agents.base import AgentOutput, BaseAgent
from nexus.core.types import SignalType
from nexus.llm.providers import create_provider
from nexus.llm.prompts import get_system_prompt, render_prompt


class BearAgent(BaseAgent):
    def __init__(self, **kwargs: Any) -> None:
        if "provider" not in kwargs or kwargs.get("provider") is None:
            kwargs["provider"] = create_provider("anthropic", "claude-sonnet-4-20250514")
        super().__init__(name="bear", agent_type="debate", **kwargs)

    async def analyze(self, ticker: str, data: dict[str, Any]) -> AgentOutput:
        analyses = json.dumps(data.get("agent_analyses", {}), default=str)
        current_price = data.get("current_price", 0)
        context = json.dumps(data.get("position_context", {}), default=str)

        prompt = render_prompt(
            "bear",
            ticker=ticker,
            analyses=analyses,
            current_price=str(current_price),
            context=context,
        )
        system = get_system_prompt("bear")
        response = await self.call_llm(prompt, system, temperature=0.7)
        parsed = self.parse_json_response(response.content)

        conviction = int(parsed.get("conviction", 5))
        action = parsed.get("action", "avoid")

        action_signal_map = {
            "sell": SignalType.STRONG_SELL,
            "reduce": SignalType.SELL,
            "avoid": SignalType.HOLD,
        }
        signal_type = action_signal_map.get(action, SignalType.HOLD)
        confidence = conviction / 10.0

        signal = self.build_signal(
            ticker=ticker,
            signal_type=signal_type,
            confidence=confidence,
            reasoning=parsed.get("reasoning", ""),
            stop_loss=parsed.get("stop_loss"),
            target_price=parsed.get("downside_target"),
            metadata={
                "conviction": conviction,
                "action": action,
                "downside_scenarios": parsed.get("downside_scenarios", []),
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
