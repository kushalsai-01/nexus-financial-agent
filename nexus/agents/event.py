from __future__ import annotations

import json
from typing import Any

from nexus.agents.base import AgentOutput, BaseAgent
from nexus.core.types import SignalType
from nexus.llm.prompts import get_system_prompt, render_prompt


class EventAgent(BaseAgent):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(name="event", agent_type="research", **kwargs)

    async def analyze(self, ticker: str, data: dict[str, Any]) -> AgentOutput:
        events = json.dumps(data.get("calendar_events", []), default=str)
        history = json.dumps(data.get("event_history", {}), default=str)

        prompt = render_prompt(
            "event",
            ticker=ticker,
            events=events,
            history=history,
        )
        system = get_system_prompt("event")
        response = await self.call_llm(prompt, system)
        parsed = self.parse_json_response(response.content)

        catalyst_score = float(parsed.get("net_catalyst_score", 0))

        if catalyst_score > 0.5:
            signal_type = SignalType.BUY
        elif catalyst_score > 0.2:
            signal_type = SignalType.BUY
        elif catalyst_score < -0.5:
            signal_type = SignalType.SELL
        elif catalyst_score < -0.2:
            signal_type = SignalType.SELL
        else:
            signal_type = SignalType.HOLD

        confidence = min(abs(catalyst_score), 1.0)

        signal = self.build_signal(
            ticker=ticker,
            signal_type=signal_type,
            confidence=confidence,
            reasoning=parsed.get("reasoning", ""),
            metadata={
                "events": parsed.get("events", []),
                "net_catalyst_score": catalyst_score,
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
