from __future__ import annotations

import json
from typing import Any

from nexus.agents.base import AgentOutput, BaseAgent
from nexus.core.types import SignalType
from nexus.llm.prompts import get_system_prompt, render_prompt


class CoordinatorAgent(BaseAgent):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(name="coordinator", agent_type="meta", **kwargs)

    async def analyze(self, ticker: str, data: dict[str, Any]) -> AgentOutput:
        agent_outputs = json.dumps(data.get("agent_outputs", {}), default=str)
        current_price = data.get("current_price", 0)
        threshold = data.get("consensus_threshold", 0.6)

        prompt = render_prompt(
            "coordinator",
            ticker=ticker,
            agent_outputs=agent_outputs,
            current_price=str(current_price),
            threshold=str(threshold),
        )
        system = get_system_prompt("coordinator")
        response = await self.call_llm(prompt, system)
        parsed = self.parse_json_response(response.content)

        consensus = parsed.get("consensus_reached", False)
        consensus_signal = parsed.get("consensus_signal", "HOLD")
        signal_map = {
            "BUY": SignalType.BUY,
            "SELL": SignalType.SELL,
            "HOLD": SignalType.HOLD,
        }
        signal_type = signal_map.get(consensus_signal, SignalType.HOLD)
        confidence = float(parsed.get("consensus_confidence", 0.0))

        if not consensus:
            signal_type = SignalType.HOLD
            confidence *= 0.5

        signal = self.build_signal(
            ticker=ticker,
            signal_type=signal_type,
            confidence=confidence,
            reasoning=parsed.get("reasoning", ""),
            metadata={
                "consensus_reached": consensus,
                "agent_agreement_pct": parsed.get("agent_agreement_pct", 0),
                "dissenting_agents": parsed.get("dissenting_agents", []),
                "recommended_action": parsed.get("recommended_action", {}),
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
