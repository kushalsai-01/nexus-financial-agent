from __future__ import annotations

import json
from typing import Any

from nexus.agents.base import AgentOutput, BaseAgent
from nexus.core.types import SignalType
from nexus.llm.prompts import get_system_prompt, render_prompt


class RLAgent(BaseAgent):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(name="rl_agent", agent_type="strategy", **kwargs)

    async def analyze(self, ticker: str, data: dict[str, Any]) -> AgentOutput:
        prediction = json.dumps(data.get("model_prediction", {}), default=str)
        features = json.dumps(data.get("feature_importance", {}), default=str)
        accuracy = json.dumps(data.get("historical_accuracy", {}), default=str)

        prompt = render_prompt(
            "rl_agent",
            ticker=ticker,
            prediction=prediction,
            features=features,
            accuracy=accuracy,
        )
        system = get_system_prompt("rl_agent")
        response = await self.call_llm(prompt, system)
        parsed = self.parse_json_response(response.content)

        signal_map = {
            "BUY": SignalType.BUY,
            "SELL": SignalType.SELL,
            "HOLD": SignalType.HOLD,
        }
        ml_signal = parsed.get("ml_signal", "HOLD")
        signal_type = signal_map.get(ml_signal, SignalType.HOLD)
        confidence = float(parsed.get("confidence", 0.5))
        uncertainty = float(parsed.get("model_uncertainty", 0.5))

        adjusted_confidence = confidence * (1 - uncertainty)

        signal = self.build_signal(
            ticker=ticker,
            signal_type=signal_type,
            confidence=adjusted_confidence,
            reasoning=parsed.get("reasoning", ""),
            metadata={
                "predicted_return": parsed.get("predicted_return", 0),
                "model_uncertainty": uncertainty,
                "feature_drivers": parsed.get("feature_drivers", []),
            },
        )

        return AgentOutput(
            agent_name=self.name,
            agent_type=self.agent_type,
            ticker=ticker,
            raw_response=response.content,
            parsed_output=parsed,
            signal=signal,
            confidence=adjusted_confidence,
            llm_cost_usd=response.cost_usd,
            model_used=response.model,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
        )
