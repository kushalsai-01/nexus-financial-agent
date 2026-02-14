from __future__ import annotations

import json
from typing import Any

from nexus.agents.base import AgentOutput, BaseAgent
from nexus.core.types import SignalType
from nexus.llm.prompts import get_system_prompt, render_prompt


class QuantitativeAgent(BaseAgent):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(name="quantitative", agent_type="strategy", **kwargs)

    async def analyze(self, ticker: str, data: dict[str, Any]) -> AgentOutput:
        factors = json.dumps(data.get("factor_data", {}), default=str)
        statistics = json.dumps(data.get("statistics", {}), default=str)
        correlations = json.dumps(data.get("correlations", {}), default=str)

        prompt = render_prompt(
            "quantitative",
            ticker=ticker,
            factors=factors,
            statistics=statistics,
            correlations=correlations,
        )
        system = get_system_prompt("quantitative")
        response = await self.call_llm(prompt, system)
        parsed = self.parse_json_response(response.content)

        alpha = float(parsed.get("alpha_score", 0))
        if alpha > 1.5:
            signal_type = SignalType.STRONG_BUY
        elif alpha > 0.5:
            signal_type = SignalType.BUY
        elif alpha < -1.5:
            signal_type = SignalType.STRONG_SELL
        elif alpha < -0.5:
            signal_type = SignalType.SELL
        else:
            signal_type = SignalType.HOLD

        confidence = min(abs(alpha) / 3.0, 1.0)

        signal = self.build_signal(
            ticker=ticker,
            signal_type=signal_type,
            confidence=confidence,
            reasoning=parsed.get("reasoning", ""),
            metadata={"factor_breakdown": parsed.get("factor_breakdown", {})},
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
