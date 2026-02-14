from __future__ import annotations

import json
from typing import Any

from nexus.agents.base import AgentOutput, BaseAgent
from nexus.core.types import SignalType
from nexus.llm.prompts import get_system_prompt, render_prompt


class FundamentalAgent(BaseAgent):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(name="fundamental", agent_type="research", **kwargs)

    async def analyze(self, ticker: str, data: dict[str, Any]) -> AgentOutput:
        financials = json.dumps(data.get("financials", {}), default=str)
        metrics = json.dumps(data.get("key_metrics", {}), default=str)
        industry = json.dumps(data.get("industry_comparison", {}), default=str)

        prompt = render_prompt(
            "fundamental",
            ticker=ticker,
            financials=financials,
            metrics=metrics,
            industry=industry,
        )
        system = get_system_prompt("fundamental")
        response = await self.call_llm(prompt, system)
        parsed = self.parse_json_response(response.content)

        fair_value = parsed.get("fair_value", 0)
        current_price = data.get("current_price", 0)
        upside = parsed.get("upside_pct", 0)

        if upside > 15:
            signal_type = SignalType.STRONG_BUY
        elif upside > 5:
            signal_type = SignalType.BUY
        elif upside < -15:
            signal_type = SignalType.STRONG_SELL
        elif upside < -5:
            signal_type = SignalType.SELL
        else:
            signal_type = SignalType.HOLD

        quality = float(parsed.get("quality_score", 0.5))

        signal = self.build_signal(
            ticker=ticker,
            signal_type=signal_type,
            confidence=quality,
            reasoning=parsed.get("reasoning", ""),
            target_price=fair_value if fair_value else None,
        )

        return AgentOutput(
            agent_name=self.name,
            agent_type=self.agent_type,
            ticker=ticker,
            raw_response=response.content,
            parsed_output=parsed,
            signal=signal,
            confidence=quality,
            llm_cost_usd=response.cost_usd,
            model_used=response.model,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
        )
