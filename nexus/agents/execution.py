from __future__ import annotations

import json
from typing import Any

from nexus.agents.base import AgentOutput, BaseAgent
from nexus.core.types import SignalType
from nexus.llm.prompts import get_system_prompt, render_prompt


class ExecutionAgent(BaseAgent):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(name="execution", agent_type="execution", **kwargs)

    async def analyze(self, ticker: str, data: dict[str, Any]) -> AgentOutput:
        order = json.dumps(data.get("trade_order", {}), default=str)
        market = json.dumps(data.get("market_conditions", {}), default=str)
        liquidity = json.dumps(data.get("liquidity_data", {}), default=str)

        prompt = render_prompt(
            "execution",
            ticker=ticker,
            order=order,
            market=market,
            liquidity=liquidity,
        )
        system = get_system_prompt("execution")
        response = await self.call_llm(prompt, system)
        parsed = self.parse_json_response(response.content)

        strategy = parsed.get("strategy", "MARKET")
        slippage_bps = float(parsed.get("expected_slippage_bps", 5))

        signal = self.build_signal(
            ticker=ticker,
            signal_type=SignalType.HOLD,
            confidence=0.9,
            reasoning=parsed.get("reasoning", ""),
            metadata={
                "execution_strategy": strategy,
                "num_slices": parsed.get("num_slices", 1),
                "time_horizon_minutes": parsed.get("time_horizon_minutes", 5),
                "max_participation_rate": parsed.get("max_participation_rate", 0.1),
                "expected_slippage_bps": slippage_bps,
                "urgency": parsed.get("urgency", "medium"),
            },
        )

        return AgentOutput(
            agent_name=self.name,
            agent_type=self.agent_type,
            ticker=ticker,
            raw_response=response.content,
            parsed_output=parsed,
            signal=signal,
            confidence=0.9,
            llm_cost_usd=response.cost_usd,
            model_used=response.model,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
        )
