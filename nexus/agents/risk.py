from __future__ import annotations

import json
from typing import Any

from nexus.agents.base import AgentOutput, BaseAgent
from nexus.core.types import SignalType
from nexus.llm.prompts import get_system_prompt, render_prompt


class RiskAgent(BaseAgent):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(name="risk", agent_type="risk", **kwargs)

    async def analyze(self, ticker: str, data: dict[str, Any]) -> AgentOutput:
        action = json.dumps(data.get("proposed_action", {}), default=str)
        portfolio = json.dumps(data.get("portfolio_state", {}), default=str)
        limits = json.dumps(data.get("risk_limits", {}), default=str)
        debate = json.dumps(data.get("bull_bear_debate", {}), default=str)

        prompt = render_prompt(
            "risk",
            ticker=ticker,
            action=action,
            portfolio=portfolio,
            limits=limits,
            debate=debate,
        )
        system = get_system_prompt("risk")
        response = await self.call_llm(prompt, system)
        parsed = self.parse_json_response(response.content)

        decision = parsed.get("decision", "VETO")
        risk_score = float(parsed.get("risk_score", 1.0))

        approved = decision == "APPROVE"
        signal_type = SignalType.HOLD if not approved else SignalType.BUY

        signal = self.build_signal(
            ticker=ticker,
            signal_type=signal_type,
            confidence=1.0 - risk_score,
            reasoning=parsed.get("reasoning", ""),
            metadata={
                "decision": decision,
                "adjusted_size_pct": parsed.get("adjusted_size_pct", 0),
                "adjusted_stop_loss": parsed.get("adjusted_stop_loss", 0),
                "portfolio_impact": parsed.get("portfolio_impact", {}),
                "veto_reasons": parsed.get("veto_reasons", []),
                "conditions": parsed.get("conditions", []),
            },
        )

        return AgentOutput(
            agent_name=self.name,
            agent_type=self.agent_type,
            ticker=ticker,
            raw_response=response.content,
            parsed_output=parsed,
            signal=signal,
            confidence=1.0 - risk_score,
            llm_cost_usd=response.cost_usd,
            model_used=response.model,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
        )
