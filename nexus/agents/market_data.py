from __future__ import annotations

import json
from typing import Any

from nexus.agents.base import AgentOutput, BaseAgent
from nexus.core.types import SignalType, TimeFrame
from nexus.llm.prompts import get_system_prompt, render_prompt


class MarketDataAgent(BaseAgent):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(name="market_data", agent_type="research", **kwargs)

    async def analyze(self, ticker: str, data: dict[str, Any]) -> AgentOutput:
        price_data = data.get("price_data", {})
        volume = data.get("volume_profile", {})

        summary = {
            "ticker": ticker,
            "current_price": price_data.get("close", 0),
            "change_1d": price_data.get("change_1d", 0),
            "change_5d": price_data.get("change_5d", 0),
            "change_20d": price_data.get("change_20d", 0),
            "avg_volume_20d": volume.get("avg_20d", 0),
            "relative_volume": volume.get("relative", 1.0),
            "high_52w": price_data.get("high_52w", 0),
            "low_52w": price_data.get("low_52w", 0),
            "data_points": price_data.get("data_points", 0),
        }

        return AgentOutput(
            agent_name=self.name,
            agent_type=self.agent_type,
            ticker=ticker,
            parsed_output=summary,
            confidence=1.0,
            metadata={"source": "market_data_aggregator"},
        )
