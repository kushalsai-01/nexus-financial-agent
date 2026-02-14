from __future__ import annotations

import json
from typing import Any

from nexus.agents.base import AgentOutput, BaseAgent
from nexus.core.types import Sentiment, SignalType
from nexus.llm.prompts import get_system_prompt, render_prompt


class SentimentAgent(BaseAgent):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(name="sentiment", agent_type="research", **kwargs)

    async def analyze(self, ticker: str, data: dict[str, Any]) -> AgentOutput:
        news = json.dumps(data.get("news", []), default=str)
        social = json.dumps(data.get("social_data", {}), default=str)
        analyst = json.dumps(data.get("analyst_ratings", {}), default=str)

        prompt = render_prompt(
            "sentiment",
            ticker=ticker,
            news=news,
            social=social,
            analyst=analyst,
        )
        system = get_system_prompt("sentiment")
        response = await self.call_llm(prompt, system)
        parsed = self.parse_json_response(response.content)

        score = float(parsed.get("score", 0))
        sentiment_str = parsed.get("sentiment", "neutral")

        sentiment_signal_map = {
            "very_bullish": SignalType.STRONG_BUY,
            "bullish": SignalType.BUY,
            "neutral": SignalType.HOLD,
            "bearish": SignalType.SELL,
            "very_bearish": SignalType.STRONG_SELL,
        }
        signal_type = sentiment_signal_map.get(sentiment_str, SignalType.HOLD)
        confidence = min(abs(score), 1.0)

        signal = self.build_signal(
            ticker=ticker,
            signal_type=signal_type,
            confidence=confidence,
            reasoning=parsed.get("reasoning", ""),
            metadata={
                "sentiment": sentiment_str,
                "key_topics": parsed.get("key_topics", []),
                "catalyst_events": parsed.get("catalyst_events", []),
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
