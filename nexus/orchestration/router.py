from __future__ import annotations

from typing import Any

from nexus.core.config import get_config
from nexus.core.logging import get_logger
from nexus.llm.providers import BaseLLMProvider, create_provider

logger = get_logger("orchestration.router")


class LLMRouter:
    TIER_CONFIGS: dict[str, dict[str, str]] = {
        "cheap": {"provider": "grok", "model": "grok-3"},
        "medium": {"provider": "grok", "model": "grok-3"},
        "expensive": {"provider": "grok", "model": "grok-3"},
    }

    AGENT_TIERS: dict[str, str] = {
        "market_data": "cheap",
        "technical": "medium",
        "fundamental": "medium",
        "quantitative": "medium",
        "sentiment": "medium",
        "macro": "medium",
        "event": "cheap",
        "rl_agent": "cheap",
        "bull": "expensive",
        "bear": "expensive",
        "risk": "medium",
        "portfolio": "expensive",
        "execution": "cheap",
        "coordinator": "medium",
    }

    def __init__(self, overrides: dict[str, dict[str, str]] | None = None) -> None:
        self._providers: dict[str, BaseLLMProvider] = {}
        self._overrides = overrides or {}
        self._total_cost = 0.0
        self._calls_by_tier: dict[str, int] = {"cheap": 0, "medium": 0, "expensive": 0}
        self._initialize()

    def _initialize(self) -> None:
        config = get_config()
        for tier, tier_cfg in self.TIER_CONFIGS.items():
            override = self._overrides.get(tier, {})
            provider_name = override.get("provider", tier_cfg["provider"])
            model = override.get("model", tier_cfg["model"])

            # Fail-fast when the required API key is missing
            if provider_name == "grok" and not getattr(config, "grok_api_key", ""):
                logger.error(
                    f"GROK_API_KEY (or XAI_API_KEY) is not set — "
                    f"cannot initialise {tier} tier. "
                    f"Add the key to your .env file and restart."
                )
                continue

            try:
                self._providers[tier] = create_provider(provider_name, model)
                logger.info(f"Router initialized {tier} tier: {provider_name}/{model}")
            except Exception as e:
                logger.error(f"Failed to initialize {tier} tier: {e}")

    def get_provider(self, tier: str) -> BaseLLMProvider:
        if tier not in self._providers:
            logger.warning(f"Tier {tier} not available, falling back to cheap")
            tier = "cheap"
        if tier not in self._providers:
            raise ValueError("No LLM providers available")
        self._calls_by_tier[tier] = self._calls_by_tier.get(tier, 0) + 1
        return self._providers[tier]

    def get_provider_for_agent(self, agent_name: str) -> BaseLLMProvider:
        tier = self.AGENT_TIERS.get(agent_name, "cheap")
        return self.get_provider(tier)

    def get_tier_for_agent(self, agent_name: str) -> str:
        return self.AGENT_TIERS.get(agent_name, "cheap")

    def track_cost(self, cost: float, tier: str) -> None:
        self._total_cost += cost

    @property
    def total_cost(self) -> float:
        return self._total_cost

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "total_cost_usd": self._total_cost,
            "calls_by_tier": dict(self._calls_by_tier),
            "available_tiers": list(self._providers.keys()),
            "agent_tier_mapping": dict(self.AGENT_TIERS),
        }
