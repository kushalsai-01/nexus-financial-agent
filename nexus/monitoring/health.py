from __future__ import annotations

import asyncio
import time
from datetime import datetime
from enum import Enum
from typing import Any

import httpx

from nexus.core.logging import get_logger

logger = get_logger("monitoring.health")


class HealthStatus(str, Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


class ComponentHealth:
    def __init__(self, name: str, status: HealthStatus, latency_ms: float = 0.0, message: str = "") -> None:
        self.name = name
        self.status = status
        self.latency_ms = latency_ms
        self.message = message
        self.checked_at = datetime.now()

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status.value,
            "latency_ms": round(self.latency_ms, 2),
            "message": self.message,
            "checked_at": self.checked_at.isoformat(),
        }


class HealthChecker:
    def __init__(self) -> None:
        self._checks: dict[str, Any] = {}
        self._last_results: dict[str, ComponentHealth] = {}

    def register_check(self, name: str, check_fn: Any) -> None:
        self._checks[name] = check_fn

    async def check_database(self) -> ComponentHealth:
        start = time.monotonic()
        try:
            import sqlalchemy
            from nexus.core.config import get_config
            config = get_config()
            engine = sqlalchemy.create_engine(config.storage.postgres.url, pool_pre_ping=True)
            with engine.connect() as conn:
                conn.execute(sqlalchemy.text("SELECT 1"))
            latency = (time.monotonic() - start) * 1000
            return ComponentHealth("database", HealthStatus.HEALTHY, latency)
        except Exception as e:
            latency = (time.monotonic() - start) * 1000
            return ComponentHealth("database", HealthStatus.UNHEALTHY, latency, str(e))

    async def check_llm_provider(self, provider: str = "anthropic") -> ComponentHealth:
        start = time.monotonic()
        try:
            if provider == "anthropic":
                url = "https://api.anthropic.com/v1/messages"
            elif provider == "openai":
                url = "https://api.openai.com/v1/models"
            elif provider == "grok":
                url = "https://api.x.ai/v1/models"
            else:
                url = f"https://api.{provider}.com"

            async with httpx.AsyncClient() as client:
                resp = await client.head(url, timeout=5)
                latency = (time.monotonic() - start) * 1000
                if resp.status_code in (200, 401, 405):
                    return ComponentHealth(f"llm_{provider}", HealthStatus.HEALTHY, latency)
                return ComponentHealth(f"llm_{provider}", HealthStatus.DEGRADED, latency, f"HTTP {resp.status_code}")
        except Exception as e:
            latency = (time.monotonic() - start) * 1000
            return ComponentHealth(f"llm_{provider}", HealthStatus.UNHEALTHY, latency, str(e))

    async def check_market_data(self) -> ComponentHealth:
        start = time.monotonic()
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get("https://query1.finance.yahoo.com/v8/finance/chart/SPY?range=1d", timeout=5)
                latency = (time.monotonic() - start) * 1000
                if resp.status_code == 200:
                    return ComponentHealth("market_data", HealthStatus.HEALTHY, latency)
                return ComponentHealth("market_data", HealthStatus.DEGRADED, latency, f"HTTP {resp.status_code}")
        except Exception as e:
            latency = (time.monotonic() - start) * 1000
            return ComponentHealth("market_data", HealthStatus.UNHEALTHY, latency, str(e))

    async def check_broker(self) -> ComponentHealth:
        start = time.monotonic()
        try:
            from nexus.core.config import get_config
            config = get_config()
            if config.execution.broker == "alpaca":
                base_url = "https://paper-api.alpaca.markets"
                async with httpx.AsyncClient() as client:
                    resp = await client.get(
                        f"{base_url}/v2/account",
                        headers={
                            "APCA-API-KEY-ID": config.alpaca_api_key,
                            "APCA-API-SECRET-KEY": config.alpaca_api_secret,
                        },
                        timeout=5,
                    )
                    latency = (time.monotonic() - start) * 1000
                    if resp.status_code == 200:
                        return ComponentHealth("broker", HealthStatus.HEALTHY, latency)
                    if resp.status_code == 401:
                        return ComponentHealth("broker", HealthStatus.DEGRADED, latency, "Invalid API keys")
                    return ComponentHealth("broker", HealthStatus.DEGRADED, latency, f"HTTP {resp.status_code}")
            latency = (time.monotonic() - start) * 1000
            return ComponentHealth("broker", HealthStatus.HEALTHY, latency, "No broker configured")
        except Exception as e:
            latency = (time.monotonic() - start) * 1000
            return ComponentHealth("broker", HealthStatus.UNHEALTHY, latency, str(e))

    async def check_api_keys(self) -> ComponentHealth:
        start = time.monotonic()
        try:
            import os
            grok_key = os.getenv("GROK_API_KEY", "")
            anthropic_key = os.getenv("ANTHROPIC_API_KEY", "")
            openai_key = os.getenv("OPENAI_API_KEY", "")

            latency = (time.monotonic() - start) * 1000
            # Healthy if at least one LLM key is set (Grok-only mode is fine)
            if grok_key or (anthropic_key and openai_key):
                return ComponentHealth("api_keys", HealthStatus.HEALTHY, latency)
            if anthropic_key or openai_key:
                return ComponentHealth("api_keys", HealthStatus.DEGRADED, latency, "Only one legacy LLM key set; consider setting GROK_API_KEY")
            return ComponentHealth("api_keys", HealthStatus.UNHEALTHY, latency, "Missing LLM API key: set GROK_API_KEY (or ANTHROPIC_API_KEY + OPENAI_API_KEY)")
        except Exception as e:
            latency = (time.monotonic() - start) * 1000
            return ComponentHealth("api_keys", HealthStatus.UNHEALTHY, latency, str(e))

    async def run_all_checks(self) -> dict[str, Any]:
        checks = [
            self.check_api_keys(),
            self.check_market_data(),
            self.check_llm_provider("grok"),
        ]

        results = await asyncio.gather(*checks, return_exceptions=True)
        components: list[dict[str, Any]] = []
        overall = HealthStatus.HEALTHY

        for result in results:
            if isinstance(result, Exception):
                comp = ComponentHealth("unknown", HealthStatus.UNHEALTHY, message=str(result))
            else:
                comp = result

            self._last_results[comp.name] = comp
            components.append(comp.to_dict())

            if comp.status == HealthStatus.UNHEALTHY:
                overall = HealthStatus.UNHEALTHY
            elif comp.status == HealthStatus.DEGRADED and overall != HealthStatus.UNHEALTHY:
                overall = HealthStatus.DEGRADED

        for name, check_fn in self._checks.items():
            try:
                result = await check_fn() if asyncio.iscoroutinefunction(check_fn) else check_fn()
                if isinstance(result, ComponentHealth):
                    self._last_results[name] = result
                    components.append(result.to_dict())
                    if result.status == HealthStatus.UNHEALTHY:
                        overall = HealthStatus.UNHEALTHY
                    elif result.status == HealthStatus.DEGRADED and overall != HealthStatus.UNHEALTHY:
                        overall = HealthStatus.DEGRADED
            except Exception as e:
                comp = ComponentHealth(name, HealthStatus.UNHEALTHY, message=str(e))
                components.append(comp.to_dict())
                overall = HealthStatus.UNHEALTHY

        return {
            "status": overall.value,
            "timestamp": datetime.now().isoformat(),
            "components": components,
        }

    def get_last_results(self) -> dict[str, dict[str, Any]]:
        return {name: comp.to_dict() for name, comp in self._last_results.items()}

    @property
    def is_healthy(self) -> bool:
        if not self._last_results:
            return True
        return all(c.status != HealthStatus.UNHEALTHY for c in self._last_results.values())
