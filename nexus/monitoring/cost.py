from __future__ import annotations

import threading
from collections import defaultdict
from datetime import date, datetime, timedelta
from typing import Any

from nexus.core.logging import get_logger

logger = get_logger("monitoring.cost")


class LLMUsageRecord:
    __slots__ = ("agent_name", "model", "input_tokens", "output_tokens", "cost_usd", "timestamp", "ticker", "cached")

    def __init__(
        self,
        agent_name: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cost_usd: float,
        ticker: str = "",
        cached: bool = False,
    ) -> None:
        self.agent_name = agent_name
        self.model = model
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.cost_usd = cost_usd
        self.timestamp = datetime.now()
        self.ticker = ticker
        self.cached = cached


class CostTracker:
    _instance: CostTracker | None = None

    def __init__(self, daily_budget_usd: float = 100.0) -> None:
        self.daily_budget_usd = daily_budget_usd
        self._records: list[LLMUsageRecord] = []
        self._lock = threading.Lock()
        self._daily_cost: dict[str, float] = defaultdict(float)
        self._agent_cost: dict[str, float] = defaultdict(float)
        self._model_cost: dict[str, float] = defaultdict(float)
        self._agent_tokens: dict[str, int] = defaultdict(int)

    @classmethod
    def get_instance(cls) -> CostTracker:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def record(
        self,
        agent_name: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cost_usd: float,
        ticker: str = "",
        cached: bool = False,
    ) -> None:
        record = LLMUsageRecord(
            agent_name=agent_name,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost_usd,
            ticker=ticker,
            cached=cached,
        )

        with self._lock:
            self._records.append(record)
            day_key = record.timestamp.strftime("%Y-%m-%d")
            self._daily_cost[day_key] += cost_usd
            self._agent_cost[agent_name] += cost_usd
            self._model_cost[model] += cost_usd
            self._agent_tokens[agent_name] += input_tokens + output_tokens

        if self._daily_cost[day_key] > self.daily_budget_usd:
            logger.warning(
                f"Daily LLM cost ${self._daily_cost[day_key]:.2f} exceeds budget ${self.daily_budget_usd:.2f}"
            )

    @property
    def total_cost(self) -> float:
        return sum(self._daily_cost.values())

    @property
    def today_cost(self) -> float:
        return self._daily_cost.get(date.today().isoformat(), 0.0)

    @property
    def budget_remaining(self) -> float:
        return max(0.0, self.daily_budget_usd - self.today_cost)

    @property
    def budget_utilization_pct(self) -> float:
        if self.daily_budget_usd <= 0:
            return 0.0
        return (self.today_cost / self.daily_budget_usd) * 100

    def cost_by_agent(self) -> dict[str, float]:
        with self._lock:
            return dict(self._agent_cost)

    def cost_by_model(self) -> dict[str, float]:
        with self._lock:
            return dict(self._model_cost)

    def cost_by_day(self, last_n_days: int = 30) -> dict[str, float]:
        cutoff = (datetime.now() - timedelta(days=last_n_days)).strftime("%Y-%m-%d")
        with self._lock:
            return {k: v for k, v in sorted(self._daily_cost.items()) if k >= cutoff}

    def tokens_by_agent(self) -> dict[str, int]:
        with self._lock:
            return dict(self._agent_tokens)

    def cost_per_decision(self, agent_name: str | None = None) -> float:
        with self._lock:
            if agent_name:
                agent_records = [r for r in self._records if r.agent_name == agent_name]
            else:
                agent_records = self._records
            if not agent_records:
                return 0.0
            return sum(r.cost_usd for r in agent_records) / len(agent_records)

    def cost_per_ticker(self) -> dict[str, float]:
        result: dict[str, float] = defaultdict(float)
        with self._lock:
            for record in self._records:
                if record.ticker:
                    result[record.ticker] += record.cost_usd
        return dict(result)

    def get_summary(self) -> dict[str, Any]:
        with self._lock:
            total_records = len(self._records)
            total_tokens = sum(r.input_tokens + r.output_tokens for r in self._records)
            cached_count = sum(1 for r in self._records if r.cached)

        return {
            "total_cost_usd": round(self.total_cost, 4),
            "today_cost_usd": round(self.today_cost, 4),
            "daily_budget_usd": self.daily_budget_usd,
            "budget_remaining_usd": round(self.budget_remaining, 4),
            "budget_utilization_pct": round(self.budget_utilization_pct, 2),
            "total_records": total_records,
            "total_tokens": total_tokens,
            "cached_requests": cached_count,
            "cost_by_agent": {k: round(v, 4) for k, v in self.cost_by_agent().items()},
            "cost_by_model": {k: round(v, 4) for k, v in self.cost_by_model().items()},
            "avg_cost_per_decision": round(self.cost_per_decision(), 6),
        }

    def get_agent_breakdown(self, agent_name: str) -> dict[str, Any]:
        with self._lock:
            records = [r for r in self._records if r.agent_name == agent_name]

        if not records:
            return {"agent": agent_name, "total_cost": 0, "total_calls": 0}

        return {
            "agent": agent_name,
            "total_cost_usd": round(sum(r.cost_usd for r in records), 4),
            "total_calls": len(records),
            "total_tokens": sum(r.input_tokens + r.output_tokens for r in records),
            "avg_cost_per_call": round(sum(r.cost_usd for r in records) / len(records), 6),
            "avg_input_tokens": sum(r.input_tokens for r in records) // len(records),
            "avg_output_tokens": sum(r.output_tokens for r in records) // len(records),
            "models_used": list({r.model for r in records}),
            "cached_pct": round(sum(1 for r in records if r.cached) / len(records) * 100, 1),
        }

    def cleanup_old_records(self, days: int = 90) -> int:
        cutoff = datetime.now() - timedelta(days=days)
        with self._lock:
            before = len(self._records)
            self._records = [r for r in self._records if r.timestamp >= cutoff]
            removed = before - len(self._records)
        if removed > 0:
            logger.info(f"Cleaned up {removed} cost records older than {days} days")
        return removed
