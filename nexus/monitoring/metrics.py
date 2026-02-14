from __future__ import annotations

import threading
import time
from collections import defaultdict, deque
from datetime import datetime, timedelta
from typing import Any

from nexus.core.logging import get_logger

logger = get_logger("monitoring.metrics")


class MetricPoint:
    __slots__ = ("name", "value", "timestamp", "labels")

    def __init__(self, name: str, value: float, labels: dict[str, str] | None = None) -> None:
        self.name = name
        self.value = value
        self.timestamp = datetime.now()
        self.labels = labels or {}


class Counter:
    def __init__(self, name: str, description: str = "") -> None:
        self.name = name
        self.description = description
        self._value: float = 0.0
        self._lock = threading.Lock()

    def inc(self, amount: float = 1.0) -> None:
        with self._lock:
            self._value += amount

    @property
    def value(self) -> float:
        return self._value

    def reset(self) -> None:
        with self._lock:
            self._value = 0.0


class Gauge:
    def __init__(self, name: str, description: str = "") -> None:
        self.name = name
        self.description = description
        self._value: float = 0.0
        self._lock = threading.Lock()

    def set(self, value: float) -> None:
        with self._lock:
            self._value = value

    def inc(self, amount: float = 1.0) -> None:
        with self._lock:
            self._value += amount

    def dec(self, amount: float = 1.0) -> None:
        with self._lock:
            self._value -= amount

    @property
    def value(self) -> float:
        return self._value


class Histogram:
    def __init__(self, name: str, description: str = "", buckets: list[float] | None = None) -> None:
        self.name = name
        self.description = description
        self._buckets = buckets or [0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
        self._values: deque[float] = deque(maxlen=100_000)
        self._lock = threading.Lock()

    def observe(self, value: float) -> None:
        with self._lock:
            self._values.append(value)

    @property
    def count(self) -> int:
        return len(self._values)

    @property
    def sum(self) -> float:
        return sum(self._values)

    @property
    def avg(self) -> float:
        if not self._values:
            return 0.0
        return self.sum / len(self._values)

    def percentile(self, p: float) -> float:
        if not self._values:
            return 0.0
        sorted_vals = sorted(self._values)
        idx = int(len(sorted_vals) * p / 100)
        return sorted_vals[min(idx, len(sorted_vals) - 1)]


class RateTracker:
    def __init__(self, window_seconds: int = 60) -> None:
        self._window = window_seconds
        self._timestamps: deque[float] = deque()
        self._lock = threading.Lock()

    def record(self) -> None:
        now = time.monotonic()
        with self._lock:
            self._timestamps.append(now)
            cutoff = now - self._window
            while self._timestamps and self._timestamps[0] < cutoff:
                self._timestamps.popleft()

    @property
    def rate_per_minute(self) -> float:
        now = time.monotonic()
        with self._lock:
            cutoff = now - self._window
            while self._timestamps and self._timestamps[0] < cutoff:
                self._timestamps.popleft()
            if not self._timestamps:
                return 0.0
            elapsed = now - self._timestamps[0]
            if elapsed <= 0:
                return 0.0
            return len(self._timestamps) / elapsed * 60


class NexusMetrics:
    _instance: NexusMetrics | None = None

    def __init__(self) -> None:
        self.decisions_total = Counter("nexus_decisions_total", "Total trading decisions made")
        self.decisions_rate = RateTracker(window_seconds=300)
        self.agent_latency = Histogram("nexus_agent_latency_ms", "Agent execution latency in ms")
        self.llm_cost_total = Counter("nexus_llm_cost_usd", "Total LLM cost in USD")
        self.llm_cost_by_agent: dict[str, float] = defaultdict(float)
        self.llm_cost_by_model: dict[str, float] = defaultdict(float)
        self.api_calls_total = Counter("nexus_api_calls_total", "Total API calls")
        self.api_calls_failed = Counter("nexus_api_calls_failed", "Failed API calls")
        self.api_latency = Histogram("nexus_api_latency_ms", "API call latency in ms")
        self.data_fetch_latency = Histogram("nexus_data_fetch_latency_ms", "Data fetch latency")
        self.data_fetch_failures = Counter("nexus_data_fetch_failures", "Data fetch failures")
        self.active_positions = Gauge("nexus_active_positions", "Number of active positions")
        self.portfolio_value = Gauge("nexus_portfolio_value", "Current portfolio value")
        self.daily_pnl = Gauge("nexus_daily_pnl", "Today's P&L")
        self.backtest_queue_depth = Gauge("nexus_backtest_queue_depth", "Pending backtests")
        self.system_uptime_start = time.monotonic()
        self.decision_accuracy = Gauge("nexus_decision_accuracy", "Overall decision accuracy")
        self.agent_accuracy: dict[str, float] = {}
        self.agent_confidence: dict[str, list[float]] = defaultdict(list)
        self._history: deque[MetricPoint] = deque(maxlen=100_000)
        self._prom_registry: Any = None
        self._lock = threading.Lock()

    @classmethod
    def get_instance(cls) -> NexusMetrics:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def record_decision(
        self,
        agent_name: str,
        latency_ms: float,
        cost_usd: float,
        model: str = "",
        confidence: float = 0.0,
    ) -> None:
        self.decisions_total.inc()
        self.decisions_rate.record()
        self.agent_latency.observe(latency_ms)
        self.llm_cost_total.inc(cost_usd)

        with self._lock:
            self.llm_cost_by_agent[agent_name] += cost_usd
            if model:
                self.llm_cost_by_model[model] += cost_usd
            if confidence > 0:
                self.agent_confidence[agent_name].append(confidence)
                if len(self.agent_confidence[agent_name]) > 10000:
                    self.agent_confidence[agent_name] = self.agent_confidence[agent_name][-5000:]

        self._history.append(MetricPoint("decision", cost_usd, {"agent": agent_name}))

    def record_api_call(self, success: bool, latency_ms: float) -> None:
        self.api_calls_total.inc()
        self.api_latency.observe(latency_ms)
        if not success:
            self.api_calls_failed.inc()

    def record_data_fetch(self, success: bool, latency_ms: float) -> None:
        self.data_fetch_latency.observe(latency_ms)
        if not success:
            self.data_fetch_failures.inc()

    def update_portfolio(self, value: float, daily_pnl: float, position_count: int) -> None:
        self.portfolio_value.set(value)
        self.daily_pnl.set(daily_pnl)
        self.active_positions.set(position_count)

    @property
    def uptime_seconds(self) -> float:
        return time.monotonic() - self.system_uptime_start

    @property
    def decisions_per_minute(self) -> float:
        return self.decisions_rate.rate_per_minute

    @property
    def cost_per_decision(self) -> float:
        if self.decisions_total.value == 0:
            return 0.0
        return self.llm_cost_total.value / self.decisions_total.value

    @property
    def api_success_rate(self) -> float:
        total = self.api_calls_total.value
        if total == 0:
            return 1.0
        return 1.0 - (self.api_calls_failed.value / total)

    def avg_confidence_by_agent(self) -> dict[str, float]:
        result: dict[str, float] = {}
        with self._lock:
            for agent, values in self.agent_confidence.items():
                if values:
                    result[agent] = sum(values) / len(values)
        return result

    def get_snapshot(self) -> dict[str, Any]:
        return {
            "uptime_seconds": self.uptime_seconds,
            "decisions_total": self.decisions_total.value,
            "decisions_per_minute": self.decisions_per_minute,
            "cost_per_decision": self.cost_per_decision,
            "llm_cost_total_usd": self.llm_cost_total.value,
            "llm_cost_by_agent": dict(self.llm_cost_by_agent),
            "llm_cost_by_model": dict(self.llm_cost_by_model),
            "agent_latency_avg_ms": self.agent_latency.avg,
            "agent_latency_p99_ms": self.agent_latency.percentile(99),
            "api_success_rate": self.api_success_rate,
            "api_calls_total": self.api_calls_total.value,
            "data_fetch_failures": self.data_fetch_failures.value,
            "portfolio_value": self.portfolio_value.value,
            "daily_pnl": self.daily_pnl.value,
            "active_positions": self.active_positions.value,
            "backtest_queue_depth": self.backtest_queue_depth.value,
            "decision_accuracy": self.decision_accuracy.value,
            "avg_confidence_by_agent": self.avg_confidence_by_agent(),
        }

    def setup_prometheus(self, port: int = 9090) -> None:
        try:
            from prometheus_client import Counter as PromCounter
            from prometheus_client import Gauge as PromGauge
            from prometheus_client import Histogram as PromHistogram
            from prometheus_client import start_http_server

            self._prom_registry = {
                "decisions": PromCounter("nexus_decisions_total", "Total decisions"),
                "llm_cost": PromCounter("nexus_llm_cost_usd_total", "Total LLM cost"),
                "latency": PromHistogram("nexus_agent_latency_seconds", "Agent latency"),
                "portfolio": PromGauge("nexus_portfolio_value_usd", "Portfolio value"),
                "positions": PromGauge("nexus_active_positions", "Active positions"),
                "pnl": PromGauge("nexus_daily_pnl_usd", "Daily P&L"),
            }
            start_http_server(port)
            logger.info(f"Prometheus metrics server started on port {port}")
        except ImportError:
            logger.warning("prometheus-client not installed")
        except Exception as e:
            logger.error(f"Failed to start Prometheus server: {e}")

    def export_prometheus(self) -> str:
        lines = [
            f"# HELP nexus_uptime_seconds System uptime",
            f"# TYPE nexus_uptime_seconds gauge",
            f"nexus_uptime_seconds {self.uptime_seconds:.1f}",
            f"# HELP nexus_decisions_total Total decisions",
            f"# TYPE nexus_decisions_total counter",
            f"nexus_decisions_total {self.decisions_total.value}",
            f"# HELP nexus_llm_cost_usd Total LLM cost",
            f"# TYPE nexus_llm_cost_usd counter",
            f"nexus_llm_cost_usd {self.llm_cost_total.value:.6f}",
            f"# HELP nexus_portfolio_value Portfolio value",
            f"# TYPE nexus_portfolio_value gauge",
            f"nexus_portfolio_value {self.portfolio_value.value:.2f}",
            f"# HELP nexus_active_positions Active positions",
            f"# TYPE nexus_active_positions gauge",
            f"nexus_active_positions {self.active_positions.value}",
            f"# HELP nexus_daily_pnl Daily P&L",
            f"# TYPE nexus_daily_pnl gauge",
            f"nexus_daily_pnl {self.daily_pnl.value:.2f}",
            f"# HELP nexus_api_success_rate API success rate",
            f"# TYPE nexus_api_success_rate gauge",
            f"nexus_api_success_rate {self.api_success_rate:.4f}",
        ]
        return "\n".join(lines) + "\n"
