from __future__ import annotations

import time

import pytest

from nexus.monitoring.metrics import Counter, Gauge, Histogram, RateTracker, NexusMetrics


class TestCounter:
    def test_initial_value(self) -> None:
        c = Counter("test")
        assert c.value == 0

    def test_increment(self) -> None:
        c = Counter("test")
        c.inc()
        assert c.value == 1

    def test_increment_by(self) -> None:
        c = Counter("test")
        c.inc(5)
        assert c.value == 5

    def test_reset(self) -> None:
        c = Counter("test")
        c.inc(10)
        c.reset()
        assert c.value == 0


class TestGauge:
    def test_set(self) -> None:
        g = Gauge("test")
        g.set(42.5)
        assert g.value == 42.5

    def test_inc_dec(self) -> None:
        g = Gauge("test")
        g.set(10)
        g.inc(5)
        assert g.value == 15
        g.dec(3)
        assert g.value == 12


class TestHistogram:
    def test_observe(self) -> None:
        h = Histogram("test")
        h.observe(1.0)
        h.observe(2.0)
        h.observe(3.0)
        assert h.count == 3
        assert h.sum == 6.0
        assert h.avg == 2.0

    def test_percentile(self) -> None:
        h = Histogram("test")
        for i in range(100):
            h.observe(float(i))
        p50 = h.percentile(50)
        assert 45 <= p50 <= 55


class TestRateTracker:
    def test_record_and_rate(self) -> None:
        rt = RateTracker()
        for _ in range(10):
            rt.record()
        time.sleep(0.02)
        rt.record()
        rate = rt.rate_per_minute
        assert rate > 0


class TestNexusMetrics:
    def test_singleton(self) -> None:
        m1 = NexusMetrics.get_instance()
        m2 = NexusMetrics.get_instance()
        assert m1 is m2

    def test_record_decision(self) -> None:
        m = NexusMetrics.get_instance()
        initial = m.decisions_total.value
        m.record_decision(agent_name="test_agent", latency_ms=50, cost_usd=0.001)
        assert m.decisions_total.value == initial + 1

    def test_record_api_call(self) -> None:
        m = NexusMetrics.get_instance()
        initial = m.api_calls_total.value
        m.record_api_call(success=True, latency_ms=50.0)
        assert m.api_calls_total.value == initial + 1

    def test_update_portfolio(self) -> None:
        m = NexusMetrics.get_instance()
        m.update_portfolio(value=100000, daily_pnl=5000, position_count=5)
        assert m.portfolio_value.value == 100000
        assert m.daily_pnl.value == 5000
        assert m.active_positions.value == 5

    def test_get_snapshot(self) -> None:
        m = NexusMetrics.get_instance()
        snap = m.get_snapshot()
        assert "decisions_total" in snap
        assert "portfolio_value" in snap
        assert "uptime_seconds" in snap
