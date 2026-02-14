from __future__ import annotations

import pytest

from nexus.monitoring.health import HealthStatus, ComponentHealth, HealthChecker


class TestHealthStatus:
    def test_values(self) -> None:
        assert HealthStatus.HEALTHY.value == "healthy"
        assert HealthStatus.DEGRADED.value == "degraded"
        assert HealthStatus.UNHEALTHY.value == "unhealthy"


class TestComponentHealth:
    def test_healthy(self) -> None:
        ch = ComponentHealth(
            name="test",
            status=HealthStatus.HEALTHY,
            latency_ms=5.0,
        )
        assert ch.status == HealthStatus.HEALTHY
        assert ch.message == ""

    def test_unhealthy(self) -> None:
        ch = ComponentHealth(
            name="test",
            status=HealthStatus.UNHEALTHY,
            message="Connection refused",
        )
        assert ch.message == "Connection refused"

    def test_to_dict(self) -> None:
        ch = ComponentHealth(name="db", status=HealthStatus.DEGRADED, latency_ms=10.5, message="slow")
        d = ch.to_dict()
        assert d["name"] == "db"
        assert d["status"] == "degraded"
        assert d["latency_ms"] == 10.5
        assert d["message"] == "slow"


class TestHealthChecker:
    def test_register_check(self) -> None:
        checker = HealthChecker()

        async def custom_check() -> ComponentHealth:
            return ComponentHealth(name="custom", status=HealthStatus.HEALTHY)

        checker.register_check("custom", custom_check)
        assert "custom" in checker._checks

    @pytest.mark.asyncio
    async def test_run_all_checks(self) -> None:
        checker = HealthChecker()

        async def ok_check() -> ComponentHealth:
            return ComponentHealth(name="ok", status=HealthStatus.HEALTHY, latency_ms=1.0)

        checker.register_check("ok", ok_check)
        results = await checker.run_all_checks()
        assert isinstance(results, dict)
        assert "status" in results
        assert "components" in results
        component_names = [c["name"] for c in results["components"]]
        assert "ok" in component_names

    @pytest.mark.asyncio
    async def test_is_healthy(self) -> None:
        checker = HealthChecker()
        await checker.run_all_checks()
        assert isinstance(checker.is_healthy, bool)
