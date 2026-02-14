from __future__ import annotations

import pytest

from nexus.monitoring.alerts import AlertSeverity, AlertChannel, Alert, AlertRule, AlertManager


class TestAlertSeverity:
    def test_values(self) -> None:
        assert AlertSeverity.INFO.value == "info"
        assert AlertSeverity.CRITICAL.value == "critical"
        assert AlertSeverity.EMERGENCY.value == "emergency"


class TestAlertChannel:
    def test_values(self) -> None:
        assert AlertChannel.SLACK.value == "slack"
        assert AlertChannel.PAGERDUTY.value == "pagerduty"


class TestAlert:
    def test_creation(self) -> None:
        alert = Alert(
            title="Test Alert",
            message="Test alert message",
            severity=AlertSeverity.WARNING,
        )
        assert alert.title == "Test Alert"
        assert alert.severity == AlertSeverity.WARNING
        assert alert.acknowledged is False

    def test_to_dict(self) -> None:
        alert = Alert(title="Test", message="msg", severity=AlertSeverity.INFO)
        d = alert.to_dict()
        assert "alert_id" in d
        assert d["title"] == "Test"
        assert d["severity"] == "info"


class TestAlertRule:
    def test_rule_trigger(self) -> None:
        rule = AlertRule(
            name="test",
            condition_fn=lambda ctx: ctx.get("value", 0) > 100,
            title="Value Exceeded",
            message_template="Value exceeded: {value}",
            severity=AlertSeverity.WARNING,
            cooldown_seconds=0,
        )
        result = rule.check({"value": 150})
        assert result is not None
        assert result.title == "Value Exceeded"

        result2 = rule.check({"value": 50})
        assert result2 is None

    def test_cooldown(self) -> None:
        rule = AlertRule(
            name="test",
            condition_fn=lambda ctx: True,
            title="Cooldown Test",
            message_template="Fired",
            severity=AlertSeverity.INFO,
            cooldown_seconds=10,
        )
        first = rule.check({})
        assert first is not None
        second = rule.check({})
        assert second is None


class TestAlertManager:
    def test_add_webhook(self) -> None:
        mgr = AlertManager()
        mgr.add_webhook("https://example.com/webhook")
        assert AlertChannel.WEBHOOK in mgr._channels

    def test_setup_default_rules(self) -> None:
        mgr = AlertManager()
        mgr.setup_default_rules()
        assert len(mgr._rules) >= 5

    def test_get_history(self) -> None:
        mgr = AlertManager()
        assert mgr.get_history() == []

    @pytest.mark.asyncio
    async def test_check_rules_no_fire(self) -> None:
        mgr = AlertManager()
        mgr.setup_default_rules()
        context = {
            "daily_pnl_pct": 0,
            "risk_breached": False,
            "consecutive_fetch_failures": 0,
            "daily_llm_cost": 0,
            "drawdown_pct": 0,
            "max_drawdown_limit": 20,
            "system_healthy": True,
        }
        fired = await mgr.check_rules(context)
        assert len(fired) == 0
