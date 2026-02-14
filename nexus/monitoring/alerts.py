from __future__ import annotations

import json
from datetime import datetime
from enum import Enum
from typing import Any

import httpx

from nexus.core.logging import get_logger

logger = get_logger("monitoring.alerts")


class AlertSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class AlertChannel(str, Enum):
    SLACK = "slack"
    PAGERDUTY = "pagerduty"
    EMAIL = "email"
    WEBHOOK = "webhook"
    LOG = "log"


class Alert:
    def __init__(
        self,
        title: str,
        message: str,
        severity: AlertSeverity = AlertSeverity.WARNING,
        source: str = "nexus",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.alert_id = f"{source}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        self.title = title
        self.message = message
        self.severity = severity
        self.source = source
        self.metadata = metadata or {}
        self.timestamp = datetime.now()
        self.acknowledged = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "alert_id": self.alert_id,
            "title": self.title,
            "message": self.message,
            "severity": self.severity.value,
            "source": self.source,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat(),
            "acknowledged": self.acknowledged,
        }


class SlackNotifier:
    def __init__(self, webhook_url: str) -> None:
        self._webhook_url = webhook_url

    async def send(self, alert: Alert) -> bool:
        severity_emoji = {
            AlertSeverity.INFO: ":information_source:",
            AlertSeverity.WARNING: ":warning:",
            AlertSeverity.CRITICAL: ":rotating_light:",
            AlertSeverity.EMERGENCY: ":fire:",
        }
        emoji = severity_emoji.get(alert.severity, ":bell:")

        payload = {
            "blocks": [
                {
                    "type": "header",
                    "text": {"type": "plain_text", "text": f"{emoji} {alert.title}"},
                },
                {
                    "type": "section",
                    "text": {"type": "mrkdwn", "text": alert.message},
                },
                {
                    "type": "context",
                    "elements": [
                        {"type": "mrkdwn", "text": f"*Severity:* {alert.severity.value}"},
                        {"type": "mrkdwn", "text": f"*Source:* {alert.source}"},
                        {"type": "mrkdwn", "text": f"*Time:* {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}"},
                    ],
                },
            ],
        }

        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(self._webhook_url, json=payload, timeout=10)
                return resp.status_code == 200
        except Exception as e:
            logger.error(f"Slack notification failed: {e}")
            return False


class PagerDutyNotifier:
    def __init__(self, routing_key: str) -> None:
        self._routing_key = routing_key
        self._api_url = "https://events.pagerduty.com/v2/enqueue"

    async def send(self, alert: Alert) -> bool:
        severity_map = {
            AlertSeverity.INFO: "info",
            AlertSeverity.WARNING: "warning",
            AlertSeverity.CRITICAL: "critical",
            AlertSeverity.EMERGENCY: "critical",
        }

        payload = {
            "routing_key": self._routing_key,
            "event_action": "trigger",
            "payload": {
                "summary": f"[NEXUS] {alert.title}: {alert.message}",
                "severity": severity_map.get(alert.severity, "warning"),
                "source": alert.source,
                "timestamp": alert.timestamp.isoformat(),
                "custom_details": alert.metadata,
            },
        }

        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(self._api_url, json=payload, timeout=10)
                return resp.status_code == 202
        except Exception as e:
            logger.error(f"PagerDuty notification failed: {e}")
            return False


class EmailNotifier:
    def __init__(
        self,
        smtp_host: str = "localhost",
        smtp_port: int = 587,
        sender: str = "",
        recipients: list[str] | None = None,
        username: str = "",
        password: str = "",
    ) -> None:
        self._smtp_host = smtp_host
        self._smtp_port = smtp_port
        self._sender = sender
        self._recipients = recipients or []
        self._username = username
        self._password = password

    async def send(self, alert: Alert) -> bool:
        try:
            import smtplib
            from email.mime.text import MIMEText

            msg = MIMEText(
                f"Alert: {alert.title}\n\n"
                f"Severity: {alert.severity.value}\n"
                f"Source: {alert.source}\n"
                f"Time: {alert.timestamp.isoformat()}\n\n"
                f"{alert.message}\n\n"
                f"Metadata: {json.dumps(alert.metadata, indent=2)}"
            )
            msg["Subject"] = f"[NEXUS {alert.severity.value.upper()}] {alert.title}"
            msg["From"] = self._sender
            msg["To"] = ", ".join(self._recipients)

            with smtplib.SMTP(self._smtp_host, self._smtp_port) as server:
                if self._username:
                    server.starttls()
                    server.login(self._username, self._password)
                server.sendmail(self._sender, self._recipients, msg.as_string())
            return True
        except Exception as e:
            logger.error(f"Email notification failed: {e}")
            return False


class WebhookNotifier:
    def __init__(self, url: str, headers: dict[str, str] | None = None) -> None:
        self._url = url
        self._headers = headers or {"Content-Type": "application/json"}

    async def send(self, alert: Alert) -> bool:
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    self._url, json=alert.to_dict(), headers=self._headers, timeout=10,
                )
                return 200 <= resp.status_code < 300
        except Exception as e:
            logger.error(f"Webhook notification failed: {e}")
            return False


class AlertRule:
    def __init__(
        self,
        name: str,
        condition_fn: Any,
        title: str,
        message_template: str,
        severity: AlertSeverity = AlertSeverity.WARNING,
        cooldown_seconds: int = 300,
    ) -> None:
        self.name = name
        self.condition_fn = condition_fn
        self.title = title
        self.message_template = message_template
        self.severity = severity
        self.cooldown_seconds = cooldown_seconds
        self._last_triggered: datetime | None = None

    def check(self, context: dict[str, Any]) -> Alert | None:
        if self._last_triggered:
            elapsed = (datetime.now() - self._last_triggered).total_seconds()
            if elapsed < self.cooldown_seconds:
                return None

        try:
            if self.condition_fn(context):
                self._last_triggered = datetime.now()
                message = self.message_template.format(**context)
                return Alert(
                    title=self.title,
                    message=message,
                    severity=self.severity,
                    metadata={"rule": self.name},
                )
        except Exception as e:
            logger.error(f"Alert rule {self.name} check failed: {e}")
        return None


class AlertManager:
    def __init__(self) -> None:
        self._channels: dict[AlertChannel, Any] = {}
        self._rules: list[AlertRule] = []
        self._history: list[Alert] = []
        self._max_history = 10000
        self._severity_channels: dict[AlertSeverity, list[AlertChannel]] = {
            AlertSeverity.INFO: [AlertChannel.LOG],
            AlertSeverity.WARNING: [AlertChannel.LOG, AlertChannel.SLACK],
            AlertSeverity.CRITICAL: [AlertChannel.LOG, AlertChannel.SLACK, AlertChannel.PAGERDUTY],
            AlertSeverity.EMERGENCY: [AlertChannel.LOG, AlertChannel.SLACK, AlertChannel.PAGERDUTY, AlertChannel.EMAIL],
        }

    def add_slack(self, webhook_url: str) -> None:
        self._channels[AlertChannel.SLACK] = SlackNotifier(webhook_url)

    def add_pagerduty(self, routing_key: str) -> None:
        self._channels[AlertChannel.PAGERDUTY] = PagerDutyNotifier(routing_key)

    def add_email(self, **kwargs: Any) -> None:
        self._channels[AlertChannel.EMAIL] = EmailNotifier(**kwargs)

    def add_webhook(self, url: str, headers: dict[str, str] | None = None) -> None:
        self._channels[AlertChannel.WEBHOOK] = WebhookNotifier(url, headers)

    def add_rule(self, rule: AlertRule) -> None:
        self._rules.append(rule)

    def setup_default_rules(self) -> None:
        self._rules = [
            AlertRule(
                name="daily_loss",
                condition_fn=lambda ctx: ctx.get("daily_pnl_pct", 0) < -5.0,
                title="Abnormal Daily Loss",
                message_template="Portfolio down {daily_pnl_pct:.2f}% today (>${daily_loss:.0f})",
                severity=AlertSeverity.CRITICAL,
                cooldown_seconds=3600,
            ),
            AlertRule(
                name="risk_limit_breached",
                condition_fn=lambda ctx: ctx.get("risk_breached", False),
                title="Risk Limit Breached",
                message_template="Risk check failed: {risk_message}",
                severity=AlertSeverity.CRITICAL,
                cooldown_seconds=600,
            ),
            AlertRule(
                name="data_fetch_failures",
                condition_fn=lambda ctx: ctx.get("consecutive_fetch_failures", 0) > 10,
                title="Data Fetch Failures",
                message_template="{consecutive_fetch_failures} consecutive data fetch failures",
                severity=AlertSeverity.WARNING,
                cooldown_seconds=300,
            ),
            AlertRule(
                name="llm_cost_spike",
                condition_fn=lambda ctx: ctx.get("daily_llm_cost", 0) > 100,
                title="LLM Cost Spike",
                message_template="Daily LLM cost: ${daily_llm_cost:.2f} (limit: $100)",
                severity=AlertSeverity.WARNING,
                cooldown_seconds=3600,
            ),
            AlertRule(
                name="max_drawdown",
                condition_fn=lambda ctx: ctx.get("drawdown_pct", 0) > ctx.get("max_drawdown_limit", 20),
                title="Max Drawdown Exceeded",
                message_template="Drawdown at {drawdown_pct:.2f}% exceeds limit of {max_drawdown_limit}%",
                severity=AlertSeverity.EMERGENCY,
                cooldown_seconds=1800,
            ),
            AlertRule(
                name="system_down",
                condition_fn=lambda ctx: not ctx.get("system_healthy", True),
                title="System Health Check Failed",
                message_template="System unhealthy: {health_message}",
                severity=AlertSeverity.EMERGENCY,
                cooldown_seconds=60,
            ),
        ]

    async def check_rules(self, context: dict[str, Any]) -> list[Alert]:
        triggered: list[Alert] = []
        for rule in self._rules:
            alert = rule.check(context)
            if alert:
                triggered.append(alert)
                await self.send_alert(alert)
        return triggered

    async def send_alert(self, alert: Alert) -> None:
        self._history.append(alert)
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history:]

        logger.warning(f"ALERT [{alert.severity.value}] {alert.title}: {alert.message}")

        channels = self._severity_channels.get(alert.severity, [AlertChannel.LOG])
        for channel in channels:
            notifier = self._channels.get(channel)
            if notifier:
                try:
                    await notifier.send(alert)
                except Exception as e:
                    logger.error(f"Failed to send alert via {channel.value}: {e}")

    def get_history(self, limit: int = 100, severity: AlertSeverity | None = None) -> list[dict[str, Any]]:
        alerts = self._history[-limit:]
        if severity:
            alerts = [a for a in alerts if a.severity == severity]
        return [a.to_dict() for a in alerts]

    def acknowledge(self, alert_id: str) -> bool:
        for alert in self._history:
            if alert.alert_id == alert_id:
                alert.acknowledged = True
                return True
        return False
