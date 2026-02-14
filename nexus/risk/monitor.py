from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from nexus.core.exceptions import DrawdownLimitError, RiskViolation
from nexus.core.logging import get_logger
from nexus.core.types import Action, Portfolio
from nexus.risk.limits import RiskLimitChecker, RiskLimits
from nexus.risk.portfolio import PortfolioRiskAnalyzer

logger = get_logger("risk.monitor")


class RiskAlert:
    def __init__(
        self,
        alert_type: str,
        severity: str,
        message: str,
        current_value: float = 0.0,
        threshold: float = 0.0,
        timestamp: datetime | None = None,
    ) -> None:
        self.alert_type = alert_type
        self.severity = severity
        self.message = message
        self.current_value = current_value
        self.threshold = threshold
        self.timestamp = timestamp or datetime.now()

    def to_dict(self) -> dict[str, Any]:
        return {
            "alert_type": self.alert_type,
            "severity": self.severity,
            "message": self.message,
            "current_value": self.current_value,
            "threshold": self.threshold,
            "timestamp": self.timestamp.isoformat(),
        }


class RiskMonitor:
    def __init__(
        self,
        limits: RiskLimits | None = None,
        vix_reduction_threshold: float = 30.0,
        vix_halt_threshold: float = 45.0,
        max_alerts_stored: int = 1000,
    ) -> None:
        self.limits = limits or RiskLimits()
        self.limit_checker = RiskLimitChecker(self.limits)
        self.risk_analyzer = PortfolioRiskAnalyzer()
        self.vix_reduction_threshold = vix_reduction_threshold
        self.vix_halt_threshold = vix_halt_threshold
        self.alerts: list[RiskAlert] = []
        self.max_alerts = max_alerts_stored
        self._daily_high_water: float = 0.0
        self._daily_pnl_start: float = 0.0
        self._trading_halted: bool = False

    @property
    def is_halted(self) -> bool:
        return self._trading_halted

    def update_portfolio_state(self, portfolio: Portfolio) -> list[RiskAlert]:
        new_alerts: list[RiskAlert] = []

        if portfolio.total_value > self._daily_high_water:
            self._daily_high_water = portfolio.total_value

        if self._daily_pnl_start == 0:
            self._daily_pnl_start = portfolio.total_value

        daily_pnl_pct = (portfolio.total_value / self._daily_pnl_start - 1) if self._daily_pnl_start > 0 else 0

        if daily_pnl_pct < -self.limits.max_daily_loss_pct:
            alert = RiskAlert(
                alert_type="daily_loss_limit",
                severity="critical",
                message=f"Daily loss {daily_pnl_pct:.2%} exceeds limit {self.limits.max_daily_loss_pct:.2%}",
                current_value=daily_pnl_pct,
                threshold=-self.limits.max_daily_loss_pct,
            )
            new_alerts.append(alert)
            self._trading_halted = True
            logger.critical(f"TRADING HALTED: {alert.message}")

        if abs(portfolio.max_drawdown) > self.limits.max_drawdown_pct:
            alert = RiskAlert(
                alert_type="max_drawdown",
                severity="critical",
                message=f"Drawdown {portfolio.max_drawdown:.2%} exceeds limit {self.limits.max_drawdown_pct:.2%}",
                current_value=portfolio.max_drawdown,
                threshold=-self.limits.max_drawdown_pct,
            )
            new_alerts.append(alert)

        exposure = sum(abs(p.market_value) for p in portfolio.positions.values())
        exposure_ratio = exposure / portfolio.total_value if portfolio.total_value > 0 else 0
        if exposure_ratio > self.limits.max_leverage * 0.9:
            alert = RiskAlert(
                alert_type="high_exposure",
                severity="warning",
                message=f"Exposure {exposure_ratio:.1%} approaching limit {self.limits.max_leverage:.1%}",
                current_value=exposure_ratio,
                threshold=self.limits.max_leverage,
            )
            new_alerts.append(alert)

        self._store_alerts(new_alerts)
        return new_alerts

    def check_vix(self, vix_level: float) -> dict[str, Any]:
        result: dict[str, Any] = {
            "vix_level": vix_level,
            "position_size_multiplier": 1.0,
            "action": "normal",
        }

        if vix_level >= self.vix_halt_threshold:
            result["position_size_multiplier"] = 0.0
            result["action"] = "halt_trading"
            self._trading_halted = True
            alert = RiskAlert(
                alert_type="vix_halt",
                severity="critical",
                message=f"VIX at {vix_level:.1f} — trading halted",
                current_value=vix_level,
                threshold=self.vix_halt_threshold,
            )
            self._store_alerts([alert])
            logger.critical(f"VIX HALT: {alert.message}")
        elif vix_level >= self.vix_reduction_threshold:
            reduction = 1.0 - (vix_level - self.vix_reduction_threshold) / (self.vix_halt_threshold - self.vix_reduction_threshold)
            result["position_size_multiplier"] = max(0.25, min(1.0, reduction))
            result["action"] = "reduce_size"
            alert = RiskAlert(
                alert_type="vix_elevated",
                severity="warning",
                message=f"VIX at {vix_level:.1f} — reducing position sizes to {result['position_size_multiplier']:.0%}",
                current_value=vix_level,
                threshold=self.vix_reduction_threshold,
            )
            self._store_alerts([alert])

        return result

    def pre_trade_check(
        self,
        action: Action,
        portfolio: Portfolio,
        vix_level: float | None = None,
    ) -> dict[str, Any]:
        if self._trading_halted:
            return {
                "approved": False,
                "reason": "Trading is halted",
                "alerts": [a.to_dict() for a in self.alerts[-5:]],
            }

        risk_results = self.limit_checker.run_all_checks(action, portfolio)
        veto = self.limit_checker.should_veto(risk_results)

        vix_adjustment = 1.0
        if vix_level is not None:
            vix_result = self.check_vix(vix_level)
            vix_adjustment = vix_result["position_size_multiplier"]
            if vix_result["action"] == "halt_trading":
                return {
                    "approved": False,
                    "reason": f"VIX halt: {vix_level}",
                    "vix_check": vix_result,
                }

        failures = [r for r in risk_results if not r.passed]

        return {
            "approved": not veto,
            "risk_checks": [
                {
                    "check": r.check_name,
                    "passed": r.passed,
                    "message": r.message,
                    "severity": r.severity,
                }
                for r in risk_results
            ],
            "failures": len(failures),
            "vix_size_multiplier": vix_adjustment,
            "reason": "; ".join(r.message for r in failures) if failures else "All checks passed",
        }

    def reset_daily(self, portfolio_value: float) -> None:
        self._daily_pnl_start = portfolio_value
        self._daily_high_water = portfolio_value
        self._trading_halted = False
        logger.info("Daily risk counters reset")

    def get_recent_alerts(self, n: int = 20) -> list[dict[str, Any]]:
        return [a.to_dict() for a in self.alerts[-n:]]

    def _store_alerts(self, alerts: list[RiskAlert]) -> None:
        self.alerts.extend(alerts)
        if len(self.alerts) > self.max_alerts:
            self.alerts = self.alerts[-self.max_alerts:]
