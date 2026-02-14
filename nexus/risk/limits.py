from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from nexus.core.exceptions import (
    DrawdownLimitError,
    ExposureLimitError,
    PositionLimitError,
    RiskViolation,
)
from nexus.core.logging import get_logger
from nexus.core.types import Action, OrderSide, Portfolio

logger = get_logger("risk.limits")


@dataclass
class RiskLimits:
    max_position_pct: float = 0.15
    max_sector_pct: float = 0.30
    max_leverage: float = 1.0
    max_correlation: float = 0.80
    max_beta: float = 1.5
    max_drawdown_pct: float = 0.20
    max_daily_loss_pct: float = 0.03
    max_open_positions: int = 20
    min_cash_pct: float = 0.05
    max_single_trade_pct: float = 0.10
    max_concentration_top3: float = 0.40


@dataclass
class RiskCheckResult:
    passed: bool
    check_name: str
    message: str = ""
    current_value: float = 0.0
    limit_value: float = 0.0
    severity: str = "warning"


class RiskLimitChecker:
    def __init__(self, limits: RiskLimits | None = None) -> None:
        self.limits = limits or RiskLimits()

    def run_all_checks(
        self,
        action: Action,
        portfolio: Portfolio,
    ) -> list[RiskCheckResult]:
        results: list[RiskCheckResult] = []

        results.append(self._check_position_size(action, portfolio))
        results.append(self._check_total_exposure(action, portfolio))
        results.append(self._check_drawdown(portfolio))
        results.append(self._check_cash_reserve(action, portfolio))
        results.append(self._check_position_count(action, portfolio))
        results.append(self._check_concentration(action, portfolio))

        return results

    def should_veto(self, results: list[RiskCheckResult]) -> bool:
        critical_failures = [
            r for r in results
            if not r.passed and r.severity == "critical"
        ]
        return len(critical_failures) > 0

    def _check_position_size(self, action: Action, portfolio: Portfolio) -> RiskCheckResult:
        if portfolio.total_value <= 0:
            return RiskCheckResult(
                passed=False,
                check_name="position_size",
                message="Portfolio value is zero",
                severity="critical",
            )

        trade_value = action.quantity * (action.limit_price or 0)
        position_pct = trade_value / portfolio.total_value

        existing = portfolio.positions.get(action.ticker)
        if existing:
            total_value = existing.market_value + trade_value
            position_pct = total_value / portfolio.total_value

        passed = position_pct <= self.limits.max_position_pct

        return RiskCheckResult(
            passed=passed,
            check_name="position_size",
            message=f"Position {action.ticker}: {position_pct:.1%} vs limit {self.limits.max_position_pct:.1%}",
            current_value=position_pct,
            limit_value=self.limits.max_position_pct,
            severity="critical" if not passed else "info",
        )

    def _check_total_exposure(self, action: Action, portfolio: Portfolio) -> RiskCheckResult:
        if portfolio.total_value <= 0:
            return RiskCheckResult(passed=False, check_name="total_exposure", severity="critical")

        total_exposure = sum(abs(p.market_value) for p in portfolio.positions.values())
        trade_value = action.quantity * (action.limit_price or 0)
        if action.side == OrderSide.BUY:
            total_exposure += trade_value

        exposure_ratio = total_exposure / portfolio.total_value
        passed = exposure_ratio <= self.limits.max_leverage

        return RiskCheckResult(
            passed=passed,
            check_name="total_exposure",
            message=f"Exposure: {exposure_ratio:.1%} vs limit {self.limits.max_leverage:.1%}",
            current_value=exposure_ratio,
            limit_value=self.limits.max_leverage,
            severity="critical" if not passed else "info",
        )

    def _check_drawdown(self, portfolio: Portfolio) -> RiskCheckResult:
        passed = abs(portfolio.max_drawdown) <= self.limits.max_drawdown_pct

        return RiskCheckResult(
            passed=passed,
            check_name="drawdown",
            message=f"Drawdown: {portfolio.max_drawdown:.1%} vs limit {self.limits.max_drawdown_pct:.1%}",
            current_value=abs(portfolio.max_drawdown),
            limit_value=self.limits.max_drawdown_pct,
            severity="critical" if not passed else "info",
        )

    def _check_cash_reserve(self, action: Action, portfolio: Portfolio) -> RiskCheckResult:
        if action.side == OrderSide.SELL:
            return RiskCheckResult(passed=True, check_name="cash_reserve")

        trade_cost = action.quantity * (action.limit_price or 0)
        remaining_cash = portfolio.cash - trade_cost
        cash_pct = remaining_cash / portfolio.total_value if portfolio.total_value > 0 else 0

        passed = cash_pct >= self.limits.min_cash_pct

        return RiskCheckResult(
            passed=passed,
            check_name="cash_reserve",
            message=f"Remaining cash: {cash_pct:.1%} vs minimum {self.limits.min_cash_pct:.1%}",
            current_value=cash_pct,
            limit_value=self.limits.min_cash_pct,
            severity="warning" if not passed else "info",
        )

    def _check_position_count(self, action: Action, portfolio: Portfolio) -> RiskCheckResult:
        current_count = len(portfolio.positions)
        if action.side == OrderSide.BUY and action.ticker not in portfolio.positions:
            current_count += 1

        passed = current_count <= self.limits.max_open_positions

        return RiskCheckResult(
            passed=passed,
            check_name="position_count",
            message=f"Positions: {current_count} vs limit {self.limits.max_open_positions}",
            current_value=float(current_count),
            limit_value=float(self.limits.max_open_positions),
            severity="warning" if not passed else "info",
        )

    def _check_concentration(self, action: Action, portfolio: Portfolio) -> RiskCheckResult:
        if portfolio.total_value <= 0:
            return RiskCheckResult(passed=True, check_name="concentration")

        weights = sorted(
            [abs(p.market_value / portfolio.total_value) for p in portfolio.positions.values()],
            reverse=True,
        )

        top3_weight = sum(weights[:3]) if len(weights) >= 3 else sum(weights)
        passed = top3_weight <= self.limits.max_concentration_top3

        return RiskCheckResult(
            passed=passed,
            check_name="concentration",
            message=f"Top-3 concentration: {top3_weight:.1%} vs limit {self.limits.max_concentration_top3:.1%}",
            current_value=top3_weight,
            limit_value=self.limits.max_concentration_top3,
            severity="warning" if not passed else "info",
        )

    def enforce(self, action: Action, portfolio: Portfolio) -> Action:
        results = self.run_all_checks(action, portfolio)

        for result in results:
            if not result.passed:
                logger.warning(f"Risk check failed: {result.check_name} — {result.message}")

        if self.should_veto(results):
            critical = [r for r in results if not r.passed and r.severity == "critical"]
            messages = "; ".join(r.message for r in critical)
            raise RiskViolation(
                f"Trade vetoed: {messages}",
                code="RISK_POSITION_LIMIT",
            )

        return self._adjust_size(action, portfolio, results)

    def _adjust_size(
        self,
        action: Action,
        portfolio: Portfolio,
        results: list[RiskCheckResult],
    ) -> Action:
        max_value = portfolio.total_value * self.limits.max_position_pct
        price = action.limit_price or 1.0
        max_quantity = max_value / price if price > 0 else action.quantity

        if action.quantity > max_quantity:
            logger.info(f"Reducing position size from {action.quantity} to {max_quantity:.0f}")
            action.quantity = float(int(max_quantity))

        return action
