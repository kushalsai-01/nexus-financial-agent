from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from nexus.core.logging import get_logger

logger = get_logger("risk.stress")


@dataclass
class StressScenario:
    name: str
    description: str
    equity_shock: float
    volatility_multiplier: float
    correlation_shift: float
    duration_days: int
    recovery_days: int


HISTORICAL_SCENARIOS: dict[str, StressScenario] = {
    "2008_financial_crisis": StressScenario(
        name="2008 Financial Crisis",
        description="GFC: Lehman collapse, credit freeze, systemic risk",
        equity_shock=-0.55,
        volatility_multiplier=4.0,
        correlation_shift=0.3,
        duration_days=350,
        recovery_days=1400,
    ),
    "2020_covid_crash": StressScenario(
        name="2020 COVID Crash",
        description="Pandemic sell-off: fastest bear market in history",
        equity_shock=-0.34,
        volatility_multiplier=5.0,
        correlation_shift=0.4,
        duration_days=23,
        recovery_days=148,
    ),
    "2000_dotcom_bust": StressScenario(
        name="2000 Dot-Com Bust",
        description="Tech bubble burst, NASDAQ -78%",
        equity_shock=-0.49,
        volatility_multiplier=2.5,
        correlation_shift=0.15,
        duration_days=630,
        recovery_days=2500,
    ),
    "2011_euro_crisis": StressScenario(
        name="2011 Euro Sovereign Debt Crisis",
        description="European sovereign debt contagion",
        equity_shock=-0.19,
        volatility_multiplier=2.0,
        correlation_shift=0.2,
        duration_days=150,
        recovery_days=200,
    ),
    "2018_vol_shock": StressScenario(
        name="2018 Volatility Shock",
        description="VIX spike, XIV collapse, vol-selling unwind",
        equity_shock=-0.10,
        volatility_multiplier=3.5,
        correlation_shift=0.25,
        duration_days=10,
        recovery_days=120,
    ),
    "flash_crash": StressScenario(
        name="Flash Crash",
        description="Rapid intraday crash and recovery",
        equity_shock=-0.09,
        volatility_multiplier=6.0,
        correlation_shift=0.5,
        duration_days=1,
        recovery_days=5,
    ),
    "rate_shock": StressScenario(
        name="Interest Rate Shock",
        description="Sudden 200bp rate increase",
        equity_shock=-0.15,
        volatility_multiplier=2.0,
        correlation_shift=0.2,
        duration_days=60,
        recovery_days=180,
    ),
}


class StressTester:
    def __init__(self, seed: int = 42) -> None:
        self.rng = np.random.RandomState(seed)

    def apply_scenario(
        self,
        returns: pd.Series,
        scenario: StressScenario,
        portfolio_value: float = 100_000,
    ) -> dict[str, Any]:
        stressed = returns.copy()
        n = len(stressed)

        stress_days = min(scenario.duration_days, n)
        stress_start = n - stress_days

        daily_shock = (1 + scenario.equity_shock) ** (1 / max(stress_days, 1)) - 1
        noise = self.rng.normal(0, abs(daily_shock) * 0.3, stress_days)

        stressed.iloc[stress_start:] = daily_shock + noise
        stressed.iloc[stress_start:] *= scenario.volatility_multiplier

        cumulative = (1 + stressed).cumprod()
        equity = portfolio_value * cumulative

        peak = equity.expanding().max()
        drawdown = (equity - peak) / peak
        max_dd = float(drawdown.min())
        final_value = float(equity.iloc[-1])

        return {
            "scenario": scenario.name,
            "description": scenario.description,
            "equity_shock": scenario.equity_shock,
            "portfolio_impact": final_value - portfolio_value,
            "portfolio_impact_pct": (final_value / portfolio_value) - 1,
            "max_drawdown": max_dd,
            "final_value": final_value,
            "recovery_time_est_days": scenario.recovery_days,
            "vol_multiplier": scenario.volatility_multiplier,
        }

    def run_all_scenarios(
        self,
        returns: pd.Series,
        portfolio_value: float = 100_000,
        scenarios: dict[str, StressScenario] | None = None,
    ) -> dict[str, dict[str, Any]]:
        if scenarios is None:
            scenarios = HISTORICAL_SCENARIOS

        results: dict[str, dict[str, Any]] = {}
        for key, scenario in scenarios.items():
            result = self.apply_scenario(returns, scenario, portfolio_value)
            results[key] = result
            logger.info(
                f"Stress test [{scenario.name}]: "
                f"impact={result['portfolio_impact_pct']:.1%}, "
                f"max_dd={result['max_drawdown']:.1%}"
            )

        return results

    def custom_stress_test(
        self,
        returns: pd.Series,
        equity_shocks: list[float],
        portfolio_value: float = 100_000,
    ) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for shock in equity_shocks:
            scenario = StressScenario(
                name=f"Custom {shock:.0%} shock",
                description=f"Custom equity shock of {shock:.0%}",
                equity_shock=shock,
                volatility_multiplier=2.0,
                correlation_shift=0.2,
                duration_days=30,
                recovery_days=90,
            )
            results.append(self.apply_scenario(returns, scenario, portfolio_value))

        return results

    def compute_stress_summary(
        self,
        results: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        impacts = [r["portfolio_impact_pct"] for r in results.values()]
        drawdowns = [r["max_drawdown"] for r in results.values()]
        worst_scenario = min(results.items(), key=lambda x: x[1]["portfolio_impact_pct"])

        return {
            "total_scenarios": len(results),
            "avg_impact_pct": float(np.mean(impacts)),
            "worst_impact_pct": float(min(impacts)),
            "worst_scenario": worst_scenario[0],
            "avg_max_drawdown": float(np.mean(drawdowns)),
            "worst_max_drawdown": float(min(drawdowns)),
            "scenarios_with_20pct_loss": sum(1 for i in impacts if i < -0.20),
        }
