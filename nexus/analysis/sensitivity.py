from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from nexus.backtest.engine import BacktestConfig, BacktestEngine
from nexus.backtest.strategy import BaseStrategy
from nexus.core.logging import get_logger

logger = get_logger("analysis.sensitivity")


class SensitivityAnalyzer:
    def __init__(
        self,
        n_steps: int = 10,
        variation_pct: float = 0.50,
    ) -> None:
        self.n_steps = n_steps
        self.variation_pct = variation_pct

    def analyze_parameter(
        self,
        data: pd.DataFrame,
        strategy_class: type[BaseStrategy],
        base_params: dict[str, Any],
        param_name: str,
        param_range: tuple[float, float] | None = None,
        benchmark: pd.Series | None = None,
    ) -> dict[str, Any]:
        base_value = base_params.get(param_name)
        if base_value is None:
            return {"error": f"Parameter {param_name} not found"}

        if param_range:
            low, high = param_range
        else:
            low = base_value * (1 - self.variation_pct)
            high = base_value * (1 + self.variation_pct)

        if isinstance(base_value, int):
            test_values = [int(v) for v in np.linspace(low, high, self.n_steps)]
            test_values = sorted(set(test_values))
        else:
            test_values = np.linspace(low, high, self.n_steps).tolist()

        results: list[dict[str, Any]] = []
        engine = BacktestEngine()

        for value in test_values:
            params = base_params.copy()
            params[param_name] = value

            try:
                strategy = strategy_class(**params)
                result = engine.run(data, strategy, benchmark)
                results.append({
                    "value": value,
                    "sharpe_ratio": result.sharpe_ratio,
                    "total_return_pct": result.total_return_pct,
                    "max_drawdown": result.max_drawdown,
                    "win_rate": result.win_rate,
                    "total_trades": result.total_trades,
                    "profit_factor": result.profit_factor,
                    "volatility": result.volatility,
                })
            except Exception as e:
                logger.debug(f"Sensitivity test failed for {param_name}={value}: {e}")
                results.append({"value": value, "error": str(e)})

        valid_results = [r for r in results if "sharpe_ratio" in r]
        if not valid_results:
            return {"param_name": param_name, "results": results, "analysis": {}}

        sharpes = [r["sharpe_ratio"] for r in valid_results]
        values = [r["value"] for r in valid_results]

        return {
            "param_name": param_name,
            "base_value": base_value,
            "range": [low, high],
            "results": results,
            "analysis": {
                "best_value": values[int(np.argmax(sharpes))],
                "best_sharpe": max(sharpes),
                "worst_value": values[int(np.argmin(sharpes))],
                "worst_sharpe": min(sharpes),
                "sharpe_range": max(sharpes) - min(sharpes),
                "sensitivity": float(np.std(sharpes)),
                "monotonic": self._is_monotonic(sharpes),
            },
        }

    def analyze_all_parameters(
        self,
        data: pd.DataFrame,
        strategy_class: type[BaseStrategy],
        base_params: dict[str, Any],
        param_ranges: dict[str, tuple[float, float]] | None = None,
        benchmark: pd.Series | None = None,
    ) -> dict[str, Any]:
        param_ranges = param_ranges or {}
        results: dict[str, Any] = {}

        numeric_params = {
            k: v for k, v in base_params.items()
            if isinstance(v, (int, float)) and k != "name"
        }

        for param_name in numeric_params:
            logger.info(f"Analyzing sensitivity of {param_name}")
            param_range = param_ranges.get(param_name)
            results[param_name] = self.analyze_parameter(
                data, strategy_class, base_params, param_name, param_range, benchmark,
            )

        sensitivities = {}
        for name, result in results.items():
            analysis = result.get("analysis", {})
            sensitivities[name] = analysis.get("sensitivity", 0)

        ranked = sorted(sensitivities.items(), key=lambda x: x[1], reverse=True)

        return {
            "parameters": results,
            "sensitivity_ranking": [
                {"param": name, "sensitivity": round(score, 4)}
                for name, score in ranked
            ],
            "most_sensitive": ranked[0][0] if ranked else "none",
            "least_sensitive": ranked[-1][0] if ranked else "none",
        }

    def cost_sensitivity(
        self,
        data: pd.DataFrame,
        strategy: BaseStrategy,
        cost_multipliers: list[float] | None = None,
        benchmark: pd.Series | None = None,
    ) -> dict[str, Any]:
        from nexus.backtest.costs import TransactionCostModel

        if cost_multipliers is None:
            cost_multipliers = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0]

        results: list[dict[str, Any]] = []
        base_model = TransactionCostModel()

        for mult in cost_multipliers:
            cost_model = TransactionCostModel(
                commission_pct=base_model.commission_pct * mult,
                spread_bps=base_model.spread_bps * mult,
                market_impact_factor=base_model.market_impact_factor * mult,
            )
            engine = BacktestEngine(cost_model=cost_model)

            try:
                result = engine.run(data, strategy, benchmark)
                results.append({
                    "cost_multiplier": mult,
                    "sharpe_ratio": result.sharpe_ratio,
                    "total_return_pct": result.total_return_pct,
                    "max_drawdown": result.max_drawdown,
                    "profit_factor": result.profit_factor,
                })
            except Exception as e:
                results.append({"cost_multiplier": mult, "error": str(e)})

        valid = [r for r in results if "sharpe_ratio" in r]
        breakeven_mult = None
        for r in valid:
            if r["sharpe_ratio"] <= 0:
                breakeven_mult = r["cost_multiplier"]
                break

        return {
            "results": results,
            "breakeven_cost_multiplier": breakeven_mult,
            "cost_resilient": breakeven_mult is None or breakeven_mult > 2.0,
        }

    @staticmethod
    def _is_monotonic(values: list[float]) -> bool:
        if len(values) < 3:
            return True
        diffs = [values[i + 1] - values[i] for i in range(len(values) - 1)]
        return all(d >= 0 for d in diffs) or all(d <= 0 for d in diffs)
