from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd

from nexus.backtest.engine import BacktestConfig, BacktestEngine
from nexus.backtest.strategy import BaseStrategy, MomentumStrategy, MeanReversionStrategy
from nexus.core.logging import get_logger

logger = get_logger("analysis.optimization")


class HyperparameterOptimizer:
    def __init__(
        self,
        n_trials: int = 100,
        n_jobs: int = 1,
        seed: int = 42,
        max_drawdown_limit: float = 0.20,
        min_win_rate: float = 0.45,
        min_trades: int = 30,
    ) -> None:
        self.n_trials = n_trials
        self.n_jobs = n_jobs
        self.seed = seed
        self.max_drawdown_limit = max_drawdown_limit
        self.min_win_rate = min_win_rate
        self.min_trades = min_trades

    def optimize_strategy(
        self,
        data: pd.DataFrame,
        strategy_class: type[BaseStrategy],
        param_space: dict[str, dict[str, Any]],
        objective: str = "sharpe_ratio",
        benchmark: pd.Series | None = None,
    ) -> dict[str, Any]:
        try:
            import optuna
            optuna.logging.set_verbosity(optuna.logging.WARNING)
        except ImportError:
            logger.warning("optuna not installed, using grid search fallback")
            return self._grid_search(data, strategy_class, param_space, objective, benchmark)

        def objective_fn(trial: Any) -> float:
            params = self._sample_params(trial, param_space)

            try:
                strategy = strategy_class(**params)
                engine = BacktestEngine()
                result = engine.run(data, strategy, benchmark)

                if abs(result.max_drawdown) > self.max_drawdown_limit:
                    return -10.0
                if result.win_rate < self.min_win_rate and result.total_trades > self.min_trades:
                    return -5.0
                if result.total_trades < self.min_trades:
                    return -3.0

                if objective == "sharpe_ratio":
                    return result.sharpe_ratio
                elif objective == "sortino_ratio":
                    return result.sortino_ratio
                elif objective == "calmar_ratio":
                    return result.calmar_ratio
                elif objective == "total_return":
                    return result.total_return_pct
                elif objective == "profit_factor":
                    return result.profit_factor
                else:
                    return result.sharpe_ratio
            except Exception as e:
                logger.debug(f"Trial failed: {e}")
                return -100.0

        sampler = optuna.samplers.TPESampler(seed=self.seed)
        study = optuna.create_study(
            direction="maximize",
            sampler=sampler,
            study_name=f"optimize_{strategy_class.__name__}",
        )
        study.optimize(objective_fn, n_trials=self.n_trials, n_jobs=self.n_jobs)

        best = study.best_trial
        logger.info(f"Best {objective}: {best.value:.4f} with params: {best.params}")

        return {
            "best_params": best.params,
            "best_value": best.value,
            "objective": objective,
            "n_trials": self.n_trials,
            "all_trials": [
                {
                    "number": t.number,
                    "value": t.value,
                    "params": t.params,
                    "state": str(t.state),
                }
                for t in study.trials
            ],
        }

    def optimize_agent_weights(
        self,
        data: pd.DataFrame,
        agent_names: list[str],
        run_backtest_fn: Callable,
        n_trials: int | None = None,
    ) -> dict[str, Any]:
        try:
            import optuna
            optuna.logging.set_verbosity(optuna.logging.WARNING)
        except ImportError:
            logger.warning("optuna not installed")
            return {"error": "optuna not installed"}

        trials = n_trials or self.n_trials

        def objective_fn(trial: Any) -> float:
            weights: dict[str, float] = {}
            for name in agent_names:
                weights[name] = trial.suggest_float(f"weight_{name}", 0.0, 1.0)

            total = sum(weights.values())
            if total > 0:
                weights = {k: v / total for k, v in weights.items()}

            try:
                result = run_backtest_fn(data, weights)
                if abs(result.max_drawdown) > self.max_drawdown_limit:
                    return -10.0
                return result.sharpe_ratio
            except Exception:
                return -100.0

        sampler = optuna.samplers.TPESampler(seed=self.seed)
        study = optuna.create_study(direction="maximize", sampler=sampler)
        study.optimize(objective_fn, n_trials=trials)

        best_weights = {}
        total = 0.0
        for name in agent_names:
            w = study.best_params.get(f"weight_{name}", 0)
            best_weights[name] = w
            total += w
        if total > 0:
            best_weights = {k: round(v / total, 4) for k, v in best_weights.items()}

        return {
            "best_weights": best_weights,
            "best_sharpe": study.best_value,
            "n_trials": trials,
        }

    @staticmethod
    def _sample_params(trial: Any, param_space: dict[str, dict[str, Any]]) -> dict[str, Any]:
        params: dict[str, Any] = {}
        for name, spec in param_space.items():
            ptype = spec.get("type", "float")
            if ptype == "int":
                params[name] = trial.suggest_int(name, spec["low"], spec["high"], step=spec.get("step", 1))
            elif ptype == "float":
                params[name] = trial.suggest_float(name, spec["low"], spec["high"], log=spec.get("log", False))
            elif ptype == "categorical":
                params[name] = trial.suggest_categorical(name, spec["choices"])
        return params

    def _grid_search(
        self,
        data: pd.DataFrame,
        strategy_class: type[BaseStrategy],
        param_space: dict[str, dict[str, Any]],
        objective: str,
        benchmark: pd.Series | None,
    ) -> dict[str, Any]:
        from itertools import product

        param_names = list(param_space.keys())
        param_values: list[list[Any]] = []
        for name in param_names:
            spec = param_space[name]
            ptype = spec.get("type", "float")
            if ptype == "categorical":
                param_values.append(spec["choices"])
            else:
                low, high = spec["low"], spec["high"]
                steps = spec.get("n_grid", 5)
                if ptype == "int":
                    param_values.append(list(range(low, high + 1, max(1, (high - low) // steps))))
                else:
                    param_values.append(np.linspace(low, high, steps).tolist())

        best_score = -float("inf")
        best_params: dict[str, Any] = {}

        for combo in product(*param_values):
            params = dict(zip(param_names, combo))
            try:
                strategy = strategy_class(**params)
                engine = BacktestEngine()
                result = engine.run(data, strategy, benchmark)

                score = getattr(result, objective, result.sharpe_ratio)
                if score > best_score:
                    best_score = score
                    best_params = params
            except Exception:
                continue

        return {
            "best_params": best_params,
            "best_value": best_score,
            "objective": objective,
            "method": "grid_search",
        }
