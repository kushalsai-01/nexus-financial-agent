from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from nexus.core.logging import get_logger

logger = get_logger("risk.scenarios")


class MonteCarloSimulator:
    def __init__(
        self,
        n_simulations: int = 10_000,
        seed: int = 42,
    ) -> None:
        self.n_simulations = n_simulations
        self.rng = np.random.RandomState(seed)

    def simulate_returns(
        self,
        historical_returns: pd.Series,
        horizon_days: int = 252,
        method: str = "parametric",
    ) -> np.ndarray:
        if method == "parametric":
            return self._parametric(historical_returns, horizon_days)
        elif method == "bootstrap":
            return self._bootstrap(historical_returns, horizon_days)
        elif method == "block_bootstrap":
            return self._block_bootstrap(historical_returns, horizon_days)
        else:
            raise ValueError(f"Unknown method: {method}")

    def _parametric(self, returns: pd.Series, horizon: int) -> np.ndarray:
        mu = float(returns.mean())
        sigma = float(returns.std())
        skew = float(returns.skew())
        kurt = float(returns.kurtosis())

        base = self.rng.normal(mu, sigma, (self.n_simulations, horizon))

        if abs(skew) > 0.5:
            skew_adj = skew / 6 * (base ** 2 - 1)
            kurt_adj = (kurt - 3) / 24 * (base ** 3 - 3 * base)
            base = base + sigma * (skew_adj + kurt_adj)

        return base

    def _bootstrap(self, returns: pd.Series, horizon: int) -> np.ndarray:
        values = returns.values
        indices = self.rng.randint(0, len(values), (self.n_simulations, horizon))
        return values[indices]

    def _block_bootstrap(
        self,
        returns: pd.Series,
        horizon: int,
        block_size: int = 21,
    ) -> np.ndarray:
        values = returns.values
        n = len(values)
        result = np.zeros((self.n_simulations, horizon))

        for sim in range(self.n_simulations):
            idx = 0
            while idx < horizon:
                start = self.rng.randint(0, max(1, n - block_size))
                block = values[start:start + block_size]
                end = min(idx + len(block), horizon)
                result[sim, idx:end] = block[:end - idx]
                idx = end

        return result

    def simulate_portfolio_paths(
        self,
        returns: pd.Series,
        initial_value: float = 100_000,
        horizon_days: int = 252,
        method: str = "parametric",
    ) -> np.ndarray:
        sim_returns = self.simulate_returns(returns, horizon_days, method)
        cumulative = np.cumprod(1 + sim_returns, axis=1)
        return initial_value * cumulative

    def compute_distribution_stats(
        self,
        paths: np.ndarray,
        initial_value: float = 100_000,
    ) -> dict[str, Any]:
        terminal_values = paths[:, -1]
        terminal_returns = (terminal_values / initial_value) - 1

        percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
        pct_values = {
            f"p{p}": float(np.percentile(terminal_returns, p))
            for p in percentiles
        }

        peak = np.maximum.accumulate(paths, axis=1)
        drawdowns = (paths - peak) / peak
        max_drawdowns = drawdowns.min(axis=1)

        return {
            "mean_return": float(terminal_returns.mean()),
            "median_return": float(np.median(terminal_returns)),
            "std_return": float(terminal_returns.std()),
            "skewness": float(pd.Series(terminal_returns).skew()),
            "kurtosis": float(pd.Series(terminal_returns).kurtosis()),
            "prob_loss": float((terminal_returns < 0).mean()),
            "prob_loss_10pct": float((terminal_returns < -0.10).mean()),
            "prob_loss_20pct": float((terminal_returns < -0.20).mean()),
            "prob_gain_10pct": float((terminal_returns > 0.10).mean()),
            "prob_gain_20pct": float((terminal_returns > 0.20).mean()),
            "percentiles": pct_values,
            "mean_max_drawdown": float(max_drawdowns.mean()),
            "worst_max_drawdown": float(max_drawdowns.min()),
            "mean_terminal_value": float(terminal_values.mean()),
            "median_terminal_value": float(np.median(terminal_values)),
            "var_95": float(np.percentile(terminal_returns, 5)),
            "cvar_95": float(terminal_returns[terminal_returns <= np.percentile(terminal_returns, 5)].mean()),
        }

    def run_full_simulation(
        self,
        returns: pd.Series,
        initial_value: float = 100_000,
        horizon_days: int = 252,
    ) -> dict[str, Any]:
        results: dict[str, Any] = {}

        for method in ["parametric", "bootstrap", "block_bootstrap"]:
            paths = self.simulate_portfolio_paths(
                returns, initial_value, horizon_days, method,
            )
            stats = self.compute_distribution_stats(paths, initial_value)
            results[method] = stats
            logger.info(
                f"Monte Carlo ({method}): mean={stats['mean_return']:.2%}, "
                f"P(loss)={stats['prob_loss']:.1%}, VaR95={stats['var_95']:.2%}"
            )

        return results
