from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from nexus.core.logging import get_logger

logger = get_logger("risk.portfolio")


class PortfolioRiskAnalyzer:
    def __init__(
        self,
        confidence_level: float = 0.95,
        risk_free_rate: float = 0.04,
        trading_days: int = 252,
    ) -> None:
        self.confidence_level = confidence_level
        self.risk_free_rate = risk_free_rate
        self.trading_days = trading_days

    def compute_var_historical(
        self,
        returns: pd.Series,
        horizon: int = 1,
        portfolio_value: float = 100_000,
    ) -> float:
        if len(returns) < 30:
            return 0.0
        if horizon > 1:
            rolling_returns = returns.rolling(horizon).sum().dropna()
            var_pct = float(np.percentile(rolling_returns, (1 - self.confidence_level) * 100))
        else:
            var_pct = float(np.percentile(returns, (1 - self.confidence_level) * 100))
        return abs(var_pct * portfolio_value)

    def compute_var_parametric(
        self,
        returns: pd.Series,
        horizon: int = 1,
        portfolio_value: float = 100_000,
    ) -> float:
        if len(returns) < 30:
            return 0.0
        mu = float(returns.mean()) * horizon
        sigma = float(returns.std()) * np.sqrt(horizon)
        z_score = stats.norm.ppf(1 - self.confidence_level)
        var_pct = mu + z_score * sigma
        return abs(var_pct * portfolio_value)

    def compute_var_monte_carlo(
        self,
        returns: pd.Series,
        horizon: int = 1,
        portfolio_value: float = 100_000,
        n_simulations: int = 10_000,
        seed: int = 42,
    ) -> float:
        if len(returns) < 30:
            return 0.0
        rng = np.random.RandomState(seed)
        mu = float(returns.mean())
        sigma = float(returns.std())

        simulated = rng.normal(mu, sigma, (n_simulations, horizon))
        cumulative = simulated.sum(axis=1)
        var_pct = float(np.percentile(cumulative, (1 - self.confidence_level) * 100))
        return abs(var_pct * portfolio_value)

    def compute_cvar(
        self,
        returns: pd.Series,
        horizon: int = 1,
        portfolio_value: float = 100_000,
    ) -> float:
        if len(returns) < 30:
            return 0.0
        if horizon > 1:
            returns = returns.rolling(horizon).sum().dropna()
        cutoff = float(np.percentile(returns, (1 - self.confidence_level) * 100))
        tail = returns[returns <= cutoff]
        if len(tail) == 0:
            return 0.0
        return abs(float(tail.mean()) * portfolio_value)

    def compute_var_all_horizons(
        self,
        returns: pd.Series,
        portfolio_value: float = 100_000,
    ) -> dict[str, dict[str, float]]:
        horizons = {"daily": 1, "10_day": 10, "30_day": 30}
        result: dict[str, dict[str, float]] = {}

        for name, h in horizons.items():
            result[name] = {
                "historical": self.compute_var_historical(returns, h, portfolio_value),
                "parametric": self.compute_var_parametric(returns, h, portfolio_value),
                "monte_carlo": self.compute_var_monte_carlo(returns, h, portfolio_value),
                "cvar": self.compute_cvar(returns, h, portfolio_value),
            }

        return result

    def compute_correlation_matrix(
        self,
        returns_dict: dict[str, pd.Series],
    ) -> pd.DataFrame:
        df = pd.DataFrame(returns_dict)
        return df.corr()

    def compute_covariance_matrix(
        self,
        returns_dict: dict[str, pd.Series],
        annualize: bool = True,
    ) -> pd.DataFrame:
        df = pd.DataFrame(returns_dict)
        cov = df.cov()
        if annualize:
            cov *= self.trading_days
        return cov

    def compute_portfolio_beta(
        self,
        portfolio_returns: pd.Series,
        benchmark_returns: pd.Series,
    ) -> float:
        if len(portfolio_returns) < 30 or len(benchmark_returns) < 30:
            return 1.0
        aligned = pd.DataFrame({"port": portfolio_returns, "bench": benchmark_returns}).dropna()
        if len(aligned) < 30:
            return 1.0
        cov = aligned.cov().iloc[0, 1]
        var = aligned["bench"].var()
        if var == 0:
            return 1.0
        return float(cov / var)

    def compute_portfolio_risk_summary(
        self,
        returns: pd.Series,
        positions: dict[str, float] | None = None,
        benchmark_returns: pd.Series | None = None,
        portfolio_value: float = 100_000,
    ) -> dict[str, Any]:
        vol = float(returns.std() * np.sqrt(self.trading_days))
        var_results = self.compute_var_all_horizons(returns, portfolio_value)

        summary: dict[str, Any] = {
            "annualized_volatility": round(vol, 4),
            "var": var_results,
            "skewness": float(returns.skew()),
            "kurtosis": float(returns.kurtosis()),
            "max_daily_loss": float(returns.min()),
            "max_daily_gain": float(returns.max()),
        }

        if benchmark_returns is not None:
            summary["beta"] = self.compute_portfolio_beta(returns, benchmark_returns)

        if positions:
            total = sum(abs(v) for v in positions.values())
            summary["gross_exposure"] = total / portfolio_value if portfolio_value > 0 else 0
            long_val = sum(v for v in positions.values() if v > 0)
            short_val = abs(sum(v for v in positions.values() if v < 0))
            summary["net_exposure"] = (long_val - short_val) / portfolio_value if portfolio_value > 0 else 0
            summary["long_exposure"] = long_val / portfolio_value if portfolio_value > 0 else 0
            summary["short_exposure"] = short_val / portfolio_value if portfolio_value > 0 else 0

        return summary
