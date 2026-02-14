from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd


def compute_total_return(equity_curve: pd.Series) -> float:
    if len(equity_curve) < 2 or equity_curve.iloc[0] == 0:
        return 0.0
    return (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1.0


def compute_cagr(equity_curve: pd.Series, trading_days: int = 252) -> float:
    if len(equity_curve) < 2 or equity_curve.iloc[0] <= 0:
        return 0.0
    total_return = equity_curve.iloc[-1] / equity_curve.iloc[0]
    years = len(equity_curve) / trading_days
    if years <= 0 or total_return <= 0:
        return 0.0
    return total_return ** (1.0 / years) - 1.0


def compute_volatility(returns: pd.Series, trading_days: int = 252) -> float:
    if len(returns) < 2:
        return 0.0
    return float(returns.std() * math.sqrt(trading_days))


def compute_sharpe(returns: pd.Series, risk_free_rate: float = 0.04, trading_days: int = 252) -> float:
    if len(returns) < 2:
        return 0.0
    daily_rf = (1 + risk_free_rate) ** (1 / trading_days) - 1
    excess = returns - daily_rf
    std = excess.std()
    if std == 0:
        return 0.0
    return float(excess.mean() / std * math.sqrt(trading_days))


def compute_sortino(returns: pd.Series, risk_free_rate: float = 0.04, trading_days: int = 252) -> float:
    if len(returns) < 2:
        return 0.0
    daily_rf = (1 + risk_free_rate) ** (1 / trading_days) - 1
    excess = returns - daily_rf
    downside = excess[excess < 0]
    if len(downside) < 1:
        return 0.0
    downside_std = float(downside.std())
    if downside_std == 0:
        return 0.0
    return float(excess.mean() / downside_std * math.sqrt(trading_days))


def compute_max_drawdown(equity_curve: pd.Series) -> tuple[float, int]:
    if len(equity_curve) < 2:
        return 0.0, 0
    peak = equity_curve.expanding().max()
    drawdown = (equity_curve - peak) / peak
    max_dd = float(drawdown.min())

    dd_start = None
    dd_end = None
    max_duration = 0
    current_start = None

    for i in range(len(drawdown)):
        if drawdown.iloc[i] < 0:
            if current_start is None:
                current_start = i
        else:
            if current_start is not None:
                duration = i - current_start
                if duration > max_duration:
                    max_duration = duration
                current_start = None

    if current_start is not None:
        duration = len(drawdown) - current_start
        if duration > max_duration:
            max_duration = duration

    return max_dd, max_duration


def compute_calmar(equity_curve: pd.Series, trading_days: int = 252) -> float:
    cagr = compute_cagr(equity_curve, trading_days)
    max_dd, _ = compute_max_drawdown(equity_curve)
    if max_dd == 0:
        return 0.0
    return cagr / abs(max_dd)


def compute_win_rate(trade_pnls: list[float]) -> float:
    if len(trade_pnls) == 0:
        return 0.0
    wins = sum(1 for p in trade_pnls if p > 0)
    return wins / len(trade_pnls)


def compute_profit_factor(trade_pnls: list[float]) -> float:
    if len(trade_pnls) == 0:
        return 0.0
    gross_profit = sum(p for p in trade_pnls if p > 0)
    gross_loss = abs(sum(p for p in trade_pnls if p < 0))
    if gross_loss == 0:
        return float("inf") if gross_profit > 0 else 0.0
    return gross_profit / gross_loss


def compute_avg_win_loss(trade_pnls: list[float]) -> tuple[float, float]:
    wins = [p for p in trade_pnls if p > 0]
    losses = [p for p in trade_pnls if p < 0]
    avg_win = sum(wins) / len(wins) if wins else 0.0
    avg_loss = sum(losses) / len(losses) if losses else 0.0
    return avg_win, avg_loss


def compute_rolling_sharpe(
    returns: pd.Series,
    window: int = 126,
    risk_free_rate: float = 0.04,
    trading_days: int = 252,
) -> pd.Series:
    daily_rf = (1 + risk_free_rate) ** (1 / trading_days) - 1
    excess = returns - daily_rf
    rolling_mean = excess.rolling(window).mean()
    rolling_std = excess.rolling(window).std()
    rolling_sharpe = rolling_mean / rolling_std * math.sqrt(trading_days)
    return rolling_sharpe


def compute_turnover(weights: pd.DataFrame) -> pd.Series:
    if weights.empty:
        return pd.Series(dtype=float)
    return weights.diff().abs().sum(axis=1) / 2


def compute_exposure_time(positions_active: pd.Series) -> float:
    if len(positions_active) == 0:
        return 0.0
    return float(positions_active.astype(bool).mean())


def compute_monthly_returns(equity_curve: pd.Series) -> pd.Series:
    if not isinstance(equity_curve.index, pd.DatetimeIndex):
        return pd.Series(dtype=float)
    monthly = equity_curve.resample("ME").last()
    return monthly.pct_change().dropna()


def compute_yearly_returns(equity_curve: pd.Series) -> pd.Series:
    if not isinstance(equity_curve.index, pd.DatetimeIndex):
        return pd.Series(dtype=float)
    yearly = equity_curve.resample("YE").last()
    return yearly.pct_change().dropna()


def compute_beta(returns: pd.Series, benchmark_returns: pd.Series) -> float:
    if len(returns) < 2 or len(benchmark_returns) < 2:
        return 0.0
    aligned = pd.concat([returns, benchmark_returns], axis=1).dropna()
    if len(aligned) < 2:
        return 0.0
    cov = aligned.cov()
    benchmark_var = cov.iloc[1, 1]
    if benchmark_var == 0:
        return 0.0
    return float(cov.iloc[0, 1] / benchmark_var)


def compute_alpha(
    returns: pd.Series,
    benchmark_returns: pd.Series,
    risk_free_rate: float = 0.04,
    trading_days: int = 252,
) -> float:
    beta = compute_beta(returns, benchmark_returns)
    portfolio_return = float(returns.mean() * trading_days)
    benchmark_return = float(benchmark_returns.mean() * trading_days)
    return portfolio_return - (risk_free_rate + beta * (benchmark_return - risk_free_rate))


def compute_information_ratio(returns: pd.Series, benchmark_returns: pd.Series, trading_days: int = 252) -> float:
    aligned = pd.concat([returns, benchmark_returns], axis=1).dropna()
    if len(aligned) < 2:
        return 0.0
    active = aligned.iloc[:, 0] - aligned.iloc[:, 1]
    te = active.std()
    if te == 0:
        return 0.0
    return float(active.mean() / te * math.sqrt(trading_days))


def compute_all_metrics(
    equity_curve: pd.Series,
    trade_pnls: list[float],
    benchmark_returns: pd.Series | None = None,
    risk_free_rate: float = 0.04,
    trading_days: int = 252,
) -> dict[str, Any]:
    returns = equity_curve.pct_change().dropna()
    max_dd, max_dd_duration = compute_max_drawdown(equity_curve)
    avg_win, avg_loss = compute_avg_win_loss(trade_pnls)

    metrics: dict[str, Any] = {
        "total_return": compute_total_return(equity_curve),
        "cagr": compute_cagr(equity_curve, trading_days),
        "volatility": compute_volatility(returns, trading_days),
        "sharpe_ratio": compute_sharpe(returns, risk_free_rate, trading_days),
        "sortino_ratio": compute_sortino(returns, risk_free_rate, trading_days),
        "max_drawdown": max_dd,
        "max_drawdown_duration_days": max_dd_duration,
        "calmar_ratio": compute_calmar(equity_curve, trading_days),
        "win_rate": compute_win_rate(trade_pnls),
        "profit_factor": compute_profit_factor(trade_pnls),
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "total_trades": len(trade_pnls),
        "best_trade": max(trade_pnls) if len(trade_pnls) > 0 else 0.0,
        "worst_trade": min(trade_pnls) if len(trade_pnls) > 0 else 0.0,
        "avg_trade_pnl": sum(trade_pnls) / len(trade_pnls) if len(trade_pnls) > 0 else 0.0,
    }

    if benchmark_returns is not None:
        metrics["beta"] = compute_beta(returns, benchmark_returns)
        metrics["alpha"] = compute_alpha(returns, benchmark_returns, risk_free_rate, trading_days)
        metrics["information_ratio"] = compute_information_ratio(returns, benchmark_returns, trading_days)

    return metrics
