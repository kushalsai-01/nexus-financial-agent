from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any
from uuid import uuid4

import numpy as np
import pandas as pd

from nexus.backtest.costs import TransactionCostModel
from nexus.backtest.metrics import (
    compute_all_metrics,
    compute_max_drawdown,
    compute_total_return,
)
from nexus.backtest.regime import MarketRegime, RegimeClassifier
from nexus.backtest.strategy import BaseStrategy
from nexus.core.exceptions import DataValidationError, ExecutionError
from nexus.core.logging import get_logger
from nexus.core.types import (
    BacktestResult,
    OrderSide,
    Trade,
)

logger = get_logger("backtest.engine")


@dataclass
class BacktestConfig:
    initial_capital: float = 100_000.0
    max_position_pct: float = 0.15
    max_total_exposure: float = 1.0
    stop_loss_pct: float = 0.08
    trailing_stop_pct: float = 0.15
    atr_stop_multiplier: float = 2.0
    time_stop_days: int = 60
    rebalance_frequency: str = "daily"
    seed: int = 42


class BacktestEngine:
    def __init__(
        self,
        config: BacktestConfig | None = None,
        cost_model: TransactionCostModel | None = None,
    ) -> None:
        self.config = config or BacktestConfig()
        self.cost_model = cost_model or TransactionCostModel()
        self._rng = np.random.RandomState(self.config.seed)

    def run(
        self,
        data: pd.DataFrame,
        strategy: BaseStrategy,
        benchmark: pd.Series | None = None,
    ) -> BacktestResult:
        self._validate_data(data)

        signals = strategy.generate_signals(data)
        self._validate_no_lookahead(data, signals)

        return self._simulate(data, signals, strategy.name, benchmark)

    def _validate_data(self, data: pd.DataFrame) -> None:
        required = {"open", "high", "low", "close", "volume"}
        missing = required - set(data.columns)
        if missing:
            raise DataValidationError(
                f"Missing columns: {missing}",
                code="DATA_VALIDATION_FAILED",
            )

        if not isinstance(data.index, pd.DatetimeIndex):
            raise DataValidationError(
                "Data index must be DatetimeIndex",
                code="DATA_VALIDATION_FAILED",
            )

        if not data.index.is_monotonic_increasing:
            raise DataValidationError(
                "Data must be sorted by date ascending",
                code="DATA_VALIDATION_FAILED",
            )

        if data["close"].isna().sum() > 0:
            logger.warning("NaN values detected in close prices, forward-filling")
            data["close"] = data["close"].ffill()

    def _validate_no_lookahead(self, data: pd.DataFrame, signals: pd.DataFrame) -> None:
        if len(signals) != len(data):
            raise DataValidationError(
                f"Signal length ({len(signals)}) != data length ({len(data)})",
                code="DATA_VALIDATION_FAILED",
            )

    def _simulate(
        self,
        data: pd.DataFrame,
        signals: pd.DataFrame,
        strategy_name: str,
        benchmark: pd.Series | None,
    ) -> BacktestResult:
        cash = self.config.initial_capital
        positions: dict[str, _SimPosition] = {}
        trades: list[Trade] = []
        equity_curve: list[float] = []
        dates: list[datetime] = []

        close_col = data["close"]
        volume_col = data["volume"]

        atr = self._compute_atr(data, 14)

        for i in range(len(data)):
            current_date = data.index[i]
            current_price = float(close_col.iloc[i])
            current_volume = float(volume_col.iloc[i])
            current_atr = float(atr.iloc[i]) if not np.isnan(atr.iloc[i]) else current_price * 0.02

            positions_value = sum(
                p.quantity * current_price for p in positions.values()
            )
            total_value = cash + positions_value

            exit_tickers: list[str] = []
            for ticker, pos in positions.items():
                pos.current_price = current_price
                pos.days_held += 1
                pos.peak_price = max(pos.peak_price, current_price)

                should_exit, reason = self._check_exit(pos, current_price, current_atr)
                if should_exit:
                    exit_tickers.append(ticker)
                    pnl = (current_price - pos.entry_price) * pos.quantity
                    costs = self.cost_model.total_cost(
                        current_price, pos.quantity, current_volume,
                    )
                    net_pnl = pnl - costs["total_cost"]
                    cash += current_price * pos.quantity - costs["total_cost"]

                    trades.append(Trade(
                        trade_id=str(uuid4()),
                        ticker=ticker,
                        side=OrderSide.SELL,
                        quantity=pos.quantity,
                        price=current_price,
                        commission=costs["commission"],
                        slippage=costs["slippage_per_share"],
                        timestamp=current_date,
                        agent_name=strategy_name,
                        pnl=net_pnl,
                        pnl_pct=(net_pnl / (pos.entry_price * pos.quantity)) * 100 if pos.entry_price > 0 else 0,
                    ))

            for ticker in exit_tickers:
                del positions[ticker]

            if i > 0 and "signal" in signals.columns:
                signal_val = int(signals["signal"].iloc[i - 1])
                position_size = float(signals.get("position_size", pd.Series(0.5)).iloc[i - 1])
                ticker = data.get("ticker", pd.Series("ASSET")).iloc[0] if "ticker" in data.columns else "ASSET"

                if signal_val > 0 and ticker not in positions:
                    max_notional = total_value * self.config.max_position_pct
                    target_notional = total_value * position_size * self.config.max_position_pct
                    target_notional = min(target_notional, max_notional, cash * 0.95)

                    if target_notional > 0 and current_price > 0:
                        quantity = int(target_notional / current_price)
                        if quantity > 0:
                            costs = self.cost_model.total_cost(
                                current_price, quantity, current_volume,
                            )
                            total_buy_cost = current_price * quantity + costs["total_cost"]

                            if total_buy_cost <= cash:
                                cash -= total_buy_cost
                                positions[ticker] = _SimPosition(
                                    ticker=ticker,
                                    quantity=quantity,
                                    entry_price=current_price + costs["slippage_per_share"],
                                    current_price=current_price,
                                    entry_date=current_date,
                                    peak_price=current_price,
                                    atr_at_entry=current_atr,
                                )
                                trades.append(Trade(
                                    trade_id=str(uuid4()),
                                    ticker=ticker,
                                    side=OrderSide.BUY,
                                    quantity=quantity,
                                    price=current_price + costs["slippage_per_share"],
                                    commission=costs["commission"],
                                    slippage=costs["slippage_per_share"],
                                    timestamp=current_date,
                                    agent_name=strategy_name,
                                ))

                elif signal_val < 0 and ticker in positions:
                    pos = positions[ticker]
                    pnl = (current_price - pos.entry_price) * pos.quantity
                    costs = self.cost_model.total_cost(
                        current_price, pos.quantity, current_volume,
                    )
                    net_pnl = pnl - costs["total_cost"]
                    cash += current_price * pos.quantity - costs["total_cost"]

                    trades.append(Trade(
                        trade_id=str(uuid4()),
                        ticker=ticker,
                        side=OrderSide.SELL,
                        quantity=pos.quantity,
                        price=current_price,
                        commission=costs["commission"],
                        slippage=costs["slippage_per_share"],
                        timestamp=current_date,
                        agent_name=strategy_name,
                        pnl=net_pnl,
                        pnl_pct=(net_pnl / (pos.entry_price * pos.quantity)) * 100 if pos.entry_price > 0 else 0,
                    ))
                    del positions[ticker]

            positions_value = sum(p.quantity * current_price for p in positions.values())
            total_value = cash + positions_value
            equity_curve.append(total_value)
            dates.append(current_date)

        equity_series = pd.Series(equity_curve, index=dates)
        returns = equity_series.pct_change().dropna()
        trade_pnls = [t.pnl for t in trades if t.pnl is not None]

        metrics = compute_all_metrics(
            equity_curve=equity_series,
            trade_pnls=trade_pnls,
            benchmark_returns=benchmark,
        )

        max_dd, max_dd_duration = compute_max_drawdown(equity_series)
        peak = equity_series.expanding().max()
        drawdown_series = ((equity_series - peak) / peak).tolist()

        return BacktestResult(
            strategy_name=strategy_name,
            start_date=data.index[0].date(),
            end_date=data.index[-1].date(),
            initial_capital=self.config.initial_capital,
            final_value=equity_curve[-1] if equity_curve else self.config.initial_capital,
            total_return=equity_curve[-1] - self.config.initial_capital if equity_curve else 0,
            total_return_pct=metrics.get("total_return", 0) * 100,
            cagr=metrics.get("cagr", 0),
            sharpe_ratio=metrics.get("sharpe_ratio", 0),
            sortino_ratio=metrics.get("sortino_ratio", 0),
            calmar_ratio=metrics.get("calmar_ratio", 0),
            max_drawdown=max_dd,
            max_drawdown_duration_days=max_dd_duration,
            volatility=metrics.get("volatility", 0),
            win_rate=metrics.get("win_rate", 0),
            profit_factor=metrics.get("profit_factor", 0),
            total_trades=len(trades),
            avg_trade_pnl=float(np.mean(trade_pnls)) if trade_pnls else 0,
            best_trade=max(trade_pnls) if trade_pnls else 0,
            worst_trade=min(trade_pnls) if trade_pnls else 0,
            avg_holding_days=0,
            beta=metrics.get("beta", 0),
            alpha=metrics.get("alpha", 0),
            trades=trades,
            equity_curve=equity_curve,
            drawdown_series=drawdown_series,
        )

    def _check_exit(
        self,
        pos: _SimPosition,
        current_price: float,
        current_atr: float,
    ) -> tuple[bool, str]:
        pnl_pct = (current_price - pos.entry_price) / pos.entry_price

        if pnl_pct < -self.config.stop_loss_pct:
            return True, "stop_loss"

        atr_stop = pos.entry_price - self.config.atr_stop_multiplier * pos.atr_at_entry
        if current_price < atr_stop:
            return True, "atr_stop"

        trailing_stop = pos.peak_price * (1 - self.config.trailing_stop_pct)
        if current_price < trailing_stop and pnl_pct > 0:
            return True, "trailing_stop"

        if pos.days_held >= self.config.time_stop_days:
            return True, "time_stop"

        return False, ""

    @staticmethod
    def _compute_atr(data: pd.DataFrame, period: int = 14) -> pd.Series:
        high = data["high"]
        low = data["low"]
        close = data["close"]

        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(period).mean()


@dataclass
class _SimPosition:
    ticker: str
    quantity: int
    entry_price: float
    current_price: float
    entry_date: Any = None
    peak_price: float = 0.0
    days_held: int = 0
    atr_at_entry: float = 0.0
