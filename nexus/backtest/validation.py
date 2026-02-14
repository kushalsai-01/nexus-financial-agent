from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from nexus.backtest.metrics import compute_all_metrics, compute_sharpe
from nexus.backtest.strategy import BaseStrategy
from nexus.core.logging import get_logger

logger = get_logger("backtest.validation")


class WalkForwardValidator:
    def __init__(
        self,
        train_days: int = 504,
        test_days: int = 63,
        step_days: int = 21,
        min_train_days: int = 252,
        expanding: bool = False,
    ) -> None:
        self.train_days = train_days
        self.test_days = test_days
        self.step_days = step_days
        self.min_train_days = min_train_days
        self.expanding = expanding

    def generate_windows(self, data: pd.DataFrame) -> list[dict[str, Any]]:
        n = len(data)
        windows: list[dict[str, Any]] = []
        idx = 0

        while True:
            if self.expanding:
                train_start = 0
            else:
                train_start = idx

            train_end = idx + self.train_days
            test_start = train_end
            test_end = min(test_start + self.test_days, n)

            if test_start >= n:
                break

            actual_train = train_end - train_start
            if actual_train < self.min_train_days:
                idx += self.step_days
                continue

            windows.append({
                "fold": len(windows),
                "train_start_idx": train_start,
                "train_end_idx": train_end,
                "test_start_idx": test_start,
                "test_end_idx": test_end,
                "train_start": data.index[train_start] if hasattr(data.index, "date") else train_start,
                "train_end": data.index[min(train_end - 1, n - 1)] if hasattr(data.index, "date") else train_end,
                "test_start": data.index[test_start] if hasattr(data.index, "date") else test_start,
                "test_end": data.index[min(test_end - 1, n - 1)] if hasattr(data.index, "date") else test_end,
            })

            idx += self.step_days

            if test_end >= n:
                break

        return windows

    def validate(
        self,
        data: pd.DataFrame,
        strategy: BaseStrategy,
        run_backtest_fn: Any,
    ) -> dict[str, Any]:
        windows = self.generate_windows(data)
        if not windows:
            return {"error": "No valid windows", "folds": []}

        fold_results: list[dict[str, Any]] = []
        is_sharpes: list[float] = []
        oos_sharpes: list[float] = []

        for window in windows:
            train_data = data.iloc[window["train_start_idx"]:window["train_end_idx"]]
            test_data = data.iloc[window["test_start_idx"]:window["test_end_idx"]]

            try:
                train_signals = strategy.generate_signals(train_data)
                train_returns = train_data["close"].pct_change().dropna()
                is_sharpe = compute_sharpe(train_returns * train_signals["signal"].reindex(train_returns.index).fillna(0))

                test_signals = strategy.generate_signals(test_data)
                test_returns = test_data["close"].pct_change().dropna()
                oos_sharpe = compute_sharpe(test_returns * test_signals["signal"].reindex(test_returns.index).fillna(0))

                is_sharpes.append(is_sharpe)
                oos_sharpes.append(oos_sharpe)

                fold_results.append({
                    "fold": window["fold"],
                    "train_start": str(window["train_start"]),
                    "train_end": str(window["train_end"]),
                    "test_start": str(window["test_start"]),
                    "test_end": str(window["test_end"]),
                    "is_sharpe": is_sharpe,
                    "oos_sharpe": oos_sharpe,
                    "sharpe_decay": is_sharpe - oos_sharpe if is_sharpe != 0 else 0,
                })
            except Exception as e:
                logger.error(f"Fold {window['fold']} failed: {e}")
                fold_results.append({
                    "fold": window["fold"],
                    "error": str(e),
                })

        avg_is = np.mean(is_sharpes) if is_sharpes else 0
        avg_oos = np.mean(oos_sharpes) if oos_sharpes else 0
        avg_decay = avg_is - avg_oos

        overfit_detected = avg_decay > 0.5 and avg_is > 0.5

        return {
            "total_folds": len(windows),
            "successful_folds": len(is_sharpes),
            "avg_is_sharpe": float(avg_is),
            "avg_oos_sharpe": float(avg_oos),
            "avg_sharpe_decay": float(avg_decay),
            "overfit_detected": overfit_detected,
            "oos_sharpe_std": float(np.std(oos_sharpes)) if oos_sharpes else 0,
            "pct_positive_oos": float(sum(1 for s in oos_sharpes if s > 0) / len(oos_sharpes)) if oos_sharpes else 0,
            "folds": fold_results,
        }


class LookaheadBiasChecker:
    @staticmethod
    def check_timestamp_ordering(data: pd.DataFrame) -> bool:
        if not isinstance(data.index, pd.DatetimeIndex):
            return True
        return bool(data.index.is_monotonic_increasing)

    @staticmethod
    def check_signal_timing(signals: pd.DataFrame, data: pd.DataFrame) -> list[str]:
        violations: list[str] = []

        if "signal" not in signals.columns:
            return violations

        for i in range(1, len(signals)):
            if signals["signal"].iloc[i] != 0:
                signal_time = signals.index[i]
                if i > 0:
                    data_time = data.index[i - 1]
                    if hasattr(signal_time, "date") and hasattr(data_time, "date"):
                        if signal_time < data_time:
                            violations.append(f"Signal at {signal_time} uses data from {data_time}")

        return violations

    @staticmethod
    def shuffle_future_test(
        data: pd.DataFrame,
        strategy: BaseStrategy,
        n_shuffles: int = 10,
    ) -> dict[str, Any]:
        original_signals = strategy.generate_signals(data)
        original_signal_sum = original_signals["signal"].abs().sum()

        shuffle_results: list[float] = []
        for _ in range(n_shuffles):
            shuffled = data.copy()
            future_half = len(shuffled) // 2
            future_block = shuffled.iloc[future_half:].copy()
            future_block = future_block.sample(frac=1)
            future_block.index = shuffled.index[future_half:]
            shuffled.iloc[future_half:] = future_block.values

            shuffled_signals = strategy.generate_signals(shuffled)
            shuffle_results.append(float(shuffled_signals["signal"].abs().sum()))

        avg_shuffled = np.mean(shuffle_results)
        ratio = avg_shuffled / original_signal_sum if original_signal_sum > 0 else 1.0

        return {
            "original_signal_count": float(original_signal_sum),
            "avg_shuffled_signal_count": float(avg_shuffled),
            "ratio": float(ratio),
            "potential_lookahead": ratio > 0.9,
        }
