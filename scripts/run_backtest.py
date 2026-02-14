from __future__ import annotations

import argparse
import asyncio
import sys
from datetime import datetime

from nexus.core.config import get_config
from nexus.core.logging import get_logger, setup_logging

logger = get_logger("scripts.run_backtest")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run NEXUS backtest")
    parser.add_argument(
        "--tickers",
        nargs="+",
        default=["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"],
        help="Tickers to backtest",
    )
    parser.add_argument("--start", type=str, default="2022-01-01", help="Start date")
    parser.add_argument("--end", type=str, default="2024-01-01", help="End date")
    parser.add_argument(
        "--capital", type=float, default=1_000_000.0, help="Initial capital"
    )
    parser.add_argument(
        "--benchmark", type=str, default="SPY", help="Benchmark ticker"
    )
    parser.add_argument(
        "--output", type=str, default="backtest_results.json", help="Output file"
    )
    return parser.parse_args()


async def run_backtest(args: argparse.Namespace) -> None:
    config = get_config()
    logger.info(f"Starting backtest: {args.tickers}")
    logger.info(f"Period: {args.start} to {args.end}")
    logger.info(f"Capital: ${args.capital:,.2f}")
    logger.info(f"Benchmark: {args.benchmark}")

    logger.info(
        "Backtest engine will be available in Prompt 3. "
        "This script provides the CLI interface."
    )


def main() -> None:
    setup_logging()
    args = parse_args()
    asyncio.run(run_backtest(args))


if __name__ == "__main__":
    main()
