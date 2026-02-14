from __future__ import annotations

import asyncio
import sys
from datetime import datetime, timedelta

from nexus.core.config import get_config
from nexus.core.logging import get_logger, setup_logging
from nexus.data.providers.market import MarketDataProvider
from nexus.data.storage.postgres import PostgresStorage

logger = get_logger("scripts.seed_data")

DEFAULT_TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA",
    "META", "TSLA", "JPM", "V", "JNJ",
    "WMT", "PG", "UNH", "HD", "BAC",
    "XOM", "KO", "PFE", "ABBV", "SPY",
]


async def seed_market_data(
    tickers: list[str] | None = None,
    days: int = 365,
) -> None:
    tickers = tickers or DEFAULT_TICKERS
    market = MarketDataProvider()
    storage = PostgresStorage()
    storage.create_tables()

    end = datetime.now()
    start = end - timedelta(days=days)

    logger.info(f"Seeding market data for {len(tickers)} tickers, {days} days")

    for ticker in tickers:
        try:
            df = await market.fetch_bars(ticker, start, end)
            if df.empty:
                logger.warning(f"No data for {ticker}")
                continue

            records = []
            for _, row in df.iterrows():
                records.append(
                    {
                        "ticker": ticker,
                        "timestamp": row["timestamp"],
                        "open": float(row["open"]),
                        "high": float(row["high"]),
                        "low": float(row["low"]),
                        "close": float(row["close"]),
                        "volume": int(row["volume"]),
                    }
                )

            count = storage.save_prices(records)
            logger.info(f"Seeded {count} price records for {ticker}")
        except Exception as e:
            logger.error(f"Failed to seed {ticker}: {e}")


def main() -> None:
    setup_logging()
    tickers = sys.argv[1:] if len(sys.argv) > 1 else None
    asyncio.run(seed_market_data(tickers))
    logger.info("Seed data complete")


if __name__ == "__main__":
    main()
