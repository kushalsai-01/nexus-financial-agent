from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Any

import pandas as pd
import yfinance as yf

from nexus.core.config import get_config
from nexus.core.exceptions import DataFetchError
from nexus.core.logging import get_logger
from nexus.core.types import MarketData, TimeFrame
from nexus.data.providers.base import BaseDataProvider

logger = get_logger("data.providers.market")

_YF_TIMEFRAME_MAP: dict[TimeFrame, str] = {
    TimeFrame.MINUTE_1: "1m",
    TimeFrame.MINUTE_5: "5m",
    TimeFrame.MINUTE_15: "15m",
    TimeFrame.MINUTE_30: "30m",
    TimeFrame.HOUR_1: "1h",
    TimeFrame.HOUR_4: "1h",
    TimeFrame.DAILY: "1d",
    TimeFrame.WEEKLY: "1wk",
    TimeFrame.MONTHLY: "1mo",
}


class MarketDataProvider(BaseDataProvider):
    def __init__(self) -> None:
        config = get_config()
        super().__init__(
            cache_ttl=config.data.market.cache_ttl,
            rate_limit=config.data.market.rate_limit,
        )
        self._provider = config.data.market.provider

    async def fetch_bars(
        self,
        ticker: str,
        start: datetime,
        end: datetime,
        timeframe: TimeFrame = TimeFrame.DAILY,
    ) -> pd.DataFrame:
        cache_key = f"bars:{ticker}:{start.isoformat()}:{end.isoformat()}:{timeframe.value}"
        cached = self._cache.get(cache_key)
        if cached is not None:
            logger.debug(f"Cache hit for {cache_key}")
            return cached

        await self._rate_limiter.acquire()

        try:
            df = await self._fetch_yfinance_bars(ticker, start, end, timeframe)
            if df.empty:
                raise DataFetchError(f"No data returned for {ticker}")
            self._cache.set(cache_key, df)
            logger.info(f"Fetched {len(df)} bars for {ticker}")
            return df
        except DataFetchError:
            raise
        except Exception as e:
            raise DataFetchError(f"Failed to fetch bars for {ticker}: {e}") from e

    async def _fetch_yfinance_bars(
        self,
        ticker: str,
        start: datetime,
        end: datetime,
        timeframe: TimeFrame,
    ) -> pd.DataFrame:
        yf_interval = _YF_TIMEFRAME_MAP.get(timeframe, "1d")

        def _download() -> pd.DataFrame:
            tick = yf.Ticker(ticker)
            df = tick.history(start=start, end=end, interval=yf_interval)
            if df.empty:
                return pd.DataFrame()
            df.columns = [c.lower().replace(" ", "_") for c in df.columns]
            df.index.name = "timestamp"
            df = df.reset_index()
            df["ticker"] = ticker
            return df[["ticker", "timestamp", "open", "high", "low", "close", "volume"]]

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _download)

    async def fetch_quote(self, ticker: str) -> MarketData:
        cache_key = f"quote:{ticker}"
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        await self._rate_limiter.acquire()

        try:
            quote = await self._fetch_yfinance_quote(ticker)
            self._cache.set(cache_key, quote)
            return quote
        except Exception as e:
            raise DataFetchError(f"Failed to fetch quote for {ticker}: {e}") from e

    async def _fetch_yfinance_quote(self, ticker: str) -> MarketData:
        def _get_quote() -> dict[str, Any]:
            tick = yf.Ticker(ticker)
            info = tick.info
            hist = tick.history(period="1d")
            if hist.empty:
                raise DataFetchError(f"No quote data for {ticker}")
            latest = hist.iloc[-1]
            return {
                "ticker": ticker,
                "timestamp": datetime.now(),
                "open": float(latest.get("Open", 0)),
                "high": float(latest.get("High", 0)),
                "low": float(latest.get("Low", 0)),
                "close": float(latest.get("Close", 0)),
                "volume": int(latest.get("Volume", 0)),
                "vwap": info.get("vwap"),
            }

        loop = asyncio.get_event_loop()
        data = await loop.run_in_executor(None, _get_quote)
        return MarketData(**data)

    async def fetch_multiple_bars(
        self,
        tickers: list[str],
        start: datetime,
        end: datetime,
        timeframe: TimeFrame = TimeFrame.DAILY,
    ) -> dict[str, pd.DataFrame]:
        tasks = [self.fetch_bars(t, start, end, timeframe) for t in tickers]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        output: dict[str, pd.DataFrame] = {}
        for ticker, result in zip(tickers, results):
            if isinstance(result, Exception):
                logger.error(f"Failed to fetch {ticker}: {result}")
                continue
            output[ticker] = result
        return output

    async def fetch_multiple_quotes(
        self, tickers: list[str]
    ) -> dict[str, MarketData]:
        tasks = [self.fetch_quote(t) for t in tickers]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        output: dict[str, MarketData] = {}
        for ticker, result in zip(tickers, results):
            if isinstance(result, Exception):
                logger.error(f"Failed to fetch quote for {ticker}: {result}")
                continue
            output[ticker] = result
        return output

    async def health_check(self) -> bool:
        try:
            tick = yf.Ticker("SPY")
            hist = tick.history(period="1d")
            return not hist.empty
        except Exception:
            return False
