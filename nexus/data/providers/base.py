from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any

import pandas as pd

from nexus.core.config import get_config
from nexus.core.logging import get_logger
from nexus.core.types import MarketData, NewsEvent, TimeFrame

logger = get_logger("data.providers")


class RateLimiter:
    def __init__(self, calls_per_second: int = 5) -> None:
        self._calls_per_second = calls_per_second
        self._call_times: list[float] = []

    async def acquire(self) -> None:
        import asyncio
        import time

        now = time.monotonic()
        self._call_times = [t for t in self._call_times if now - t < 1.0]
        if len(self._call_times) >= self._calls_per_second:
            sleep_time = 1.0 - (now - self._call_times[0])
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)
        self._call_times.append(time.monotonic())


class Cache:
    def __init__(self, ttl: int = 300) -> None:
        self._ttl = ttl
        self._store: dict[str, tuple[float, Any]] = {}

    def get(self, key: str) -> Any | None:
        import time

        if key in self._store:
            ts, value = self._store[key]
            if time.monotonic() - ts < self._ttl:
                return value
            del self._store[key]
        return None

    def set(self, key: str, value: Any) -> None:
        import time

        self._store[key] = (time.monotonic(), value)

    def clear(self) -> None:
        self._store.clear()


class BaseDataProvider(ABC):
    def __init__(self, cache_ttl: int = 300, rate_limit: int = 5) -> None:
        self._cache = Cache(ttl=cache_ttl)
        self._rate_limiter = RateLimiter(calls_per_second=rate_limit)
        self._config = get_config()

    @abstractmethod
    async def fetch_bars(
        self,
        ticker: str,
        start: datetime,
        end: datetime,
        timeframe: TimeFrame = TimeFrame.DAILY,
    ) -> pd.DataFrame:
        ...

    @abstractmethod
    async def fetch_quote(self, ticker: str) -> MarketData:
        ...

    @abstractmethod
    async def health_check(self) -> bool:
        ...


class BaseNewsProvider(ABC):
    def __init__(self, cache_ttl: int = 300) -> None:
        self._cache = Cache(ttl=cache_ttl)
        self._config = get_config()

    @abstractmethod
    async def fetch_news(
        self,
        query: str | None = None,
        tickers: list[str] | None = None,
        max_results: int = 50,
    ) -> list[NewsEvent]:
        ...

    @abstractmethod
    async def health_check(self) -> bool:
        ...


class BaseFundamentalsProvider(ABC):
    def __init__(self, cache_ttl: int = 86400) -> None:
        self._cache = Cache(ttl=cache_ttl)
        self._config = get_config()

    @abstractmethod
    async def fetch_financials(
        self, ticker: str, periods: int = 4
    ) -> pd.DataFrame:
        ...

    @abstractmethod
    async def fetch_ratios(self, ticker: str) -> dict[str, float]:
        ...

    @abstractmethod
    async def health_check(self) -> bool:
        ...


class BaseSocialProvider(ABC):
    def __init__(self, cache_ttl: int = 600) -> None:
        self._cache = Cache(ttl=cache_ttl)
        self._config = get_config()

    @abstractmethod
    async def fetch_posts(
        self,
        query: str | None = None,
        tickers: list[str] | None = None,
        max_results: int = 50,
    ) -> list[dict[str, Any]]:
        ...

    @abstractmethod
    async def health_check(self) -> bool:
        ...
