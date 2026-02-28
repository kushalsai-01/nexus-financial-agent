from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from typing import Any

import pandas as pd

from nexus.core.config import get_config
from nexus.core.exceptions import DataFetchError
from nexus.core.logging import get_logger
from nexus.core.types import Fundamental, MarketData, NewsEvent, TimeFrame
from nexus.data.processors.features import FeatureEngineer
from nexus.data.processors.technical import TechnicalAnalyzer

try:
    from nexus.data.processors.sentiment import SentimentAnalyzer
except ImportError:
    SentimentAnalyzer = None  # type: ignore[assignment,misc]
from nexus.data.providers.fundamentals import FundamentalsProvider
from nexus.data.providers.market import MarketDataProvider

try:
    from nexus.data.providers.news import NewsProvider
except ImportError:
    NewsProvider = None  # type: ignore[assignment,misc]

try:
    from nexus.data.providers.social import SocialProvider
except ImportError:
    SocialProvider = None  # type: ignore[assignment,misc]

logger = get_logger("data.pipeline")


class DataPipeline:
    def __init__(self) -> None:
        self._config = get_config()
        self._market = MarketDataProvider()
        self._news = NewsProvider() if NewsProvider is not None else None
        self._fundamentals = FundamentalsProvider()
        self._social = SocialProvider() if SocialProvider is not None else None
        self._technical = TechnicalAnalyzer()
        self._features = FeatureEngineer()
        self._sentiment: SentimentAnalyzer | None = None

    def _get_sentiment_analyzer(self) -> SentimentAnalyzer:
        if SentimentAnalyzer is None:
            raise ImportError(
                "SentimentAnalyzer requires torch and transformers. "
                "Install them with: pip install torch transformers"
            )
        if self._sentiment is None:
            self._sentiment = SentimentAnalyzer()
        return self._sentiment

    async def get_market_data(
        self,
        ticker: str,
        start: datetime | None = None,
        end: datetime | None = None,
        timeframe: TimeFrame = TimeFrame.DAILY,
        include_technicals: bool = True,
        include_features: bool = False,
    ) -> pd.DataFrame:
        end = end or datetime.now()
        start = start or end - timedelta(days=self._config.data.market.max_history_days)

        df = await self._market.fetch_bars(ticker, start, end, timeframe)

        if include_technicals and not df.empty:
            df = self._technical.compute_all(df)

        if include_features and not df.empty:
            df = self._features.build_features(df)

        return df

    async def get_multi_ticker_data(
        self,
        tickers: list[str],
        start: datetime | None = None,
        end: datetime | None = None,
        timeframe: TimeFrame = TimeFrame.DAILY,
        include_technicals: bool = True,
    ) -> dict[str, pd.DataFrame]:
        end = end or datetime.now()
        start = start or end - timedelta(days=self._config.data.market.max_history_days)

        raw_data = await self._market.fetch_multiple_bars(tickers, start, end, timeframe)

        if include_technicals:
            for ticker, df in raw_data.items():
                if not df.empty:
                    raw_data[ticker] = self._technical.compute_all(df)

        return raw_data

    async def get_quotes(self, tickers: list[str]) -> dict[str, MarketData]:
        return await self._market.fetch_multiple_quotes(tickers)

    async def get_news(
        self,
        tickers: list[str] | None = None,
        query: str | None = None,
        max_results: int = 50,
        include_sentiment: bool = True,
    ) -> list[NewsEvent]:
        if self._news is None:
            logger.warning("NewsProvider unavailable (feedparser not installed)")
            return []
        news = await self._news.fetch_news(
            query=query, tickers=tickers, max_results=max_results
        )

        if include_sentiment and news:
            analyzer = self._get_sentiment_analyzer()
            texts = [f"{n.title} {n.summary}" for n in news]
            sentiments = await analyzer.analyze_batch_async(texts)
            for event, sent in zip(news, sentiments):
                event.sentiment_score = sent["compound_score"]

        return news

    async def get_fundamentals(
        self, ticker: str, periods: int = 4
    ) -> pd.DataFrame:
        return await self._fundamentals.fetch_financials(ticker, periods)

    async def get_fundamental(self, ticker: str) -> Fundamental | None:
        return await self._fundamentals.fetch_fundamental(ticker)

    async def get_financial_ratios(self, ticker: str) -> dict[str, float]:
        return await self._fundamentals.fetch_ratios(ticker)

    async def get_social_sentiment(
        self, ticker: str, max_posts: int = 50
    ) -> dict[str, Any]:
        if self._social is None:
            logger.warning("SocialProvider unavailable (praw not installed)")
            return {}
        social_summary = await self._social.fetch_sentiment_summary(
            ticker, max_posts
        )

        posts = await self._social.fetch_posts(tickers=[ticker], max_results=max_posts)
        if posts:
            analyzer = self._get_sentiment_analyzer()
            texts = [f"{p.get('title', '')} {p.get('content', '')[:200]}" for p in posts]
            sentiments = await analyzer.analyze_batch_async(texts)
            aggregated = analyzer.aggregate_sentiment(sentiments)
            social_summary["nlp_sentiment"] = aggregated

        return social_summary

    async def get_full_analysis(
        self,
        ticker: str,
        lookback_days: int = 90,
    ) -> dict[str, Any]:
        end = datetime.now()
        start = end - timedelta(days=lookback_days)

        market_task = self.get_market_data(
            ticker, start, end, include_technicals=True, include_features=True
        )
        news_task = self.get_news(tickers=[ticker], max_results=20)
        fundamentals_task = self.get_fundamental(ticker)
        social_task = self.get_social_sentiment(ticker, max_posts=30)

        results = await asyncio.gather(
            market_task,
            news_task,
            fundamentals_task,
            social_task,
            return_exceptions=True,
        )

        analysis: dict[str, Any] = {"ticker": ticker, "timestamp": datetime.now().isoformat()}

        if isinstance(results[0], pd.DataFrame) and not results[0].empty:
            analysis["market_data"] = {
                "bars_count": len(results[0]),
                "latest_close": float(results[0]["close"].iloc[-1]),
                "technical_summary": self._technical.get_summary(results[0]),
            }
        elif isinstance(results[0], Exception):
            logger.error(f"Market data failed for {ticker}: {results[0]}")
            analysis["market_data"] = {"error": str(results[0])}

        if isinstance(results[1], list):
            analysis["news"] = {
                "count": len(results[1]),
                "avg_sentiment": (
                    sum(n.sentiment_score or 0 for n in results[1]) / len(results[1])
                    if results[1]
                    else 0.0
                ),
                "latest": [
                    {
                        "title": n.title,
                        "source": n.source,
                        "sentiment": n.sentiment_score,
                    }
                    for n in results[1][:5]
                ],
            }
        elif isinstance(results[1], Exception):
            logger.error(f"News failed for {ticker}: {results[1]}")

        if isinstance(results[2], Fundamental):
            analysis["fundamentals"] = results[2].model_dump()
        elif isinstance(results[2], Exception):
            logger.error(f"Fundamentals failed for {ticker}: {results[2]}")

        if isinstance(results[3], dict):
            analysis["social"] = results[3]
        elif isinstance(results[3], Exception):
            logger.error(f"Social failed for {ticker}: {results[3]}")

        return analysis

    async def health_check(self) -> dict[str, bool]:
        async def _false() -> bool:
            return False

        health_tasks = [
            self._market.health_check(),
            self._news.health_check() if self._news else _false(),
            self._fundamentals.health_check(),
            self._social.health_check() if self._social else _false(),
        ]
        checks = await asyncio.gather(*health_tasks, return_exceptions=True)

        return {
            "market": checks[0] if isinstance(checks[0], bool) else False,
            "news": checks[1] if isinstance(checks[1], bool) else False,
            "fundamentals": checks[2] if isinstance(checks[2], bool) else False,
            "social": checks[3] if isinstance(checks[3], bool) else False,
        }
