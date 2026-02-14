from __future__ import annotations

import asyncio
import hashlib
from datetime import datetime, timedelta
from typing import Any

import feedparser
import httpx
from bs4 import BeautifulSoup

from nexus.core.config import get_config
from nexus.core.exceptions import DataFetchError
from nexus.core.logging import get_logger
from nexus.core.types import NewsEvent
from nexus.data.providers.base import BaseNewsProvider

logger = get_logger("data.providers.news")


class NewsProvider(BaseNewsProvider):
    def __init__(self) -> None:
        config = get_config()
        super().__init__(cache_ttl=config.data.news.dedup_window_hours * 3600)
        self._newsapi_base = config.data.news.newsapi_base_url
        self._newsapi_key = config.newsapi_key
        self._rss_feeds = config.data.news.rss_feeds
        self._max_articles = config.data.news.max_articles
        self._scrape_timeout = config.data.news.scrape_timeout
        self._seen_hashes: set[str] = set()

    async def fetch_news(
        self,
        query: str | None = None,
        tickers: list[str] | None = None,
        max_results: int = 50,
    ) -> list[NewsEvent]:
        all_news: list[NewsEvent] = []

        if self._newsapi_key:
            search_query = query or (
                " OR ".join(tickers) if tickers else "stock market"
            )
            api_news = await self._fetch_from_newsapi(search_query, max_results)
            all_news.extend(api_news)

        if self._rss_feeds:
            rss_news = await self._fetch_from_rss(tickers)
            all_news.extend(rss_news)

        deduped = self._deduplicate(all_news)
        deduped.sort(key=lambda n: n.published_at, reverse=True)
        return deduped[:max_results]

    async def _fetch_from_newsapi(
        self, query: str, max_results: int
    ) -> list[NewsEvent]:
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                response = await client.get(
                    f"{self._newsapi_base}/everything",
                    params={
                        "q": query,
                        "pageSize": min(max_results, 100),
                        "sortBy": "publishedAt",
                        "language": "en",
                        "apiKey": self._newsapi_key,
                    },
                )
                response.raise_for_status()
                data = response.json()

            articles = data.get("articles", [])
            events: list[NewsEvent] = []
            for article in articles:
                published = article.get("publishedAt", "")
                try:
                    pub_dt = datetime.fromisoformat(published.replace("Z", "+00:00"))
                except (ValueError, AttributeError):
                    pub_dt = datetime.now()

                events.append(
                    NewsEvent(
                        title=article.get("title", ""),
                        source=article.get("source", {}).get("name", ""),
                        url=article.get("url", ""),
                        published_at=pub_dt,
                        content=article.get("content", ""),
                        summary=article.get("description", ""),
                    )
                )
            logger.info(f"Fetched {len(events)} articles from NewsAPI")
            return events
        except httpx.HTTPError as e:
            logger.error(f"NewsAPI request failed: {e}")
            return []

    async def _fetch_from_rss(
        self, tickers: list[str] | None = None
    ) -> list[NewsEvent]:
        tasks = [self._parse_feed(url) for url in self._rss_feeds]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        all_items: list[NewsEvent] = []
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"RSS fetch failed: {result}")
                continue
            all_items.extend(result)

        if tickers:
            ticker_set = {t.upper() for t in tickers}
            filtered = []
            for item in all_items:
                title_upper = item.title.upper()
                if any(t in title_upper for t in ticker_set):
                    item.tickers = [t for t in ticker_set if t in title_upper]
                    filtered.append(item)
            return filtered

        return all_items

    async def _parse_feed(self, url: str) -> list[NewsEvent]:
        def _parse() -> list[dict[str, Any]]:
            feed = feedparser.parse(url)
            return feed.entries

        loop = asyncio.get_event_loop()
        entries = await loop.run_in_executor(None, _parse)
        events: list[NewsEvent] = []
        for entry in entries:
            published = entry.get("published_parsed")
            if published:
                pub_dt = datetime(*published[:6])
            else:
                pub_dt = datetime.now()

            events.append(
                NewsEvent(
                    title=entry.get("title", ""),
                    source=url,
                    url=entry.get("link", ""),
                    published_at=pub_dt,
                    summary=self._clean_html(entry.get("summary", "")),
                )
            )
        logger.info(f"Parsed {len(events)} items from RSS: {url}")
        return events

    async def scrape_article(self, url: str) -> str:
        try:
            async with httpx.AsyncClient(timeout=self._scrape_timeout) as client:
                response = await client.get(url, follow_redirects=True)
                response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")
            for tag in soup(["script", "style", "nav", "header", "footer"]):
                tag.decompose()
            text = soup.get_text(separator=" ", strip=True)
            return text[:5000]
        except Exception as e:
            logger.error(f"Failed to scrape {url}: {e}")
            return ""

    def _deduplicate(self, events: list[NewsEvent]) -> list[NewsEvent]:
        unique: list[NewsEvent] = []
        for event in events:
            content_hash = hashlib.md5(
                f"{event.title}{event.url}".encode()
            ).hexdigest()
            if content_hash not in self._seen_hashes:
                self._seen_hashes.add(content_hash)
                unique.append(event)
        return unique

    @staticmethod
    def _clean_html(text: str) -> str:
        if "<" in text:
            soup = BeautifulSoup(text, "html.parser")
            return soup.get_text(separator=" ", strip=True)
        return text

    async def health_check(self) -> bool:
        if self._newsapi_key:
            try:
                async with httpx.AsyncClient(timeout=10) as client:
                    response = await client.get(
                        f"{self._newsapi_base}/top-headlines",
                        params={
                            "country": "us",
                            "pageSize": 1,
                            "apiKey": self._newsapi_key,
                        },
                    )
                    return response.status_code == 200
            except Exception:
                return False
        return True
