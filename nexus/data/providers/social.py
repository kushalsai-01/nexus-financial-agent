from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Any

import praw

from nexus.core.config import get_config
from nexus.core.logging import get_logger
from nexus.data.providers.base import BaseSocialProvider

logger = get_logger("data.providers.social")


class SocialProvider(BaseSocialProvider):
    def __init__(self) -> None:
        config = get_config()
        super().__init__(cache_ttl=600)
        self._subreddits = config.data.social.reddit_subreddits
        self._max_posts = config.data.social.max_posts
        self._sentiment_threshold = config.data.social.sentiment_threshold
        self._reddit_client_id = config.reddit_client_id
        self._reddit_client_secret = config.reddit_client_secret
        self._reddit: praw.Reddit | None = None

    def _get_reddit(self) -> praw.Reddit:
        if self._reddit is None:
            self._reddit = praw.Reddit(
                client_id=self._reddit_client_id,
                client_secret=self._reddit_client_secret,
                user_agent="Nexus Financial Agent v1.0",
            )
        return self._reddit

    async def fetch_posts(
        self,
        query: str | None = None,
        tickers: list[str] | None = None,
        max_results: int = 50,
    ) -> list[dict[str, Any]]:
        cache_key = f"social:{query}:{tickers}:{max_results}"
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        posts: list[dict[str, Any]] = []

        if self._reddit_client_id and self._reddit_client_secret:
            reddit_posts = await self._fetch_reddit(query, tickers, max_results)
            posts.extend(reddit_posts)

        posts.sort(key=lambda p: p.get("score", 0), reverse=True)
        result = posts[:max_results]
        self._cache.set(cache_key, result)
        return result

    async def _fetch_reddit(
        self,
        query: str | None = None,
        tickers: list[str] | None = None,
        max_results: int = 50,
    ) -> list[dict[str, Any]]:
        def _scrape() -> list[dict[str, Any]]:
            reddit = self._get_reddit()
            all_posts: list[dict[str, Any]] = []

            for sub_name in self._subreddits:
                try:
                    subreddit = reddit.subreddit(sub_name)

                    if query:
                        submissions = subreddit.search(query, limit=max_results)
                    else:
                        submissions = subreddit.hot(limit=max_results)

                    for submission in submissions:
                        mentioned_tickers = self._extract_tickers(
                            f"{submission.title} {submission.selftext}", tickers
                        )

                        if tickers and not mentioned_tickers:
                            continue

                        all_posts.append(
                            {
                                "platform": "reddit",
                                "subreddit": sub_name,
                                "title": submission.title,
                                "content": submission.selftext[:2000],
                                "url": f"https://reddit.com{submission.permalink}",
                                "score": submission.score,
                                "num_comments": submission.num_comments,
                                "created_at": datetime.fromtimestamp(
                                    submission.created_utc
                                ).isoformat(),
                                "author": str(submission.author),
                                "tickers": mentioned_tickers,
                                "upvote_ratio": submission.upvote_ratio,
                            }
                        )
                except Exception as e:
                    logger.error(f"Failed to fetch from r/{sub_name}: {e}")

            return all_posts

        loop = asyncio.get_event_loop()
        posts = await loop.run_in_executor(None, _scrape)
        logger.info(f"Fetched {len(posts)} Reddit posts")
        return posts

    @staticmethod
    def _extract_tickers(
        text: str, known_tickers: list[str] | None = None
    ) -> list[str]:
        import re

        found: list[str] = []

        if known_tickers:
            for ticker in known_tickers:
                pattern = rf"\b\$?{re.escape(ticker.upper())}\b"
                if re.search(pattern, text.upper()):
                    found.append(ticker.upper())
        else:
            matches = re.findall(r"\$([A-Z]{1,5})\b", text)
            found = list(set(matches))

        return found

    async def fetch_sentiment_summary(
        self, ticker: str, max_posts: int = 50
    ) -> dict[str, Any]:
        posts = await self.fetch_posts(tickers=[ticker], max_results=max_posts)

        if not posts:
            return {
                "ticker": ticker,
                "total_posts": 0,
                "avg_score": 0.0,
                "total_comments": 0,
                "avg_upvote_ratio": 0.0,
                "platforms": [],
            }

        total_score = sum(p.get("score", 0) for p in posts)
        total_comments = sum(p.get("num_comments", 0) for p in posts)
        avg_upvote = sum(p.get("upvote_ratio", 0.5) for p in posts) / len(posts)
        platforms = list({p.get("platform", "unknown") for p in posts})

        return {
            "ticker": ticker,
            "total_posts": len(posts),
            "avg_score": total_score / len(posts) if posts else 0.0,
            "total_comments": total_comments,
            "avg_upvote_ratio": avg_upvote,
            "platforms": platforms,
            "top_posts": [
                {"title": p["title"], "score": p["score"], "url": p["url"]}
                for p in posts[:5]
            ],
        }

    async def health_check(self) -> bool:
        if not self._reddit_client_id or not self._reddit_client_secret:
            return True
        try:
            reddit = self._get_reddit()
            subreddit = reddit.subreddit("stocks")
            next(subreddit.hot(limit=1))
            return True
        except Exception:
            return False
