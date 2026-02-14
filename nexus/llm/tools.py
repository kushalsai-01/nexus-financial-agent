from __future__ import annotations

import json
from datetime import datetime, timedelta
from typing import Any

from nexus.core.logging import get_logger

logger = get_logger("llm.tools")


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, dict[str, Any]] = {}

    def register(self, name: str, description: str, handler: Any, parameters: dict[str, Any] | None = None) -> None:
        self._tools[name] = {
            "name": name,
            "description": description,
            "handler": handler,
            "parameters": parameters or {},
        }

    def get(self, name: str) -> dict[str, Any] | None:
        return self._tools.get(name)

    def list_tools(self) -> list[dict[str, str]]:
        return [
            {"name": t["name"], "description": t["description"]}
            for t in self._tools.values()
        ]

    async def execute(self, name: str, **kwargs: Any) -> Any:
        tool = self._tools.get(name)
        if not tool:
            raise ValueError(f"Tool not found: {name}")
        handler = tool["handler"]
        if callable(handler):
            result = handler(**kwargs)
            if hasattr(result, "__await__"):
                return await result
            return result
        raise TypeError(f"Tool handler is not callable: {name}")


async def fetch_price_data(ticker: str, days: int = 30) -> dict[str, Any]:
    from nexus.data.providers.market import YFinanceProvider
    provider = YFinanceProvider()
    end = datetime.now()
    start = end - timedelta(days=days)
    try:
        data = await provider.get_historical(ticker, start, end)
        return {
            "ticker": ticker,
            "period_days": days,
            "data_points": len(data) if data else 0,
            "data": data,
        }
    except Exception as e:
        logger.error(f"Failed to fetch price data for {ticker}: {e}")
        return {"ticker": ticker, "error": str(e)}


async def fetch_fundamentals(ticker: str) -> dict[str, Any]:
    from nexus.data.providers.fundamentals import SECEdgarProvider
    provider = SECEdgarProvider()
    try:
        data = await provider.get_financials(ticker)
        return {"ticker": ticker, "fundamentals": data}
    except Exception as e:
        logger.error(f"Failed to fetch fundamentals for {ticker}: {e}")
        return {"ticker": ticker, "error": str(e)}


async def fetch_news(ticker: str, max_articles: int = 20) -> dict[str, Any]:
    from nexus.data.providers.news import NewsProvider
    provider = NewsProvider()
    try:
        articles = await provider.get_news(ticker, max_results=max_articles)
        return {"ticker": ticker, "articles": articles}
    except Exception as e:
        logger.error(f"Failed to fetch news for {ticker}: {e}")
        return {"ticker": ticker, "error": str(e)}


async def compute_technicals(ticker: str, period: int = 90) -> dict[str, Any]:
    from nexus.data.processors.technical import TechnicalAnalyzer
    analyzer = TechnicalAnalyzer()
    try:
        import pandas as pd
        from nexus.data.providers.market import YFinanceProvider
        provider = YFinanceProvider()
        end = datetime.now()
        start = end - timedelta(days=period)
        raw = await provider.get_historical(ticker, start, end)
        if not raw:
            return {"ticker": ticker, "error": "No data"}
        df = pd.DataFrame(raw)
        indicators = analyzer.compute_all(df)
        return {"ticker": ticker, "indicators": indicators.to_dict() if hasattr(indicators, "to_dict") else str(indicators)}
    except Exception as e:
        logger.error(f"Failed to compute technicals for {ticker}: {e}")
        return {"ticker": ticker, "error": str(e)}


async def analyze_sentiment(ticker: str) -> dict[str, Any]:
    from nexus.data.processors.sentiment import SentimentAnalyzer
    analyzer = SentimentAnalyzer()
    try:
        from nexus.data.providers.news import NewsProvider
        provider = NewsProvider()
        articles = await provider.get_news(ticker, max_results=10)
        texts = [a.get("title", "") + " " + a.get("content", "") for a in articles] if articles else []
        if not texts:
            return {"ticker": ticker, "sentiment": "neutral", "score": 0.0}
        results = [analyzer.analyze(t) for t in texts[:10]]
        avg_score = sum(r.get("score", 0.0) for r in results) / len(results) if results else 0.0
        return {"ticker": ticker, "sentiment_score": avg_score, "num_articles": len(texts)}
    except Exception as e:
        logger.error(f"Failed to analyze sentiment for {ticker}: {e}")
        return {"ticker": ticker, "error": str(e)}


def build_default_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(
        "fetch_price_data",
        "Fetch historical price data for a ticker",
        fetch_price_data,
        {"ticker": "str", "days": "int"},
    )
    registry.register(
        "fetch_fundamentals",
        "Fetch fundamental financial data for a ticker",
        fetch_fundamentals,
        {"ticker": "str"},
    )
    registry.register(
        "fetch_news",
        "Fetch recent news articles for a ticker",
        fetch_news,
        {"ticker": "str", "max_articles": "int"},
    )
    registry.register(
        "compute_technicals",
        "Compute technical indicators for a ticker",
        compute_technicals,
        {"ticker": "str", "period": "int"},
    )
    registry.register(
        "analyze_sentiment",
        "Analyze market sentiment for a ticker",
        analyze_sentiment,
        {"ticker": "str"},
    )
    return registry
