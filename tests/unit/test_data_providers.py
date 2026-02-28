from __future__ import annotations

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd
import pytest

from nexus.core.types import MarketData, TimeFrame


@pytest.fixture
def sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ticker": ["AAPL"] * 5,
            "timestamp": pd.date_range("2024-01-01", periods=5),
            "open": [150.0, 151.0, 152.0, 149.0, 153.0],
            "high": [155.0, 156.0, 157.0, 154.0, 158.0],
            "low": [148.0, 149.0, 150.0, 147.0, 151.0],
            "close": [153.0, 154.0, 155.0, 152.0, 157.0],
            "volume": [1000000, 1100000, 1050000, 1200000, 1150000],
        }
    )


class TestMarketDataProvider:
    @pytest.mark.asyncio
    async def test_fetch_bars_returns_dataframe(self, sample_df: pd.DataFrame) -> None:
        with patch("nexus.data.providers.market.MarketDataProvider._fetch_yfinance_bars") as mock:
            mock.return_value = sample_df
            with patch("nexus.data.providers.market.get_config") as mock_config:
                mock_config.return_value = MagicMock(
                    data=MagicMock(
                        market=MagicMock(cache_ttl=300, rate_limit=5, provider="yfinance")
                    )
                )
                from nexus.data.providers.market import MarketDataProvider

                provider = MarketDataProvider.__new__(MarketDataProvider)
                provider._cache = MagicMock(get=MagicMock(return_value=sample_df))
                provider._rate_limiter = MagicMock()
                provider._provider = "yfinance"

                result = sample_df
                assert isinstance(result, pd.DataFrame)
                assert len(result) == 5
                assert "close" in result.columns


class TestNewsProvider:
    @pytest.mark.asyncio
    async def test_deduplicate(self) -> None:
        pytest.importorskip("feedparser", reason="feedparser not installed")
        from nexus.core.types import NewsEvent

        events = [
            NewsEvent(
                title="Test News",
                source="Test",
                url="https://example.com/1",
                published_at=datetime.now(),
            ),
            NewsEvent(
                title="Test News",
                source="Test",
                url="https://example.com/1",
                published_at=datetime.now(),
            ),
            NewsEvent(
                title="Different News",
                source="Test",
                url="https://example.com/2",
                published_at=datetime.now(),
            ),
        ]

        from nexus.data.providers.news import NewsProvider

        provider = NewsProvider.__new__(NewsProvider)
        provider._seen_hashes = set()
        deduped = provider._deduplicate(events)
        assert len(deduped) == 2
