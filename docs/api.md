# NEXUS Data Layer API Reference

## Data Pipeline

### `DataPipeline`

Main entry point for all data operations.

```python
from nexus.data.pipeline import DataPipeline

pipeline = DataPipeline()
```

#### Methods

##### `get_market_data(ticker, start, end, timeframe, include_technicals, include_features) -> pd.DataFrame`
Fetch OHLCV data with optional technical indicators and feature engineering.

##### `get_multi_ticker_data(tickers, start, end, timeframe, include_technicals) -> dict[str, pd.DataFrame]`
Parallel fetch for multiple tickers.

##### `get_quotes(tickers) -> dict[str, MarketData]`
Real-time quotes for multiple tickers.

##### `get_news(tickers, query, max_results, include_sentiment) -> list[NewsEvent]`
Aggregate news from all sources with optional sentiment scoring.

##### `get_fundamentals(ticker, periods) -> pd.DataFrame`
SEC EDGAR financial statements.

##### `get_financial_ratios(ticker) -> dict[str, float]`
Computed financial ratios (ROE, ROA, margins, etc.).

##### `get_social_sentiment(ticker, max_posts) -> dict[str, Any]`
Social media analysis from Reddit.

##### `get_full_analysis(ticker, lookback_days) -> dict[str, Any]`
Complete analysis combining all data sources in parallel.

##### `health_check() -> dict[str, bool]`
Check connectivity to all data sources.

---

## Providers

### `MarketDataProvider`

```python
from nexus.data.providers.market import MarketDataProvider

provider = MarketDataProvider()
df = await provider.fetch_bars("AAPL", start, end, TimeFrame.DAILY)
quote = await provider.fetch_quote("AAPL")
```

### `NewsProvider`

```python
from nexus.data.providers.news import NewsProvider

provider = NewsProvider()
news = await provider.fetch_news(tickers=["AAPL", "MSFT"])
```

### `FundamentalsProvider`

```python
from nexus.data.providers.fundamentals import FundamentalsProvider

provider = FundamentalsProvider()
financials = await provider.fetch_financials("AAPL", periods=4)
ratios = await provider.fetch_ratios("AAPL")
```

### `SocialProvider`

```python
from nexus.data.providers.social import SocialProvider

provider = SocialProvider()
posts = await provider.fetch_posts(tickers=["AAPL"])
summary = await provider.fetch_sentiment_summary("AAPL")
```

---

## Storage

### `PostgresStorage`

```python
from nexus.data.storage.postgres import PostgresStorage

storage = PostgresStorage()
storage.create_tables()
storage.save_prices([...])
trades = storage.get_trades(ticker="AAPL")
```

### `TimeseriesStorage`

```python
from nexus.data.storage.timeseries import TimeseriesStorage

ts = TimeseriesStorage()
ts.write_market_data("AAPL", timestamp, open, high, low, close, volume)
data = ts.query_market_data("AAPL", start, end)
```

### `VectorStorage`

```python
from nexus.data.storage.vector import VectorStorage

vs = VectorStorage()
vs.add_news([{"title": "...", "content": "...", "tickers": ["AAPL"]}])
results = vs.search_news("Apple earnings report", n_results=5)
```

---

## Processors

### `TechnicalAnalyzer`

```python
from nexus.data.processors.technical import TechnicalAnalyzer

df = TechnicalAnalyzer.compute_all(market_df)
summary = TechnicalAnalyzer.get_summary(df)
```

**Indicators included**: SMA (10/20/50/200), EMA (9/21/55), MACD, RSI, Stochastic, Bollinger Bands, ATR, ADX, Ichimoku, OBV, CMF, CCI, Williams %R, MFI, Aroon, Keltner, Donchian, and more.

### `FeatureEngineer`

```python
from nexus.data.processors.features import FeatureEngineer

df = FeatureEngineer.build_features(market_df)
selected = FeatureEngineer.select_features(df)
normalized = FeatureEngineer.normalize(df, selected)
```

### `SentimentAnalyzer`

```python
from nexus.data.processors.sentiment import SentimentAnalyzer

analyzer = SentimentAnalyzer()
result = analyzer.analyze("Apple reports record quarterly earnings")
batch = analyzer.analyze_batch(["text1", "text2", ...])
aggregate = analyzer.aggregate_sentiment(batch)
```

---

## Domain Types

All types are Pydantic v2 models defined in `nexus/core/types.py`:

- `Signal` — Trading signal with confidence, target, stop-loss
- `Action` — Executable trade action
- `MarketData` — Single OHLCV bar
- `OHLCV` — Time series of bars
- `NewsEvent` — News article with sentiment
- `Fundamental` — Financial statement data
- `Trade` — Executed trade record
- `Position` — Open position with P&L tracking
- `Portfolio` — Full portfolio state
- `AgentState` — Agent status and metrics
- `BacktestResult` — Backtest performance summary
