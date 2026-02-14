# NEXUS API Reference

## CLI Commands

```bash
nexus init                                    # Initialize project
nexus run -t AAPL -t MSFT --demo              # Run with terminal UI
nexus run -t AAPL --capital 100000 --no-ui    # Run headless
nexus backtest -t AAPL -s 2023-01-01 -o out.html --format html
nexus analyze -t TSLA --depth deep
nexus report --type daily --format html -o report.html
nexus status                                   # Health checks
nexus dashboard                                # Launch Streamlit
nexus costs                                    # LLM cost summary
```

---

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

---

## Orchestration

### `TradingGraph`

```python
from nexus.orchestration.graph import TradingGraph
from nexus.core.config import get_config

graph = TradingGraph(config=get_config())
result = await graph.run(ticker="AAPL", capital=100_000)
```

Returns a dict with `recommendation`, `confidence`, `exposure`, `reasoning`, `cost_usd`.

---

## Monitoring

### `NexusMetrics`

```python
from nexus.monitoring.metrics import NexusMetrics

metrics = NexusMetrics()
metrics.record_decision(agent="technical", latency_ms=120, cost_usd=0.003)
metrics.record_api_call(provider="yfinance", success=True)
metrics.update_portfolio(value=105000, pnl=5000, positions=5)
snapshot = metrics.get_snapshot()
```

### `AlertManager`

```python
from nexus.monitoring.alerts import AlertManager

alerts = AlertManager()
alerts.add_slack("https://hooks.slack.com/...")
alerts.setup_default_rules()
await alerts.check_rules(metrics.get_snapshot())
```

### `CostTracker`

```python
from nexus.monitoring.cost import CostTracker

tracker = CostTracker()
tracker.record(agent="technical", model="claude-sonnet-4-20250514", input_tokens=500, output_tokens=200, cost_usd=0.003)
summary = tracker.get_summary()
breakdown = tracker.get_agent_breakdown()
```

### `HealthChecker`

```python
from nexus.monitoring.health import HealthChecker

checker = HealthChecker()
results = await checker.run_all_checks()
```

---

## Reports

```python
from nexus.reports import DailyReport, WeeklyReport, BacktestReport

daily = DailyReport(portfolio=portfolio)
daily.add_trades(trades)
print(daily.to_text())
Path("daily.html").write_text(daily.to_html())

weekly = WeeklyReport(portfolio=portfolio)
weekly.add_trades(trades)
weekly.add_daily_values([{"date": "2024-01-15", "value": 105000}, ...])
print(weekly.to_text())

backtest_report = BacktestReport(result=backtest_result)
Path("backtest.html").write_text(backtest_report.to_html())
```

---

## Terminal UI

```python
from nexus.ui.tui import TradingTUI

tui = TradingTUI()
tui.run_sync(demo=True)                       # Demo mode
await tui.run_with_graph(graph, ["AAPL"], 100000)  # Live mode
```

---

## Web Dashboard

```bash
nexus dashboard --port 8501
# or
streamlit run nexus/ui/dashboard/app.py
```
