# NEXUS Architecture

## System Overview

NEXUS is an autonomous multi-agent hedge fund system. The architecture follows a layered design with clear separation of concerns.

## Layers

### 1. Core Layer (`nexus/core/`)

The foundation of the system.

- **config.py** — Pydantic-based configuration with YAML loading, environment variable resolution, and singleton pattern
- **types.py** — All domain models using Pydantic v2: `Signal`, `Action`, `MarketData`, `Trade`, `Position`, `Portfolio`, `BacktestResult`
- **exceptions.py** — Hierarchical exception system: `NexusError` → `DataError`, `AgentError`, `ExecutionError`, `RiskViolation`, `ConfigError`
- **logging.py** — Structured JSON logging with file rotation and console output

### 2. Data Layer (`nexus/data/`)

Handles all data ingestion, storage, and processing.

#### Providers (`nexus/data/providers/`)
- **MarketDataProvider** — yfinance integration with caching and rate limiting
- **NewsProvider** — NewsAPI + RSS feed aggregation with deduplication
- **FundamentalsProvider** — SEC EDGAR API for financial statements and ratios
- **SocialProvider** — Reddit data scraping with ticker extraction

#### Storage (`nexus/data/storage/`)
- **PostgresStorage** — SQLAlchemy ORM for structured data (prices, trades, decisions)
- **TimeseriesStorage** — InfluxDB for high-frequency time series data
- **VectorStorage** — ChromaDB for embeddings, RAG, and similarity search

#### Processors (`nexus/data/processors/`)
- **TechnicalAnalyzer** — 30+ indicators: SMA, EMA, MACD, RSI, Bollinger Bands, ADX, Ichimoku, ATR, OBV, etc.
- **FeatureEngineer** — Returns, price features, volume features, statistical features, lag features, time features
- **SentimentAnalyzer** — FinBERT model for financial text sentiment analysis

#### Pipeline (`nexus/data/pipeline.py`)
- Orchestrates all data sources
- Parallel data fetching with `asyncio.gather`
- Full analysis combining market data, news, fundamentals, and social sentiment

### 3. Agent Layer (Prompt 2)

14 specialized agents organized in 5 teams.

### 4. Execution Layer (Prompt 3)

Order management, broker integration, backtesting engine.

### 5. Monitoring & UI Layer (Prompt 4)

Dashboard, CLI, alerts, cost tracking.

## Data Flow

```
External APIs → Providers → Processors → Pipeline → Agents → Execution → Storage
                                                          ↑                  │
                                                          └──────────────────┘
                                                           (Feedback Loop)
```

## Configuration Flow

```
config/default.yaml → config/{env}.yaml → Environment Variables → .env → NexusConfig
```

## Technology Decisions

| Decision | Rationale |
|----------|-----------|
| Pydantic v2 | Type safety, validation, serialization |
| asyncio | Non-blocking I/O for data fetching |
| SQLAlchemy 2.0 | Typed ORM with async support |
| InfluxDB | Purpose-built for time series |
| ChromaDB | Lightweight vector DB for RAG |
| FinBERT | Domain-specific financial NLP |
| YAML config | Human-readable, supports nesting |
| JSON logging | Machine-parseable, structured |
