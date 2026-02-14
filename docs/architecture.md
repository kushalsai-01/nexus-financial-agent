# NEXUS Architecture Guide

## System Overview

NEXUS is an autonomous multi-agent hedge fund system structured as a layered architecture with clear separation of concerns.

```
┌─────────────────────┐
│     CLI / UI        │  Click CLI, Rich TUI (4 FPS), Streamlit Dashboard (5 pages)
├─────────────────────┤
│   Orchestration     │  LangGraph workflow, state management, TradingGraph.run()
├─────────────────────┤
│   Agent Layer       │  14 specialized agents across 5 teams
├─────────────────────┤
│   Analysis          │  Correlation, performance attribution, portfolio optimization
├─────────────────────┤
│   Risk / Execution  │  Position limits, VaR, order management, broker gateway
├─────────────────────┤
│   Backtest          │  Walk-forward engine, Monte Carlo, Sharpe/Sortino/Calmar
├─────────────────────┤
│   Data Layer        │  Providers, processors, storage (Postgres/InfluxDB/ChromaDB)
├─────────────────────┤
│   Monitoring        │  Langfuse tracing, Prometheus, alerting, health, cost tracking
├─────────────────────┤
│   Core              │  Config (Pydantic), types, exceptions, structured logging
└─────────────────────┘
```

## Layers

### 1. Core Layer (`nexus/core/`)

The foundation of the system.

- **config.py** — Pydantic-based configuration with YAML loading, environment variable resolution, and singleton pattern
- **types.py** — All domain models using Pydantic v2: `Signal`, `Action`, `MarketData`, `OHLCV`, `NewsEvent`, `Fundamental`, `Trade`, `Position`, `Portfolio`, `AgentState`, `BacktestResult`
- **exceptions.py** — Hierarchical exception system: `NexusError` → `DataError`, `AgentError`, `LLMError`, `ExecutionError`, `RiskViolation`, `ConfigError` (14 exception types)
- **logging.py** — Structured JSON logging with file rotation and console output

### 2. Data Layer (`nexus/data/`)

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
- **TechnicalAnalyzer** — 30+ indicators: SMA, EMA, MACD, RSI, Bollinger Bands, ADX, Ichimoku, ATR, OBV, Stochastic, CCI, Williams %R, MFI, Aroon, Keltner Channels
- **FeatureEngineer** — Returns (1d–252d), volatility, z-scores, percentile ranks, price ratios, volume metrics
- **SentimentAnalyzer** — FinBERT model for financial text sentiment analysis

#### Pipeline (`nexus/data/pipeline.py`)
- Orchestrates all data sources with parallel `asyncio.gather`
- Full analysis combining market data, news, fundamentals, and social sentiment

### 3. Agent Layer (`nexus/agents/`)

14 specialized agents organized in 5 teams:

| Team | Agents | Role |
|------|--------|------|
| **Analyst** | market_data, sentiment, fundamental | Data collection & analysis |
| **Research** | bull, bear, coordinator | Debate & consensus building |
| **Quant** | technical, quantitative, macro | Quantitative modeling |
| **Strategy** | event, rl_agent, risk | Strategy & risk assessment |
| **Execution** | portfolio, execution | Portfolio management & trade execution |

#### Agent Flow
```
market_data → [technical, fundamental, sentiment, quantitative, macro, event]
    → rl_agent → [bull, bear] → coordinator → risk → portfolio → execution
```

Each agent receives `TradingState` (TypedDict), calls its LLM with a specialized system prompt, produces a `Signal` (action, confidence, reasoning), and appends results to shared state.

### 4. LLM Integration (`nexus/llm/`)

- **LLMRouter** — Provider selection with automatic fallback
- **Primary**: Anthropic Claude (claude-sonnet-4-20250514)
- **Fallback**: OpenAI GPT-4o
- Automatic retry with exponential backoff
- Cost tracking per-call (input/output tokens × pricing table)
- Response caching for identical prompts

### 5. Backtest & Analysis (`nexus/backtest/`, `nexus/analysis/`)

- **BacktestEngine** — Walk-forward backtesting with configurable windows
- **Metrics** — Sharpe, Sortino, Calmar, max drawdown, profit factor, win rate
- **Correlation** — Regime detection and correlation analysis
- **Performance** — Return attribution by agent, ticker, strategy
- **Optimization** — Portfolio optimization (mean-variance, risk parity)

### 6. Risk & Execution (`nexus/risk/`, `nexus/execution/`)

- **RiskMonitor** — Position limits, drawdown controls, VaR, exposure monitoring
- **ExecutionEngine** — SimulatedBroker, PaperBroker with slippage modeling
- **OrderManager** — Order lifecycle, fills, commission tracking

### 7. Monitoring (`nexus/monitoring/`)

- **Tracing** — Langfuse/LangSmith integration for LLM call tracing
- **Metrics** — Prometheus counters, gauges, histograms (decisions/sec, latency, cost)
- **Alerts** — Slack, PagerDuty, Email, Webhook with severity routing and cooldown
- **Health** — Database, LLM provider, market data, broker connectivity checks
- **Cost** — Per-agent, per-model LLM cost tracking with daily budgets

### 8. Reports (`nexus/reports/`)

- **DailyReport** — P&L, positions, trades, alerts (text + HTML)
- **WeeklyReport** — Weekly performance, ticker breakdown, agent performance
- **BacktestReport** — Monthly returns heatmap, trade statistics, ticker breakdown
- **Templates** — Jinja2 HTML templates with dark theme

### 9. User Interfaces

#### CLI (`nexus/cli/`)
Click-based commands: `run`, `backtest`, `analyze`, `report`, `status`, `dashboard`, `costs`, `init`

#### Terminal UI (`nexus/ui/`)
Rich-powered live dashboard:
- 4 FPS refresh rate
- Agent progress table (14 agents across 5 teams)
- Live message feed
- Final report panel
- Status bar with metrics

#### Web Dashboard (`nexus/ui/dashboard/`)
Streamlit + Plotly with 5 pages:
- **Overview** — Portfolio value, positions, allocation
- **Agents** — Accuracy, cost, decisions, confidence
- **Backtests** — Equity curve, drawdown, monthly returns heatmap
- **Trades** — Filterable trade history, P&L analysis
- **Risk** — VaR, correlation, sector exposure

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

## Deployment Options

| Method | Use Case |
|--------|----------|
| `pip install -e .` | Local development |
| Docker Compose | Full stack with databases |
| Kubernetes | Production cluster deployment |
| Terraform | AWS infrastructure provisioning |

## Technology Decisions

| Decision | Rationale |
|----------|-----------|
| Pydantic v2 | Type safety, validation, serialization |
| asyncio | Non-blocking I/O for data fetching |
| LangGraph | Stateful multi-agent orchestration |
| SQLAlchemy 2.0 | Typed ORM with async support |
| InfluxDB | Purpose-built for time series |
| ChromaDB | Lightweight vector DB for RAG |
| FinBERT | Domain-specific financial NLP |
| Rich | Beautiful terminal UI with live updates |
| Streamlit + Plotly | Interactive web dashboards |
| Click | Production CLI framework |
| Langfuse | LLM observability with cost tracking |
| Prometheus | Industry-standard metrics |
| YAML config | Human-readable, supports nesting |
| JSON logging | Machine-parseable, structured |
