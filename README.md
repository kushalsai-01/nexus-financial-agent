# NEXUS — Autonomous Multi-Agent Hedge Fund

```
 ███╗   ██╗███████╗██╗  ██╗██╗   ██╗███████╗
 ████╗  ██║██╔════╝╚██╗██╔╝██║   ██║██╔════╝
 ██╔██╗ ██║█████╗   ╚███╔╝ ██║   ██║███████╗
 ██║╚██╗██║██╔══╝   ██╔██╗ ██║   ██║╚════██║
 ██║ ╚████║███████╗██╔╝ ╚██╗╚██████╔╝███████║
 ╚═╝  ╚═══╝╚══════╝╚═╝   ╚═╝ ╚═════╝ ╚══════╝
         Autonomous Financial Intelligence
```

[![CI](https://github.com/kushalsai-01/nexus-financial-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/kushalsai-01/nexus-financial-agent/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Type checked: mypy](https://img.shields.io/badge/type%20check-mypy-blue.svg)](https://mypy-lang.org/)

An autonomous hedge fund system powered by a multi-agent architecture where 14 specialized AI agents collaborate across 5 teams to analyze markets, generate trading signals, manage risk, and execute trades.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        NEXUS SYSTEM                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐       │
│  │ Market   │  │  News    │  │  Social  │  │   SEC    │       │
│  │  Data    │  │  Feeds   │  │  Media   │  │  EDGAR   │       │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘       │
│       │              │              │              │             │
│  ┌────▼──────────────▼──────────────▼──────────────▼─────┐      │
│  │                 DATA PIPELINE                          │      │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────────┐           │      │
│  │  │Technical │ │ Feature  │ │  Sentiment   │           │      │
│  │  │Indicators│ │Engineering│ │  Analysis    │           │      │
│  │  └──────────┘ └──────────┘ └──────────────┘           │      │
│  └───────────────────────┬───────────────────────────────┘      │
│                          │                                      │
│  ┌───────────────────────▼───────────────────────────────┐      │
│  │              AI AGENT TEAMS (14 Agents)                │      │
│  │                                                        │      │
│  │  Team 1: Research    Team 2: Strategy    Team 3: Risk  │      │
│  │  ├─ Technical       ├─ Momentum         ├─ Risk Mgr   │      │
│  │  ├─ Fundamental     ├─ Value            ├─ Compliance  │      │
│  │  └─ Sentiment       ├─ Statistical                     │      │
│  │                     └─ ML Quant         Team 4: Exec   │      │
│  │  Team 5: Meta                           ├─ Executor    │      │
│  │  ├─ Debate          Team 4 cont:        └─ Portfolio   │      │
│  │  └─ Coordinator     └─ Allocation                      │      │
│  └───────────────────────┬───────────────────────────────┘      │
│                          │                                      │
│  ┌───────────────────────▼───────────────────────────────┐      │
│  │              EXECUTION ENGINE                          │      │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────────┐           │      │
│  │  │  Order   │ │  Broker  │ │   Position   │           │      │
│  │  │ Manager  │ │ Gateway  │ │   Tracker    │           │      │
│  │  └──────────┘ └──────────┘ └──────────────┘           │      │
│  └───────────────────────────────────────────────────────┘      │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  STORAGE: PostgreSQL │ InfluxDB │ ChromaDB              │    │
│  └─────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

## Features

- **14 Specialized AI Agents** across 5 collaborative teams
- **Multi-Source Data Pipeline** — market data, news, fundamentals, social sentiment
- **30+ Technical Indicators** with automatic computation
- **Advanced Feature Engineering** — returns, volatility, statistical features
- **NLP Sentiment Analysis** using FinBERT for financial text
- **Risk Management** — position limits, drawdown controls, exposure monitoring
- **Backtesting Engine** with comprehensive performance metrics
- **Triple Storage Layer** — PostgreSQL, InfluxDB, ChromaDB
- **Production-Grade Infrastructure** — Docker, CI/CD, monitoring

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.11+ |
| AI/LLM | Claude (Anthropic), GPT-4o (OpenAI) |
| Orchestration | LangGraph, LangChain |
| Market Data | yfinance, Alpaca |
| NLP | FinBERT (HuggingFace Transformers) |
| Relational DB | PostgreSQL 16 |
| Time Series DB | InfluxDB 2.7 |
| Vector DB | ChromaDB |
| Technical Analysis | ta (Python) |
| Data Processing | pandas, NumPy |
| Type Safety | Pydantic v2, mypy (strict) |
| CI/CD | GitHub Actions |
| Containerization | Docker, Docker Compose |

## Quick Start

### Prerequisites

- Python 3.11+
- Docker & Docker Compose
- API keys (Anthropic, Alpaca, NewsAPI)

### Installation

```bash
git clone https://github.com/kushalsai-01/nexus-financial-agent.git
cd nexus-financial-agent

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

# Install with dev dependencies
make install-dev
# or: pip install -e ".[dev]"

# Copy and configure environment
cp .env.example .env
# Edit .env with your API keys
```

### Infrastructure Setup

```bash
# Start all services (Postgres, InfluxDB, ChromaDB)
make docker-up

# Initialize database
make setup-db

# Seed historical data
make seed
```

### Running

```bash
# Run the system
python -m nexus

# Run backtest
python scripts/run_backtest.py --tickers AAPL MSFT GOOGL --start 2022-01-01

# Seed data for specific tickers
python scripts/seed_data.py AAPL MSFT NVDA
```

## Project Structure

```
nexus-financial-agent/
├── nexus/
│   ├── core/               # Foundation layer
│   │   ├── config.py        # Pydantic settings, YAML loading
│   │   ├── types.py         # All domain models & enums
│   │   ├── exceptions.py    # Custom exception hierarchy
│   │   └── logging.py       # Structured JSON logging
│   ├── data/
│   │   ├── providers/       # Data source integrations
│   │   │   ├── base.py      # Abstract providers + caching + rate limiting
│   │   │   ├── market.py    # yfinance market data
│   │   │   ├── news.py      # NewsAPI + RSS feeds
│   │   │   ├── fundamentals.py  # SEC EDGAR financials
│   │   │   └── social.py    # Reddit social data
│   │   ├── storage/         # Database integrations
│   │   │   ├── postgres.py  # SQLAlchemy models & CRUD
│   │   │   ├── timeseries.py # InfluxDB time series
│   │   │   └── vector.py    # ChromaDB vector store
│   │   ├── processors/      # Data transformation
│   │   │   ├── technical.py # 30+ TA indicators
│   │   │   ├── features.py  # Feature engineering
│   │   │   └── sentiment.py # FinBERT NLP analysis
│   │   └── pipeline.py      # Data orchestration
│   └── __init__.py
├── config/
│   ├── default.yaml         # Full configuration
│   ├── development.yaml     # Dev overrides
│   └── production.yaml      # Prod overrides
├── docker/
│   ├── Dockerfile           # Multi-stage build
│   ├── docker-compose.yml   # Full stack
│   └── postgres/init.sql    # DB schema
├── scripts/
│   ├── setup_db.py          # Database initialization
│   ├── seed_data.py         # Historical data loading
│   └── run_backtest.py      # Backtesting CLI
├── tests/
│   ├── unit/                # Unit tests
│   └── integration/         # Integration tests
├── .github/workflows/
│   ├── ci.yml               # CI pipeline
│   └── tests.yml            # Test matrix
├── pyproject.toml           # Project config & deps
├── Makefile                 # Dev commands
└── .env.example             # Environment template
```

## Configuration

Configuration uses a layered approach:

1. `config/default.yaml` — All defaults
2. `config/{environment}.yaml` — Environment overrides
3. Environment variables — Runtime overrides
4. `.env` file — Local secrets

```yaml
# Example: config/default.yaml
risk:
  max_position_size_pct: 5.0
  max_drawdown_pct: 10.0
  daily_loss_limit_pct: 3.0
  position_limit: 20

agents:
  consensus_threshold: 0.6
  teams:
    - name: research
      agents: [technical_analyst, fundamental_analyst, sentiment_analyst]
```

## Development

```bash
# Run tests
make test

# Run tests with coverage
make test-cov

# Lint
make lint

# Format
make format

# Type check
make typecheck

# All checks
make all
```

## Roadmap

- [x] **Prompt 1**: Foundation, Data Layer, Infrastructure
- [ ] **Prompt 2**: 14 AI Agents, LLM Integration, Orchestration
- [ ] **Prompt 3**: Backtesting Engine, Risk Management, Execution
- [ ] **Prompt 4**: Monitoring, CLI, Dashboard, Deployment

## License

Private — All rights reserved.
