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
[![License: Private](https://img.shields.io/badge/license-private-red.svg)]()

A production-grade autonomous hedge fund system powered by a multi-agent architecture where **14 specialized AI agents** collaborate across **5 teams** to analyze markets, generate trading signals, manage risk, and execute trades — with a beautiful terminal UI, web dashboard, comprehensive monitoring, and full deployment tooling.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│                            NEXUS SYSTEM                                  │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  DATA LAYER             AGENT LAYER              EXECUTION LAYER         │
│  ┌────────────────┐     ┌────────────────────┐   ┌──────────────────┐   │
│  │ Market Data    │────▶│ 🔬 Analyst Team    │   │ Order Manager    │   │
│  │ News Feeds     │     │  market_data        │   │ Broker Gateway   │   │
│  │ SEC Filings    │     │  sentiment          │──▶│ Position Tracker │   │
│  │ Social Media   │     │  fundamental        │   │ Slippage Model   │   │
│  └────────┬───────┘     ├────────────────────┤   └──────────────────┘   │
│           │             │ 📊 Quant Team       │                          │
│  ┌────────▼───────┐     │  technical          │   MONITORING             │
│  │ 30+ Indicators │     │  quantitative       │   ┌──────────────────┐   │
│  │ Feature Eng.   │     │  macro              │   │ Langfuse Tracing │   │
│  │ FinBERT NLP    │     ├────────────────────┤   │ Prometheus       │   │
│  └────────────────┘     │ 🔎 Research Team    │   │ Cost Tracking    │   │
│                         │  bull / bear         │   │ Health Checks    │   │
│  STORAGE                │  coordinator         │   │ Alerting         │   │
│  ┌────────────────┐     ├────────────────────┤   └──────────────────┘   │
│  │ PostgreSQL 16  │     │ ⚡ Strategy Team    │                          │
│  │ InfluxDB 2.7   │     │  event / rl_agent   │   UI LAYER               │
│  │ ChromaDB       │     │  risk               │   ┌──────────────────┐   │
│  └────────────────┘     ├────────────────────┤   │ Rich Terminal UI │   │
│                         │ 💰 Execution Team   │   │ Streamlit Dash.  │   │
│                         │  portfolio           │   │ HTML Reports     │   │
│                         │  execution           │   │ Click CLI        │   │
│                         └────────────────────┘   └──────────────────┘   │
└──────────────────────────────────────────────────────────────────────────┘
```

## Features

### Core Intelligence
- **14 Specialized AI Agents** across 5 collaborative teams (Analyst, Research, Quant, Strategy, Execution)
- **Multi-LLM Support** — Claude (Anthropic), GPT-4o (OpenAI) with automatic fallback & cost tracking
- **LangGraph Orchestration** — Stateful agent workflows with debate & consensus mechanisms
- **Multi-Source Data Pipeline** — Market data, news, fundamentals, social sentiment

### Analysis & Trading
- **30+ Technical Indicators** with automatic computation (RSI, MACD, Bollinger, etc.)
- **Advanced Feature Engineering** — Returns, volatility, rolling statistics, z-scores
- **FinBERT Sentiment Analysis** — NLP on financial news and social media
- **Backtesting Engine** — Walk-forward, Monte Carlo, comprehensive metrics (Sharpe, Sortino, Calmar)
- **Risk Management** — Position limits, drawdown controls, VaR, exposure monitoring
- **Execution Engine** — Simulated and paper trading with slippage modeling

### Observability & Monitoring
- **Langfuse/LangSmith Tracing** — Full LLM call tracing with cost attribution
- **Prometheus Metrics** — Decisions/sec, API latency, portfolio value, agent accuracy
- **Smart Alerting** — Slack, PagerDuty, Email, Webhook with cooldown & severity routing
- **Health Checks** — Database, LLM providers, market data, broker connectivity
- **LLM Cost Tracking** — Per-agent, per-model, daily budgets with automatic warnings

### User Interface
- **Beautiful Terminal UI** — Rich-powered live dashboard with 4 FPS refresh, agent progress, message feed
- **Web Dashboard** — Streamlit + Plotly with 5 pages (Overview, Agents, Backtests, Trades, Risk)
- **HTML Reports** — Daily, weekly, and backtest reports with dark-themed templates
- **Click CLI** — Full command-line interface (run, backtest, analyze, report, status, dashboard, costs, init)

### Infrastructure
- **Docker + Docker Compose** — One-command development stack
- **Kubernetes Manifests** — Production deployment with probes, resource limits, ingress
- **Terraform** — AWS infrastructure (VPC, RDS, ECR, security groups)
- **CI/CD** — GitHub Actions with lint, type-check, test matrix

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.11+ |
| AI/LLM | Claude (Anthropic), GPT-4o (OpenAI) |
| Orchestration | LangGraph, LangChain |
| Market Data | yfinance, Alpaca |
| NLP | FinBERT (HuggingFace Transformers) |
| Databases | PostgreSQL 16, InfluxDB 2.7, ChromaDB |
| Terminal UI | Rich (Live, Layout, Panel, Table) |
| Dashboard | Streamlit, Plotly |
| CLI | Click |
| Reports | Jinja2 HTML templates |
| Monitoring | Langfuse, Prometheus, Slack/PagerDuty |
| Type Safety | Pydantic v2, mypy (strict) |
| Testing | pytest, pytest-asyncio, pytest-cov |
| CI/CD | GitHub Actions |
| Deployment | Docker, Kubernetes, Terraform |

## Quick Start

### Prerequisites

- Python 3.11+
- Docker & Docker Compose (optional, for full stack)
- API keys (Anthropic and/or OpenAI)

### Installation

```bash
git clone https://github.com/kushalsai-01/nexus-financial-agent.git
cd nexus-financial-agent

python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

pip install -e ".[dev]"

cp .env.example .env
# Edit .env with your API keys
```

### CLI Usage

```bash
# Initialize a new project
nexus init

# Run with terminal UI (demo mode)
nexus run -t AAPL -t MSFT --demo

# Run live analysis
nexus run -t AAPL -t NVDA -t GOOGL --capital 100000

# Backtest a strategy
nexus backtest -t AAPL -s 2023-01-01 -e 2024-01-01 --format html -o report.html

# Quick analysis
nexus analyze -t TSLA --depth deep

# Generate reports
nexus report --type daily --format html -o daily.html

# System health check
nexus status

# Launch web dashboard
nexus dashboard

# LLM cost summary
nexus costs
```

### Docker

```bash
# Start full stack (Postgres, InfluxDB, ChromaDB, NEXUS)
cd docker && docker compose up -d

# Or use the deploy script
bash deploy/deploy.sh up
bash deploy/deploy.sh status
bash deploy/deploy.sh logs
```

### Kubernetes

```bash
# Deploy to K8s cluster
bash deploy/deploy.sh deploy-k8s

# Or manually
kubectl apply -f deploy/k8s/namespace.yaml
kubectl apply -f deploy/k8s/configmap.yaml
kubectl apply -f deploy/k8s/services.yaml
kubectl apply -f deploy/k8s/deployment.yaml
```

## Project Structure

```
nexus-financial-agent/
├── nexus/
│   ├── core/                    # Foundation layer
│   │   ├── config.py             # Pydantic settings, YAML loading
│   │   ├── types.py              # Domain models (Signal, Trade, Portfolio, etc.)
│   │   ├── exceptions.py         # NexusError hierarchy (14 exception types)
│   │   └── logging.py            # Structured JSON logging
│   ├── data/
│   │   ├── providers/            # Data sources (market, news, fundamentals, social)
│   │   ├── storage/              # PostgreSQL, InfluxDB, ChromaDB
│   │   ├── processors/           # Technical indicators, features, sentiment
│   │   └── pipeline.py           # Data orchestration
│   ├── agents/                   # 14 AI agents
│   │   ├── base.py               # BaseAgent with LLM integration
│   │   ├── market_data.py        # Market data collection
│   │   ├── technical.py          # Technical analysis
│   │   ├── fundamental.py        # Fundamental analysis
│   │   ├── sentiment.py          # Sentiment analysis
│   │   ├── quantitative.py       # Quantitative modeling
│   │   ├── macro.py              # Macroeconomic analysis
│   │   ├── event.py              # Event-driven strategies
│   │   ├── rl_agent.py           # Reinforcement learning
│   │   ├── bull.py / bear.py     # Debate agents
│   │   ├── risk.py               # Risk assessment
│   │   ├── portfolio.py          # Portfolio optimization
│   │   ├── execution.py          # Trade execution
│   │   └── coordinator.py        # Multi-agent coordination
│   ├── orchestration/            # LangGraph workflow
│   │   ├── graph.py              # TradingGraph with run()
│   │   └── state.py              # TradingState TypedDict
│   ├── llm/                      # LLM integration
│   │   ├── providers.py          # Anthropic, OpenAI, LLMRouter
│   │   └── prompts.py            # System & agent prompts
│   ├── backtest/                 # Backtesting engine
│   │   ├── engine.py             # Walk-forward backtester
│   │   ├── metrics.py            # Sharpe, Sortino, Calmar, etc.
│   │   └── reports.py            # Backtest result formatting
│   ├── risk/                     # Risk management
│   │   ├── monitor.py            # RiskMonitor with alerts
│   │   └── limits.py             # Position/exposure limits
│   ├── execution/                # Order execution
│   │   ├── engine.py             # SimulatedBroker, PaperBroker
│   │   └── models.py             # Order, Fill, SlippageModel
│   ├── analysis/                 # Advanced analytics
│   │   ├── correlation.py        # Correlation & regime detection
│   │   ├── performance.py        # Performance attribution
│   │   └── optimization.py       # Portfolio optimization
│   ├── monitoring/               # Observability stack
│   │   ├── trace.py              # Langfuse/LangSmith tracing
│   │   ├── metrics.py            # Prometheus metrics & counters
│   │   ├── alerts.py             # Slack/PagerDuty/Email alerting
│   │   ├── health.py             # Health checks
│   │   └── cost.py               # LLM cost tracking
│   ├── reports/                  # Report generation
│   │   ├── daily.py              # Daily P&L report
│   │   ├── weekly.py             # Weekly performance summary
│   │   ├── backtest.py           # Backtest report with monthly heatmap
│   │   └── templates/            # Jinja2 HTML templates
│   ├── ui/                       # User interfaces
│   │   ├── tui.py                # Rich terminal UI (4 FPS live)
│   │   ├── themes.py             # Color themes & styling
│   │   ├── components.py         # UI components (tables, panels, bars)
│   │   ├── layouts.py            # Layout managers
│   │   └── dashboard/            # Streamlit web dashboard
│   │       ├── app.py             # Main app with navigation
│   │       └── pages/             # Overview, Agents, Backtests, Trades, Risk
│   └── cli/                      # Click CLI
│       └── main.py               # Commands: run, backtest, analyze, report, etc.
├── deploy/
│   ├── k8s/                      # Kubernetes manifests
│   ├── terraform/                # AWS infrastructure
│   └── deploy.sh                 # Deployment script
├── docker/
│   ├── Dockerfile                # Multi-stage build
│   └── docker-compose.yml        # Full development stack
├── config/                       # YAML configuration
├── tests/                        # Unit & integration tests
├── .github/workflows/            # CI/CD pipelines
├── pyproject.toml                # Project config & all dependencies
└── Makefile                      # Developer shortcuts
```

## Configuration

Layered configuration: `config/default.yaml` → `config/{env}.yaml` → env vars → `.env`

```yaml
risk:
  max_position_size_pct: 5.0
  max_drawdown_pct: 10.0
  daily_loss_limit_pct: 3.0
  position_limit: 20

monitoring:
  prometheus:
    enabled: true
    port: 9090
  langfuse:
    enabled: true
```

## Development

```bash
make test          # Run test suite
make test-cov      # Tests with coverage
make lint          # Ruff linting
make format        # Black formatting
make typecheck     # mypy strict
make all           # All checks
```

## License

Private — All rights reserved.
