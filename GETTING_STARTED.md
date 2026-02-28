# NEXUS — Getting Started Guide

## What Is This?

NEXUS is an **AI-powered hedge fund simulation** that uses 14 specialized AI agents working together to analyze stocks and make trading decisions — like having a team of Wall Street analysts powered by AI.

## How The Agents Work (Simple Version)

Think of NEXUS like a trading desk with specialists:

| Agent | Role | What It Does |
|-------|------|-------------|
| **Market Data** | Data Collector | Pulls live prices, volumes, and charts |
| **Technical** | Chart Reader | Analyzes patterns (moving averages, RSI, MACD) |
| **Fundamental** | Balance Sheet Analyst | Reads SEC filings, revenue, earnings |
| **Quantitative** | Quant Strategist | Runs statistical models and signals |
| **Sentiment** | News/Social Reader | Scans Reddit, news headlines for mood |
| **Macro** | Economist | Tracks interest rates, inflation, GDP |
| **Event** | Catalyst Tracker | Watches for earnings dates, FDA approvals |
| **RL Agent** | Learning Bot | Reinforcement-learning model that adapts |
| **Bull** | Optimist | Argues *for* buying a stock |
| **Bear** | Pessimist | Argues *against* buying a stock |
| **Risk** | Risk Manager | Sets stop-losses, position limits, drawdown caps |
| **Portfolio** | Portfolio Manager | Decides final allocation across all positions |
| **Execution** | Trader | Places the orders with optimal timing |
| **Coordinator** | Team Lead | Orchestrates debate and reaches consensus |

### Flow:
1. **Data agents** (Market Data, Macro, Event) gather information
2. **Analysis agents** (Technical, Fundamental, Quant, Sentiment) each form opinions
3. **Debate agents** (Bull vs Bear) argue for/against each trade
4. **Decision agents** (Risk, Portfolio, Coordinator) reach consensus
5. **Execution agent** places the trade

Every agent is powered by the **same LLM** (Gemini by default) but with different system prompts that give each one its specialty.

---

## Setup (5 Minutes)

### 1. Clone & Install

```bash
git clone https://github.com/YOUR_USERNAME/Nexus-financial-agent.git
cd Nexus-financial-agent
pip install -e ".[dev]"
```

### 2. Get a Free Gemini API Key

1. Go to **https://aistudio.google.com/apikey**
2. Sign in with Google
3. Click "Create API Key"
4. Copy the key

### 3. Set Your Key

**Option A — .env file (recommended):**
```bash
cp .env.example .env
```
Edit `.env` and paste your key:
```
GEMINI_API_KEY=AIzaSy...your-key-here
```

**Option B — environment variable:**
```bash
# Windows PowerShell
$env:GEMINI_API_KEY = "AIzaSy...your-key-here"

# macOS/Linux
export GEMINI_API_KEY="AIzaSy...your-key-here"
```

### 4. Verify It Works

```bash
python -m nexus status    # Check system health
python -m nexus --help    # See all commands
```

---

## Running NEXUS

### Quick Analysis (No Trading)
```bash
python -m nexus analyze AAPL MSFT TSLA
```

### Run a Backtest
```bash
python -m nexus backtest --tickers AAPL,MSFT --start 2024-01-01 --end 2024-12-31
```

### Start the Dashboard
```bash
python -m nexus dashboard
```
Opens a Streamlit web UI at `http://localhost:8501`

### Full Trading Run (Paper Mode)
```bash
python -m nexus run --tickers AAPL,MSFT,GOOGL,AMZN,NVDA
```

---

## LLM Provider Options

| Provider | Cost | How to Get Key |
|----------|------|---------------|
| **Gemini** (default) | **FREE** | https://aistudio.google.com/apikey |
| **Ollama** (fallback) | **FREE** (local) | Install from https://ollama.ai then `ollama pull llama3.1:8b` |
| Grok (xAI) | Paid | https://console.x.ai |
| Groq | Free tier | https://console.groq.com |
| OpenAI | Paid | https://platform.openai.com |
| Anthropic | Paid | https://console.anthropic.com |

**Default setup:** Gemini for all agents (free), Ollama as offline fallback.

To switch providers, edit `config/default.yaml`:
```yaml
llm:
  primary:
    provider: gemini          # or: grok, openai, anthropic, groq, ollama
    model: gemini-2.0-flash   # model name for chosen provider
```

---

## Deploying for Friends

### Option 1: Share Repo + Key Instructions
1. Push to GitHub
2. Share this guide — they just need `pip install -e .` + a free Gemini key

### Option 2: HuggingFace Spaces (Web Demo)
The project includes `app.py` for HuggingFace deployment:
```bash
# Push to HF Spaces (auto-deploys via GitHub Actions)
git push origin main
```
Set `GEMINI_API_KEY` as a Space secret in Settings.

### Option 3: Docker
```bash
cd docker
docker-compose up
```
Access dashboard at `http://localhost:8501`

---

## Project Structure (Quick Reference)

```
nexus/
├── agents/       # 14 AI agents (bull, bear, risk, etc.)
├── llm/          # LLM providers (Gemini, Ollama, OpenAI, etc.)
├── orchestration/# Agent coordination & routing
├── data/         # Market data fetching & processing
├── risk/         # Risk management & limits
├── execution/    # Order placement & simulation
├── backtest/     # Historical backtesting engine
├── ui/           # Terminal UI & Streamlit dashboard
├── monitoring/   # Health checks & cost tracking
└── reports/      # Daily/weekly report generation
```

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| "No LLM API key set" | Set `GEMINI_API_KEY` in `.env` |
| "Cannot connect to Ollama" | Run `ollama serve` first, or just use Gemini |
| Import errors with torch | Normal — torch is optional, only needed for RL agent |
| Tests fail on `influxdb_client` | Optional dep — install with `pip install influxdb-client` if needed |
