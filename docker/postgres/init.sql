CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

CREATE TABLE IF NOT EXISTS prices (
    id SERIAL PRIMARY KEY,
    ticker VARCHAR(20) NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    open DOUBLE PRECISION NOT NULL,
    high DOUBLE PRECISION NOT NULL,
    low DOUBLE PRECISION NOT NULL,
    close DOUBLE PRECISION NOT NULL,
    volume BIGINT NOT NULL,
    vwap DOUBLE PRECISION,
    timeframe VARCHAR(10) DEFAULT '1Day'
);

CREATE INDEX IF NOT EXISTS ix_prices_ticker ON prices(ticker);
CREATE INDEX IF NOT EXISTS ix_prices_ticker_timestamp ON prices(ticker, timestamp);

CREATE TABLE IF NOT EXISTS fundamentals (
    id SERIAL PRIMARY KEY,
    ticker VARCHAR(20) NOT NULL,
    period VARCHAR(20) NOT NULL,
    fiscal_date TIMESTAMP NOT NULL,
    revenue DOUBLE PRECISION,
    net_income DOUBLE PRECISION,
    eps DOUBLE PRECISION,
    pe_ratio DOUBLE PRECISION,
    pb_ratio DOUBLE PRECISION,
    debt_to_equity DOUBLE PRECISION,
    roe DOUBLE PRECISION,
    roa DOUBLE PRECISION,
    current_ratio DOUBLE PRECISION,
    free_cash_flow DOUBLE PRECISION,
    gross_margin DOUBLE PRECISION,
    operating_margin DOUBLE PRECISION,
    market_cap DOUBLE PRECISION,
    data JSONB
);

CREATE INDEX IF NOT EXISTS ix_fundamentals_ticker ON fundamentals(ticker);
CREATE INDEX IF NOT EXISTS ix_fundamentals_ticker_date ON fundamentals(ticker, fiscal_date);

CREATE TABLE IF NOT EXISTS trades (
    id SERIAL PRIMARY KEY,
    trade_id VARCHAR(64) UNIQUE NOT NULL,
    ticker VARCHAR(20) NOT NULL,
    side VARCHAR(10) NOT NULL,
    quantity DOUBLE PRECISION NOT NULL,
    price DOUBLE PRECISION NOT NULL,
    commission DOUBLE PRECISION DEFAULT 0.0,
    slippage DOUBLE PRECISION DEFAULT 0.0,
    timestamp TIMESTAMP DEFAULT NOW(),
    order_id VARCHAR(64),
    agent_name VARCHAR(64) DEFAULT '',
    pnl DOUBLE PRECISION,
    pnl_pct DOUBLE PRECISION,
    metadata JSONB
);

CREATE INDEX IF NOT EXISTS ix_trades_ticker ON trades(ticker);
CREATE INDEX IF NOT EXISTS ix_trades_ticker_timestamp ON trades(ticker, timestamp);

CREATE TABLE IF NOT EXISTS positions (
    id SERIAL PRIMARY KEY,
    ticker VARCHAR(20) NOT NULL UNIQUE,
    quantity DOUBLE PRECISION NOT NULL,
    entry_price DOUBLE PRECISION NOT NULL,
    current_price DOUBLE PRECISION DEFAULT 0.0,
    market_value DOUBLE PRECISION DEFAULT 0.0,
    unrealized_pnl DOUBLE PRECISION DEFAULT 0.0,
    side VARCHAR(10) DEFAULT 'buy',
    opened_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS agent_decisions (
    id SERIAL PRIMARY KEY,
    agent_name VARCHAR(64) NOT NULL,
    ticker VARCHAR(20) NOT NULL,
    signal_type VARCHAR(20) NOT NULL,
    confidence DOUBLE PRECISION NOT NULL,
    reasoning TEXT DEFAULT '',
    target_price DOUBLE PRECISION,
    stop_loss DOUBLE PRECISION,
    timestamp TIMESTAMP DEFAULT NOW(),
    cost_usd DOUBLE PRECISION DEFAULT 0.0,
    latency_ms DOUBLE PRECISION DEFAULT 0.0,
    metadata JSONB
);

CREATE INDEX IF NOT EXISTS ix_decisions_agent ON agent_decisions(agent_name);
CREATE INDEX IF NOT EXISTS ix_decisions_ticker ON agent_decisions(ticker);
CREATE INDEX IF NOT EXISTS ix_decisions_agent_timestamp ON agent_decisions(agent_name, timestamp);

CREATE TABLE IF NOT EXISTS news (
    id SERIAL PRIMARY KEY,
    title TEXT NOT NULL,
    source VARCHAR(128) NOT NULL,
    url TEXT NOT NULL,
    published_at TIMESTAMP NOT NULL,
    content TEXT DEFAULT '',
    tickers JSONB,
    sentiment_score DOUBLE PRECISION,
    content_hash VARCHAR(64) UNIQUE
);

CREATE INDEX IF NOT EXISTS ix_news_published ON news(published_at);

CREATE TABLE IF NOT EXISTS portfolio_snapshots (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP DEFAULT NOW(),
    cash DOUBLE PRECISION NOT NULL,
    total_value DOUBLE PRECISION NOT NULL,
    positions_value DOUBLE PRECISION NOT NULL,
    daily_pnl DOUBLE PRECISION DEFAULT 0.0,
    total_pnl DOUBLE PRECISION DEFAULT 0.0,
    total_pnl_pct DOUBLE PRECISION DEFAULT 0.0,
    max_drawdown DOUBLE PRECISION DEFAULT 0.0,
    sharpe_ratio DOUBLE PRECISION DEFAULT 0.0,
    positions_data JSONB
);

CREATE INDEX IF NOT EXISTS ix_snapshots_timestamp ON portfolio_snapshots(timestamp);
