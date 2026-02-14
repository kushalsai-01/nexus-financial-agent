from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class LoggingConfig(BaseModel):
    level: str = "INFO"
    log_dir: str = "logs"
    format: str = "json"


class LLMProviderConfig(BaseModel):
    provider: str = "anthropic"
    model: str = "claude-sonnet-4-20250514"
    max_tokens: int = 4096
    temperature: float = 0.1
    timeout: int = 60
    max_retries: int = 3


class LLMConfig(BaseModel):
    primary: LLMProviderConfig = Field(default_factory=LLMProviderConfig)
    fallback: LLMProviderConfig = Field(
        default_factory=lambda: LLMProviderConfig(
            provider="openai",
            model="gpt-4o",
            max_tokens=4096,
            temperature=0.1,
            timeout=60,
            max_retries=3,
        )
    )


class MarketDataConfig(BaseModel):
    provider: str = "yfinance"
    cache_ttl: int = 300
    rate_limit: int = 5
    default_timeframe: str = "1Day"
    max_history_days: int = 365


class NewsDataConfig(BaseModel):
    newsapi_base_url: str = "https://newsapi.org/v2"
    rss_feeds: list[str] = Field(default_factory=list)
    max_articles: int = 100
    dedup_window_hours: int = 24
    scrape_timeout: int = 10


class FundamentalsConfig(BaseModel):
    sec_base_url: str = "https://data.sec.gov"
    cache_ttl: int = 86400
    user_agent: str = "Nexus Fund admin@nexusfund.ai"
    filings_lookback: int = 8


class SocialConfig(BaseModel):
    reddit_subreddits: list[str] = Field(
        default_factory=lambda: ["wallstreetbets", "stocks", "investing"]
    )
    max_posts: int = 100
    sentiment_threshold: float = 0.3


class DataConfig(BaseModel):
    market: MarketDataConfig = Field(default_factory=MarketDataConfig)
    news: NewsDataConfig = Field(default_factory=NewsDataConfig)
    fundamentals: FundamentalsConfig = Field(default_factory=FundamentalsConfig)
    social: SocialConfig = Field(default_factory=SocialConfig)


class PostgresConfig(BaseModel):
    host: str = "localhost"
    port: int = 5432
    database: str = "nexus"
    user: str = "nexus"
    password: str = "nexus_dev"
    pool_size: int = 10
    max_overflow: int = 20
    echo: bool = False

    @property
    def url(self) -> str:
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"


class InfluxDBConfig(BaseModel):
    url: str = "http://localhost:8086"
    token: str = ""
    org: str = "nexus"
    bucket: str = "market_data"
    timeout: int = 30000


class ChromaDBConfig(BaseModel):
    host: str = "localhost"
    port: int = 8000
    collection_prefix: str = "nexus"
    embedding_model: str = "all-MiniLM-L6-v2"


class StorageConfig(BaseModel):
    postgres: PostgresConfig = Field(default_factory=PostgresConfig)
    influxdb: InfluxDBConfig = Field(default_factory=InfluxDBConfig)
    chromadb: ChromaDBConfig = Field(default_factory=ChromaDBConfig)


class AgentTeamConfig(BaseModel):
    name: str
    agents: list[str] = Field(default_factory=list)
    weight: float = 1.0


class AgentsConfig(BaseModel):
    consensus_threshold: float = 0.6
    max_concurrent: int = 5
    timeout_seconds: int = 120
    teams: list[AgentTeamConfig] = Field(default_factory=list)


class RiskConfig(BaseModel):
    max_position_size_pct: float = 5.0
    max_portfolio_risk_pct: float = 15.0
    max_sector_exposure_pct: float = 30.0
    max_correlation: float = 0.7
    max_drawdown_pct: float = 10.0
    daily_loss_limit_pct: float = 3.0
    max_leverage: float = 1.0
    min_cash_pct: float = 10.0
    position_limit: int = 20
    var_confidence: float = 0.95
    stop_loss_pct: float = 5.0
    trailing_stop_pct: float = 3.0


class ExecutionConfig(BaseModel):
    broker: str = "alpaca"
    mode: str = "paper"
    max_slippage_pct: float = 0.1
    retry_attempts: int = 3
    retry_delay: float = 1.0
    order_timeout: int = 60


class LangfuseConfig(BaseModel):
    enabled: bool = False
    public_key: str = ""
    secret_key: str = ""
    host: str = "https://cloud.langfuse.com"


class WandbConfig(BaseModel):
    enabled: bool = False
    project: str = "nexus"
    entity: str = ""


class PrometheusConfig(BaseModel):
    enabled: bool = False
    port: int = 9090


class MonitoringConfig(BaseModel):
    langfuse: LangfuseConfig = Field(default_factory=LangfuseConfig)
    wandb: WandbConfig = Field(default_factory=WandbConfig)
    prometheus: PrometheusConfig = Field(default_factory=PrometheusConfig)


class BacktestConfig(BaseModel):
    initial_capital: float = 1_000_000.0
    commission_pct: float = 0.001
    slippage_pct: float = 0.0005
    benchmark: str = "SPY"
    data_start: str = "2020-01-01"
    data_end: str = "2024-01-01"


class NexusConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="NEXUS_",
        env_nested_delimiter="__",
        extra="ignore",
    )

    env: str = "development"
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    agents: AgentsConfig = Field(default_factory=AgentsConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)
    backtest: BacktestConfig = Field(default_factory=BacktestConfig)

    anthropic_api_key: str = ""
    openai_api_key: str = ""
    alpaca_api_key: str = ""
    alpaca_api_secret: str = ""
    newsapi_key: str = ""
    reddit_client_id: str = ""
    reddit_client_secret: str = ""


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = base.copy()
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _resolve_env_vars(data: Any) -> Any:
    if isinstance(data, str) and data.startswith("${") and data.endswith("}"):
        env_expr = data[2:-1]
        if ":-" in env_expr:
            var_name, default = env_expr.split(":-", 1)
            return os.getenv(var_name, default)
        return os.getenv(env_expr, "")
    if isinstance(data, dict):
        return {k: _resolve_env_vars(v) for k, v in data.items()}
    if isinstance(data, list):
        return [_resolve_env_vars(item) for item in data]
    return data


def _load_yaml_config(config_dir: Path, env: str) -> dict[str, Any]:
    default_path = config_dir / "default.yaml"
    env_path = config_dir / f"{env}.yaml"

    config: dict[str, Any] = {}

    if default_path.exists():
        with open(default_path) as f:
            default_data = yaml.safe_load(f)
            if default_data:
                config = default_data

    if env_path.exists():
        with open(env_path) as f:
            env_data = yaml.safe_load(f)
            if env_data:
                config = _deep_merge(config, env_data)

    return _resolve_env_vars(config)


@lru_cache(maxsize=1)
def get_config(
    config_dir: str | None = None,
    env: str | None = None,
) -> NexusConfig:
    environment = env or os.getenv("NEXUS_ENV", "development")

    if config_dir:
        cfg_path = Path(config_dir)
    else:
        cfg_path = Path(__file__).parent.parent.parent / "config"

    yaml_data = _load_yaml_config(cfg_path, environment)
    yaml_data["env"] = environment

    for key in [
        "ANTHROPIC_API_KEY",
        "OPENAI_API_KEY",
        "ALPACA_API_KEY",
        "ALPACA_API_SECRET",
        "NEWSAPI_KEY",
        "REDDIT_CLIENT_ID",
        "REDDIT_CLIENT_SECRET",
    ]:
        env_val = os.getenv(key, "")
        if env_val:
            yaml_data[key.lower()] = env_val

    return NexusConfig(**yaml_data)


def reset_config() -> None:
    get_config.cache_clear()
