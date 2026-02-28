from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from nexus.core.config import (
    NexusConfig,
    _deep_merge,
    _resolve_env_vars,
    get_config,
    reset_config,
)


class TestDeepMerge:
    def test_simple_merge(self) -> None:
        base = {"a": 1, "b": 2}
        override = {"b": 3, "c": 4}
        result = _deep_merge(base, override)
        assert result == {"a": 1, "b": 3, "c": 4}

    def test_nested_merge(self) -> None:
        base = {"a": {"x": 1, "y": 2}, "b": 3}
        override = {"a": {"y": 5, "z": 6}}
        result = _deep_merge(base, override)
        assert result == {"a": {"x": 1, "y": 5, "z": 6}, "b": 3}

    def test_empty_override(self) -> None:
        base = {"a": 1}
        result = _deep_merge(base, {})
        assert result == {"a": 1}


class TestResolveEnvVars:
    def test_resolve_env_var(self) -> None:
        with patch.dict(os.environ, {"TEST_VAR": "hello"}):
            result = _resolve_env_vars("${TEST_VAR}")
            assert result == "hello"

    def test_resolve_with_default(self) -> None:
        result = _resolve_env_vars("${NONEXISTENT_VAR:-fallback}")
        assert result == "fallback"

    def test_resolve_nested_dict(self) -> None:
        with patch.dict(os.environ, {"DB_HOST": "localhost"}):
            data = {"host": "${DB_HOST}", "port": 5432}
            result = _resolve_env_vars(data)
            assert result == {"host": "localhost", "port": 5432}

    def test_resolve_list(self) -> None:
        with patch.dict(os.environ, {"VAL": "x"}):
            data = ["${VAL}", "static"]
            result = _resolve_env_vars(data)
            assert result == ["x", "static"]

    def test_no_resolution_needed(self) -> None:
        assert _resolve_env_vars("plain_text") == "plain_text"
        assert _resolve_env_vars(42) == 42


class TestNexusConfig:
    def setup_method(self) -> None:
        reset_config()

    def test_default_config(self) -> None:
        config = NexusConfig()
        # conftest sets NEXUS_ENV=test, so pydantic-settings picks that up
        assert config.env in ("development", "test")
        assert config.logging.level == "INFO"
        assert config.risk.max_position_size_pct == 5.0

    def test_config_risk_defaults(self) -> None:
        config = NexusConfig()
        assert config.risk.max_drawdown_pct == 10.0
        assert config.risk.daily_loss_limit_pct == 3.0
        assert config.risk.position_limit == 20

    def test_config_execution_defaults(self) -> None:
        config = NexusConfig()
        assert config.execution.broker == "alpaca"
        assert config.execution.mode == "paper"

    def test_postgres_url(self) -> None:
        config = NexusConfig()
        url = config.storage.postgres.url
        assert "postgresql://" in url
        assert "nexus" in url

    def test_config_from_yaml(self) -> None:
        config_dir = Path(__file__).parent.parent.parent / "config"
        if config_dir.exists():
            config = get_config(config_dir=str(config_dir), env="development")
            assert config.env == "development"
