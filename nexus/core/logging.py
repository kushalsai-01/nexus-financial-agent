from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class JSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        log_data: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        if record.exc_info and record.exc_info[1]:
            log_data["exception"] = {
                "type": type(record.exc_info[1]).__name__,
                "message": str(record.exc_info[1]),
            }

        if hasattr(record, "extra_data"):
            log_data["data"] = record.extra_data

        return json.dumps(log_data, default=str)


class NexusLogger:
    _instances: dict[str, NexusLogger] = {}

    def __init__(self, name: str, log_dir: str = "./logs", level: str = "INFO") -> None:
        self._logger = logging.getLogger(name)
        self._logger.setLevel(getattr(logging, level.upper(), logging.INFO))
        self._logger.propagate = False

        if not self._logger.handlers:
            self._setup_console_handler()
            self._setup_file_handler(log_dir, name)

    def _setup_console_handler(self) -> None:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(JSONFormatter())
        self._logger.addHandler(handler)

    def _setup_file_handler(self, log_dir: str, name: str) -> None:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)

        category = name.split(".")[1] if "." in name else "general"
        file_handler = logging.FileHandler(log_path / f"{category}.log")
        file_handler.setFormatter(JSONFormatter())
        self._logger.addHandler(file_handler)

    def debug(self, msg: str, *args: Any, **kwargs: Any) -> None:
        self._logger.debug(msg, *args, **kwargs)

    def info(self, msg: str, *args: Any, **kwargs: Any) -> None:
        self._logger.info(msg, *args, **kwargs)

    def warning(self, msg: str, *args: Any, **kwargs: Any) -> None:
        self._logger.warning(msg, *args, **kwargs)

    def error(self, msg: str, *args: Any, **kwargs: Any) -> None:
        self._logger.error(msg, *args, **kwargs)

    def critical(self, msg: str, *args: Any, **kwargs: Any) -> None:
        self._logger.critical(msg, *args, **kwargs)

    def with_data(self, **data: Any) -> NexusLogger:
        adapter = logging.LoggerAdapter(self._logger, {"extra_data": data})
        wrapper = NexusLogger.__new__(NexusLogger)
        wrapper._logger = adapter
        return wrapper


def get_logger(name: str, log_dir: str = "./logs", level: str = "INFO") -> NexusLogger:
    if name not in NexusLogger._instances:
        NexusLogger._instances[name] = NexusLogger(name, log_dir, level)
    return NexusLogger._instances[name]


def setup_logging(log_dir: str = "./logs", log_level: str = "INFO", log_format: str = "json") -> None:
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    logging.getLogger().setLevel(getattr(logging, log_level.upper(), logging.INFO))
