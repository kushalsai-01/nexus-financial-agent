from __future__ import annotations

try:
    from nexus.data.pipeline import DataPipeline
except ImportError:
    DataPipeline = None  # type: ignore[assignment,misc]

__all__ = ["DataPipeline"]
