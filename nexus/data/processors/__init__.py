from __future__ import annotations

from nexus.data.processors.features import FeatureEngineer
from nexus.data.processors.technical import TechnicalAnalyzer

# SentimentAnalyzer depends on torch/transformers — import lazily
try:
    from nexus.data.processors.sentiment import SentimentAnalyzer
except ImportError:
    SentimentAnalyzer = None  # type: ignore[assignment,misc]

__all__ = [
    "TechnicalAnalyzer",
    "FeatureEngineer",
    "SentimentAnalyzer",
]
