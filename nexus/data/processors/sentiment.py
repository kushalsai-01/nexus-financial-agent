from __future__ import annotations

import asyncio
from typing import Any

import numpy as np

try:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

from nexus.core.logging import get_logger
from nexus.core.types import Sentiment

logger = get_logger("data.processors.sentiment")

_MODEL_NAME = "ProsusAI/finbert"


class SentimentAnalyzer:
    def __init__(self, model_name: str = _MODEL_NAME, device: str | None = None) -> None:
        self._model_name = model_name
        if _HAS_TORCH:
            self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self._device = device or "cpu"
        self._tokenizer: Any = None
        self._model: Any = None
        self._labels = ["positive", "negative", "neutral"]

    def _load_model(self) -> None:
        if not _HAS_TORCH:
            raise ImportError(
                "torch and transformers are required for SentimentAnalyzer. "
                "Install them with: pip install torch transformers"
            )
        if self._tokenizer is None:
            logger.info(f"Loading sentiment model: {self._model_name}")
            self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)
            self._model = AutoModelForSequenceClassification.from_pretrained(
                self._model_name
            )
            self._model.to(self._device)
            self._model.eval()

    def analyze(self, text: str) -> dict[str, Any]:
        self._load_model()

        inputs = self._tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            probs_np = probs.cpu().numpy()[0]

        scores = {label: float(prob) for label, prob in zip(self._labels, probs_np)}
        sentiment_idx = np.argmax(probs_np)
        sentiment_label = self._labels[sentiment_idx]

        compound_score = scores.get("positive", 0) - scores.get("negative", 0)

        return {
            "label": sentiment_label,
            "compound_score": compound_score,
            "scores": scores,
            "confidence": float(probs_np[sentiment_idx]),
        }

    def analyze_batch(self, texts: list[str]) -> list[dict[str, Any]]:
        self._load_model()

        results: list[dict[str, Any]] = []
        batch_size = 16

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            inputs = self._tokenizer(
                batch,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True,
            )
            inputs = {k: v.to(self._device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self._model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
                probs_np = probs.cpu().numpy()

            for j, prob in enumerate(probs_np):
                scores = {label: float(p) for label, p in zip(self._labels, prob)}
                idx = np.argmax(prob)
                compound = scores.get("positive", 0) - scores.get("negative", 0)
                results.append(
                    {
                        "text": batch[j][:100],
                        "label": self._labels[idx],
                        "compound_score": compound,
                        "scores": scores,
                        "confidence": float(prob[idx]),
                    }
                )

        return results

    async def analyze_async(self, text: str) -> dict[str, Any]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.analyze, text)

    async def analyze_batch_async(self, texts: list[str]) -> list[dict[str, Any]]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.analyze_batch, texts)

    def to_sentiment_enum(self, compound_score: float) -> Sentiment:
        if compound_score > 0.5:
            return Sentiment.VERY_BULLISH
        elif compound_score > 0.15:
            return Sentiment.BULLISH
        elif compound_score > -0.15:
            return Sentiment.NEUTRAL
        elif compound_score > -0.5:
            return Sentiment.BEARISH
        else:
            return Sentiment.VERY_BEARISH

    def aggregate_sentiment(
        self, results: list[dict[str, Any]]
    ) -> dict[str, Any]:
        if not results:
            return {
                "overall_sentiment": Sentiment.NEUTRAL.value,
                "avg_compound": 0.0,
                "positive_pct": 0.0,
                "negative_pct": 0.0,
                "neutral_pct": 0.0,
                "total_analyzed": 0,
            }

        compounds = [r["compound_score"] for r in results]
        avg_compound = np.mean(compounds)
        labels = [r["label"] for r in results]
        total = len(labels)

        return {
            "overall_sentiment": self.to_sentiment_enum(avg_compound).value,
            "avg_compound": float(avg_compound),
            "median_compound": float(np.median(compounds)),
            "std_compound": float(np.std(compounds)),
            "positive_pct": labels.count("positive") / total * 100,
            "negative_pct": labels.count("negative") / total * 100,
            "neutral_pct": labels.count("neutral") / total * 100,
            "total_analyzed": total,
            "avg_confidence": float(np.mean([r["confidence"] for r in results])),
        }
