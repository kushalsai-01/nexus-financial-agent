from __future__ import annotations

import hashlib
import json
import time
from typing import Any

from nexus.core.logging import get_logger

logger = get_logger("llm.cache")


class LLMCache:
    def __init__(self, ttl_seconds: int = 3600, max_size: int = 1000) -> None:
        self._ttl = ttl_seconds
        self._max_size = max_size
        self._cache: dict[str, dict[str, Any]] = {}
        self._access_order: list[str] = []
        self._hits = 0
        self._misses = 0

    @staticmethod
    def _make_key(prompt: str, system: str, model: str, temperature: float) -> str:
        raw = f"{model}|{temperature:.4f}|{system}|{prompt}"
        return hashlib.sha256(raw.encode()).hexdigest()

    def get(self, prompt: str, system: str, model: str, temperature: float) -> dict[str, Any] | None:
        key = self._make_key(prompt, system, model, temperature)
        entry = self._cache.get(key)
        if entry is None:
            self._misses += 1
            return None

        if time.time() - entry["timestamp"] > self._ttl:
            del self._cache[key]
            if key in self._access_order:
                self._access_order.remove(key)
            self._misses += 1
            return None

        self._hits += 1
        if key in self._access_order:
            self._access_order.remove(key)
        self._access_order.append(key)
        return entry["response"]

    def put(self, prompt: str, system: str, model: str, temperature: float, response: dict[str, Any]) -> None:
        key = self._make_key(prompt, system, model, temperature)

        if len(self._cache) >= self._max_size and key not in self._cache:
            self._evict()

        self._cache[key] = {
            "response": response,
            "timestamp": time.time(),
        }
        if key in self._access_order:
            self._access_order.remove(key)
        self._access_order.append(key)

    def _evict(self) -> None:
        while len(self._cache) >= self._max_size and self._access_order:
            oldest = self._access_order.pop(0)
            self._cache.pop(oldest, None)

    def clear(self) -> None:
        self._cache.clear()
        self._access_order.clear()
        self._hits = 0
        self._misses = 0

    @property
    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "size": len(self._cache),
            "max_size": self._max_size,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self.hit_rate,
            "ttl_seconds": self._ttl,
        }


class SemanticCache:
    def __init__(self, ttl_seconds: int = 7200, similarity_threshold: float = 0.92) -> None:
        self._ttl = ttl_seconds
        self._threshold = similarity_threshold
        self._entries: list[dict[str, Any]] = []
        self._hits = 0
        self._misses = 0

    def get(self, prompt: str, model: str) -> dict[str, Any] | None:
        now = time.time()
        prompt_lower = prompt.lower().strip()
        prompt_words = set(prompt_lower.split())

        best_match: dict[str, Any] | None = None
        best_score = 0.0

        for entry in self._entries:
            if now - entry["timestamp"] > self._ttl:
                continue
            if entry["model"] != model:
                continue

            entry_words = set(entry["prompt_lower"].split())
            if not prompt_words or not entry_words:
                continue
            intersection = prompt_words & entry_words
            union = prompt_words | entry_words
            jaccard = len(intersection) / len(union)

            if jaccard > best_score and jaccard >= self._threshold:
                best_score = jaccard
                best_match = entry

        if best_match:
            self._hits += 1
            return best_match["response"]

        self._misses += 1
        return None

    def put(self, prompt: str, model: str, response: dict[str, Any]) -> None:
        self._entries.append({
            "prompt_lower": prompt.lower().strip(),
            "model": model,
            "response": response,
            "timestamp": time.time(),
        })

        now = time.time()
        self._entries = [e for e in self._entries if now - e["timestamp"] <= self._ttl]

    @property
    def stats(self) -> dict[str, Any]:
        total = self._hits + self._misses
        return {
            "size": len(self._entries),
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self._hits / total if total > 0 else 0.0,
        }
