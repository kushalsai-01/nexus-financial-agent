from __future__ import annotations

import json
import time
from typing import Any
from uuid import uuid4

from nexus.core.logging import get_logger

logger = get_logger("orchestration.memory")


class DecisionMemory:
    def __init__(self, max_entries: int = 10000) -> None:
        self._entries: list[dict[str, Any]] = []
        self._max_entries = max_entries

    def store(
        self,
        ticker: str,
        decision: dict[str, Any],
        outcome: dict[str, Any] | None = None,
        context: dict[str, Any] | None = None,
    ) -> str:
        entry_id = str(uuid4())
        entry = {
            "id": entry_id,
            "ticker": ticker,
            "decision": decision,
            "outcome": outcome or {},
            "context": context or {},
            "timestamp": time.time(),
        }
        self._entries.append(entry)

        if len(self._entries) > self._max_entries:
            self._entries = self._entries[-self._max_entries:]

        return entry_id

    def update_outcome(self, entry_id: str, outcome: dict[str, Any]) -> bool:
        for entry in reversed(self._entries):
            if entry["id"] == entry_id:
                entry["outcome"] = outcome
                return True
        return False

    def query(
        self,
        ticker: str | None = None,
        limit: int = 20,
        signal_type: str | None = None,
    ) -> list[dict[str, Any]]:
        results = self._entries

        if ticker:
            results = [e for e in results if e["ticker"] == ticker]

        if signal_type:
            results = [
                e for e in results
                if e.get("decision", {}).get("signal") == signal_type
            ]

        return results[-limit:]

    def get_accuracy(self, ticker: str | None = None, lookback: int = 100) -> dict[str, Any]:
        entries = self.query(ticker=ticker, limit=lookback)
        entries_with_outcome = [e for e in entries if e.get("outcome")]

        if not entries_with_outcome:
            return {"total": 0, "correct": 0, "accuracy": 0.0}

        correct = sum(
            1 for e in entries_with_outcome
            if e["outcome"].get("profitable", False)
        )
        total = len(entries_with_outcome)

        return {
            "total": total,
            "correct": correct,
            "accuracy": correct / total if total > 0 else 0.0,
            "ticker": ticker,
        }

    def get_similar_decisions(self, ticker: str, context: dict[str, Any], limit: int = 5) -> list[dict[str, Any]]:
        ticker_entries = [e for e in self._entries if e["ticker"] == ticker]

        if not ticker_entries:
            return []

        scored = []
        context_keys = set(context.keys())
        for entry in ticker_entries:
            entry_keys = set(entry.get("context", {}).keys())
            if not context_keys or not entry_keys:
                continue
            overlap = len(context_keys & entry_keys) / max(len(context_keys | entry_keys), 1)
            scored.append((overlap, entry))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [entry for _, entry in scored[:limit]]


class VectorMemory:
    def __init__(self, collection_name: str = "nexus_decisions") -> None:
        self._collection_name = collection_name
        self._store: list[dict[str, Any]] = []
        self._initialized = False

    async def initialize(self) -> None:
        try:
            import chromadb
            self._client = chromadb.Client()
            self._collection = self._client.get_or_create_collection(
                name=self._collection_name,
                metadata={"hnsw:space": "cosine"},
            )
            self._initialized = True
            logger.info(f"VectorMemory initialized with collection: {self._collection_name}")
        except ImportError:
            logger.warning("ChromaDB not available, using in-memory fallback")
            self._initialized = False

    async def store(self, text: str, metadata: dict[str, Any] | None = None) -> str:
        doc_id = str(uuid4())

        if self._initialized:
            try:
                self._collection.add(
                    documents=[text],
                    metadatas=[metadata or {}],
                    ids=[doc_id],
                )
            except Exception as e:
                logger.error(f"ChromaDB store failed: {e}")
                self._store.append({"id": doc_id, "text": text, "metadata": metadata or {}})
        else:
            self._store.append({"id": doc_id, "text": text, "metadata": metadata or {}})

        return doc_id

    async def query(self, text: str, n_results: int = 5) -> list[dict[str, Any]]:
        if self._initialized:
            try:
                results = self._collection.query(
                    query_texts=[text],
                    n_results=n_results,
                )
                docs = results.get("documents", [[]])[0]
                metas = results.get("metadatas", [[]])[0]
                ids = results.get("ids", [[]])[0]
                return [
                    {"id": ids[i], "text": docs[i], "metadata": metas[i]}
                    for i in range(len(docs))
                ]
            except Exception as e:
                logger.error(f"ChromaDB query failed: {e}")

        query_words = set(text.lower().split())
        scored = []
        for entry in self._store:
            entry_words = set(entry["text"].lower().split())
            if not query_words or not entry_words:
                continue
            overlap = len(query_words & entry_words) / max(len(query_words | entry_words), 1)
            scored.append((overlap, entry))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [entry for _, entry in scored[:n_results]]

    @property
    def stats(self) -> dict[str, Any]:
        if self._initialized:
            try:
                count = self._collection.count()
            except Exception:
                count = 0
        else:
            count = len(self._store)

        return {
            "collection": self._collection_name,
            "initialized": self._initialized,
            "document_count": count,
        }
