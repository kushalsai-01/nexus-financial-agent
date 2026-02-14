from __future__ import annotations

from typing import Any

import chromadb
from chromadb.config import Settings

from nexus.core.config import get_config
from nexus.core.exceptions import DataStorageError
from nexus.core.logging import get_logger

logger = get_logger("data.storage.vector")


class VectorStorage:
    def __init__(self) -> None:
        config = get_config()
        self._host = config.storage.chromadb.host
        self._port = config.storage.chromadb.port
        self._prefix = config.storage.chromadb.collection_prefix
        self._embedding_model = config.storage.chromadb.embedding_model
        self._client: chromadb.HttpClient | None = None
        self._collections: dict[str, chromadb.Collection] = {}

    def _get_client(self) -> chromadb.HttpClient:
        if self._client is None:
            self._client = chromadb.HttpClient(
                host=self._host,
                port=self._port,
                settings=Settings(anonymized_telemetry=False),
            )
        return self._client

    def _get_collection(self, name: str) -> chromadb.Collection:
        full_name = f"{self._prefix}_{name}"
        if full_name not in self._collections:
            client = self._get_client()
            self._collections[full_name] = client.get_or_create_collection(
                name=full_name,
                metadata={"hnsw:space": "cosine"},
            )
        return self._collections[full_name]

    def add_documents(
        self,
        collection_name: str,
        documents: list[str],
        ids: list[str],
        metadatas: list[dict[str, Any]] | None = None,
        embeddings: list[list[float]] | None = None,
    ) -> int:
        try:
            collection = self._get_collection(collection_name)
            kwargs: dict[str, Any] = {
                "documents": documents,
                "ids": ids,
            }
            if metadatas:
                kwargs["metadatas"] = metadatas
            if embeddings:
                kwargs["embeddings"] = embeddings
            collection.add(**kwargs)
            logger.info(
                f"Added {len(documents)} documents to '{collection_name}'"
            )
            return len(documents)
        except Exception as e:
            raise DataStorageError(
                f"Failed to add documents to '{collection_name}': {e}"
            ) from e

    def query(
        self,
        collection_name: str,
        query_texts: list[str] | None = None,
        query_embeddings: list[list[float]] | None = None,
        n_results: int = 10,
        where: dict[str, Any] | None = None,
        include: list[str] | None = None,
    ) -> dict[str, Any]:
        try:
            collection = self._get_collection(collection_name)
            kwargs: dict[str, Any] = {"n_results": n_results}
            if query_texts:
                kwargs["query_texts"] = query_texts
            if query_embeddings:
                kwargs["query_embeddings"] = query_embeddings
            if where:
                kwargs["where"] = where
            if include:
                kwargs["include"] = include
            else:
                kwargs["include"] = ["documents", "metadatas", "distances"]

            results = collection.query(**kwargs)
            return results
        except Exception as e:
            raise DataStorageError(
                f"Failed to query '{collection_name}': {e}"
            ) from e

    def search_similar(
        self,
        collection_name: str,
        text: str,
        n_results: int = 5,
        min_relevance: float = 0.0,
    ) -> list[dict[str, Any]]:
        results = self.query(
            collection_name=collection_name,
            query_texts=[text],
            n_results=n_results,
        )

        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]
        ids = results.get("ids", [[]])[0]

        output: list[dict[str, Any]] = []
        for doc, meta, dist, doc_id in zip(documents, metadatas, distances, ids):
            relevance = 1.0 - dist
            if relevance >= min_relevance:
                output.append(
                    {
                        "id": doc_id,
                        "document": doc,
                        "metadata": meta,
                        "distance": dist,
                        "relevance": relevance,
                    }
                )
        return output

    def add_news(self, news_items: list[dict[str, Any]]) -> int:
        documents = []
        ids = []
        metadatas = []
        for item in news_items:
            content = f"{item.get('title', '')} {item.get('content', '')}"
            documents.append(content)
            ids.append(item.get("id", str(hash(content))))
            metadatas.append(
                {
                    "source": item.get("source", ""),
                    "published_at": item.get("published_at", ""),
                    "tickers": ",".join(item.get("tickers", [])),
                    "sentiment": item.get("sentiment_score", 0.0),
                }
            )
        return self.add_documents("news", documents, ids, metadatas)

    def search_news(
        self, query: str, n_results: int = 10
    ) -> list[dict[str, Any]]:
        return self.search_similar("news", query, n_results)

    def add_research(self, research_items: list[dict[str, Any]]) -> int:
        documents = []
        ids = []
        metadatas = []
        for item in research_items:
            documents.append(item.get("content", ""))
            ids.append(item.get("id", str(hash(item.get("content", "")))))
            metadatas.append(
                {
                    "ticker": item.get("ticker", ""),
                    "agent": item.get("agent", ""),
                    "analysis_type": item.get("analysis_type", ""),
                    "timestamp": item.get("timestamp", ""),
                }
            )
        return self.add_documents("research", documents, ids, metadatas)

    def search_research(
        self, query: str, ticker: str | None = None, n_results: int = 5
    ) -> list[dict[str, Any]]:
        where = {"ticker": ticker} if ticker else None
        results = self.query(
            collection_name="research",
            query_texts=[query],
            n_results=n_results,
            where=where,
        )

        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]
        ids = results.get("ids", [[]])[0]

        return [
            {
                "id": doc_id,
                "content": doc,
                "metadata": meta,
                "relevance": 1.0 - dist,
            }
            for doc, meta, dist, doc_id in zip(documents, metadatas, distances, ids)
        ]

    def delete_collection(self, name: str) -> None:
        full_name = f"{self._prefix}_{name}"
        try:
            client = self._get_client()
            client.delete_collection(full_name)
            self._collections.pop(full_name, None)
            logger.info(f"Deleted collection '{full_name}'")
        except Exception as e:
            logger.error(f"Failed to delete collection '{full_name}': {e}")

    def list_collections(self) -> list[str]:
        client = self._get_client()
        collections = client.list_collections()
        return [c.name for c in collections]

    def health_check(self) -> bool:
        try:
            client = self._get_client()
            client.heartbeat()
            return True
        except Exception:
            return False
