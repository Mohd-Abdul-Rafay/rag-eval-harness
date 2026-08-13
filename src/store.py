"""Persist chunk embeddings in ChromaDB and run k-NN search."""

from pathlib import Path
from typing import List, Dict

import chromadb
from chromadb.config import Settings

from src.embedder import Embedder


class VectorStore:
    def __init__(self,
                 persist_dir: str = "data/chroma",
                 collection_name: str = "papers",
                 embedder: Embedder | None = None):
        Path(persist_dir).mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection_name = collection_name
        self.embedder = embedder or Embedder()
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add(self, records: List[Dict], batch_size: int = 64) -> None:
        """Embed and store chunk records produced by chunker.chunk_corpus()."""
        texts = [r["text"] for r in records]
        vectors = self.embedder.encode(texts, batch_size=batch_size)

        self.collection.add(
            ids=[r["id"] for r in records],
            embeddings=vectors,
            documents=texts,
            metadatas=[{"source": r["source"],
                        "chunk_index": r["chunk_index"]} for r in records],
        )

    def search(self, query: str, k: int = 5) -> List[Dict]:
        """Return the k most similar chunks to a query."""
        qvec = self.embedder.encode([query], show_progress=False)[0]
        res = self.collection.query(query_embeddings=[qvec], n_results=k)

        hits = []
        for i in range(len(res["ids"][0])):
            hits.append({
                "id": res["ids"][0][i],
                "text": res["documents"][0][i],
                "source": res["metadatas"][0][i]["source"],
                "chunk_index": res["metadatas"][0][i]["chunk_index"],
                # Chroma returns cosine DISTANCE; convert to similarity
                "similarity": 1.0 - res["distances"][0][i],
            })
        return hits

    def count(self) -> int:
        return self.collection.count()

    def reset(self) -> None:
        """Drop and recreate the collection. Needed when re-ingesting."""
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )