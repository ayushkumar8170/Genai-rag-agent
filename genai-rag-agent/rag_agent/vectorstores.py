"""Vector stores: a Chroma-backed store (via LangChain) and an in-memory numpy store."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Protocol, Sequence

import numpy as np

from .embeddings import Embedder
from .models import Chunk


class VectorStore(Protocol):
    def reset(self) -> None: ...

    def add(self, chunks: Sequence[Chunk]) -> None: ...

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        """Return ``(chunk_id, cosine_similarity)`` pairs, best first."""

    def save(self, directory: Path) -> None: ...

    def load(self, directory: Path) -> bool: ...


class MemoryVectorStore:
    """Brute-force cosine search over a numpy matrix. Fast enough for thousands of chunks."""

    def __init__(self, embedder: Embedder) -> None:
        self._embedder = embedder
        self._ids: list[str] = []
        self._matrix: np.ndarray | None = None

    def reset(self) -> None:
        self._ids, self._matrix = [], None

    def add(self, chunks: Sequence[Chunk]) -> None:
        if not chunks:
            return
        vectors = np.asarray(self._embedder.embed_documents([c.text for c in chunks]), dtype=np.float32)
        self._ids.extend(c.id for c in chunks)
        self._matrix = vectors if self._matrix is None else np.vstack([self._matrix, vectors])

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        if self._matrix is None or not self._ids:
            return []
        q = np.asarray(self._embedder.embed_query(query), dtype=np.float32)
        norm = np.linalg.norm(q)
        if norm == 0:
            return []
        norms = np.linalg.norm(self._matrix, axis=1)
        norms[norms == 0] = 1.0
        scores = (self._matrix @ q) / (norms * norm)
        order = np.argsort(-scores, kind="stable")[:k]
        return [(self._ids[i], float(scores[i])) for i in order]

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        if self._matrix is not None:
            np.save(directory / "vectors.npy", self._matrix)
        (directory / "vector_ids.json").write_text(json.dumps(self._ids), encoding="utf-8")

    def load(self, directory: Path) -> bool:
        vectors, ids = directory / "vectors.npy", directory / "vector_ids.json"
        if not (vectors.exists() and ids.exists()):
            return False
        self._matrix = np.load(vectors)
        self._ids = json.loads(ids.read_text(encoding="utf-8"))
        return True


class ChromaVectorStore:
    """Persistent Chroma collection managed through ``langchain-chroma``."""

    _BATCH = 500

    def __init__(self, embedder: Embedder, persist_dir: Path, collection: str = "rag_agent") -> None:
        try:
            from langchain_chroma import Chroma  # noqa: F401
        except ImportError as exc:
            raise RuntimeError("Install langchain-chroma and chromadb to use VECTOR_STORE=chroma.") from exc
        self._embedder = embedder
        self._persist_dir = Path(persist_dir)
        self._collection = collection
        self._db = self._new_db()

    def _new_db(self):
        from langchain_chroma import Chroma

        return Chroma(
            collection_name=self._collection,
            embedding_function=self._embedder,
            persist_directory=str(self._persist_dir),
            collection_metadata={"hnsw:space": "cosine"},
        )

    def reset(self) -> None:
        try:
            self._db.delete_collection()
        except Exception:  # collection may not exist yet
            pass
        self._db = self._new_db()

    def add(self, chunks: Sequence[Chunk]) -> None:
        from langchain_core.documents import Document

        for start in range(0, len(chunks), self._BATCH):
            batch = chunks[start : start + self._BATCH]
            docs = [
                Document(
                    page_content=c.text,
                    metadata={"chunk_id": c.id, "doc_id": c.doc_id, "source": c.source},
                )
                for c in batch
            ]
            self._db.add_documents(docs, ids=[c.id for c in batch])

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        results = self._db.similarity_search_with_score(query, k=k)
        # With the cosine HNSW space Chroma returns a distance in [0, 2].
        return [(doc.metadata["chunk_id"], 1.0 - float(distance)) for doc, distance in results]

    def save(self, directory: Path) -> None:  # Chroma persists on write
        return None

    def load(self, directory: Path) -> bool:
        return True


def make_store(kind: str, embedder: Embedder, index_dir: Path) -> VectorStore:
    kind = kind.lower()
    if kind == "auto":
        kind = "chroma" if importlib.util.find_spec("langchain_chroma") else "memory"
    if kind == "memory":
        return MemoryVectorStore(embedder)
    if kind == "chroma":
        return ChromaVectorStore(embedder, Path(index_dir) / "chroma")
    raise ValueError(f"Unknown vector store: {kind!r}")
