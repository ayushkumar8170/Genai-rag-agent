"""Small framework-independent data containers shared across the package."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RawDoc:
    """A source document as loaded from disk."""

    doc_id: str  # human-readable id, e.g. the file name
    text: str
    source: str  # path or URI the text came from


@dataclass(frozen=True)
class Chunk:
    """A retrievable passage of a document."""

    id: str  # "<doc_id>::<index>"
    doc_id: str
    text: str
    source: str
    index: int


@dataclass
class Hit:
    """A retrieval result with the scores that produced it."""

    chunk: Chunk
    score: float  # final score used for ranking (RRF score in hybrid mode)
    dense: float = 0.0  # cosine similarity from the vector store (0 if absent)
    bm25: float = 0.0  # BM25 score (0 if absent)
