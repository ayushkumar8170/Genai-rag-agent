"""Embedding backends.

* ``HuggingFaceEmbedder`` - sentence-transformers via ``langchain-huggingface`` (semantic).
* ``HashEmbedder`` - deterministic feature-hashing embedder that needs no model download.
  It is lexical rather than semantic, so it is used for offline runs and unit tests.

Both expose LangChain's ``embed_documents`` / ``embed_query`` interface, so they can be
passed straight to LangChain vector stores such as Chroma.
"""

from __future__ import annotations

import hashlib
import logging
import math
from collections import Counter
from typing import Protocol, Sequence

import numpy as np

from .text import tokenize

logger = logging.getLogger(__name__)

DEFAULT_HF_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class Embedder(Protocol):
    name: str

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class EmbeddingUnavailable(RuntimeError):
    """The requested embedding backend cannot be used in this environment."""


class HashEmbedder:
    """Signed feature hashing of stemmed unigrams and bigrams, L2-normalised."""

    def __init__(self, dim: int = 1024) -> None:
        self.dim = dim
        self.name = f"hash-{dim}"

    @staticmethod
    def _bucket(feature: str, dim: int) -> tuple[int, float]:
        digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
        h = int.from_bytes(digest, "big")
        return h % dim, (1.0 if (h >> 40) & 1 else -1.0)

    def _vector(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float64)
        tokens = tokenize(text)
        for token, count in Counter(tokens).items():
            idx, sign = self._bucket(token, self.dim)
            vec[idx] += sign * (1.0 + math.log(count))
        for left, right in zip(tokens, tokens[1:]):
            idx, sign = self._bucket(f"{left}_{right}", self.dim)
            vec[idx] += sign * 0.5
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._vector(t).tolist() for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text).tolist()


class HuggingFaceEmbedder:
    """Sentence-transformers embeddings through LangChain's Hugging Face integration."""

    def __init__(self, model_name: str = DEFAULT_HF_MODEL) -> None:
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
        except ImportError as exc:
            raise EmbeddingUnavailable(
                "Install langchain-huggingface and sentence-transformers to use Hugging Face embeddings."
            ) from exc
        self._model = HuggingFaceEmbeddings(
            model_name=model_name,
            encode_kwargs={"normalize_embeddings": True},
        )
        self.name = f"hf:{model_name}"

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return self._model.embed_documents(list(texts))

    def embed_query(self, text: str) -> list[float]:
        return self._model.embed_query(text)


def get_embedder(backend: str = "auto", model: str | None = None) -> Embedder:
    """Create an embedder. ``auto`` tries Hugging Face first and falls back to hashing."""
    backend = backend.lower()
    if backend == "hash":
        return HashEmbedder()
    if backend in {"auto", "huggingface", "hf"}:
        try:
            embedder = HuggingFaceEmbedder(model or DEFAULT_HF_MODEL)
            embedder.embed_query("warm-up")  # fail fast if the model cannot be loaded
            return embedder
        except Exception as exc:
            if backend != "auto":
                raise
            logger.warning(
                "Hugging Face embeddings unavailable (%s); falling back to the offline hash embedder. "
                "Retrieval will be lexical rather than semantic.",
                exc,
            )
            return HashEmbedder()
    raise ValueError(f"Unknown embedding backend: {backend!r}")
