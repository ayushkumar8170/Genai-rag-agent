"""Hybrid retrieval: dense vectors + BM25, merged with Reciprocal Rank Fusion (RRF)."""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Optional, Sequence

from .embeddings import Embedder
from .models import Chunk, Hit
from .text import tokenize
from .vectorstores import VectorStore


class BM25Index:
    """Okapi BM25 over tokenised chunks (no external dependency)."""

    def __init__(self, ids: Sequence[str], texts: Sequence[str], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self.ids = list(ids)
        self._tf = [Counter(tokenize(t)) for t in texts]
        self._len = [sum(tf.values()) for tf in self._tf]
        self._avg_len = (sum(self._len) / len(self._len)) if self._len else 0.0
        self._df: Counter[str] = Counter()
        for tf in self._tf:
            self._df.update(tf.keys())

    def doc_freq(self, term: str) -> int:
        return self._df.get(term, 0)

    def _idf(self, term: str) -> float:
        n, df = len(self.ids), self._df.get(term, 0)
        return math.log(1.0 + (n - df + 0.5) / (df + 0.5))

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        terms = set(tokenize(query))
        scored: list[tuple[str, float]] = []
        for idx, tf in enumerate(self._tf):
            score = 0.0
            for term in terms:
                freq = tf.get(term, 0)
                if not freq:
                    continue
                denom = freq + self.k1 * (1 - self.b + self.b * self._len[idx] / (self._avg_len or 1.0))
                score += self._idf(term) * freq * (self.k1 + 1) / denom
            if score > 0:
                scored.append((self.ids[idx], score))
        scored.sort(key=lambda p: (-p[1], p[0]))
        return scored[:k]


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[str]], k: int = 60, weights: Optional[Sequence[float]] = None
) -> dict[str, float]:
    """Fuse ranked id lists: ``score(d) = sum_i w_i / (k + rank_i(d))`` with 1-based ranks."""
    weights = list(weights) if weights is not None else [1.0] * len(rankings)
    fused: dict[str, float] = {}
    for ranking, weight in zip(rankings, weights):
        for rank, item in enumerate(ranking, start=1):
            fused[item] = fused.get(item, 0.0) + weight / (k + rank)
    return fused


class KnowledgeBase:
    """Chunks + a vector store + a BM25 index, searchable in dense, sparse, or hybrid mode."""

    MODES = ("dense", "bm25", "hybrid")

    def __init__(self, chunks: Sequence[Chunk], store: VectorStore, embedder: Embedder) -> None:
        self.chunks = {c.id: c for c in chunks}
        self.store = store
        self.embedder = embedder
        self.bm25 = BM25Index([c.id for c in chunks], [c.text for c in chunks])
        terms_by_doc: dict[str, set[str]] = defaultdict(set)
        for c in chunks:
            terms_by_doc[c.doc_id].update(tokenize(c.text))
        self._doc_df: Counter[str] = Counter()
        for terms in terms_by_doc.values():
            self._doc_df.update(terms)
        self._n_docs = len(terms_by_doc)

    def idf(self, term: str) -> float:
        """BM25 inverse document frequency (higher = rarer = more informative)."""
        return self.bm25._idf(term)

    def informative_terms(self, terms: Iterable[str], max_df_ratio: float = 0.5) -> set[str]:
        """Terms that occur in the corpus but not in most *documents* (so they carry topical signal).

        Terms absent from the corpus (e.g. "football") say nothing about whether the knowledge base
        can answer, and near-ubiquitous terms (e.g. the product name) match almost everything.
        Document-level counts are used so the result does not depend on the chunk size.
        """
        informative = set()
        for term in terms:
            df = self._doc_df.get(term, 0)
            if df == 0:
                continue
            if self._n_docs >= 4 and df / self._n_docs >= max_df_ratio:
                continue
            informative.add(term)
        return informative

    @classmethod
    def build(cls, chunks: Sequence[Chunk], store: VectorStore, embedder: Embedder) -> "KnowledgeBase":
        store.reset()
        store.add(chunks)
        return cls(chunks, store, embedder)

    # -- persistence -----------------------------------------------------------------
    def save(self, index_dir: Path) -> None:
        index_dir = Path(index_dir)
        index_dir.mkdir(parents=True, exist_ok=True)
        with (index_dir / "chunks.jsonl").open("w", encoding="utf-8") as fh:
            for c in self.chunks.values():
                fh.write(json.dumps({"id": c.id, "doc_id": c.doc_id, "text": c.text, "source": c.source, "index": c.index}) + "\n")
        (index_dir / "meta.json").write_text(
            json.dumps({"embedder": self.embedder.name, "chunks": len(self.chunks)}), encoding="utf-8"
        )
        self.store.save(index_dir)

    @classmethod
    def load(cls, index_dir: Path, store: VectorStore, embedder: Embedder) -> "KnowledgeBase":
        index_dir = Path(index_dir)
        meta_path, chunks_path = index_dir / "meta.json", index_dir / "chunks.jsonl"
        if not (meta_path.exists() and chunks_path.exists()):
            raise FileNotFoundError(f"No index found in {index_dir}. Run `python -m rag_agent ingest` first.")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("embedder") != embedder.name:
            raise RuntimeError(
                f"The index was built with embeddings '{meta.get('embedder')}' but the current backend is "
                f"'{embedder.name}'. Re-run `python -m rag_agent ingest`."
            )
        chunks = [Chunk(**json.loads(line)) for line in chunks_path.read_text(encoding="utf-8").splitlines() if line]
        if not store.load(index_dir):
            raise FileNotFoundError(f"Vector data missing in {index_dir}. Re-run `python -m rag_agent ingest`.")
        return cls(chunks, store, embedder)

    # -- search ----------------------------------------------------------------------
    def search(self, query: str, k: int = 4, mode: str = "hybrid", candidates: int = 20) -> list[Hit]:
        if mode not in self.MODES:
            raise ValueError(f"mode must be one of {self.MODES}")
        dense = self.store.search(query, candidates) if mode in {"dense", "hybrid"} else []
        sparse = self.bm25.search(query, candidates) if mode in {"bm25", "hybrid"} else []
        dense_scores, sparse_scores = dict(dense), dict(sparse)
        if mode == "dense":
            fused = dict(dense)
        elif mode == "bm25":
            fused = dict(sparse)
        else:
            fused = reciprocal_rank_fusion([[i for i, _ in dense], [i for i, _ in sparse]])
        ranked = sorted(fused, key=lambda cid: (-fused[cid], cid))[:k]
        return [
            Hit(
                chunk=self.chunks[cid],
                score=fused[cid],
                dense=dense_scores.get(cid, 0.0),
                bm25=sparse_scores.get(cid, 0.0),
            )
            for cid in ranked
        ]
