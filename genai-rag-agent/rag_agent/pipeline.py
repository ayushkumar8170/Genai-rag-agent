"""Build or load the knowledge base from settings."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from .config import Settings
from .embeddings import get_embedder
from .ingest import chunk_documents, load_documents
from .retrieval import KnowledgeBase
from .vectorstores import make_store

logger = logging.getLogger(__name__)


@dataclass
class IngestReport:
    documents: int
    chunks: int
    embedder: str
    redactions: dict[str, int] = field(default_factory=dict)


def build_index(settings: Settings) -> tuple[KnowledgeBase, IngestReport]:
    docs = load_documents(settings.data_dir)
    chunks, redactions = chunk_documents(docs, settings)
    embedder = get_embedder(settings.embedding_backend, settings.embedding_model)
    store = make_store(settings.vector_store, embedder, settings.index_dir)
    kb = KnowledgeBase.build(chunks, store, embedder)
    kb.save(settings.index_dir)
    return kb, IngestReport(len(docs), len(chunks), embedder.name, redactions)


def load_index(settings: Settings) -> KnowledgeBase:
    embedder = get_embedder(settings.embedding_backend, settings.embedding_model)
    store = make_store(settings.vector_store, embedder, settings.index_dir)
    return KnowledgeBase.load(settings.index_dir, store, embedder)
