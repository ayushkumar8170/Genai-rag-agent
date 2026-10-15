from __future__ import annotations

from pathlib import Path

from rag_agent.config import Settings
from rag_agent.embeddings import HashEmbedder
from rag_agent.ingest import chunk_documents, load_documents
from rag_agent.retrieval import KnowledgeBase
from rag_agent.vectorstores import MemoryVectorStore

ROOT = Path(__file__).resolve().parent.parent
SAMPLE_DOCS = ROOT / "data" / "sample_docs"
EVAL_QUESTIONS = ROOT / "data" / "eval_questions.json"


def make_settings(**overrides) -> Settings:
    base = dict(
        data_dir=SAMPLE_DOCS,
        embedding_backend="hash",
        vector_store="memory",
        agent_engine="local",
        llm_provider="none",
    )
    base.update(overrides)
    return Settings(**base)


def build_kb(settings: Settings | None = None) -> KnowledgeBase:
    settings = settings or make_settings()
    chunks, _ = chunk_documents(load_documents(settings.data_dir), settings)
    embedder = HashEmbedder()
    return KnowledgeBase.build(chunks, MemoryVectorStore(embedder), embedder)


class FakeLLM:
    """Scripted LLM: returns queued replies in order, recording every prompt it receives."""

    name = "fake"

    def __init__(self, replies: list[str]) -> None:
        self.replies = list(replies)
        self.calls: list[tuple[str, str]] = []

    def complete(self, system: str, user: str) -> str:
        self.calls.append((system, user))
        return self.replies.pop(0) if self.replies else ""
