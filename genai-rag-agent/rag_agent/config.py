"""Runtime configuration, read from environment variables (and an optional .env file)."""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from pathlib import Path

try:  # python-dotenv is optional
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover
    pass


def _bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _int(name: str, default: int) -> int:
    return int(os.getenv(name, default))


def _float(name: str, default: float) -> float:
    return float(os.getenv(name, default))


@dataclass(frozen=True)
class Settings:
    # Data / index
    data_dir: Path = Path("data/sample_docs")
    index_dir: Path = Path("index")
    chunk_size: int = 90
    chunk_overlap: int = 15
    # Retrieval
    embedding_backend: str = "auto"  # auto | huggingface | hash
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    vector_store: str = "auto"  # auto | chroma | memory
    top_k: int = 4
    # Agent
    llm_provider: str = "none"  # none | openai | anthropic | ollama
    llm_model: str = ""
    agent_engine: str = "langgraph"  # langgraph | local
    max_agent_steps: int = 4
    min_coverage: float = 0.5  # fraction of question terms a hit must contain
    min_dense: float = 0.45  # ...or a dense cosine similarity at least this high
    # Privacy
    privacy_mode: str = "standard"  # standard | strict
    redact_pii: bool = True

    @classmethod
    def from_env(cls, **overrides) -> "Settings":
        base = cls(
            data_dir=Path(os.getenv("RAG_DATA_DIR", cls.data_dir)),
            index_dir=Path(os.getenv("RAG_INDEX_DIR", cls.index_dir)),
            chunk_size=_int("CHUNK_SIZE", cls.chunk_size),
            chunk_overlap=_int("CHUNK_OVERLAP", cls.chunk_overlap),
            embedding_backend=os.getenv("EMBEDDING_BACKEND", cls.embedding_backend),
            embedding_model=os.getenv("EMBEDDING_MODEL", cls.embedding_model),
            vector_store=os.getenv("VECTOR_STORE", cls.vector_store),
            top_k=_int("TOP_K", cls.top_k),
            llm_provider=os.getenv("LLM_PROVIDER", cls.llm_provider),
            llm_model=os.getenv("LLM_MODEL", cls.llm_model),
            agent_engine=os.getenv("AGENT_ENGINE", cls.agent_engine),
            max_agent_steps=_int("MAX_AGENT_STEPS", cls.max_agent_steps),
            min_coverage=_float("MIN_COVERAGE", cls.min_coverage),
            min_dense=_float("MIN_DENSE", cls.min_dense),
            privacy_mode=os.getenv("PRIVACY_MODE", cls.privacy_mode),
            redact_pii=_bool("REDACT_PII", cls.redact_pii),
        )
        clean = {k: v for k, v in overrides.items() if v is not None}
        return replace(base, **clean)
