"""Document loading and chunking.

LlamaIndex (``SimpleDirectoryReader`` + ``SentenceSplitter``) is used when installed. If it
cannot be imported or fails (for example no tokenizer data offline), a small sentence-aware
splitter is used instead, so ingestion never depends on a single framework.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from pathlib import Path

from .config import Settings
from .models import Chunk, RawDoc
from .privacy import redact

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = (".md", ".txt", ".rst")


def _load_with_llamaindex(data_dir: Path) -> list[RawDoc]:
    from llama_index.core import SimpleDirectoryReader

    reader = SimpleDirectoryReader(
        input_dir=str(data_dir), required_exts=list(SUPPORTED_EXTENSIONS), recursive=True
    )
    docs: list[RawDoc] = []
    for doc in reader.load_data():
        source = str(doc.metadata.get("file_path") or doc.metadata.get("file_name") or "unknown")
        docs.append(RawDoc(doc_id=Path(source).name, text=doc.text, source=source))
    return docs


def _load_plain(data_dir: Path) -> list[RawDoc]:
    docs = []
    for path in sorted(data_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            docs.append(RawDoc(doc_id=path.name, text=path.read_text(encoding="utf-8"), source=str(path)))
    return docs


def load_documents(data_dir: Path) -> list[RawDoc]:
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    try:
        docs = _load_with_llamaindex(data_dir)
    except Exception as exc:  # ImportError or reader failure
        logger.info("LlamaIndex loader unavailable (%s); using the plain-text loader.", exc)
        docs = _load_plain(data_dir)
    docs.sort(key=lambda d: d.doc_id)
    if not docs:
        raise FileNotFoundError(f"No {', '.join(SUPPORTED_EXTENSIONS)} files found in {data_dir}")
    return docs


def split_text(text: str, chunk_size: int = 90, overlap: int = 15) -> list[str]:
    """Sentence-aware splitter. Sizes are measured in words."""
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n{2,}", text) if s.strip()]
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    for sentence in sentences:
        n = len(sentence.split())
        if current and current_len + n > chunk_size:
            chunks.append("\n".join(current))
            carry: list[str] = []
            carry_len = 0
            for prev in reversed(current):
                length = len(prev.split())
                if carry_len + length > overlap:
                    break
                carry.insert(0, prev)
                carry_len += length
            current, current_len = carry, carry_len
        current.append(sentence)
        current_len += n
    if current:
        chunks.append("\n".join(current))
    return chunks


def _split(text: str, settings: Settings) -> list[str]:
    try:
        from llama_index.core.node_parser import SentenceSplitter

        splitter = SentenceSplitter(chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap)
        pieces = splitter.split_text(text)
        return [p for p in pieces if p.strip()]
    except Exception as exc:
        logger.info("LlamaIndex splitter unavailable (%s); using the built-in splitter.", exc)
        return split_text(text, settings.chunk_size, settings.chunk_overlap)


def chunk_documents(docs: list[RawDoc], settings: Settings) -> tuple[list[Chunk], dict[str, int]]:
    """Redact (optionally) and split documents. Returns chunks and aggregate redaction counts."""
    chunks: list[Chunk] = []
    redactions: Counter[str] = Counter()
    for doc in docs:
        text = doc.text
        if settings.redact_pii:
            result = redact(text)
            text = result.text
            redactions.update(result.counts)
        for i, piece in enumerate(_split(text, settings)):
            chunks.append(Chunk(id=f"{doc.doc_id}::{i}", doc_id=doc.doc_id, text=piece, source=doc.source, index=i))
    return chunks, dict(redactions)
