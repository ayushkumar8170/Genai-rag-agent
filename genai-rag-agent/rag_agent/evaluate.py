"""Retrieval evaluation: hit@k and MRR for dense, BM25 and hybrid search on a labelled set."""

from __future__ import annotations

import json
from pathlib import Path

from .retrieval import KnowledgeBase


def _ranked_docs(kb: KnowledgeBase, question: str, mode: str, depth: int) -> list[str]:
    docs: list[str] = []
    for hit in kb.search(question, k=depth, mode=mode):
        if hit.chunk.doc_id not in docs:
            docs.append(hit.chunk.doc_id)
    return docs


def evaluate(kb: KnowledgeBase, questions_path: Path, depth: int = 10) -> dict[str, dict[str, float]]:
    """Each item is ``{"question": ..., "expected_doc": "<file name>"}``."""
    items = json.loads(Path(questions_path).read_text(encoding="utf-8"))
    results: dict[str, dict[str, float]] = {}
    for mode in KnowledgeBase.MODES:
        hit1 = hit3 = rr = 0.0
        for item in items:
            ranked = _ranked_docs(kb, item["question"], mode, depth)
            expected = item["expected_doc"]
            if expected in ranked:
                position = ranked.index(expected) + 1
                hit1 += position == 1
                hit3 += position <= 3
                rr += 1.0 / position
        n = len(items)
        results[mode] = {"hit@1": hit1 / n, "hit@3": hit3 / n, "mrr": rr / n, "questions": n}
    return results


def format_table(results: dict[str, dict[str, float]]) -> str:
    lines = ["mode    hit@1  hit@3  MRR", "------  -----  -----  -----"]
    for mode, m in results.items():
        lines.append(f"{mode:<6}  {m['hit@1']:.2f}   {m['hit@3']:.2f}   {m['mrr']:.2f}")
    lines.append(f"({int(next(iter(results.values()))['questions'])} questions)")
    return "\n".join(lines)
