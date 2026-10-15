"""Tests for the framework-backed paths. Each class skips itself if its dependency is missing."""

import importlib.util
import tempfile
import unittest
from pathlib import Path

from rag_agent.embeddings import HashEmbedder
from rag_agent.ingest import chunk_documents, load_documents
from rag_agent.retrieval import KnowledgeBase
from rag_agent.vectorstores import make_store
from tests.helpers import SAMPLE_DOCS, make_settings


def _has(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


@unittest.skipUnless(_has("llama_index"), "llama-index-core not installed")
class LlamaIndexIngestTests(unittest.TestCase):
    def test_loads_and_chunks_sample_docs(self):
        docs = load_documents(SAMPLE_DOCS)
        self.assertEqual(len(docs), 7)
        chunks, _ = chunk_documents(docs, make_settings())
        self.assertGreaterEqual(len(chunks), 7)
        self.assertTrue(all(c.text.strip() for c in chunks))


@unittest.skipUnless(_has("langchain_chroma") and _has("chromadb"), "langchain-chroma / chromadb not installed")
class ChromaStoreTests(unittest.TestCase):
    def test_chroma_hybrid_search_roundtrip(self):
        settings = make_settings(vector_store="chroma")
        chunks, _ = chunk_documents(load_documents(SAMPLE_DOCS), settings)
        embedder = HashEmbedder()
        with tempfile.TemporaryDirectory() as tmp:
            store = make_store("chroma", embedder, Path(tmp))
            kb = KnowledgeBase.build(chunks, store, embedder)
            hits = kb.search("how long does a refund take", k=3)
            self.assertIn("refund_policy.md", [h.chunk.doc_id for h in hits])
            self.assertGreater(hits[0].dense, 0)


@unittest.skipUnless(_has("langchain_core"), "langchain-core not installed")
class LangChainToolsTests(unittest.TestCase):
    def test_tools_convert_to_structured_tools(self):
        from rag_agent.tools import as_langchain_tools, build_tools
        from tests.helpers import build_kb

        tools = as_langchain_tools(build_tools(build_kb(), make_settings()))
        by_name = {t.name: t for t in tools}
        self.assertEqual(by_name["calculator"].invoke({"input": "6 * 7"}), "6 * 7 = 42")


@unittest.skipUnless(_has("langchain_huggingface") and _has("sentence_transformers"), "Hugging Face embeddings not installed")
class HuggingFaceEmbeddingTests(unittest.TestCase):
    def test_semantic_similarity(self):
        from rag_agent.embeddings import HuggingFaceEmbedder

        try:
            e = HuggingFaceEmbedder()
            q = e.embed_query("Can I get my money back?")
        except Exception as exc:  # model download may be blocked (offline CI)
            self.skipTest(f"model unavailable: {exc}")
        related = e.embed_query("We offer a 30-day refund guarantee.")
        unrelated = e.embed_query("Maintenance runs on Sunday nights.")
        dot = lambda x, y: sum(i * j for i, j in zip(x, y))
        self.assertGreater(dot(q, related), dot(q, unrelated))


if __name__ == "__main__":
    unittest.main()
