import json
import math
import tempfile
import unittest
from pathlib import Path

from rag_agent.embeddings import HashEmbedder
from rag_agent.evaluate import evaluate
from rag_agent.ingest import chunk_documents, split_text
from rag_agent.models import RawDoc
from rag_agent.retrieval import BM25Index, KnowledgeBase, reciprocal_rank_fusion
from rag_agent.text import stem, tokenize
from rag_agent.vectorstores import MemoryVectorStore
from tests.helpers import EVAL_QUESTIONS, build_kb, make_settings


class TextTests(unittest.TestCase):
    def test_tokenize_drops_stopwords_and_stems(self):
        self.assertEqual(tokenize("The refunds were processed quickly"), ["refund", "process", "quickly"])

    def test_stem_keeps_short_words(self):
        self.assertEqual(stem("is"), "is")
        self.assertEqual(stem("uses"), "use")


class EmbedderTests(unittest.TestCase):
    def test_deterministic_and_normalised(self):
        e = HashEmbedder()
        a, b = e.embed_query("refund policy for annual plans"), e.embed_query("refund policy for annual plans")
        self.assertEqual(a, b)
        self.assertAlmostEqual(math.sqrt(sum(x * x for x in a)), 1.0, places=6)

    def test_related_text_scores_higher_than_unrelated(self):
        e = HashEmbedder()
        q = e.embed_query("annual subscription refund")
        related = e.embed_query("Refunds on annual subscriptions are issued within 30 days")
        unrelated = e.embed_query("Planned maintenance happens on Sundays at night")
        dot = lambda x, y: sum(i * j for i, j in zip(x, y))
        self.assertGreater(dot(q, related), dot(q, unrelated))


class BM25Tests(unittest.TestCase):
    def test_exact_term_ranks_first(self):
        index = BM25Index(["a", "b", "c"], ["rate limit is 60 per minute", "refund within 30 days", "encrypted at rest"])
        self.assertEqual(index.search("what is the rate limit", 3)[0][0], "a")

    def test_no_match_returns_nothing(self):
        index = BM25Index(["a"], ["refund within 30 days"])
        self.assertEqual(index.search("zebra", 3), [])

    def test_rare_terms_outweigh_common_ones(self):
        index = BM25Index(["a", "b"], ["plan plan plan plan", "plan enterprise"])
        self.assertEqual(index.search("enterprise plan", 2)[0][0], "b")


class FusionTests(unittest.TestCase):
    def test_rrf_scores(self):
        fused = reciprocal_rank_fusion([["a", "b"], ["b", "c"]], k=60)
        self.assertAlmostEqual(fused["a"], 1 / 61)
        self.assertAlmostEqual(fused["b"], 1 / 62 + 1 / 61)
        self.assertAlmostEqual(fused["c"], 1 / 62)
        self.assertEqual(max(fused, key=fused.get), "b")  # appears in both lists

    def test_weights(self):
        fused = reciprocal_rank_fusion([["a"], ["b"]], k=60, weights=[2.0, 1.0])
        self.assertGreater(fused["a"], fused["b"])


class IngestTests(unittest.TestCase):
    def test_split_respects_size_and_overlaps(self):
        text = " ".join(f"Sentence number {i} ends here." for i in range(40))
        chunks = split_text(text, chunk_size=30, overlap=10)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(c.split()) <= 30 for c in chunks))
        # consecutive chunks share their boundary sentence(s)
        for previous, following in zip(chunks, chunks[1:]):
            last_sentence = previous.split("\n")[-1]
            self.assertIn(last_sentence.rstrip("."), following)
        # nothing is lost: every sentence appears in at least one chunk
        for i in range(40):
            self.assertTrue(any(f"Sentence number {i} ends here." in c for c in chunks))

    def test_pii_is_redacted_before_indexing(self):
        docs = [RawDoc("d.md", "Contact bob@example.com or +1 415 555 0100 for help.", "d.md")]
        chunks, counts = chunk_documents(docs, make_settings(redact_pii=True))
        self.assertNotIn("bob@example.com", chunks[0].text)
        self.assertEqual(counts, {"EMAIL": 1, "PHONE": 1})
        raw_chunks, raw_counts = chunk_documents(docs, make_settings(redact_pii=False))
        self.assertIn("bob@example.com", raw_chunks[0].text)
        self.assertEqual(raw_counts, {})

    def test_sample_docs_contain_no_raw_pii_after_ingest(self):
        kb = build_kb()
        everything = " ".join(c.text for c in kb.chunks.values())
        self.assertNotIn("@nimbusdesk.example", everything)
        self.assertNotIn("555 0132", everything)


class KnowledgeBaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.kb = build_kb()

    def test_hybrid_finds_expected_document(self):
        hits = self.kb.search("how many API requests per minute on the Business plan", k=3)
        self.assertIn("api_rate_limits.md", [h.chunk.doc_id for h in hits])

    def test_modes_are_validated(self):
        with self.assertRaises(ValueError):
            self.kb.search("x", mode="magic")

    def test_hits_carry_scores(self):
        hit = self.kb.search("refund annual subscription", k=1)[0]
        self.assertEqual(hit.chunk.doc_id, "refund_policy.md")
        self.assertGreater(hit.dense, 0)
        self.assertGreater(hit.bm25, 0)

    def test_save_and_load_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.kb.save(Path(tmp))
            embedder = HashEmbedder()
            loaded = KnowledgeBase.load(Path(tmp), MemoryVectorStore(embedder), embedder)
            query = "service credit for downtime"
            self.assertEqual(
                [h.chunk.id for h in self.kb.search(query, k=3)],
                [h.chunk.id for h in loaded.search(query, k=3)],
            )

    def test_load_rejects_embedder_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.kb.save(Path(tmp))
            other = HashEmbedder(dim=64)
            with self.assertRaises(RuntimeError):
                KnowledgeBase.load(Path(tmp), MemoryVectorStore(other), other)

    def test_missing_index_has_helpful_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            embedder = HashEmbedder()
            with self.assertRaises(FileNotFoundError):
                KnowledgeBase.load(Path(tmp), MemoryVectorStore(embedder), embedder)


class EvaluationTests(unittest.TestCase):
    def test_eval_runs_and_hybrid_is_competitive(self):
        results = evaluate(build_kb(), EVAL_QUESTIONS)
        self.assertEqual(set(results), {"dense", "bm25", "hybrid"})
        for metrics in results.values():
            self.assertTrue(0.0 <= metrics["hit@1"] <= metrics["hit@3"] <= 1.0)
            self.assertEqual(metrics["questions"], len(json.loads(EVAL_QUESTIONS.read_text())))
        best_single = max(results["dense"]["mrr"], results["bm25"]["mrr"])
        self.assertGreaterEqual(results["hybrid"]["mrr"], best_single - 0.05)


if __name__ == "__main__":
    unittest.main()
