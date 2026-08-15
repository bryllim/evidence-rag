import unittest

from evidence_rag import Document, EvaluationCase, LexicalRetriever, RAGEvaluator, chunk_document


class EvidenceRAGTests(unittest.TestCase):
    def setUp(self):
        documents = [
            Document("refund", "Customers may request a refund within thirty days of purchase.", "refund.md"),
            Document("shipping", "Standard shipping takes three to five business days.", "shipping.md"),
            Document("security", "All customer data is encrypted at rest and in transit.", "security.md"),
        ]
        self.chunks = tuple(chunk for document in documents for chunk in chunk_document(document, 20))
        self.retriever = LexicalRetriever(self.chunks)

    def test_retrieves_relevant_chunk_first(self):
        hits = self.retriever.search("How long does standard shipping take?")

        self.assertEqual(hits[0].chunk.id, "shipping:0")
        self.assertGreater(hits[0].score, 0)

    def test_chunking_is_overlapping_and_stable(self):
        document = Document("long", "one two three four five six seven eight", "long.md")

        chunks = chunk_document(document, chunk_size=4, overlap=1)

        self.assertEqual([chunk.id for chunk in chunks], ["long:0", "long:1", "long:2"])
        self.assertTrue(chunks[1].text.startswith("four"))

    def test_evaluation_passes_with_grounded_citation(self):
        hits = self.retriever.search("refund thirty days")
        case = EvaluationCase(
            "refund thirty days",
            frozenset({"refund:0"}),
            ("refund:0",),
            {"refund window": frozenset({"refund:0"})},
        )

        report = RAGEvaluator().evaluate(case, hits)

        self.assertTrue(report.passed)
        self.assertEqual(report.reciprocal_rank, 1.0)

    def test_flags_citation_not_in_context(self):
        hits = self.retriever.search("shipping")
        case = EvaluationCase("shipping", frozenset({"shipping:0"}), ("invented:9",))

        report = RAGEvaluator().evaluate(case, hits)

        self.assertFalse(report.passed)
        self.assertEqual(report.citation_precision, 0.0)
        self.assertIn("citations absent from context", report.diagnostics[0])

    def test_flags_unsupported_claim(self):
        hits = self.retriever.search("refund")
        case = EvaluationCase(
            "refund",
            frozenset({"refund:0"}),
            ("refund:0",),
            {"shipping duration": frozenset({"shipping:0"})},
        )

        report = RAGEvaluator().evaluate(case, hits)

        self.assertEqual(report.claim_coverage, 0.0)
        self.assertFalse(report.passed)


if __name__ == "__main__":
    unittest.main()
