"""Evaluation contracts for retrieval and evidence grounding."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import fmean
from typing import Iterable, Mapping

from .retrieval import SearchHit


@dataclass(frozen=True)
class EvaluationCase:
    query: str
    expected_chunk_ids: frozenset[str]
    cited_chunk_ids: tuple[str, ...] = ()
    claim_evidence: Mapping[str, frozenset[str]] | None = None


@dataclass(frozen=True)
class EvaluationReport:
    recall_at_k: float
    reciprocal_rank: float
    citation_precision: float
    claim_coverage: float
    passed: bool
    diagnostics: tuple[str, ...]


class RAGEvaluator:
    def __init__(self, minimum_recall: float = 0.8, minimum_citation_precision: float = 1.0):
        self.minimum_recall = minimum_recall
        self.minimum_citation_precision = minimum_citation_precision

    def evaluate(self, case: EvaluationCase, hits: Iterable[SearchHit]) -> EvaluationReport:
        ranked_ids = tuple(hit.chunk.id for hit in hits)
        retrieved = set(ranked_ids)
        expected = set(case.expected_chunk_ids)
        recall = len(retrieved & expected) / len(expected) if expected else 1.0
        reciprocal_rank = next(
            (1 / rank for rank, chunk_id in enumerate(ranked_ids, start=1) if chunk_id in expected),
            0.0,
        )
        citations = set(case.cited_chunk_ids)
        citation_precision = len(citations & retrieved) / len(citations) if citations else 1.0

        claim_scores: list[float] = []
        for evidence_ids in (case.claim_evidence or {}).values():
            claim_scores.append(1.0 if set(evidence_ids) & citations & retrieved else 0.0)
        coverage = fmean(claim_scores) if claim_scores else 1.0

        diagnostics: list[str] = []
        missing = expected - retrieved
        invalid_citations = citations - retrieved
        if missing:
            diagnostics.append(f"missing expected chunks: {sorted(missing)}")
        if invalid_citations:
            diagnostics.append(f"citations absent from context: {sorted(invalid_citations)}")
        if coverage < 1:
            diagnostics.append("one or more claims lack retrieved, cited evidence")
        passed = (
            recall >= self.minimum_recall
            and citation_precision >= self.minimum_citation_precision
            and coverage == 1.0
        )
        return EvaluationReport(
            round(recall, 6),
            round(reciprocal_rank, 6),
            round(citation_precision, 6),
            round(coverage, 6),
            passed,
            tuple(diagnostics),
        )
