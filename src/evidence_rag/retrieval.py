"""Deterministic offline retrieval suitable for tests and baselines."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
import re
from typing import Iterable


TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> tuple[str, ...]:
    return tuple(TOKEN_PATTERN.findall(text.lower()))


@dataclass(frozen=True)
class Document:
    id: str
    text: str
    source: str


@dataclass(frozen=True)
class Chunk:
    id: str
    document_id: str
    text: str
    source: str
    position: int


@dataclass(frozen=True)
class SearchHit:
    chunk: Chunk
    score: float


def chunk_document(document: Document, chunk_size: int = 120, overlap: int = 20) -> tuple[Chunk, ...]:
    if chunk_size < 1 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("require chunk_size > overlap >= 0")
    words = document.text.split()
    chunks: list[Chunk] = []
    start = 0
    position = 0
    while start < len(words):
        segment = words[start : start + chunk_size]
        chunks.append(
            Chunk(
                id=f"{document.id}:{position}",
                document_id=document.id,
                text=" ".join(segment),
                source=document.source,
                position=position,
            )
        )
        position += 1
        start += chunk_size - overlap
    return tuple(chunks)


class LexicalRetriever:
    """A BM25-style baseline that exposes stable, inspectable scores."""

    def __init__(self, chunks: Iterable[Chunk], k1: float = 1.5, b: float = 0.75):
        self.chunks = tuple(chunks)
        if not self.chunks:
            raise ValueError("at least one chunk is required")
        self.k1 = k1
        self.b = b
        self._terms = [Counter(tokenize(chunk.text)) for chunk in self.chunks]
        self._lengths = [sum(terms.values()) for terms in self._terms]
        self._average_length = sum(self._lengths) / len(self._lengths)
        self._document_frequency = Counter(
            token for terms in self._terms for token in terms.keys()
        )

    def search(self, query: str, limit: int = 5) -> tuple[SearchHit, ...]:
        if limit < 1:
            raise ValueError("limit must be positive")
        query_terms = Counter(tokenize(query))
        hits: list[SearchHit] = []
        for chunk, terms, length in zip(self.chunks, self._terms, self._lengths):
            score = 0.0
            for term, query_frequency in query_terms.items():
                frequency = terms[term]
                if not frequency:
                    continue
                document_frequency = self._document_frequency[term]
                inverse_frequency = math.log(
                    1 + (len(self.chunks) - document_frequency + 0.5) / (document_frequency + 0.5)
                )
                normalization = frequency + self.k1 * (
                    1 - self.b + self.b * length / self._average_length
                )
                score += query_frequency * inverse_frequency * frequency * (self.k1 + 1) / normalization
            if score:
                hits.append(SearchHit(chunk, round(score, 8)))
        return tuple(sorted(hits, key=lambda hit: (-hit.score, hit.chunk.id))[:limit])
