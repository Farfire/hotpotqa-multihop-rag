"""Lexical BM25 search and rank-based fusion with dense retrieval."""

from __future__ import annotations

import re

import numpy as np
from rank_bm25 import BM25Okapi

from .index import Hit, VectorIndex

_TOKEN_RE = re.compile(r"\w+", flags=re.UNICODE)


def tokenize(text: str) -> list[str]:
    """Lowercase Unicode word tokens; intentionally avoid corpus-specific rules."""
    return _TOKEN_RE.findall(text.casefold())


class BM25Index:
    """In-memory BM25 index over the same title-plus-paragraph documents as E5."""

    def __init__(self, index: VectorIndex) -> None:
        self.index = index
        self._bm25 = BM25Okapi([tokenize(p.to_document()) for p in index.passages])
        # Store an inverted index so a query touches only documents containing
        # its terms. This is algebraically identical to rank_bm25's dense
        # zero-filled scan, but avoids iterating over every document per term.
        postings: dict[str, list[tuple[int, int]]] = {}
        for doc_id, frequencies in enumerate(self._bm25.doc_freqs):
            for term, frequency in frequencies.items():
                postings.setdefault(term, []).append((doc_id, frequency))
        self._postings = {
            term: (
                np.fromiter((doc_id for doc_id, _ in values), dtype=np.int32),
                np.fromiter((frequency for _, frequency in values), dtype=np.float32),
            )
            for term, values in postings.items()
        }
        self._doc_lengths = np.asarray(self._bm25.doc_len, dtype=np.float32)

    def search(self, query: str, k: int) -> list[Hit]:
        if k < 1:
            raise ValueError("k must be positive")
        scores = np.zeros(len(self.index), dtype=np.float64)
        for term in tokenize(query):
            posting = self._postings.get(term)
            if posting is None:
                continue
            doc_ids, frequencies = posting
            norm = self._bm25.k1 * (
                1 - self._bm25.b + self._bm25.b * self._doc_lengths[doc_ids] / self._bm25.avgdl
            )
            scores[doc_ids] += self._bm25.idf.get(term, 0.0) * (
                frequencies * (self._bm25.k1 + 1) / (frequencies + norm)
            )
        # Stable ordering makes ties reproducible and preserves corpus order.
        order = np.argsort(-scores, kind="stable")[: min(k, len(self.index))]
        return [Hit(self.index.passages[int(i)], float(scores[i])) for i in order]


def reciprocal_rank_fusion(
    *ranked_lists: list[Hit], k: int = 60, limit: int | None = None
) -> list[Hit]:
    """Combine rankings using 1/(k + rank), without comparing score scales."""
    if k < 1 or (limit is not None and limit < 1):
        raise ValueError("k and limit must be positive")
    scores: dict[str, float] = {}
    passages = {}
    insertion_order: dict[str, int] = {}
    for ranked in ranked_lists:
        seen: set[str] = set()
        for rank, hit in enumerate(ranked, start=1):
            title = hit.passage.title
            if title in seen:
                continue
            seen.add(title)
            insertion_order.setdefault(title, len(insertion_order))
            passages[title] = hit.passage
            scores[title] = scores.get(title, 0.0) + 1.0 / (k + rank)
    ordered = sorted(scores, key=lambda title: (-scores[title], insertion_order[title]))
    if limit is not None:
        ordered = ordered[:limit]
    return [Hit(passages[title], scores[title]) for title in ordered]
