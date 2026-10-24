"""Shared retrieval path for interactive questions and measured evaluation.

No answers or supporting-fact labels enter this module. The second hop follows
corpus-title mentions in retrieved text and also runs a contextual dense query.
"""

from __future__ import annotations

import html
import re
from dataclasses import dataclass

from .config import EncoderConfig
from .encoder import encode
from .index import Hit, VectorIndex, encoder_fingerprint
from .rerank import Reranker
from .sparse import BM25Index, reciprocal_rank_fusion


@dataclass
class RetrievalResult:
    hits: list[Hit]
    candidates: list[Hit]
    first_hop: list[Hit]
    bridge_titles: list[str]
    dense_hits: list[Hit]
    initial_candidates: list[Hit]
    bm25_hits: list[Hit]


def index_config(
    index: VectorIndex, *, model_name: str | None = None, max_length: int | None = None
) -> EncoderConfig:
    model = model_name or index.metadata.get("model_name") or index.fingerprint.split("@")[0]
    if not model:
        raise ValueError("index has no encoder identity; rebuild it")
    length = max_length or index.metadata.get("max_length", 256)
    return EncoderConfig(model_name=model, max_length=length)


class Retriever:
    def __init__(
        self,
        index: VectorIndex,
        encoder,
        tokenizer,
        config: EncoderConfig,
        *,
        reranker: Reranker | None = None,
        candidates: int = 50,
        hops: int = 2,
        bridge_count: int = 2,
        checkpoint=None,
        use_bm25: bool = True,
        rrf_k: int = 60,
    ) -> None:
        if candidates < 1 or hops not in (1, 2) or bridge_count < 1:
            raise ValueError("candidates/bridge count must be positive; hops must be 1 or 2")
        index.check_fingerprint(
            encoder_fingerprint(config.model_name, checkpoint, config.max_length)
        )
        self.index, self.encoder, self.tokenizer, self.config = index, encoder, tokenizer, config
        self.reranker, self.candidate_count = reranker, candidates
        self.hops, self.bridge_count = hops, bridge_count
        self.use_bm25, self.rrf_k = use_bm25, rrf_k
        self.bm25 = BM25Index(index) if use_bm25 else None
        self.title_patterns: dict[tuple[str, ...], list[tuple[int, str, re.Pattern[str]]]] = {}
        for pos, passage in enumerate(index.passages):
            title = html.unescape(passage.title)
            tokens = tuple(re.findall(r"\w+", title.casefold()))
            if len(title) >= 3 and tokens:
                pattern = re.compile(r"(?<!\w)" + re.escape(title) + r"(?!\w)", re.I)
                self.title_patterns.setdefault(tokens, []).append((pos, passage.title, pattern))
        self.max_title_tokens = max((len(tokens) for tokens in self.title_patterns), default=0)

    def dense(self, question: str, k: int) -> list[Hit]:
        vectors = encode(self.encoder, [question], self.tokenizer, self.config, role="query")
        return self.index.search(vectors, min(k, len(self.index)))[0]

    def candidates(self, question: str, k: int) -> tuple[list[Hit], list[Hit], list[Hit]]:
        """Return fused, dense, and lexical candidates for evaluation and logging."""
        dense_hits = self.dense(question, k)
        if self.bm25 is None:
            return dense_hits, dense_hits, []
        lexical_hits = self.bm25.search(question, k)
        return (
            reciprocal_rank_fusion(dense_hits, lexical_hits, k=self.rrf_k, limit=k),
            dense_hits,
            lexical_hits,
        )

    def _mentioned_titles(self, text: str, anchor_title: str) -> list[Hit]:
        """Find exact title mentions by checking only matching token n-grams."""
        words = [m.group(0) for m in re.finditer(r"\w+", text.casefold())]
        found: dict[int, Hit] = {}
        for start in range(len(words)):
            for width in range(1, min(self.max_title_tokens, len(words) - start) + 1):
                entries = self.title_patterns.get(tuple(words[start : start + width]), ())
                for pos, title, pattern in entries:
                    if title != anchor_title and pos not in found and pattern.search(text):
                        found[pos] = Hit(self.index.passages[pos], 0.0)
        return [found[pos] for pos in sorted(found)]

    def retrieve(self, question: str, k: int = 5) -> RetrievalResult:
        if k < 1 or k > self.candidate_count:
            raise ValueError("k must be positive and no larger than --candidates")
        candidates, dense_hits, bm25_hits = self.candidates(question, self.candidate_count)
        first = self.reranker.rerank(question, candidates) if self.reranker else candidates
        if self.hops == 1 or not first:
            return RetrievalResult(
                first[:k], candidates, first, [], dense_hits, candidates, bm25_hits
            )

        anchors = first[: self.bridge_count]
        pool = {h.passage.title: h for h in candidates}
        branches: list[list[Hit]] = []
        for anchor in anchors:
            context_query = f"{question}\nKnown context: {anchor.passage.to_document()[:700]}"
            # Following explicit entity mentions can recover a missing passage
            # even when dense search places it outside the original top 50.
            text = html.unescape(anchor.passage.text)
            linked = self._mentioned_titles(text, anchor.passage.title)
            followup = linked or self.candidates(context_query, min(20, self.candidate_count))[0]
            followup = list(
                {
                    h.passage.title: h for h in followup if h.passage.title != anchor.passage.title
                }.values()
            )
            pool.update({h.passage.title: h for h in followup})
            # An explicit reference in the bridge is stronger navigation evidence
            # than another semantically similar article. Rank those references
            # first; use contextual dense results when the bridge has no links.
            ranked = self.reranker.rerank(context_query, followup) if self.reranker else followup
            branches.append(ranked)

        # Reserve room for first-hop evidence and one result per bridge. Scores
        # from different contextual queries are not numerically comparable.
        ordered = list(anchors)
        used = {h.passage.title for h in ordered}
        for branch in branches:
            next_hit = next((h for h in branch if h.passage.title not in used), None)
            if next_hit is not None:
                ordered.append(next_hit)
                used.add(next_hit.passage.title)
        first_titles = {h.passage.title for h in first}
        new = [h for title, h in pool.items() if title not in first_titles]
        if self.reranker:
            # Reuse original-query scores rather than scoring the same 50
            # passages again; these logits share a query and are comparable.
            remainder = sorted(
                [*first, *self.reranker.rerank(question, new)], key=lambda h: h.score, reverse=True
            )
        else:
            remainder = [*first, *new]
        ordered.extend(h for h in remainder if h.passage.title not in used)
        return RetrievalResult(
            ordered[:k],
            list(pool.values()),
            first,
            [h.passage.title for h in anchors],
            dense_hits,
            candidates,
            bm25_hits,
        )
