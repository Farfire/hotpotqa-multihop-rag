"""Architecture 3: augment baseline evidence with grounded follow-up retrieval."""

from dataclasses import dataclass, field

from rag.index import Hit
from rag.retrieval import RetrievalResult, Retriever

from .rewrite import GeminiQueryRewriter


@dataclass
class RewriteResult(RetrievalResult):
    trace: dict = field(default_factory=dict)


def merge_evidence(
    baseline: list[Hit], proposals: list[Hit], remainder: list[Hit], k: int
) -> list[Hit]:
    """Reserve three baseline slots, up to two follow-up slots, then fill uniquely.

    Scores for different queries are not compared. This preserves baseline top-2
    and retains baseline's title-navigation results rather than discarding them.
    """
    ordered, used = [], set()
    for hit in [*baseline[:3], *proposals[:2], *baseline[3:], *remainder]:
        if hit.passage.title not in used:
            ordered.append(hit)
            used.add(hit.passage.title)
    return ordered[:k]


def retrieve_query_rewrite(
    engine: Retriever,
    rewriter: GeminiQueryRewriter,
    question: str,
    k: int = 5,
    *,
    evidence_depth: int = 5,
) -> RewriteResult:
    if not engine.use_bm25 or engine.reranker is None:
        raise ValueError("query_rewrite requires hybrid search and a cross-encoder")
    if not 1 <= k <= engine.candidate_count:
        raise ValueError("k must be between 1 and candidate_count")
    # Retain the entire legacy expanded pool, including exact-title discoveries.
    baseline = engine.retrieve(question, engine.candidate_count)
    if not 1 <= evidence_depth <= engine.candidate_count:
        raise ValueError("evidence_depth must be between 1 and candidate_count")
    evidence = baseline.first_hop[:evidence_depth]
    try:
        plan = rewriter.rewrite(question, evidence)
    except Exception as exc:
        return RewriteResult(
            **{**vars(baseline), "hits": baseline.hits[:k]},
            trace={"status": "rewrite_error", "error_type": type(exc).__name__},
        )
    if not plan["queries"]:
        return RewriteResult(
            **{**vars(baseline), "hits": baseline.hits[:k]},
            trace={"status": "no_grounded_queries", "plan": plan},
        )
    pool = {hit.passage.title: hit for hit in baseline.candidates}
    branches, proposals = [], []
    for query in plan["queries"]:
        source = evidence[query["source_id"] - 1]
        candidates, _, _ = engine.candidates(query["query"], engine.candidate_count)
        candidates = [hit for hit in candidates if hit.passage.title != source.passage.title]
        ranked = engine.reranker.rerank(query["query"], candidates)
        pool.update({hit.passage.title: hit for hit in candidates})
        # A follow-up already present in the first three needs no extra slot.
        reserved = {hit.passage.title for hit in baseline.hits[:3]} | {
            hit.passage.title for hit in proposals
        }
        selected = ranked[0] if ranked and ranked[0].passage.title not in reserved else None
        if selected is not None:
            proposals.append(selected)
        branches.append(
            {
                "query": query,
                "source_title": source.passage.title,
                "selected_title": selected.passage.title if selected else None,
                "ranking": [{"title": hit.passage.title, "score": hit.score} for hit in ranked],
            }
        )
    baseline_titles = {hit.passage.title for hit in baseline.candidates}
    new = engine.reranker.rerank(
        question, [hit for title, hit in pool.items() if title not in baseline_titles]
    )
    hits = merge_evidence(baseline.hits, proposals, new, k)
    return RewriteResult(
        hits,
        list(pool.values()),
        baseline.first_hop,
        baseline.bridge_titles,
        baseline.dense_hits,
        baseline.initial_candidates,
        baseline.bm25_hits,
        trace={
            "status": "rewritten",
            "plan": plan,
            "branches": branches,
            "baseline_titles": [hit.passage.title for hit in baseline.hits],
            "assembly": "baseline first 3; up to 2 novel follow-ups; baseline remainder; new original-query-ranked candidates",
            "version": 2,
            "evidence_count": len(evidence),
        },
    )
