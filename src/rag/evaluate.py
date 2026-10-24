"""Scoring retrieval against HotpotQA's gold labels.

HotpotQA tells us exactly which paragraphs answer each question, so retrieval
quality is measurable without an LLM judge and without eyeballing. Three numbers
at each cut-off k:

- recall@k     — of the gold paragraphs, what share landed in the top k?
- both_gold@k  — for what share of questions did we find *every* gold paragraph?
- any_gold@k   — for what share did we find *at least one*?

The gap between any_gold and both_gold is the multi-hop problem: the second
paragraph is usually only findable once you have read the first, which a
single-shot retriever never does.
"""

from __future__ import annotations

from dataclasses import dataclass

from .data import Example
from .index import Hit

DEFAULT_KS: tuple[int, ...] = (2, 5, 10, 20)


@dataclass(frozen=True)
class RetrievalMetrics:
    k: int
    recall: float
    both_gold: float
    any_gold: float


def evaluate_retrieval(
    examples: list[Example],
    hits_per_example: list[list[Hit]],
    ks: tuple[int, ...] = DEFAULT_KS,
) -> list[RetrievalMetrics]:
    """Score ranked results against the gold titles.

    `hits_per_example[i]` must be the results for `examples[i]`, ranked best
    first, and long enough to cover `max(ks)`.
    """
    if len(examples) != len(hits_per_example):
        raise ValueError("examples and hits must be the same length")

    metrics: list[RetrievalMetrics] = []
    for k in ks:
        recall_sum = 0.0
        both = 0
        any_ = 0
        scored = 0  # examples with gold labels; the denominator for all three

        for example, hits in zip(examples, hits_per_example):
            gold = set(example.gold_titles)
            if not gold:
                continue  # unlabelled, so it cannot be scored either way
            retrieved = {hit.passage.title for hit in hits[:k]}
            found = len(gold & retrieved)

            scored += 1
            recall_sum += found / len(gold)
            both += int(found == len(gold))
            any_ += int(found > 0)

        n = max(scored, 1)
        metrics.append(
            RetrievalMetrics(
                k=k,
                recall=recall_sum / n,
                both_gold=both / n,
                any_gold=any_ / n,
            )
        )
    return metrics


def format_metrics_table(metrics: list[RetrievalMetrics]) -> str:
    """Render the metrics as a markdown table."""
    lines = [
        "| k  | recall | both gold | any gold |",
        "|----|--------|-----------|----------|",
    ]
    for m in metrics:
        lines.append(f"| {m.k:<2} | {m.recall:>5.1%} | {m.both_gold:>9.1%} | {m.any_gold:>8.1%} |")
    return "\n".join(lines)
