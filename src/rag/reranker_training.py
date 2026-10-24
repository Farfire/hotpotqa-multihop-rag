"""Train with both gold passages and same-question hard negatives."""

from __future__ import annotations

import random
from dataclasses import dataclass

from .data import Passage


@dataclass(frozen=True)
class LabeledPair:
    question_id: str
    query: str
    document: str
    label: float


def split_questions(rows: list[dict], val_fraction: float, seed: int):
    if len(rows) < 2 or not 0 < val_fraction < 1:
        raise ValueError("need at least two questions and a validation fraction in (0, 1)")
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("duplicate question IDs would contaminate the holdout")
    shuffled = list(rows)
    random.Random(seed).shuffle(shuffled)
    count = min(len(rows) - 1, max(1, round(len(rows) * val_fraction)))
    return shuffled[count:], shuffled[:count]


def labeled_pairs(
    rows: list[dict], *, negatives: int = 4, seed: int = 42, contextual: bool = True
) -> list[LabeledPair]:
    if negatives < 1:
        raise ValueError("at least one hard negative is required")
    rng = random.Random(seed)
    pairs = []
    for row in rows:
        gold = set(row["supporting_facts"]["title"])
        passages = {
            title: Passage(title, "".join(sentences).strip())
            for title, sentences in zip(row["context"]["title"], row["context"]["sentences"])
        }
        positives = [p for title, p in passages.items() if title in gold]
        decoys = [p for title, p in passages.items() if title not in gold]
        if len(positives) < 2 or not decoys:
            continue
        decoys = rng.sample(decoys, min(negatives, len(decoys)))
        for passage in positives + decoys:
            pairs.append(
                LabeledPair(
                    row["id"], row["question"], passage.to_document(), float(passage.title in gold)
                )
            )
        if contextual:
            # Gold bridge text is training supervision only. In inference the
            # same query format uses retrieved text, with no access to labels.
            for anchor in positives:
                query = f"{row['question']}\nKnown context: {anchor.to_document()[:700]}"
                for passage in positives + decoys:
                    if passage.title != anchor.title:
                        pairs.append(
                            LabeledPair(
                                row["id"],
                                query,
                                passage.to_document(),
                                float(passage.title in gold),
                            )
                        )
    return pairs
