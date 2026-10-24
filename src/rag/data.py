"""Load HotpotQA and reshape it into the two records the rest of the code uses.

HotpotQA ships each example with 10 Wikipedia paragraphs: the 2 that actually
answer the question ("gold") plus 8 plausible decoys ("distractors"). We flatten
every paragraph across every example into one deduplicated corpus, which is what
we retrieve from.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from datasets import load_dataset

HOTPOTQA_REPO = "hotpotqa/hotpot_qa"
HOTPOTQA_CONFIG = "distractor"


@dataclass(frozen=True)
class Passage:
    """One Wikipedia paragraph — the unit we index and retrieve."""

    title: str
    text: str

    def to_document(self) -> str:
        """The string that actually gets embedded.

        Note this is *not* the same as what we show the user. The title carries
        a lot of signal ("Arthur's Magazine" is most of the answer to a question
        about Arthur's Magazine), so we prepend it before encoding.
        """
        return f"{self.title}. {self.text}"


@dataclass(frozen=True)
class Example:
    """One question, with the answer and the titles needed to reach it."""

    id: str
    question: str
    answer: str
    gold_titles: tuple[str, ...]


def _iter_rows(split: str, limit: int) -> Iterator[dict]:
    """Cache only the requested slice, so repeated comparisons use identical rows."""
    if limit < 1 or split not in {"train", "validation"}:
        raise ValueError("use a positive limit and the train or validation split")
    cache = Path(__file__).resolve().parents[2] / "artifacts" / "data" / f"{split}-{limit}.json"
    if not cache.exists() and cache.parent.exists():
        larger = sorted(
            (int(p.stem.rsplit("-", 1)[1]), p)
            for p in cache.parent.glob(f"{split}-*.json")
            if p.stem.rsplit("-", 1)[1].isdigit() and int(p.stem.rsplit("-", 1)[1]) >= limit
        )
        if larger:
            cache = larger[0][1]
    if cache.exists():
        yield from json.loads(cache.read_text(encoding="utf-8"))[:limit]
        return
    dataset = load_dataset(HOTPOTQA_REPO, HOTPOTQA_CONFIG, split=split, streaming=True)
    rows = []
    for i, row in enumerate(dataset):
        if i >= limit:
            break
        rows.append(row)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    for row in rows:
        yield row


def load_examples(split: str = "validation", limit: int = 200) -> list[Example]:
    """Read `limit` questions from `split`."""
    examples: list[Example] = []
    for row in _iter_rows(split, limit):
        # supporting_facts is two parallel lists: title[i] pairs with sent_id[i].
        # Several sentences can come from the same paragraph, so deduplicate
        # while preserving order.
        gold = dict.fromkeys(row["supporting_facts"]["title"])
        examples.append(
            Example(
                id=row["id"],
                question=row["question"],
                answer=row["answer"],
                gold_titles=tuple(gold),
            )
        )
    return examples


def load_passages(split: str = "validation", limit: int = 200) -> list[Passage]:
    """Build the retrieval corpus from the same rows `load_examples` reads.

    Titles are unique in HotpotQA, so we deduplicate on title: the same
    paragraph often appears as a distractor for several different questions.
    """
    seen: dict[str, Passage] = {}
    for row in _iter_rows(split, limit):
        titles = row["context"]["title"]
        sentence_lists = row["context"]["sentences"]
        for title, sentences in zip(titles, sentence_lists):
            if title not in seen:
                # HotpotQA sentences already carry their trailing spaces.
                seen[title] = Passage(title=title, text="".join(sentences).strip())
    return list(seen.values())


def load_training_pairs(split: str = "train", limit: int = 4000) -> list[tuple[str, str]]:
    """Build (question, gold document) pairs for contrastive fine-tuning.

    Include every gold paragraph. The paired loader supplies a multi-positive
    mask so correct companion passages are never trained as negatives.
    """
    pairs: list[tuple[str, str]] = []
    for row in _iter_rows(split, limit):
        gold_titles = row["supporting_facts"]["title"]
        if not gold_titles:
            continue
        title_to_sentences = dict(zip(row["context"]["title"], row["context"]["sentences"]))
        if not set(gold_titles).issubset(title_to_sentences):
            continue  # do not train an incomplete question with missing supervision
        for title in dict.fromkeys(gold_titles):
            if title in title_to_sentences:
                passage = Passage(title=title, text="".join(title_to_sentences[title]).strip())
                pairs.append((row["question"], passage.to_document()))
    return pairs
