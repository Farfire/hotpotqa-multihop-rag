"""Tutorial stage 1: Dataset and DataLoader.

A `Dataset` answers two questions — how many items are there (`__len__`) and
what is item *i* (`__getitem__`). It stays deliberately dumb: it hands back raw
Python strings.

A `DataLoader` wraps a Dataset to produce mini-batches. Turning a list of
strings into padded tensors is the job of `collate_fn`, which the DataLoader
calls once per batch. Doing it here rather than in `__getitem__` means each
batch is padded to *its own* longest item instead of the global maximum — less
wasted compute on short batches.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedTokenizerBase

# What a collated batch looks like: the tokenizer's tensors, keyed by name
# ("input_ids", "attention_mask", ...).
Batch = dict[str, torch.Tensor]


class TextDataset(Dataset[str]):
    """A flat list of strings to encode."""

    def __init__(self, texts: list[str]) -> None:
        self.texts = texts

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, index: int) -> str:
        return self.texts[index]


class PairDataset(Dataset[tuple[str, str]]):
    """(question, matching document) pairs used for contrastive training."""

    def __init__(self, pairs: list[tuple[str, str]]) -> None:
        self.pairs = pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> tuple[str, str]:
        return self.pairs[index]


def make_text_collate_fn(
    tokenizer: PreTrainedTokenizerBase, max_length: int
) -> Callable[[list[str]], Batch]:
    """Build the function that turns a list of strings into padded tensors."""

    def collate(texts: list[str]) -> Batch:
        return tokenizer(
            texts,
            padding=True,  # pad to the longest item in *this* batch
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

    return collate


def make_pair_collate_fn(
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
    positives: dict[str, set[str]] | None = None,
    query_prefix: str = "",
    passage_prefix: str = "",
) -> Callable[[list[tuple[str, str]]], tuple[Batch, Batch]]:
    """Same idea, but tokenizes the two sides of each pair separately."""
    collate_text = make_text_collate_fn(tokenizer, max_length)

    def collate(pairs: list[tuple[str, str]]) -> tuple[Batch, Batch]:
        questions, documents = zip(*pairs)
        q_batch = collate_text([query_prefix + q for q in questions])
        d_batch = collate_text([passage_prefix + d for d in documents])
        known = positives or {q: {d for q2, d in pairs if q2 == q} for q in questions}
        q_batch["positive_mask"] = torch.tensor(
            [[d in known[q] for d in documents] for q in questions], dtype=torch.bool
        )
        return q_batch, d_batch

    return collate


def make_text_loader(
    texts: list[str],
    tokenizer: PreTrainedTokenizerBase,
    *,
    batch_size: int,
    max_length: int,
    num_workers: int = 0,
) -> DataLoader:
    """DataLoader for inference — order matters, so never shuffle."""
    return DataLoader(
        TextDataset(texts),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=make_text_collate_fn(tokenizer, max_length),
    )


def make_pair_loader(
    pairs: list[tuple[str, str]],
    tokenizer: PreTrainedTokenizerBase,
    *,
    batch_size: int,
    max_length: int,
    shuffle: bool,
    num_workers: int = 0,
    query_prefix: str = "",
    passage_prefix: str = "",
) -> DataLoader:
    """DataLoader for training.

    `shuffle=True` matters more than usual here: the negatives come from
    whichever other pairs land in the batch, so a fixed order would show the
    model the same negatives every epoch.

    `drop_last=True` discards a final partial batch — a batch of 1 has no
    negatives at all and produces a meaningless loss of 0.
    """
    positives: dict[str, set[str]] = {}
    for q, d in pairs:
        positives.setdefault(q, set()).add(d)
    return DataLoader(
        PairDataset(pairs),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=shuffle,
        collate_fn=make_pair_collate_fn(
            tokenizer, max_length, positives, query_prefix, passage_prefix
        ),
    )


def to_device(batch: Batch, device: torch.device) -> Batch:
    """Move every tensor in a batch onto the compute device."""
    return {key: value.to(device) for key, value in batch.items()}
