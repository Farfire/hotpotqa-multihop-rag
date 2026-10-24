"""Cross-encoder scoring: query and document attend to each other."""

from __future__ import annotations

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .config import RERANKER_MODEL, resolve_device
from .index import Hit


class Reranker:
    def __init__(
        self,
        model_name: str = RERANKER_MODEL,
        *,
        batch_size: int = 16,
        max_length: int = 512,
        device: torch.device | None = None,
    ) -> None:
        if batch_size < 1 or max_length < 8:
            raise ValueError("invalid reranker batch size or maximum length")
        self.model_name = model_name
        self.device = device or resolve_device()
        self.batch_size = batch_size
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)
        if self.model.config.num_labels != 1:
            raise ValueError("reranker must output one relevance logit per pair")

    @torch.no_grad()
    def score(self, question: str, documents: list[str]) -> list[float]:
        self.model.eval()
        scores: list[float] = []
        for start in range(0, len(documents), self.batch_size):
            docs = documents[start : start + self.batch_size]
            batch = self.tokenizer(
                [question] * len(docs),
                docs,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)
            scores.extend(self.model(**batch).logits.flatten().float().cpu().tolist())
        return scores

    def rerank(self, question: str, hits: list[Hit], k: int | None = None) -> list[Hit]:
        unique = list({hit.passage.title: hit for hit in hits}.values())
        scores = self.score(question, [hit.passage.to_document() for hit in unique])
        ranked = sorted(
            (Hit(hit.passage, score) for hit, score in zip(unique, scores)),
            key=lambda hit: hit.score,
            reverse=True,
        )
        return ranked if k is None else ranked[:k]
