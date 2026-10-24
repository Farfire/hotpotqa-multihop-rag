"""The vector store: FAISS holds the embeddings, we hold the passages.

FAISS only knows about float32 arrays and integer row numbers. It has no idea
what a Passage is. So the contract is: row *i* of the array we add is passage
*i* of our list, and we keep the two in the same order forever. Losing that
alignment is the classic vector-store bug — the search "works" and returns
confident nonsense.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np

from .data import Passage

INDEX_FILENAME = "index.faiss"
PASSAGES_FILENAME = "passages.json"
META_FILENAME = "meta.json"


def encoder_fingerprint(model_name: str, checkpoint: Path | None, max_length: int = 256) -> str:
    """Identify the exact weights used to produce a set of embeddings.

    Queries must be encoded by the *same* weights that encoded the corpus.
    Different weights put vectors in a different space, so the similarity scores
    become meaningless — and nothing crashes, you just get quietly bad results.
    Storing this alongside the index lets us catch the mismatch instead.
    """
    weights = (
        "pretrained"
        if checkpoint is None
        else hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest()[:12]
    )
    suffix = f":length={max_length}" if max_length != 256 else ""
    if model_name.startswith("intfloat/e5-"):
        suffix += ":e5-prefix-v1"
    return f"{model_name}@{weights}{suffix}"


@dataclass(frozen=True)
class Hit:
    """One search result."""

    passage: Passage
    score: float  # stage-specific score: cosine or cross-encoder logit, not a probability


class VectorIndex:
    """A FAISS index plus the passages its rows refer to."""

    def __init__(
        self,
        index: faiss.Index,
        passages: list[Passage],
        fingerprint: str = "",
        metadata: dict | None = None,
    ) -> None:
        if index.ntotal != len(passages):
            raise ValueError(f"index has {index.ntotal} vectors but got {len(passages)} passages")
        self.index = index
        self.passages = passages
        self.fingerprint = fingerprint
        self.metadata = metadata or {}

    def check_fingerprint(self, query_fingerprint: str) -> None:
        """Fail loudly if the query encoder differs from the corpus encoder."""
        if self.fingerprint and self.fingerprint != query_fingerprint:
            raise ValueError(
                "Encoder mismatch — this index was built with "
                f"'{self.fingerprint}' but you are searching with "
                f"'{query_fingerprint}'. Results would be meaningless. "
                "Rebuild the index with the same --checkpoint you are querying with."
            )

    @classmethod
    def build(
        cls,
        embeddings: np.ndarray,
        passages: list[Passage],
        fingerprint: str = "",
    ) -> VectorIndex:
        """Exact inner-product search over unit-length vectors = cosine ranking.

        `IndexFlatIP` compares the query against every stored vector. That is
        O(corpus size) per query, which is fine up to ~1M vectors and always
        returns the true nearest neighbours. Past that you would swap in an
        approximate index (`IndexIVFFlat`, `IndexHNSWFlat`) and trade a little
        recall for a lot of speed — the rest of this file would not change.
        """
        embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)
        return cls(index, passages, fingerprint)

    def search(self, query_embeddings: np.ndarray, k: int) -> list[list[Hit]]:
        """Search many queries at once; result *i* belongs to query *i*."""
        query_embeddings = np.ascontiguousarray(query_embeddings, dtype=np.float32)
        scores, indices = self.index.search(query_embeddings, k)

        results: list[list[Hit]] = []
        for row_scores, row_indices in zip(scores, indices):
            hits = [
                Hit(passage=self.passages[idx], score=float(score))
                # FAISS pads with -1 when the corpus is smaller than k.
                for score, idx in zip(row_scores, row_indices)
                if idx != -1
            ]
            results.append(hits)
        return results

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(directory / INDEX_FILENAME))
        payload = [{"title": p.title, "text": p.text} for p in self.passages]
        (directory / PASSAGES_FILENAME).write_text(
            json.dumps(payload, ensure_ascii=False), encoding="utf-8"
        )
        (directory / META_FILENAME).write_text(
            json.dumps({**self.metadata, "fingerprint": self.fingerprint}), encoding="utf-8"
        )

    @classmethod
    def load(cls, directory: Path) -> VectorIndex:
        index_path = directory / INDEX_FILENAME
        if not index_path.exists():
            raise FileNotFoundError(f"No index at {directory}. Run scripts/build_index.py first.")
        index = faiss.read_index(str(index_path))
        payload = json.loads((directory / PASSAGES_FILENAME).read_text(encoding="utf-8"))
        passages = [Passage(title=p["title"], text=p["text"]) for p in payload]

        meta_path = directory / META_FILENAME
        fingerprint = ""
        metadata = {}
        if meta_path.exists():
            metadata = json.loads(meta_path.read_text(encoding="utf-8"))
            fingerprint = metadata.get("fingerprint", "")
        return cls(index, passages, fingerprint, metadata)

    def __len__(self) -> int:
        return len(self.passages)
