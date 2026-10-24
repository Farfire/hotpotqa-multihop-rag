"""Every tunable value in one place, so no magic numbers hide inside the logic."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import torch

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
ARTIFACTS_DIR: Path = PROJECT_ROOT / "artifacts"
INDEX_DIR: Path = ARTIFACTS_DIR / "index"
CHECKPOINT_PATH: Path = ARTIFACTS_DIR / "encoder-finetuned.pt"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"
RERANKER_SELECTION = ARTIFACTS_DIR / "reranker-selection.json"
BUNDLED_RERANKER = PROJECT_ROOT / "models" / "reranker-hardneg"


def selected_reranker() -> str:
    """Use an explicitly selected local checkpoint; fresh checkouts use the Hub model."""
    if not RERANKER_SELECTION.exists():
        if (BUNDLED_RERANKER / "config.json").exists():
            return str(BUNDLED_RERANKER)
        return RERANKER_MODEL
    selection = json.loads(RERANKER_SELECTION.read_text(encoding="utf-8"))
    checkpoint = PROJECT_ROOT / selection["checkpoint"]
    if not (checkpoint / "config.json").exists():
        raise ValueError(
            f"Selected reranker is missing at {checkpoint}; restore it or remove {RERANKER_SELECTION}."
        )
    return str(checkpoint)


def resolve_device() -> torch.device:
    """Pick the fastest available device.

    `mps` is Apple Silicon's GPU backend, `cuda` is NVIDIA's. Tensors and the
    model must live on the *same* device or PyTorch raises at the forward pass —
    that is why every loop in this project calls `.to(device)` on its batch.
    """
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@dataclass(frozen=True)
class EncoderConfig:
    """Settings for turning text into vectors."""

    # Retrieval-trained English encoder; MiniLM remains available via --model.
    model_name: str = "intfloat/e5-base-v2"

    # E5 supports 512 tokens. Include the title and preserve longer evidence.
    max_length: int = 512

    # How many texts to push through the model at once. Larger = faster, until
    # you run out of memory.
    batch_size: int = 16

    # macOS + HuggingFace tokenizers deadlock when forked into subprocesses, so
    # tokenize on the main process. Bump this on Linux if data loading is the
    # bottleneck.
    num_workers: int = 0

    device: torch.device = field(default_factory=resolve_device)

    def prepare(self, texts: list[str], role: str) -> list[str]:
        if role not in {"query", "passage"}:
            raise ValueError("role must be query or passage")
        if self.model_name.startswith("intfloat/e5-"):
            return [f"{role}: {text}" for text in texts]
        return texts


@dataclass(frozen=True)
class TrainingConfig:
    """Settings for fine-tuning the encoder (tutorial stages 3 and 4)."""

    epochs: int = 1
    learning_rate: float = 2e-5

    # Contrastive training uses the *other* items in the batch as negatives, so
    # batch size controls the number of available negatives. All known golds
    # are masked as positives, including companion and shared passages.
    batch_size: int = 8

    # Divides the similarity scores before the softmax. Lower = sharper
    # distinction between the right passage and the wrong ones. 0.05 is the
    # standard value for L2-normalised embeddings.
    temperature: float = 0.05

    # Fraction of training examples held out to check generalisation.
    val_fraction: float = 0.1

    # Gradients are clipped to this norm to stop a single bad batch from
    # wrecking the weights.
    max_grad_norm: float = 1.0
