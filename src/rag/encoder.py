"""Tutorial stage 2: the model — plus the stage-5 pattern for running it.

The model is a pretrained transformer with a pooling step bolted on. The
transformer emits one vector per *token*; we need one vector per *text*, so we
average the token vectors (ignoring padding) and normalise the result to unit
length.

Normalising matters: once every vector has length 1, the dot product between two
of them *is* their cosine similarity. That lets FAISS use its fast inner-product
index and still rank by cosine.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from transformers import AutoModel, AutoTokenizer, PreTrainedTokenizerBase

from .config import EncoderConfig
from .dataset import make_text_loader, to_device


def mean_pool(token_embeddings: Tensor, attention_mask: Tensor) -> Tensor:
    """Average token vectors, skipping padding.

    A naive `.mean(dim=1)` would average the padding vectors in too, so short
    texts in a batch with a long one would come out systematically wrong. The
    mask zeroes the padding before summing, and we divide by the real token
    count.

    Shapes: [batch, length, dim] + [batch, length] -> [batch, dim]
    """
    mask = attention_mask.unsqueeze(-1).to(token_embeddings.dtype)
    summed = (token_embeddings * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)  # clamp guards against a 0-token text
    return summed / counts


class TextEncoder(nn.Module):
    """Maps a batch of tokenized text to unit-length vectors."""

    def __init__(self, model_name: str) -> None:
        # nn.Module.__init__ sets up the machinery that tracks parameters and
        # gradients. Forgetting this line is the classic PyTorch bug: the model
        # then reports zero parameters and the optimizer has nothing to update.
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)

    @property
    def embedding_dim(self) -> int:
        return int(self.backbone.config.hidden_size)

    def forward(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        """Call this as `encoder(ids, mask)` — never `encoder.forward(...)`.

        The `__call__` wrapper is what runs PyTorch's hooks; calling `forward`
        directly silently skips them.
        """
        output = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        pooled = mean_pool(output.last_hidden_state, attention_mask)
        return F.normalize(pooled, p=2, dim=1)


def load_tokenizer(config: EncoderConfig) -> PreTrainedTokenizerBase:
    return AutoTokenizer.from_pretrained(config.model_name)


def load_encoder(config: EncoderConfig, checkpoint: Path | None = None) -> TextEncoder:
    """Build the encoder, optionally restoring fine-tuned weights."""
    encoder = TextEncoder(config.model_name)
    if checkpoint is not None:
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        encoder.load_state_dict(state)
    return encoder.to(config.device)


def save_encoder(encoder: TextEncoder, path: Path) -> None:
    """Save weights only, not the pickled object.

    A `state_dict` is just a tensor dictionary — it survives refactors of this
    file, whereas `torch.save(model)` pickles the class and breaks the moment
    you rename anything.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(encoder.state_dict(), path)


@torch.no_grad()  # tutorial stage 5: no backward pass, so don't build the graph
def encode(
    encoder: TextEncoder,
    texts: list[str],
    tokenizer: PreTrainedTokenizerBase,
    config: EncoderConfig,
    *,
    role: str = "passage",
    progress_every: int = 0,
) -> np.ndarray:
    """Turn texts into an array of shape [len(texts), embedding_dim].

    `eval()` switches dropout and batch-norm layers into inference behaviour.
    Together with `@torch.no_grad()` this is the standard inference pair: one
    changes what the layers do, the other stops PyTorch from recording
    operations for backpropagation.
    """
    encoder.eval()
    loader = make_text_loader(
        config.prepare(texts, role),
        tokenizer,
        batch_size=config.batch_size,
        max_length=config.max_length,
        num_workers=config.num_workers,
    )

    chunks: list[np.ndarray] = []
    for step, batch in enumerate(loader, start=1):
        batch = to_device(batch, config.device)
        vectors = encoder(batch["input_ids"], batch["attention_mask"])
        # FAISS wants float32 numpy on the CPU.
        chunks.append(vectors.cpu().numpy().astype(np.float32))
        if progress_every and step % progress_every == 0:
            print(f"  encoded {min(step * config.batch_size, len(texts))}/{len(texts)}")

    if not chunks:
        return np.zeros((0, encoder.embedding_dim), dtype=np.float32)
    return np.vstack(chunks)
