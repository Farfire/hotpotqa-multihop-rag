"""Tutorial stages 3 and 4: the objective, and the loop that optimises it.

HotpotQA supplies multiple gold passages per question. Score every question
against the batch's documents, then share the target probability across all
known golds present in that batch. Other documents provide in-batch negatives.
The positive mask prevents companion passages from becoming false negatives.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from .config import EncoderConfig, TrainingConfig
from .dataset import Batch, to_device
from .encoder import TextEncoder


@dataclass(frozen=True)
class EpochResult:
    """What one pass over the data produced."""

    loss: float
    accuracy: float  # share of questions whose top match was the right document


def _score_batch(
    encoder: TextEncoder,
    questions: Batch,
    documents: Batch,
    temperature: float,
) -> tuple[Tensor, Tensor]:
    """Return the [N, N] similarity matrix and the labels that go with it."""
    # One shared encoder handles both sides. Questions and documents therefore
    # land in the same vector space, which is the whole point — at search time
    # we compare an encoded question against encoded documents.
    q = encoder(questions["input_ids"], questions["attention_mask"])  # [N, D]
    d = encoder(documents["input_ids"], documents["attention_mask"])  # [N, D]

    # Both sides are unit-length, so this matrix holds cosine similarities.
    # Dividing by a small temperature spreads the values out before the softmax
    # inside the multi-positive loss, which sharpens the gradient signal.
    scores = (q @ d.T) / temperature  # [N, N]
    labels = questions["positive_mask"]
    return scores, labels


def train_one_epoch(
    encoder: TextEncoder,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    device: torch.device,
    temperature: float,
    max_grad_norm: float,
    log_every: int = 20,
) -> EpochResult:
    """One pass over the training data, five steps per mini-batch."""
    encoder.train()  # dropout on

    total_loss = 0.0
    total_correct = 0
    total_seen = 0

    for step, (questions, documents) in enumerate(loader, start=1):
        questions = to_device(questions, device)
        documents = to_device(documents, device)

        # 1. Clear gradients left over from the previous batch. PyTorch
        #    accumulates them by default, so skipping this silently trains on a
        #    running sum of every batch so far.
        optimizer.zero_grad()

        # 2. Forward pass.
        scores, labels = _score_batch(encoder, questions, documents, temperature)

        # 3. How wrong were we?
        loss = criterion(scores, labels)

        # 4. Backpropagate: autograd walks the graph and fills in .grad on every
        #    parameter that contributed to the loss.
        loss.backward()

        # 4b. Rescale gradients if their combined norm is too large. Not in the
        #     minimal recipe, but transformers are prone to one outlier batch
        #     producing a huge gradient that undoes an epoch of progress.
        torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_grad_norm)

        # 5. Apply the update.
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_correct += int(labels.gather(1, scores.argmax(dim=1, keepdim=True)).sum())
        total_seen += batch_size

        if log_every and step % log_every == 0:
            print(f"  step {step:>4}  loss {loss.item():.4f}")

    return EpochResult(
        loss=total_loss / max(total_seen, 1),
        accuracy=total_correct / max(total_seen, 1),
    )


@torch.no_grad()
def validate(
    encoder: TextEncoder,
    loader: DataLoader,
    criterion: nn.Module,
    *,
    device: torch.device,
    temperature: float,
) -> EpochResult:
    """Tutorial stage 5: same forward pass, no backward pass.

    Note what is missing compared to `train_one_epoch`: no `zero_grad`, no
    `backward`, no `step`. Calling any of them here would train on the
    validation set and make the number meaningless.

    Caveat on the numbers this returns: the negatives come from the batch, so a
    larger batch is a harder task. Loss and accuracy are only comparable across
    runs that used the same batch size — and both look flattering, because
    picking the right passage out of 31 random ones is much easier than out of
    a 2,000-paragraph corpus full of deliberate distractors. Treat them as a
    "is training working at all" signal; `scripts/evaluate.py` is the real test.
    """
    encoder.eval()  # dropout off

    total_loss = 0.0
    total_correct = 0
    total_seen = 0

    for questions, documents in loader:
        questions = to_device(questions, device)
        documents = to_device(documents, device)

        scores, labels = _score_batch(encoder, questions, documents, temperature)
        loss = criterion(scores, labels)

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_correct += int(labels.gather(1, scores.argmax(dim=1, keepdim=True)).sum())
        total_seen += batch_size

    return EpochResult(
        loss=total_loss / max(total_seen, 1),
        accuracy=total_correct / max(total_seen, 1),
    )


class MultiPositiveLoss(nn.Module):
    """Give every known gold passage a positive target, including companions."""

    def forward(self, scores: Tensor, positives: Tensor) -> Tensor:
        targets = positives.to(scores.dtype)
        targets = targets / targets.sum(dim=1, keepdim=True).clamp_min(1)
        return -(targets * scores.log_softmax(dim=1)).sum(dim=1).mean()


def build_criterion() -> nn.Module:
    return MultiPositiveLoss()


def build_optimizer(encoder: TextEncoder, training: TrainingConfig) -> torch.optim.Optimizer:
    """The optimizer — stage 3, second half.

    AdamW holds a reference to the parameters, so `optimizer.step()` knows what
    to update. Fine-tuning uses a much smaller learning rate than training from
    scratch: the pretrained weights are already good, and a large step would
    destroy what they know before learning anything new.
    """
    return torch.optim.AdamW(encoder.parameters(), lr=training.learning_rate)


def split_pairs(
    pairs: list[tuple[str, str]], val_fraction: float
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """Split by question before pairing, so companion golds stay together."""
    questions = list(dict.fromkeys(q for q, _ in pairs))
    if len(questions) < 2 or not 0 < val_fraction < 1:
        raise ValueError("need at least two questions and a valid validation fraction")
    cutoff = min(len(questions) - 1, max(1, int(len(questions) * (1 - val_fraction))))
    train_questions = set(questions[:cutoff])
    return (
        [p for p in pairs if p[0] in train_questions],
        [p for p in pairs if p[0] not in train_questions],
    )


def describe_setup(encoder_config: EncoderConfig, training: TrainingConfig) -> str:
    return (
        f"model={encoder_config.model_name} device={encoder_config.device.type} "
        f"epochs={training.epochs} batch={training.batch_size} "
        f"lr={training.learning_rate} temperature={training.temperature}"
    )
