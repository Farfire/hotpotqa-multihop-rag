#!/usr/bin/env python3
"""Embed the HotpotQA corpus and save a FAISS index.

.venv/bin/python scripts/build_index.py
.venv/bin/python scripts/build_index.py --n 500 --checkpoint artifacts/encoder-finetuned.pt
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from rag.config import CHECKPOINT_PATH, INDEX_DIR, EncoderConfig
from rag.data import load_passages
from rag.encoder import encode, load_encoder, load_tokenizer
from rag.index import VectorIndex, encoder_fingerprint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--n", type=int, default=200, help="HotpotQA examples to draw the corpus from"
    )
    parser.add_argument("--split", default="validation")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help=f"fine-tuned weights to use instead of the pretrained model "
        f"(try {CHECKPOINT_PATH.name} after training)",
    )
    parser.add_argument("--out", type=Path, default=INDEX_DIR)
    parser.add_argument("--model", default=EncoderConfig().model_name)
    parser.add_argument("--max-length", type=int, default=EncoderConfig().max_length)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = EncoderConfig(model_name=args.model, max_length=args.max_length)

    print(f"device: {config.device.type}")
    print(f"loading {args.n} examples from {args.split} ...")
    passages = load_passages(split=args.split, limit=args.n)
    print(f"corpus: {len(passages)} unique paragraphs")

    tokenizer = load_tokenizer(config)
    encoder = load_encoder(config, checkpoint=args.checkpoint)
    if args.checkpoint:
        print(f"using fine-tuned weights from {args.checkpoint}")

    print("encoding ...")
    started = time.perf_counter()
    embeddings = encode(
        encoder,
        [p.to_document() for p in passages],
        tokenizer,
        config,
        progress_every=10,
    )
    elapsed = time.perf_counter() - started
    print(f"embeddings: {embeddings.shape} in {elapsed:.1f}s")

    # Record which weights produced these vectors, so a later search with
    # different weights fails loudly instead of returning quiet nonsense.
    fingerprint = encoder_fingerprint(config.model_name, args.checkpoint, config.max_length)
    index = VectorIndex.build(embeddings, passages, fingerprint)
    index.metadata = {
        "model_name": config.model_name,
        "max_length": config.max_length,
        "split": args.split,
        "n": args.n,
    }
    index.save(args.out)
    print(f"saved index to {args.out}  (encoder: {fingerprint})")


if __name__ == "__main__":
    main()
