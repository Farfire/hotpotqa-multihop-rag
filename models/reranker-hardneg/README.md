# Fine-tuned reranker checkpoint

This is the selected hard-negative fine-tuned checkpoint used in the 200-question benchmark in `results/benchmark-results.md`.

- Base model: [`cross-encoder/ms-marco-MiniLM-L6-v2`](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2)
- Base license: Apache License 2.0 (copy included as `LICENSE`)
- Fine-tuning data: HotpotQA training split; no validation answers used for training this checkpoint.
- File format: SafeTensors (no Python pickle payload)

The retrieval encoder, `intfloat/e5-base-v2`, is downloaded from Hugging Face at runtime and is not included in this directory.
