# HotpotQA Multi-Hop RAG

A small, reproducible retrieval-augmented generation project for HotpotQA. It compares three explicit retrieval pipelines over Wikipedia paragraphs: dense E5 search, E5 + BM25 rank fusion, and hybrid retrieval followed by evidence-grounded query rewriting.

## Results at a glance

On a completed 200-question HotpotQA validation slice, the query-rewrite pipeline placed both supporting paragraphs in its top five for **94.0%** of questions, versus **90.0%** for hybrid and **89.5%** for dense retrieval. The query-rewrite pipeline averaged **4.74 seconds/question** in that run; dense and hybrid averaged **0.44s** and **0.40s**. These are retrieval metrics, not answer accuracy. See [the complete results and limitations](results/benchmark-results.md).

## What each method does

1. **Dense (`dense`)** — E5 embeds the question and paragraphs; FAISS returns the nearest 50 paragraph vectors; a cross-encoder reranks those candidates; a second retrieval pass searches from linked passage titles; then the cross-encoder ranks the expanded pool.
2. **Hybrid (`hybrid`)** — runs dense E5 and BM25 keyword retrieval, combines their rank lists with reciprocal-rank fusion (RRF), then uses the same cross-encoder and second-pass title expansion as dense.
3. **Query rewrite (`query_rewrite`)** — runs the hybrid pipeline, sends its top evidence to Gemini to propose a follow-up question only when the evidence supports an entity link, searches again for that question, then combines the original and follow-up evidence. This method needs a Gemini API key.

All methods share `intfloat/e5-base-v2` and the included hard-negative-trained cross-encoder. Device selection uses Apple MPS when available, then NVIDIA CUDA, then CPU.

## Quick start

Python 3.10+ is required. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\\Scripts\\activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
# macOS only: let FAISS and PyTorch share one OpenMP runtime
python scripts/fix_macos_openmp.py
python scripts/build_index.py --n 200 --out artifacts/demo-index
```

The index command downloads HotpotQA validation data, deduplicates its Wikipedia paragraphs, embeds them, and creates a local FAISS index. The first run also downloads E5 model weights. A 200-question demo index is small and useful for trying the app; use `--n 5000` for a broader corpus. The published benchmark used a separate 48,466-paragraph index, which is intentionally not committed.

### Retrieve passages using each method

```bash
python scripts/ask.py "Which writer was born first, Jane Austen or Charles Dickens?" --index artifacts/demo-index --architecture dense --no-answer
python scripts/ask.py "Which writer was born first, Jane Austen or Charles Dickens?" --index artifacts/demo-index --architecture hybrid --no-answer
```

Query rewriting additionally needs `google-genai` and `GEMINI_API_KEY`:

```bash
python -m pip install -e '.[gemini]'
export GEMINI_API_KEY="your-key"
python scripts/ask.py "Which writer was born first, Jane Austen or Charles Dickens?" --index artifacts/demo-index --architecture query_rewrite --no-answer
```

To have Gemini answer from the retrieved passages, omit `--no-answer`. Answer generation also uses `GEMINI_API_KEY`. For a complete benchmark-like query, build a larger index; the demo index may not contain the required evidence.

## Run tests and checks

```bash
pytest
ruff check .
```

## Reproduce the 200-question retrieval comparison

This benchmark compares all three methods on the same 200 validation questions. It requires a larger corpus than the quick-start demo and calls Gemini for query rewrites. To reproduce the published corpus scope, first build the index over validation rows `[0, 5000)`:

```bash
python scripts/build_index.py --n 5000 --split validation --out artifacts/benchmark-index
```

Then run the comparison:

```bash
python scripts/compare_architectures.py \
  --n 200 --offset 5050 --range-size 200 --seed 20261023 \
  --dataset-limit 5250 --index artifacts/benchmark-index \
  --out artifacts/benchmark-run --no-answers
```

The script writes incremental per-question records, settings, a summary, and a Markdown report to `--out`, so interrupted runs retain completed work. Set `GEMINI_API_KEY` before running because only the query-rewrite architecture calls Gemini. These commands download the validation rows and model weights if they are not cached. Index creation for a 48k-paragraph corpus is memory- and time-intensive.

## Repository map

- `src/rag/` — data loading, E5 encoding, FAISS, BM25/RRF, reranking, and the three architecture functions.
- `scripts/ask.py` — retrieve passages for one question and optionally generate a grounded answer.
- `scripts/build_index.py` — build a local paragraph index.
- `scripts/fix_macos_openmp.py` — macOS-only repair for duplicate OpenMP runtimes inside this virtual environment.
- `scripts/compare_architectures.py` — resumable paired retrieval evaluation.
- `models/reranker-hardneg/` — the 87 MiB fine-tuned cross-encoder checkpoint used by the reported benchmark.
- `results/benchmark-results.md` — final benchmark and latency findings.

## Data, models, and credentials

HotpotQA is downloaded from [Hugging Face Datasets](https://huggingface.co/datasets/hotpotqa/hotpot_qa); the dataset and source Wikipedia text retain their upstream terms. E5-base-v2 is downloaded from [intfloat/e5-base-v2](https://huggingface.co/intfloat/e5-base-v2) under MIT. The reranker starts from [cross-encoder/ms-marco-MiniLM-L6-v2](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2) (Apache-2.0) and is fine-tuned on HotpotQA training examples; its checkpoint is included here. See [NOTICE.md](NOTICE.md).

Never commit API keys. Set `GEMINI_API_KEY` as an exported shell variable; `.env` files are ignored by Git but are not loaded automatically. Retrieval-only methods do not need an API key; query rewriting and answer generation do.
