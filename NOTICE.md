# Model and data notices

- **HotpotQA**: dataset maintained by its upstream authors and distributed through Hugging Face Datasets. The corpus contains Wikipedia text; consult the dataset card and source content terms before redistribution. This repository does not redistribute the dataset.
- **E5**: `intfloat/e5-base-v2`, MIT License. Downloaded from Hugging Face at runtime; model weights are not included here.
- **Cross-encoder base**: `cross-encoder/ms-marco-MiniLM-L6-v2`, Apache License 2.0. The included fine-tuned checkpoint was trained from this model using HotpotQA training questions and hard negatives. See the upstream model card for its notices and terms.
- **Gemini**: optional hosted API used only for query rewriting and answer generation; subject to Google's current service terms and quotas.

The original source code in this repository is licensed under MIT. Dataset, model, and API terms are separate from this code license.

The complete Apache License 2.0 notice for the included fine-tuned reranker checkpoint is in `models/reranker-hardneg/LICENSE`.
