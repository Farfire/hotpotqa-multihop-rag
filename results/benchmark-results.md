# Final retrieval results

## Headline

The evidence-grounded query-rewrite pipeline produced the best top-five retrieval result on this completed 200-question sample: it retrieved **both gold supporting paragraphs for 94.0% of questions**, compared with **90.0%** for hybrid retrieval and **89.5%** for dense retrieval. It took longer: **4.74 seconds/question on average** versus **0.40–0.44 seconds** for the other two.

These percentages measure retrieval of the two annotated support paragraphs. They are **not answer accuracy**; answer generation was disabled in this experiment.

## Setup

- **Dataset:** HotpotQA distractor validation questions, dataset rows `[5050, 5250)` (200 questions), sampled with seed `20261023`.
- **Search corpus:** 48,466 unique Wikipedia paragraphs pooled from validation rows `[0, 5000)`.
- **Encoder:** `intfloat/e5-base-v2`, max input length 512 tokens.
- **Candidate retrieval:** up to 50 passages per search; two retrieval hops; reciprocal-rank fusion constant `k=60` in hybrid methods.
- **Reranker:** the included cross-encoder fine-tuned with hard negatives from the `cross-encoder/ms-marco-MiniLM-L6-v2` base model.
- **Execution:** Apple MPS. Retrieval only; Gemini answer generation excluded. Architecture order was rotated between questions.
- **Metric:** `both_gold@k` means both annotated supporting paragraphs occur in the final top `k` passages.

## Results

| Architecture | Both gold @2 | @5 | @10 | @20 | Both in expanded candidates | Mean retrieval time |
|---|---:|---:|---:|---:|---:|---:|
| Dense E5 | 60.0% | 89.5% | 92.0% | 94.5% | 95.5% | 0.438 s/question |
| Hybrid E5 + BM25/RRF | 59.0% | 90.0% | 94.0% | 96.0% | 96.0% | 0.402 s/question |
| Hybrid + query rewrite | 59.0% | **94.0%** | **97.0%** | **97.0%** | **97.0%** | 4.740 s/question |

On the paired questions at `@5`, query rewrite beat hybrid on 9 questions, lost on 1, and tied on 190: **+4 percentage points** (bootstrap 95% interval: +1 to +7 points). For `@10`, it won 6, lost 0, tied 194: **+3 points** (interval: +1 to +5.5 points). These intervals reflect resampling this one slice; they do not establish performance on every HotpotQA question.

BM25/RRF alone had a small effect on this slice: hybrid exceeded dense by 0.5 points at `@5`, 2 points at `@10`, and 1.5 points at `@20`. Its contribution appears more visible at deeper cutoffs than in the top five here.

## What the query rewrite actually did

Gemini proposed an evidence-grounded follow-up question for **78/200 questions (39%)**. For the other 122, it found no justified follow-up and the pipeline kept the baseline evidence. This avoids inventing entity links, but means the additional API call is paid even when no rewrite is accepted.

## Latency diagnostic

A separate fresh-cache timing check used 10 random questions from the validation sample and a 65,309-paragraph index. All 10 Gemini calls succeeded; 5 produced a supported rewrite and 5 abstained. Model/index loading (10.0 seconds) and answer generation were excluded.

- Total mean: **5.99 s/question** (median 6.08; range 5.12–6.84).
- Gemini rewrite stage: **5.02 s/question average**, of which the API request was about 5.02 s and explicit wait/backoff averaged 0.07 s.
- Baseline retrieval before rewriting: **0.65 s/question** (candidate search 0.23 s; cross-encoder 0.41 s).
- Follow-up retrieval and reranking add about **0.32 s/question averaged across all 10** (only 5 accepted a rewrite).

The earlier 33.84 s/question average came from an incomplete 569-question-per-method run. In that run, 266/569 query-rewrite requests failed on Gemini quota and spent about 65.3 seconds each in retries; 113 rewrote normally (7.2 s mean), and 190 abstained (5.63 s mean). The 33.84-second number accurately described that failed run's mixed outcomes, but it did not estimate normal successful-call latency.

## Larger-run status

A planned 2,000-question comparison stopped after **569 questions per architecture** because Gemini quota failures interrupted query rewriting. Its rewrite path had 266 API errors and used baseline fallback for those questions. As a result, it is an incomplete, quota-contaminated checkpoint—not a complete, fair comparison—and its scores are not presented as the final benchmark. The resumable local run data remains outside this clean repository.

## Conclusions

1. **Best retrieval quality in this sample:** hybrid + query rewrite, especially at top 5 and top 10.
2. **Best speed/quality trade-off:** hybrid. It was roughly 12× faster than query rewrite and close in top-five retrieval on this 200-question sample.
3. **BM25/RRF:** modest improvement over dense retrieval in this slice; test on a larger quota-independent run before treating it as a general gain.
4. **Main query-rewrite bottleneck:** Gemini request latency, not local search. Cache or batch/query rewrite asynchronously if latency matters; use a lower-latency model only after re-evaluation.
5. **Next evaluation:** complete a larger paired retrieval benchmark with a stable rewrite provider or pre-generated rewrites; separately measure answer exact match/F1 and supporting-fact F1. Do not use paragraph-retrieval recall as a substitute for answer quality.

Machine-readable values are in [benchmark-results.json](benchmark-results.json).
