#!/usr/bin/env python3
"""Resumable, paired diagnostic of all three retrieval/answer configurations."""

import argparse
import hashlib
import json
import os
import random
import statistics
import time
from collections import Counter
from pathlib import Path

import numpy as np

from rag.answer_metrics import score_answer
from rag.architectures import NAMES, Architecture, GeminiQueryRewriter
from rag.architectures.rewrite import PROMPT_V1
from rag.config import EncoderConfig, selected_reranker
from rag.data import Example, _iter_rows
from rag.encoder import load_encoder, load_tokenizer
from rag.generate import MODEL, SYSTEM_PROMPT, format_context
from rag.index import VectorIndex
from rag.rerank import Reranker
from rag.retrieval import Retriever


def atomic(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    os.replace(temp, path)


def generate(rewriter, question, hits, directory):
    message = f"Passages:\n\n{format_context(hits)}\n\nQuestion: {question}"
    key = hashlib.sha256(
        (rewriter.model + SYSTEM_PROMPT + message + "temperature=0,max=512").encode()
    ).hexdigest()
    path = directory / f"{key}.json"
    if path.exists():
        result = json.loads(path.read_text())
        result["cache_hit"] = True
        return result
    start = time.perf_counter()
    for attempt in range(3):
        try:
            response = rewriter.request(
                contents=message,
                config=rewriter.types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    automatic_function_calling=rewriter.types.AutomaticFunctionCallingConfig(
                        disable=True
                    ),
                    temperature=0,
                    max_output_tokens=512,
                ),
            )
            if not response.text:
                raise ValueError("empty generated answer")
            result = {
                "text": response.text.strip(),
                "model": rewriter.model,
                "model_version": getattr(response, "model_version", None),
                "usage": response.usage_metadata.model_dump(mode="json")
                if response.usage_metadata
                else {},
                "seconds": time.perf_counter() - start,
                "cache_hit": False,
            }
            atomic(path, result)
            return result
        except Exception as exc:
            if getattr(exc, "code", None) not in (429, 500, 503) or attempt == 2:
                raise
            time.sleep(2**attempt)


def both(record, k):
    return set(record["gold_titles"]) <= {h["title"] for h in record["hits"][:k]}


def summarize(out, settings, records, complete=False, *, write_report=False):
    metrics = {}
    for name in NAMES:
        rows = [r for r in records if r["architecture"] == name]
        n = len(rows)
        metrics[name] = {
            "n": n,
            "both_gold": {
                str(k): sum(both(r, k) for r in rows) / n if n else None for k in (2, 5, 10, 20)
            },
            "both_gold_in_first_hop": sum(
                set(r["gold_titles"]) <= set(r["first_hop"]) for r in rows
            )
            / n
            if n
            else None,
            "both_gold_in_expanded_candidates": sum(
                set(r["gold_titles"]) <= set(r["candidate_titles"]) for r in rows
            )
            / n
            if n
            else None,
            "mean_retrieval_seconds": statistics.mean(r["retrieval_seconds"] for r in rows)
            if n
            else None,
            "median_retrieval_seconds": statistics.median(r["retrieval_seconds"] for r in rows)
            if n
            else None,
            "p90_retrieval_seconds": float(np.quantile([r["retrieval_seconds"] for r in rows], 0.9))
            if n
            else None,
            "answer_em": statistics.mean(r["answer_scores"]["em"] for r in rows)
            if n and rows[0].get("answer_scores")
            else None,
            "answer_f1": statistics.mean(r["answer_scores"]["f1"] for r in rows)
            if n and rows[0].get("answer_scores")
            else None,
            "mean_answer_seconds": statistics.mean(r["answer"]["seconds"] for r in rows)
            if n and rows[0].get("answer")
            else None,
        }
    by_id = {(r["id"], r["architecture"]): r for r in records}
    pairs = [
        (by_id[(qid, "hybrid")], by_id[(qid, "query_rewrite")])
        for qid in settings["ids"]
        if (qid, "hybrid") in by_id and (qid, "query_rewrite") in by_id
    ]
    differences = np.array([int(both(c, 5)) - int(both(b, 5)) for b, c in pairs])
    ci = None
    if len(pairs) and complete:
        rng = np.random.default_rng(settings["seed"])
        ci = np.quantile(
            rng.choice(differences, (10000, len(pairs))).mean(axis=1), [0.025, 0.975]
        ).tolist()
    rewrite_rows = [r for r in records if r["architecture"] == "query_rewrite"]
    statuses = dict(Counter(r["trace"]["status"] for r in rewrite_rows))
    tokens = Counter()
    for r in records:
        for kind, payload in [
            ("answer", r["answer"]),
            ("rewrite", r.get("trace", {}).get("plan", {})),
        ]:
            if payload and not payload.get("cache_hit"):
                for key in (
                    "prompt_token_count",
                    "candidates_token_count",
                    "thoughts_token_count",
                    "total_token_count",
                ):
                    tokens[f"{kind}_{key}"] += payload.get("usage", {}).get(key) or 0
    payload = {
        "status": "complete" if complete else "in progress",
        "settings": settings,
        "metrics": metrics,
        "rewrite_statuses": statuses,
        "paired_at_5": {
            "n": len(pairs),
            "wins": int(np.sum(differences > 0)),
            "losses": int(np.sum(differences < 0)),
            "ties": int(np.sum(differences == 0)),
            "delta_95pct_bootstrap": ci,
        },
        "uncached_token_usage": dict(tokens),
        "cost_usd": None,
        "cost_note": "No verified billing rate applied; token usage retained. Cached responses do not create fresh API calls.",
    }
    paired_comparisons = {}
    for left_name, right_name in [
        ("query_rewrite", "hybrid"),
        ("query_rewrite", "dense"),
        ("hybrid", "dense"),
    ]:
        differences_by_k = {}
        complete_pairs = [
            (by_id[(qid, left_name)], by_id[(qid, right_name)])
            for qid in settings["ids"]
            if (qid, left_name) in by_id and (qid, right_name) in by_id
        ]
        for k in (2, 5, 10, 20):
            delta = np.array(
                [int(both(left, k)) - int(both(right, k)) for left, right in complete_pairs]
            )
            ci = (
                np.quantile(
                    np.random.default_rng(settings["seed"] + k)
                    .choice(delta, (10000, len(delta)))
                    .mean(axis=1),
                    [0.025, 0.975],
                ).tolist()
                if len(delta) and complete
                else None
            )
            differences_by_k[str(k)] = {
                "wins": int(np.sum(delta > 0)),
                "losses": int(np.sum(delta < 0)),
                "ties": int(np.sum(delta == 0)),
                "delta": float(delta.mean()) if len(delta) else None,
                "delta_95pct_bootstrap": ci,
            }
        paired_comparisons[f"{left_name}_vs_{right_name}"] = {
            "n": len(complete_pairs),
            "both_gold": differences_by_k,
        }
    payload["paired_comparisons"] = paired_comparisons
    payload["rewrite_queries"] = {
        "accepted": sum(len(r["trace"].get("plan", {}).get("queries", [])) for r in rewrite_rows),
        "rejected": sum(len(r["trace"].get("plan", {}).get("rejected", [])) for r in rewrite_rows),
        "cache_hits": sum(bool(r["trace"].get("plan", {}).get("cache_hit")) for r in rewrite_rows),
    }
    answer_pairs = [(b, c) for b, c in pairs if b.get("answer_scores") and c.get("answer_scores")]
    answer_differences = np.array(
        [c["answer_scores"]["em"] - b["answer_scores"]["em"] for b, c in answer_pairs]
    )
    answer_ci = None
    if len(answer_pairs) and complete:
        rng = np.random.default_rng(settings["seed"])
        answer_ci = np.quantile(
            rng.choice(answer_differences, (10000, len(answer_pairs))).mean(axis=1), [0.025, 0.975]
        ).tolist()
    payload["paired_answer_em"] = {
        "wins": int(np.sum(answer_differences > 0)),
        "losses": int(np.sum(answer_differences < 0)),
        "delta_95pct_bootstrap": answer_ci,
    }
    atomic(out / "summary.json", payload)
    lines = [
        "# Three-architecture diagnostic",
        "",
        f"Status: {payload['status']}; {len(records)}/{settings['n'] * 3} architecture/question records.",
        f"Question sample: {settings['n']} randomly selected validation questions from [{settings['offset']}:{settings['offset'] + settings['range_size']}], seed {settings['seed']}.",
        f"The sampled questions come from validation rows [{settings['offset']}:{settings['offset'] + settings['range_size']}]. They remain part of HotPotQA validation, so this is a fresh validation-window comparison, not an independent external test. No model training occurs.",
        f"Corpus: {settings['corpus_size']} pooled paragraphs; selected reranker: `{settings['reranker']}`; Gemini: `{settings['gemini_model']}`.",
        "Both-gold@k means both supporting paragraphs occur in the final top k. Candidate coverage means both gold paragraphs appeared anywhere in the expanded candidate pool before final ranking. This run is retrieval-only; it does not measure generated answer accuracy.",
        "",
        "| Architecture | n | First-hop both-gold | Candidate both-gold | @2 | @5 | @10 | @20 | Mean / p90 retrieval seconds |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, m in metrics.items():
        if m["n"]:
            values = [m["both_gold_in_first_hop"], m["both_gold_in_expanded_candidates"]] + [
                m["both_gold"][str(k)] for k in (2, 5, 10, 20)
            ]
            lines.append(
                f"| {name} | {m['n']} | "
                + " | ".join(f"{v:.1%}" if v is not None else "n/a" for v in values)
                + f" | {m['mean_retrieval_seconds']:.3f} / {m['p90_retrieval_seconds']:.3f} |"
            )
    lines += [
        "",
        f"Query-rewrite statuses: {statuses}.",
        f"Paired rewrite versus hybrid at @5: {payload['paired_at_5']}.",
    ]
    lines += [
        "",
        "## Paired differences",
        "",
        "Positive delta means the first architecture in the comparison retrieved both gold paragraphs more often.",
        "",
        "| Comparison | Cutoff | Wins | Losses | Ties | Delta | 95% bootstrap interval |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for comparison, details in paired_comparisons.items():
        for k, item in details["both_gold"].items():
            ci = item["delta_95pct_bootstrap"]
            delta_text = f"{item['delta']:+.1%}" if item["delta"] is not None else "n/a"
            ci_text = f"{ci[0]:+.1%} to {ci[1]:+.1%}" if ci is not None else "pending"
            lines.append(
                f"| {comparison.replace('_vs_', ' vs ')} | @{k} | {item['wins']} | {item['losses']} | {item['ties']} | {delta_text} | {ci_text} |"
            )
    lines += [
        "This is one fresh validation window, not proof of general improvement. Model calls can vary even at temperature zero. API failures are explicitly recorded as hybrid fallbacks.",
        "Retrieval latency includes query-rewrite API time for architecture 3; answer generation is not measured. Execution order is rotated; startup is excluded. This is not a rigorous load benchmark.",
        "Timing includes Gemini rewrite-call time and free-tier request pacing (5.5 seconds between fresh request starts), as well as service retries. No answer-generation calls are included.",
        "Architecture 3 changes both query formulation and evidence-pair assembly. This comparison evaluates that complete architecture; it does not isolate the causal contribution of query rewriting alone.",
        "Token usage is in summary.json; monetary billing is not estimated without a verified rate.",
        "",
        "## Question-level evidence",
        "",
    ]
    for qid in settings["ids"]:
        entries = [by_id[(qid, n)] for n in NAMES if (qid, n) in by_id]
        if not entries:
            continue
        if all(both(r, 5) for r in entries):
            continue
        row = entries[0]
        lines += [f"### {qid}: {row['question']}", f"Gold answer: {row['gold_answer']}", ""]
        for gold in row["gold"]:
            lines += [f"**Gold paragraph: {gold['title']}**", gold["text"], ""]
        for r in entries:
            lines += [
                f"**{r['architecture']}** — both-gold@5 {both(r, 5)}"
                + (f"; answer: {r['answer']['text']}" if r.get("answer") else "")
            ]
            for query in r["trace"].get("plan", {}).get("queries", []):
                lines.append(f"Follow-up query: {query}")
            lines += [
                f"{i}. **{h['title']}** — {h['text']}" for i, h in enumerate(r["hits"][:10], 1)
            ]
            lines.append("")
    if write_report or complete:
        (out / "REPORT.md.tmp").write_text("\n".join(lines))
        os.replace(out / "REPORT.md.tmp", out / "REPORT.md")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n", type=int, default=50)
    p.add_argument("--offset", type=int, default=3200)
    p.add_argument("--range-size", type=int, default=1800)
    p.add_argument("--seed", type=int, default=20261022)
    p.add_argument("--index", type=Path, default=Path("artifacts/retrieval-audit-2026-10-21"))
    p.add_argument("--out", type=Path, default=Path("artifacts/three-architectures-50-20261022"))
    p.add_argument("--gemini-model", default=MODEL)
    p.add_argument(
        "--dataset-limit",
        type=int,
        default=None,
        help="read validation rows up to this absolute row count instead of the old 5,000-row cache",
    )
    p.add_argument(
        "--no-answers",
        action="store_true",
        help="evaluate retrieval only; do not call Gemini for answer generation",
    )
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.dataset_limit:
        rows = list(_iter_rows("validation", args.dataset_limit))[
            args.offset : args.offset + args.range_size
        ]
        examples = [
            Example(
                id=row["id"],
                question=row["question"],
                answer=row["answer"],
                gold_titles=tuple(dict.fromkeys(row["supporting_facts"]["title"])),
            )
            for row in rows
        ]
        if len(examples) != args.range_size:
            raise SystemExit("dataset cache did not include the requested range")
    else:
        rows = list(_iter_rows("validation", args.offset + args.range_size))[
            args.offset : args.offset + args.range_size
        ]
        examples = [
            Example(
                id=row["id"],
                question=row["question"],
                answer=row["answer"],
                gold_titles=tuple(dict.fromkeys(row["supporting_facts"]["title"])),
            )
            for row in rows
        ]
    examples = random.Random(args.seed).sample(examples, args.n)
    index = VectorIndex.load(args.index)
    config = EncoderConfig(max_length=512)
    settings = {
        **{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "ids": [e.id for e in examples],
        "corpus_size": len(index),
        "reranker": selected_reranker(),
        "encoder": config.model_name,
        "max_length": 512,
        "candidates": 50,
        "rrf_k": 60,
        "answer_evidence_k": 5,
        "query_rewrite_revision": "retention-v2",
        "rewrite_evidence_depth": 5,
        "baseline_reserved_slots": 3,
        "rewrite_prompt_sha256": hashlib.sha256(PROMPT_V1.encode()).hexdigest(),
        "evaluation": "retrieval only" if args.no_answers else "retrieval plus generated answers",
    }
    if (args.out / "settings.json").exists() and json.loads(
        (args.out / "settings.json").read_text()
    ) != settings:
        raise SystemExit("settings mismatch; use a different output directory")
    atomic(args.out / "settings.json", settings)
    titles = {p.title: p.text for p in index.passages}
    if any(not set(e.gold_titles) <= titles.keys() for e in examples):
        raise SystemExit("corpus lacks gold paragraphs")
    tok, encoder = load_tokenizer(config), load_encoder(config)
    ce = Reranker(settings["reranker"])
    common = dict(reranker=ce, candidates=50, hops=2)
    dense = Retriever(index, encoder, tok, config, use_bm25=False, **common)
    hybrid = Retriever(index, encoder, tok, config, use_bm25=True, rrf_k=60, **common)
    rewriter = GeminiQueryRewriter(model=args.gemini_model, cache_dir=args.out / "rewrite-cache")
    architectures = {
        name: Architecture(
            name,
            dense if name == "dense" else hybrid,
            rewriter if name == "query_rewrite" else None,
        )
        for name in NAMES
    }
    answers = args.out / "answer-cache"
    answers.mkdir(exist_ok=True)
    path = args.out / "records.jsonl"
    records = []
    if path.exists():
        records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    done = {(r["id"], r["architecture"]) for r in records}
    summarize(args.out, settings, records)
    with path.open("a") as f:
        for i, e in enumerate(examples):
            # Rotate order to reduce a systematic first/last timing advantage.
            for name in NAMES[i % 3 :] + NAMES[: i % 3]:
                if (e.id, name) in done:
                    continue
                start = time.perf_counter()
                result = architectures[name].retrieve(e.question, 20)
                elapsed = time.perf_counter() - start
                answer = (
                    None
                    if args.no_answers
                    else generate(rewriter, e.question, result.hits[:5], answers)
                )
                record = {
                    "id": e.id,
                    "question": e.question,
                    "architecture": name,
                    "gold_titles": list(e.gold_titles),
                    "gold_answer": e.answer,
                    "gold": [{"title": t, "text": titles[t]} for t in e.gold_titles],
                    "hits": [
                        {"title": h.passage.title, "text": h.passage.text, "score": h.score}
                        for h in result.hits
                    ],
                    "first_hop": [h.passage.title for h in result.first_hop],
                    "candidate_titles": [h.passage.title for h in result.candidates],
                    "trace": getattr(result, "trace", {}),
                    "retrieval_seconds": elapsed,
                    "answer": answer,
                    "answer_scores": score_answer(answer["text"], e.answer) if answer else None,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                f.flush()
                os.fsync(f.fileno())
                records.append(record)
                done.add((e.id, name))
                if len(records) % 75 == 0:
                    summarize(args.out, settings, records, write_report=(len(records) % 750 == 0))
                print(
                    f"{len(records)}/{args.n * 3}: {name} both_gold@5={both(record, 5)}", flush=True
                )
    summarize(args.out, settings, records, True)
    print(f"Complete: {args.out}/REPORT.md", flush=True)


if __name__ == "__main__":
    main()
