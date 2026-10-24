#!/usr/bin/env python3
"""Retrieve passages for a question, and optionally have Gemini answer from them.

.venv/bin/python scripts/ask.py "Which magazine started first: Arthur's Magazine or First for Women?"
.venv/bin/python scripts/ask.py --example 0          # use a dataset question, show the gold answer
.venv/bin/python scripts/ask.py --example 3 --no-answer -k 3
"""

from __future__ import annotations

import argparse
import textwrap
from pathlib import Path

from rag.cli import add_retrieval_args, load_retriever

from rag.config import INDEX_DIR
from rag.data import Example, load_examples
from rag.generate import GenerationUnavailable, generate_answer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question", nargs="?", help="your question")
    parser.add_argument(
        "--example",
        type=int,
        default=None,
        help="use question N from the dataset instead, and print its gold answer",
    )
    parser.add_argument("-k", type=int, default=5, help="passages to retrieve")
    parser.add_argument("--index", type=Path, default=INDEX_DIR)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--answer-model", default=None, help="optional Gemini model override")
    add_retrieval_args(parser)
    parser.add_argument("--no-answer", action="store_true", help="skip the Gemini generation step")
    return parser.parse_args()


def resolve_question(args: argparse.Namespace) -> tuple[str, Example | None]:
    if args.question:
        return args.question, None
    index = 0 if args.example is None else args.example
    if index < 0:
        raise SystemExit("example index must be non-negative")
    examples = load_examples(split="validation", limit=index + 1)
    if index >= len(examples):
        raise SystemExit(f"No example at index {index}")
    example = examples[index]
    return example.question, example


def main() -> None:
    args = parse_args()

    question, example = resolve_question(args)
    print(f"\nQ: {question}", flush=True)

    try:
        retriever = load_retriever(args)
    except (FileNotFoundError, ValueError, GenerationUnavailable) as exc:
        raise SystemExit(f"error: {exc}")

    try:
        result = retriever.retrieve(question, k=args.k)
    except ValueError as exc:
        raise SystemExit(f"error: {exc}")
    hits = result.hits
    print(f"   Architecture: {retriever.name}")
    if hasattr(result, "trace"):
        print(f"   Query reformulation: {result.trace['status']}")
        for query in result.trace.get("plan", {}).get("queries", []):
            print(f"   Follow-up: {query['query']}")

    gold_titles = set(example.gold_titles) if example else set()

    print(
        f"   (searched {len(retriever.index)} paragraphs; {len(result.candidates)} candidates; "
        f"{args.hops} hop(s); reranking {'off' if args.no_rerank else 'on'})\n"
    )

    for number, hit in enumerate(hits, 1):
        marker = "  GOLD" if hit.passage.title in gold_titles else ""
        print(f"[{number}] {hit.passage.title}{marker}")
        body = textwrap.shorten(hit.passage.text, width=280, placeholder=" ...")
        print(textwrap.indent(textwrap.fill(body, width=76), "    "))
        print()

    if not args.no_answer:
        try:
            answer = generate_answer(
                question, hits, **({"model": args.answer_model} if args.answer_model else {})
            )
            print(f"Answer: {answer}")
        except GenerationUnavailable as exc:
            print(f"(no generated answer — {exc})")

    if example:
        print(f"Gold answer: {example.answer}")
        missing = gold_titles - {h.passage.title for h in hits}
        print(f"Missing gold passages: {', '.join(sorted(missing)) if missing else 'none'}")


if __name__ == "__main__":
    main()
