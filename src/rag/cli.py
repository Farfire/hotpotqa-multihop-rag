"""Consistent retrieval options for asking and evaluation."""

from pathlib import Path

from .architectures import NAMES, Architecture, GeminiQueryRewriter
from .config import selected_reranker
from .encoder import load_encoder, load_tokenizer
from .generate import MODEL
from .index import VectorIndex, encoder_fingerprint
from .rerank import Reranker
from .retrieval import Retriever, index_config


def add_retrieval_args(parser):
    parser.add_argument(
        "--architecture",
        choices=NAMES,
        default=None,
        help="explicit pipeline; default hybrid (or dense with --no-bm25)",
    )
    parser.add_argument("--rewrite-model", default=MODEL)
    parser.add_argument("--rewrite-cache", type=Path, default=Path("artifacts/query-rewrite-cache"))
    parser.add_argument("--model", default=None, help="encoder (inferred from index by default)")
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--candidates", type=int, default=50)
    parser.add_argument("--hops", type=int, choices=(1, 2), default=2)
    parser.add_argument("--no-rerank", action="store_true")
    parser.add_argument(
        "--no-bm25", action="store_true", help="disable BM25 + reciprocal-rank fusion"
    )
    parser.add_argument(
        "--rrf-k", type=int, default=60, help="rank-fusion smoothing constant (default: 60)"
    )
    parser.add_argument(
        "--reranker",
        default=None,
        help="Hub model or trained local directory; default is selected checkpoint or pretrained",
    )


def load_retriever(args):
    name = args.architecture or ("dense" if args.no_bm25 else "hybrid")
    if args.no_bm25 and name != "dense":
        raise ValueError("--no-bm25 conflicts with the selected architecture")
    if name == "query_rewrite" and (args.no_rerank or args.hops != 2):
        raise ValueError("query_rewrite requires reranking and --hops 2")
    if args.reranker is None and not args.no_rerank:
        args.reranker = selected_reranker()
    index = VectorIndex.load(args.index)
    config = index_config(index, model_name=args.model, max_length=args.max_length)
    index.check_fingerprint(
        encoder_fingerprint(config.model_name, args.checkpoint, config.max_length)
    )
    encoder = load_encoder(config, checkpoint=args.checkpoint)
    engine = Retriever(
        index,
        encoder,
        load_tokenizer(config),
        config,
        reranker=None if args.no_rerank else Reranker(args.reranker),
        candidates=args.candidates,
        hops=args.hops,
        checkpoint=args.checkpoint,
        use_bm25=name != "dense",
        rrf_k=args.rrf_k,
    )
    rewriter = (
        GeminiQueryRewriter(model=args.rewrite_model, cache_dir=args.rewrite_cache)
        if name == "query_rewrite"
        else None
    )
    return Architecture(name, engine, rewriter)
