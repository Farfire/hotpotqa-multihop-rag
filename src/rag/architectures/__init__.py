"""Three explicit retrieval architectures over shared, unchanged search primitives."""

from rag.retrieval import RetrievalResult, Retriever

from .dense import retrieve_dense
from .hybrid import retrieve_hybrid
from .query_rewrite import retrieve_query_rewrite
from .rewrite import GeminiQueryRewriter

NAMES = ("dense", "hybrid", "query_rewrite")


class Architecture:
    """Dispatch a named pipeline without changing the underlying search engine."""

    def __init__(
        self, name: str, engine: Retriever, rewriter: GeminiQueryRewriter | None = None
    ) -> None:
        if name not in NAMES:
            raise ValueError(f"unknown architecture: {name}")
        if name == "query_rewrite" and rewriter is None:
            raise ValueError("query_rewrite needs a query rewriter")
        self.name, self.engine, self.rewriter = name, engine, rewriter
        self.index = engine.index
        self.encoder = engine.encoder

    def retrieve(self, question: str, k: int = 5) -> RetrievalResult:
        if self.name == "query_rewrite":
            return retrieve_query_rewrite(self.engine, self.rewriter, question, k)
        function = retrieve_dense if self.name == "dense" else retrieve_hybrid
        return function(self.engine, question, k)
