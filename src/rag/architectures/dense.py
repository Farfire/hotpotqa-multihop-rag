"""Architecture 1: original E5-only candidate retrieval and legacy two-hop assembly."""

from rag.retrieval import RetrievalResult, Retriever


def retrieve_dense(engine: Retriever, question: str, k: int = 5) -> RetrievalResult:
    if engine.use_bm25:
        raise ValueError("dense architecture requires use_bm25=False")
    return engine.retrieve(question, k)
