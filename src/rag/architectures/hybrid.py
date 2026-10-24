"""Architecture 2: original E5+BM25/RRF retrieval and legacy two-hop assembly."""

from rag.retrieval import RetrievalResult, Retriever


def retrieve_hybrid(engine: Retriever, question: str, k: int = 5) -> RetrievalResult:
    if not engine.use_bm25:
        raise ValueError("hybrid architecture requires use_bm25=True")
    return engine.retrieve(question, k)
