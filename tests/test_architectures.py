import unittest
from unittest.mock import Mock, patch

from rag.architectures import retrieve_dense, retrieve_hybrid, retrieve_query_rewrite
from rag.architectures.query_rewrite import merge_evidence
from rag.architectures.rewrite import GeminiQueryRewriter, validate_plan
from rag.data import Passage
from rag.index import Hit
from rag.retrieval import RetrievalResult


def hit(title, text="", score=0.0):
    return Hit(Passage(title, text), score)


class ArchitectureTests(unittest.TestCase):
    def test_merge_keeps_baseline_evidence_and_unique_followup(self):
        base = [hit(str(i), score=10 - i) for i in range(10)]
        out = merge_evidence(base, [hit("new"), hit("1")], [], 5)
        self.assertEqual([h.passage.title for h in out], ["0", "1", "2", "new", "3"])

    def test_v2_retains_legacy_expanded_pool(self):
        source = hit("Source")
        target = hit("Legacy target")
        extra = hit("New target")
        base = RetrievalResult([source, target], [source, target], [source], [], [], [source], [])
        engine = Mock(use_bm25=True, candidate_count=50)
        engine.retrieve.return_value = base
        engine.candidates.return_value = ([extra], [], [])
        engine.reranker.rerank.side_effect = [[extra], [extra]]
        rw = Mock()
        rw.rewrite.return_value = {"queries": [{"query": "followup", "source_id": 1}]}
        result = retrieve_query_rewrite(engine, rw, "q", 5)
        self.assertEqual(
            {h.passage.title for h in result.candidates}, {"Source", "Legacy target", "New target"}
        )
        self.assertEqual(
            [h.passage.title for h in result.hits], ["Source", "Legacy target", "New target"]
        )
        engine.retrieve.assert_called_once_with("q", 50)

    def test_api_quota_retry_is_paced_and_uses_server_delay(self):
        class QuotaError(Exception):
            code = 429

        rewriter = GeminiQueryRewriter.__new__(GeminiQueryRewriter)
        rewriter.model = "test"
        rewriter.client = Mock()
        rewriter._last_request = 0.0
        response = object()
        rewriter.client.models.generate_content.side_effect = [
            QuotaError("Please retry in 20s"),
            response,
        ]
        with (
            patch("rag.architectures.rewrite.time.monotonic", return_value=1000.0),
            patch("rag.architectures.rewrite.time.sleep") as sleep,
        ):
            self.assertIs(rewriter.request(contents="q", config=None), response)
        self.assertEqual(rewriter.client.models.generate_content.call_count, 2)
        self.assertIn(unittest.mock.call(21.0), sleep.call_args_list)
        self.assertIn(unittest.mock.call(5.5), sleep.call_args_list)

    def test_existing_paths_are_unchanged_delegates(self):
        engine = Mock(use_bm25=False)
        self.assertIs(retrieve_dense(engine, "question", 20), engine.retrieve.return_value)
        engine.retrieve.assert_called_once_with("question", 20)
        engine.use_bm25 = True
        self.assertIs(retrieve_hybrid(engine, "question", 5), engine.retrieve.return_value)

    def test_rewrite_requires_grounded_source_and_exact_quote(self):
        evidence = [hit("Harry Potter", "Written by J. K. Rowling.")]
        valid = {
            "query": "When was J. K. Rowling born?",
            "entity": "J. K. Rowling",
            "source_id": 1,
            "evidence_quote": "Written by J. K. Rowling.",
        }
        accepted, rejected = validate_plan(
            {"queries": [valid, dict(valid, source_id=9)]}, "Who wrote Harry Potter?", evidence
        )
        self.assertEqual(accepted, [valid])
        self.assertEqual(len(rejected), 1)
        accepted, rejected = validate_plan(
            {"queries": [dict(valid, entity="invented person")]}, "q", evidence
        )
        self.assertFalse(accepted)
        self.assertEqual(len(rejected), 1)

    def test_failed_rewrite_is_explicit_baseline_fallback(self):
        engine = Mock(use_bm25=True, candidate_count=50)
        engine.candidates.return_value = ([], [], [])
        engine.reranker.rerank.return_value = []
        baseline = RetrievalResult([], [], [], [], [], [], [])
        engine.retrieve.return_value = baseline
        rw = Mock()
        rw.rewrite.side_effect = RuntimeError("offline")
        result = retrieve_query_rewrite(engine, rw, "q", 5)
        self.assertEqual(result.trace["status"], "rewrite_error")
        self.assertEqual(result.trace["error_type"], "RuntimeError")


if __name__ == "__main__":
    unittest.main()
