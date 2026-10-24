import unittest

from rag.answer_metrics import score_answer


class AnswerMetricTests(unittest.TestCase):
    def test_citations_and_normalization(self):
        self.assertEqual(
            score_answer("The United States. [1][2]", "United States"), {"em": 1.0, "f1": 1.0}
        )

    def test_partial_overlap_with_repeated_tokens(self):
        self.assertEqual(score_answer("John John Smith", "John Smith"), {"em": 0.0, "f1": 0.8})

    def test_boolean_answers_do_not_get_partial_credit(self):
        self.assertEqual(score_answer("yes", "no"), {"em": 0.0, "f1": 0.0})
        self.assertEqual(score_answer("yes indeed", "yes"), {"em": 0.0, "f1": 0.0})
