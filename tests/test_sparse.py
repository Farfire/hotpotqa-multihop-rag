import unittest

import numpy as np

from rag.data import Passage
from rag.index import Hit, VectorIndex
from rag.sparse import BM25Index, reciprocal_rank_fusion, tokenize


class SparseRetrievalTests(unittest.TestCase):
    def test_tokenize_is_case_insensitive_and_unicode_aware(self):
        self.assertEqual(tokenize("École, WIKI-Page!"), ["école", "wiki", "page"])

    def test_bm25_finds_exact_rare_terms(self):
        passages = [
            Passage("Common", "A general article about history."),
            Passage("Other", "An article about art and music."),
            Passage("Rare Author", "A biography mentioning Xylophonic."),
        ]
        results = BM25Index(VectorIndex.build(np.eye(3, dtype=np.float32), passages)).search(
            "Xylophonic", 1
        )
        self.assertEqual(results[0].passage.title, "Rare Author")

    def test_inverted_bm25_matches_rank_bm25_full_scan(self):
        passages = [
            Passage("A", "alpha beta beta common"),
            Passage("B", "beta gamma common common common"),
            Passage("C", "delta alpha common"),
            Passage("D", "no query words here"),
        ]
        bm25 = BM25Index(VectorIndex.build(np.eye(4, dtype=np.float32), passages))
        for query in ("alpha beta", "beta beta common", "unknown", "gamma alpha"):
            expected = bm25._bm25.get_scores(tokenize(query))
            actual = np.zeros(len(passages), dtype=np.float64)
            for term in tokenize(query):
                posting = bm25._postings.get(term)
                if posting is None:
                    continue
                ids, frequencies = posting
                norm = bm25._bm25.k1 * (
                    1 - bm25._bm25.b + bm25._bm25.b * bm25._doc_lengths[ids] / bm25._bm25.avgdl
                )
                actual[ids] += bm25._bm25.idf.get(term, 0.0) * (
                    frequencies * (bm25._bm25.k1 + 1) / (frequencies + norm)
                )
            np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-7)
            hits = bm25.search(query, len(passages))
            self.assertEqual(
                [x.passage.title for x in hits],
                [passages[i].title for i in np.argsort(-expected, kind="stable")],
            )

    def test_rrf_rewards_a_result_found_high_in_both_lists(self):
        a, b, c = [
            Hit(Passage(title, ""), score) for title, score in (("A", 100), ("B", 0.01), ("C", 1))
        ]
        fused = reciprocal_rank_fusion([a, b], [b, c], k=60, limit=2)
        self.assertEqual([hit.passage.title for hit in fused], ["B", "A"])
        self.assertGreater(fused[0].score, fused[1].score)

    def test_rrf_deduplicates_each_list_and_validates_parameters(self):
        a = Hit(Passage("A", ""), 0)
        fused = reciprocal_rank_fusion([a, a], k=60)
        self.assertEqual(len(fused), 1)
        with self.assertRaises(ValueError):
            reciprocal_rank_fusion([a], k=0)


if __name__ == "__main__":
    unittest.main()
