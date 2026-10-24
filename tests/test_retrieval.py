import unittest
from unittest.mock import patch

import numpy as np
import torch

from rag.config import EncoderConfig
from rag.data import Passage, load_training_pairs
from rag.dataset import make_pair_collate_fn
from rag.index import Hit, VectorIndex, encoder_fingerprint
from rag.rerank import Reranker
from rag.reranker_training import labeled_pairs, split_questions
from rag.retrieval import Retriever
from rag.training import MultiPositiveLoss, split_pairs, train_one_epoch, validate


def row(i="q"):
    return {
        "id": i,
        "question": f"Who made work {i}?",
        "supporting_facts": {"title": ["A", "B", "A"]},
        "context": {
            "title": ["A", "B", "C"],
            "sentences": [["Created by B."], ["Born in 1962."], ["Distractor."]],
        },
    }


class RetrievalTests(unittest.TestCase):
    def test_encoder_training_updates_and_validation_does_not(self):
        class TinyEncoder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(3, 3, bias=False)

            def forward(self, input_ids, attention_mask):
                return torch.nn.functional.normalize(self.linear(input_ids), dim=1)

        torch.manual_seed(42)
        encoder = TinyEncoder()
        questions = {
            "input_ids": torch.eye(3),
            "attention_mask": torch.ones(3, 3),
            "positive_mask": torch.tensor(
                [[True, True, False], [True, True, False], [False, False, True]]
            ),
        }
        documents = {"input_ids": torch.eye(3), "attention_mask": torch.ones(3, 3)}
        loader = [(questions, documents)]
        before = encoder.linear.weight.detach().clone()
        result = train_one_epoch(
            encoder,
            loader,
            MultiPositiveLoss(),
            torch.optim.AdamW(encoder.parameters(), lr=0.01),
            device=torch.device("cpu"),
            temperature=0.05,
            max_grad_norm=1.0,
            log_every=0,
        )
        self.assertTrue(np.isfinite(result.loss))
        self.assertFalse(torch.equal(before, encoder.linear.weight))
        before = encoder.linear.weight.detach().clone()
        validate(encoder, loader, MultiPositiveLoss(), device=torch.device("cpu"), temperature=0.05)
        self.assertTrue(torch.equal(before, encoder.linear.weight))

    def test_e5_prefix_and_fingerprint(self):
        config = EncoderConfig()
        self.assertEqual(config.prepare(["x"], "query"), ["query: x"])
        self.assertEqual(config.prepare(["x"], "passage"), ["passage: x"])
        self.assertNotEqual(
            encoder_fingerprint(config.model_name, None, 256),
            encoder_fingerprint(config.model_name, None, 512),
        )
        legacy = EncoderConfig(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.assertEqual(legacy.prepare(["x"], "query"), ["x"])

    def test_both_golds_and_no_false_negative(self):
        pairs = labeled_pairs([row()], negatives=1)
        original = [p for p in pairs if "Known context:" not in p.query]
        self.assertEqual(sum(p.label for p in original), 2)
        self.assertEqual(len(original), 3)
        for p in pairs:
            if p.document.startswith(("A.", "B.")):
                self.assertEqual(p.label, 1)
        contextual = [p for p in pairs if "Known context: A." in p.query]
        self.assertTrue(any(p.document.startswith("B.") and p.label == 1 for p in contextual))
        self.assertFalse(any(p.document.startswith("A.") for p in contextual))
        with patch("rag.data._iter_rows", return_value=iter([row()])):
            self.assertEqual(len(load_training_pairs()), 2)

    def test_question_split_no_leakage(self):
        train, val = split_questions([row(str(i)) for i in range(20)], 0.1, 42)
        self.assertFalse({r["id"] for r in train} & {r["id"] for r in val})
        pairs = [(f"q{i}", d) for i in range(10) for d in ("first", "second")]
        train, val = split_pairs(pairs, 0.1)
        self.assertFalse({q for q, _ in train} & {q for q, _ in val})
        self.assertEqual(len(val), 2)

    def test_positive_mask_includes_companion_and_shared_documents(self):
        def tokenizer(texts, **kwargs):
            return {
                "input_ids": torch.ones(len(texts), 1),
                "attention_mask": torch.ones(len(texts), 1),
            }

        collate = make_pair_collate_fn(tokenizer, 32, {"q": {"a", "b"}, "q2": {"b"}})
        q, _ = collate([("q", "a"), ("q", "b"), ("q2", "b")])
        self.assertEqual(
            q["positive_mask"].tolist(),
            [[True, True, True], [True, True, True], [False, True, True]],
        )
        scores = torch.tensor([[3.0, -2.0, 0.0]], requires_grad=True)
        MultiPositiveLoss()(scores, torch.tensor([[True, True, False]])).backward()
        self.assertLess(scores.grad[0, 1].item(), 0)  # raise the under-ranked second gold
        self.assertGreater(scores.grad[0, 2].item(), 0)  # lower the actual negative

    def test_rerank_sorts_logits_and_deduplicates(self):
        reranker = object.__new__(Reranker)
        reranker.score = lambda q, docs: [-2.0, 4.0]
        a, b = Hit(Passage("a", "one"), 0.9), Hit(Passage("b", "two"), 0.1)
        self.assertEqual([h.passage.title for h in reranker.rerank("q", [a, b, a])], ["b", "a"])
        self.assertEqual(reranker.rerank("q", [], 2), [])

    def test_second_hop_recovers_missing_title_without_gold_labels(self):
        bridge = Passage("Book", "A story by Example Author.")
        decoy = Passage("Decoy", "Irrelevant")
        author = Passage("Example Author", "Born in 1962.")
        config = EncoderConfig(model_name="test", max_length=256)
        index = VectorIndex.build(
            np.eye(3, dtype=np.float32), [bridge, decoy, author], encoder_fingerprint("test", None)
        )

        class FakeReranker:
            def rerank(self, question, hits):
                target = "Example Author" if "Known context:" in question else "Book"
                return sorted(hits, key=lambda h: h.passage.title != target)

        retriever = Retriever(
            index, None, None, config, reranker=FakeReranker(), candidates=3, bridge_count=1
        )
        retriever.dense = lambda q, k: [Hit(bridge, 0.9), Hit(decoy, 0.8)]
        result = retriever.retrieve("When was the writer born?", 2)
        self.assertEqual([h.passage.title for h in result.hits], ["Book", "Example Author"])
        self.assertEqual(len(result.candidates), 3)
        with self.assertRaises(ValueError):
            retriever.retrieve("q", 0)

    def test_indexed_title_mentions_preserve_exact_boundaries_and_corpus_order(self):
        passages = [
            Passage("Arthur's Magazine", "Periodical."),
            Passage("The Other", "Novel."),
            Passage("Other", "A word."),
            Passage("A-B Journal", "Journal."),
        ]
        config = EncoderConfig(model_name="test", max_length=256)
        index = VectorIndex.build(
            np.eye(4, dtype=np.float32), passages, encoder_fingerprint("test", None)
        )
        retriever = Retriever(index, None, None, config, use_bm25=False)
        found = retriever._mentioned_titles(
            "The Other; Arthur's Magazine; A-B Journal; Otherish", "skip"
        )
        self.assertEqual(
            [hit.passage.title for hit in found],
            ["Arthur's Magazine", "The Other", "Other", "A-B Journal"],
        )


if __name__ == "__main__":
    unittest.main()
