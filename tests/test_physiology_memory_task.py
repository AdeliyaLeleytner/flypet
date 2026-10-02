import unittest
from collections import Counter, defaultdict

try:
    import torch
except ImportError:  # language-model extras are optional, see requirements-llm.txt
    raise unittest.SkipTest("torch is not installed")
import numpy as np
from flypet.physiology_memory_task import generate, question_text, LOCATIONS
from flypet.physiology_language import PhysiologicalLanguageBridge


class MemoryTaskTests(unittest.TestCase):
    def test_independent_clipping_keeps_decoder_update_despite_large_encoder(self):
        from scripts.train_physiology_language import clip_bridge_gradients

        model = torch.nn.Module()
        model.encoder = torch.nn.Linear(1, 1, bias=False)
        model.decoder = torch.nn.Linear(1, 1, bias=False)
        model.core = torch.nn.Module()
        model.core.raw_gain = torch.nn.Parameter(torch.ones(1))
        model.encoder.weight.grad = torch.full_like(model.encoder.weight, 1e8)
        model.decoder.weight.grad = torch.full_like(model.decoder.weight, 0.1)
        model.core.raw_gain.grad = torch.full_like(model.core.raw_gain, 1e5)
        norms = clip_bridge_gradients(model, True, separate=True)
        self.assertAlmostEqual(float(model.decoder.weight.grad), 0.1, places=6)
        self.assertAlmostEqual(float(model.encoder.weight.grad), 1.0, places=6)
        self.assertAlmostEqual(float(model.core.raw_gain.grad), 1.0, places=6)
        self.assertEqual(norms["encoder_norm"], 1e8)

    def test_balanced_validation_subset_and_no_language_input_leak(self):
        from scripts.train_physiology_language import selection, Pilot

        rows = selection(generate()["validation"], 48)
        self.assertEqual(len(rows), 48)
        self.assertEqual(set(Counter(r["answer"] for r in rows).values()), {12})
        p = Pilot.__new__(Pilot)
        p.embedding = torch.nn.Embedding(12, 8)
        p.questions = {"key": [1, 2, 3]}
        p.targets = {"red": [4, 5], "blue": [6, 5]}
        prefix = torch.randn(1, 2, 8)
        a = {"query_object": "key", "answer": "red", "history": ["private fact red"]}
        b = {**a, "answer": "blue", "history": ["different private fact blue"]}
        x, labels = p.language_inputs(a, prefix, False)
        y, _ = p.language_inputs(b, prefix, False)
        torch.testing.assert_close(x, y, rtol=0, atol=0)
        self.assertTrue((labels == -100).all())
        _, teacher = p.language_inputs(a, prefix, True)
        self.assertTrue((teacher[:, :5] == -100).all())
        self.assertEqual(teacher[0, 5:].tolist(), [4, 5])

    def test_batched_language_labels_padding_and_prefix_gradients(self):
        from scripts.train_physiology_language import Pilot

        p = Pilot.__new__(Pilot)
        p.embedding = torch.nn.Embedding(12, 8).requires_grad_(False)
        p.questions = {"key": [1, 2, 3], "coin": [2, 3]}
        p.targets = {"red": [4, 5], "blue": [6, 5]}
        prefix = torch.randn(2, 2, 8, requires_grad=True)
        rows = [
            {"query_object": "key", "answer": "red"},
            {"query_object": "coin", "answer": "blue"},
        ]
        x, labels, mask = p.batch_language_inputs(rows, prefix)
        self.assertEqual(labels.tolist(), [[-100] * 5 + [4, 5], [-100] * 4 + [6, 5, -100]])
        self.assertEqual(mask.tolist(), [[1] * 7, [1] * 6 + [0]])
        x.sum().backward()
        torch.testing.assert_close(prefix.grad, torch.ones_like(prefix))

    def test_balanced_group_disjoint_counterfactuals(self):
        data = generate()
        seen = set()
        for split, rows in data.items():
            groups = defaultdict(list)
            for r in rows:
                groups[r["group"]].append(r)
                self.assertEqual(dict(r["events"])[r["query_object"]], r["answer"])
                self.assertNotIn("moved", question_text(r))
                self.assertNotIn(r["history"][0], question_text(r))
            self.assertFalse(seen & set(groups))
            seen.update(groups)
            self.assertEqual(len(set(Counter(r["answer"] for r in rows).values())), 1)
            for records in groups.values():
                self.assertEqual(len(records), 4)
                a = next(
                    r for r in records if r["assignment"] == 0 and not r["query_is_last_mentioned"]
                )
                b = next(
                    r for r in records if r["assignment"] == 1 and not r["query_is_last_mentioned"]
                )
                self.assertEqual(sorted(a["history"]), sorted(b["history"]))
                self.assertNotEqual(a["answer"], b["answer"])

    def test_bridge_state_is_only_history_channel(self):
        graph = {
            "order": np.arange(4),
            "pre": np.array([0, 1, 2]),
            "post": np.array([1, 2, 3]),
            "contacts": np.array([100, 100, 100]),
            "input_indices": np.array([0]),
            "output_indices": np.array([1, 2, 3]),
        }
        m = PhysiologicalLanguageBridge(
            graph, 8, 0.02, tokens=2, hidden=12, ticks_per_event=32, checkpoint_steps=13
        )
        x = torch.randn(2, 4, 8)
        p, stats = m(x)
        decoder_inputs = []
        hook = m.decoder.register_forward_pre_hook(
            lambda module, args: decoder_inputs.append(args[0].detach().clone())
        )
        try:
            zero, _ = m(x, intervention="zero_state")
        finally:
            hook.remove()
        self.assertEqual(p.shape, (2, 2, 8))
        self.assertEqual(torch.count_nonzero(decoder_inputs[0]).item(), 0)
        # Batched BLAS kernels can round identical rows differently in float32.
        torch.testing.assert_close(zero[0], zero[1], rtol=1e-6, atol=1e-8)
        p[:, :, 0].sum().backward()
        self.assertIsNotNone(m.core.raw_gain.grad)
        self.assertTrue(torch.isfinite(m.core.raw_gain.grad).all())
        saved = m.trainable_state()
        self.assertNotIn("core.w", saved)
        original = saved["core.raw_gain"].clone()
        with torch.no_grad():
            m.core.raw_gain.add_(1)
        torch.testing.assert_close(saved["core.raw_gain"], original)
        m.load_trainable_state(saved)
        torch.testing.assert_close(m.core.raw_gain, original)


if __name__ == "__main__":
    unittest.main()
