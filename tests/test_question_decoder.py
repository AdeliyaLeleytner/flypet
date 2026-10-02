import io
import unittest

try:
    import torch
except ImportError:  # language-model extras are optional, see requirements-llm.txt
    raise unittest.SkipTest("torch is not installed")
from torch import nn

from flypet.question_decoder import CachedStateQuestionDecoder, QUERY_OBJECTS


class QuestionDecoderTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(41)
        self.plain = CachedStateQuestionDecoder(6, 8, 0.02, tokens=2, hidden=12)
        self.conditioned = CachedStateQuestionDecoder(
            6, 8, 0.02, tokens=2, hidden=12, conditioned=True
        )
        self.conditioned.load_state_dict(self.plain.state_dict())
        self.features = torch.randn(6, 6)
        self.queries = torch.tensor([0, 1, 2, 0, 1, 2])

    def test_exact_initial_equivalence_and_matching_parameters(self):
        self.assertEqual(QUERY_OBJECTS, ("key", "coin", "ring"))
        a = self.plain(self.features, self.queries)
        b = self.conditioned(self.features, self.queries)
        self.assertEqual(a.shape, (6, 2, 8))
        torch.testing.assert_close(a, b, rtol=0, atol=0)
        self.assertEqual(
            sum(p.numel() for p in self.plain.parameters()),
            sum(p.numel() for p in self.conditioned.parameters()),
        )
        self.assertTrue(self.conditioned.conditioned)
        for i in range(len(self.features)):
            torch.testing.assert_close(
                a[i : i + 1],
                self.plain(self.features[i : i + 1], self.queries[i : i + 1]),
                rtol=1e-5,
                atol=1e-8,
            )

    def test_only_conditioned_arm_can_use_question(self):
        with torch.no_grad():
            self.conditioned.decoder[0].weight[:, 6] = torch.arange(12) / 2
            self.plain.load_state_dict(self.conditioned.state_dict())
        x = self.features[:1].expand(3, -1)
        q = torch.arange(3)
        a = self.plain(x, q)
        b = self.conditioned(x, q)
        torch.testing.assert_close(a, self.plain(x, q.flip(0)), rtol=0, atol=0)
        self.assertGreater(float((b[0] - b[1]).detach().abs().max()), 1e-6)

    def test_gradients_reach_conditioning_columns_only_in_conditioned_arm(self):
        for model in (self.plain, self.conditioned):
            model(self.features, self.queries)[:, :, 0].sum().backward()
            grad = model.decoder[0].weight.grad
            self.assertTrue(torch.isfinite(grad).all())
            self.assertGreater(float(grad[:, :6].abs().sum()), 0.0)
            if model.conditioned:
                self.assertTrue((grad[:, 6:].abs().sum(dim=0) > 0).all())
            else:
                self.assertEqual(float(grad[:, 6:].abs().sum()), 0.0)

    def test_old_bridge_initialization_and_validation_are_atomic(self):
        old = nn.Module()
        old.decoder = nn.Sequential(nn.Linear(6, 12), nn.SiLU(), nn.Linear(12, 16))
        old.output_norm = nn.LayerNorm(8)
        state = old.state_dict()
        self.conditioned.initialize_from_bridge_state(state)
        expected = old.output_norm(old.decoder(self.features).reshape(6, 2, 8)) * 0.02
        torch.testing.assert_close(
            self.conditioned(self.features, self.queries), expected, rtol=1e-6, atol=1e-8
        )
        snapshot = self.conditioned.make_checkpoint()
        bad = {**state, "decoder.2.bias": torch.zeros(2)}
        with self.assertRaises(ValueError):
            self.conditioned.initialize_from_bridge_state(bad)
        for k, value in snapshot["state_dict"].items():
            torch.testing.assert_close(self.conditioned.state_dict()[k], value, rtol=0, atol=0)

    def test_checkpoint_roundtrip_and_no_aliasing(self):
        model = CachedStateQuestionDecoder(
            6,
            8,
            0.02,
            tokens=2,
            hidden=12,
            conditioned=True,
            feature_mean=torch.arange(6.0),
            feature_std=torch.ones(6) * 2,
        )
        expected = model(self.features, self.queries)
        snapshot = model.make_checkpoint()
        with torch.no_grad():
            model.decoder[0].weight.add_(3.0)
        stream = io.BytesIO()
        torch.save(snapshot, stream)
        stream.seek(0)
        restored = CachedStateQuestionDecoder.from_checkpoint(torch.load(stream, weights_only=True))
        self.assertTrue(restored.conditioned)
        self.assertTrue(restored.normalized)
        torch.testing.assert_close(restored(self.features, self.queries), expected, rtol=0, atol=0)

    def test_normalizer_is_shared_and_explicit(self):
        mean, std = torch.randn(6), torch.rand(6) + 0.1
        model = CachedStateQuestionDecoder(
            6, 8, 0.02, tokens=2, hidden=12, feature_mean=mean, feature_std=std
        )
        reference = self.plain
        with torch.no_grad():
            reference.decoder.load_state_dict(model.decoder.state_dict())
            reference.output_norm.load_state_dict(model.output_norm.state_dict())
        torch.testing.assert_close(
            model(self.features, self.queries),
            reference(((self.features - mean) / std).clamp(-10, 10), self.queries),
            rtol=0,
            atol=0,
        )
        mean.add_(99.0)
        self.assertFalse(torch.equal(model.feature_mean, mean))

    def test_reject_bad_inputs(self):
        cases = [
            (self.features[0], [0]),
            (self.features[:, :5], self.queries),
            (self.features[:0], self.queries[:0]),
            (self.features, self.queries.float()),
            (self.features, torch.ones(6, 1)),
            (self.features, [-1] * 6),
            (self.features, [3] * 6),
            (self.features.long(), self.queries),
            (self.features * float("nan"), self.queries),
        ]
        for x, q in cases:
            with self.subTest(shape=x.shape), self.assertRaises(ValueError):
                self.plain(x, q)
        with self.assertRaises(ValueError):
            CachedStateQuestionDecoder(6, 8, 0.02, feature_mean=torch.zeros(6))
        with self.assertRaises(ValueError):
            CachedStateQuestionDecoder(
                6, 8, 0.02, feature_mean=torch.zeros(6), feature_std=-torch.ones(6)
            )

    def test_normalizer_floor_clipping_and_mean_state(self):
        model = CachedStateQuestionDecoder(
            6, 8, 0.02, tokens=2, hidden=12, feature_mean=torch.ones(6), feature_std=torch.zeros(6)
        )
        torch.testing.assert_close(model.feature_std, torch.full((6,), 0.01))
        reference = CachedStateQuestionDecoder(6, 8, 0.02, tokens=2, hidden=12)
        reference.decoder.load_state_dict(model.decoder.state_dict())
        reference.output_norm.load_state_dict(model.output_norm.state_dict())
        q = torch.tensor([0])
        torch.testing.assert_close(
            model(torch.ones(1, 6), q), reference(torch.zeros(1, 6), q), rtol=0, atol=0
        )
        torch.testing.assert_close(
            model(torch.full((1, 6), 100.0), q),
            reference(torch.full((1, 6), 10.0), q),
            rtol=0,
            atol=0,
        )


if __name__ == "__main__":
    unittest.main()
