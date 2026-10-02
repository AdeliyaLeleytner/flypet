import unittest
import numpy as np

try:
    import torch
except ModuleNotFoundError:
    raise unittest.SkipTest("Graph-reader checks require optional PyTorch dependencies")
from flypet.graph_reader import AttentionProjector, make_projector, relation_arrays


class GraphReaderTests(unittest.TestCase):
    def fixture(self):
        return {
            "metadata": np.array([[0, 0], [0, 1], [1, 0]]),
            "metadata_sizes": np.array([2, 2]),
            "src": np.array([0, 2, 1]),
            "dst": np.array([1, 1, 2]),
            "weight": np.array([2.0, 1.0, -3.0]),
        }

    def test_directed_sign_relations_and_normalization(self):
        g = self.fixture()
        result = relation_arrays(g["src"], g["dst"], g["weight"], 3)
        dense = [torch.sparse_coo_tensor(i, v, (3, 3)).to_dense().numpy() for i, v in result]
        np.testing.assert_allclose(dense[0][1], [2 / 3, 0, 1 / 3])
        np.testing.assert_allclose(dense[1][0], [0, 1, 0])
        self.assertEqual(dense[2][2, 1], 1)
        self.assertEqual(dense[3][1, 2], 1)
        for matrix in dense:
            self.assertTrue(np.isin(np.round(matrix.sum(1), 6), [0, 1]).all())

    def test_common_initialization_and_real_graph_gradient(self):
        g = self.fixture()
        torch.manual_seed(31)
        plain = make_projector("attention", 3, 4, 16, g, width=16)
        torch.manual_seed(31)
        graph = make_projector("graph", 3, 4, 16, g, width=16)
        for key, value in plain.state_dict().items():
            torch.testing.assert_close(value, graph.state_dict()[key], atol=0, rtol=0)
        x = torch.tensor([[0.0, 1.0, 2.0], [2.0, 0.0, 0.0]], requires_grad=True)
        y = graph(x)
        self.assertEqual(tuple(y.shape), (2, 4, 16))
        self.assertFalse(torch.allclose(y, plain(x)))
        (y * torch.arange(16)).sum().backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertGreater(float(x.grad.abs().sum()), 0)
        self.assertGreater(float(graph.graph_layers[0].relations[0].weight.grad.abs().sum()), 0)

    def test_single_and_batched_predictions_match(self):
        g = self.fixture()
        for mode in ("mlp", "attention", "graph"):
            torch.manual_seed(8)
            model = make_projector(mode, 3, 4, 16, g, hidden=16, width=16).eval()
            x = torch.randn(3, 3)
            torch.testing.assert_close(
                model(x), torch.cat([model(v[None]) for v in x]), atol=2e-6, rtol=2e-5
            )

    def test_no_graph_layers_is_exact_attention(self):
        g = self.fixture()
        torch.manual_seed(9)
        a = AttentionProjector(3, 4, 16, g["metadata"], g["metadata_sizes"], width=16)
        torch.manual_seed(9)
        b = AttentionProjector(
            3, 4, 16, g["metadata"], g["metadata_sizes"], width=16, graph=g, graph_layers=0
        )
        x = torch.randn(2, 3)
        torch.testing.assert_close(a(x), b(x), atol=0, rtol=0)

    def test_reject_invalid_graph(self):
        with self.assertRaises(ValueError):
            relation_arrays(np.array([3]), np.array([0]), np.array([1.0]), 3)
        with self.assertRaises(ValueError):
            relation_arrays(np.array([0]), np.array([1]), np.array([np.nan]), 3)


if __name__ == "__main__":
    unittest.main()
