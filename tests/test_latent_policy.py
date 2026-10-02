import copy
import unittest

try:
    import torch
except ImportError:
    raise unittest.SkipTest("Optional PyTorch dependencies are not installed")
from torch.distributions import (
    Normal,
    Independent,
    TransformedDistribution,
    SigmoidTransform,
    AffineTransform,
)
from flypet.latent_policy import ContinuousPolicy, group_advantages, clipped_objective


class PolicyTests(unittest.TestCase):
    def test_probability_matches_transformed_distribution(self):
        torch.manual_seed(17)
        p = ContinuousPolicy(8)
        x = torch.randn(5, 8)
        action, z, logp = p.sample(x)
        mean, log_std = p.parameters_for(x)
        distribution = TransformedDistribution(
            Independent(Normal(mean, log_std.exp()), 1),
            [SigmoidTransform(), AffineTransform(0, p.action_high, event_dim=1)],
        )
        torch.testing.assert_close(logp, distribution.log_prob(action), rtol=2e-5, atol=2e-5)
        self.assertTrue(torch.isfinite(p.log_prob(x, torch.full((5, 3), 100.0))).all())

    def test_episode_advantages_and_ratio(self):
        torch.manual_seed(3)
        p = ContinuousPolicy(8)
        reference = copy.deepcopy(p)
        x = torch.randn(8, 8)
        _, z, old = p.sample(x)
        z = z.detach()
        old = old.detach()
        adv = group_advantages(torch.tensor([[1.0, 1.0, 1.0, 1.0], [1.0, 2.0, 3.0, 4.0]]))
        torch.testing.assert_close(adv[0], torch.zeros(4))
        self.assertAlmostEqual(float(adv[1].mean()), 0.0, places=6)
        loss, metrics = clipped_objective(p, reference, x, z, old, adv.flatten())
        torch.testing.assert_close(metrics["ratio"], torch.tensor(1.0))
        torch.testing.assert_close(metrics["kl"], torch.tensor(0.0))
        loss.backward()
        self.assertGreater(float(p.net[-1].weight.grad.abs().sum()), 0)

    def test_kl_transformation_invariance(self):
        torch.manual_seed(7)
        p = ContinuousPolicy(8)
        reference = copy.deepcopy(p)
        with torch.no_grad():
            p.net[-1].bias.add_(0.1)
        x = torch.randn(20000, 8)
        _, z, logp = p.sample(x)
        estimate = (logp - reference.log_prob(x, z)).mean()
        exact = p.kl_to(reference, x).mean()
        self.assertLess(abs(float((estimate - exact).detach())), 0.004)


if __name__ == "__main__":
    unittest.main()
