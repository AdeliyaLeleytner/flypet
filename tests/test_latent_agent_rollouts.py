import json, tempfile, unittest
from pathlib import Path
from concurrent.futures import Future
import numpy as np

try:
    import torch
except ImportError:
    raise unittest.SkipTest("Optional PyTorch dependencies are not installed")
from flypet.latent_agent_rollouts import PolicyState, RolloutPool, serializable
from flypet.latent_agent_env import task_spec
from scripts.latent.train_agent_sft import fit


class FakeWorker:
    def submit(self, fn, value):
        if fn.__name__ == "reset_worker":
            self.step = 0
            self.task = value
            self.value = 0.0
            self.reference = {"fine": np.array([0.0])}
        else:
            self.step += 1
            self.value += float(np.sum(value)) / 100
        future = Future()
        future.set_result(
            {
                "features": {"fine": np.array([self.value])},
                "reference": self.reference,
                "reward_readout": {"valence": self.value, "approach_hz": 100.0, "avoid_hz": 10.0},
                "measured": {"valence": self.value},
                "reward": -abs(self.value - self.task["target"]),
            }
        )
        return future


class FakeModel:
    def agent_features(self, current, reference, goal):
        return np.arange(8, dtype=np.float32) * 0.1 + current["fine"][0]


class RolloutTests(unittest.TestCase):
    def test_sft_roundtrip_and_recorded_action_likelihoods(self):
        rng = np.random.default_rng(4)
        x = rng.normal(size=(12, 8)).astype(np.float32)
        actions = rng.random((12, 3)).astype(np.float32) * np.array([1.5, 32, 32], np.float32)
        splits = np.array(["train"] * 8 + ["validation"] * 4)
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory) / "fit"
            fit(x, actions, splits, folder, epochs=2)
            controller = PolicyState(folder)
            pool = RolloutPool.__new__(RolloutPool)
            pool.models = FakeModel()
            pool.workers = 2
            pool.pools = [FakeWorker(), FakeWorker()]
            pool.model_calls = 0
            pool.simulated_windows = 0
            episodes = pool.episodes(
                [task_spec(0), task_spec(1)], controller, stochastic=True, action_seed=99
            )
            self.assertEqual([p.step for p in pool.pools], [2, 2])
            self.assertEqual(pool.simulated_windows, 30)
            for episode in episodes:
                self.assertFalse(
                    np.array_equal(
                        episode["decisions"][0]["feature"], episode["decisions"][1]["feature"]
                    )
                )
                for decision in episode["decisions"]:
                    features = torch.tensor(controller.normalize(decision["feature"])[None])
                    z = torch.tensor(decision["z"][None])
                    logp = controller.policy.log_prob(features, z)
                    self.assertAlmostEqual(
                        float(logp.detach()[0]), decision["old_log_prob"], places=4
                    )
                    np.testing.assert_allclose(
                        (controller.policy.action_high * torch.sigmoid(z[0])).numpy(),
                        decision["action"],
                    )
            json.dumps(serializable(episodes), allow_nan=False)
            continued = Path(directory) / "continued"
            fit(x, actions, splits, continued, epochs=2, initial_directory=folder)
            other = PolicyState(continued)
            np.testing.assert_array_equal(other.mean, controller.mean)


if __name__ == "__main__":
    unittest.main()
