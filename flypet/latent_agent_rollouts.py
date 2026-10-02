"""Pinned simulator workers and frozen-model rollouts for the latent agent."""

from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from pathlib import Path
import json, time
import numpy as np
import torch
from safetensors.torch import load_file
from .latent_policy import ContinuousPolicy
from .latent_agent_env import initialize, reset_worker, step_worker, goal_text, ACTION_HIGH


class PolicyState:
    def __init__(self, folder, checkpoint="policy.safetensors"):
        folder = Path(folder)
        config = json.loads((folder / "config.json").read_text())
        self.policy = ContinuousPolicy(
            config["feature_dim"], config["hidden"], config["action_high"]
        ).eval()
        self.policy.load_state_dict(load_file(str(folder / checkpoint)))
        if not np.array_equal(self.policy.action_high.numpy(), ACTION_HIGH):
            raise ValueError("Policy/environment action bounds differ")
        with np.load(folder / "normalization.npz", allow_pickle=False) as z:
            self.mean = z["mean"]
            self.std = z["std"]

    def normalize(self, x):
        return np.clip((np.asarray(x) - self.mean) / self.std, -10, 10).astype(np.float32)


class RolloutPool:
    def __init__(self, preprocessing, models, workers=8):
        self.models = models
        self.workers = workers
        self.model_calls = 0
        self.simulated_windows = 0
        self.pools = [
            ProcessPoolExecutor(
                max_workers=1,
                mp_context=multiprocessing.get_context("spawn"),
                initializer=initialize,
                initargs=(str(preprocessing),),
            )
            for _ in range(workers)
        ]

    def close(self):
        for pool in self.pools:
            pool.shutdown(wait=True, cancel_futures=True)

    def episodes(
        self,
        tasks,
        controller=None,
        *,
        stochastic=False,
        action_seed=1,
        mode="neural",
        fixed_state=None,
        gain=32.0,
    ):
        if len(tasks) > self.workers:
            raise ValueError("Too many episodes for pinned workers")
        futures = [pool.submit(reset_worker, task) for pool, task in zip(self.pools, tasks)]
        observations = [future.result(timeout=180) for future in futures]
        self.simulated_windows += 5 * len(tasks)
        trajectories = [
            {"task": task, "initial": obs["reward_readout"], "decisions": []}
            for task, obs in zip(tasks, observations)
        ]
        generator = torch.Generator().manual_seed(action_seed)
        for turn in range(2):
            raw = []
            actions = []
            zs = []
            logps = []
            for episode_index, (task, obs) in enumerate(zip(tasks, observations)):
                if mode == "noop":
                    actions.append(np.zeros(3))
                    continue
                if mode == "scalar":
                    delta = task["target"] - obs["measured"]["valence"]
                    dose = min(ACTION_HIGH[1], gain * abs(delta))
                    actions.append(
                        np.array([1.0, dose if delta > 0 else 0.0, dose if delta < 0 else 0.0])
                    )
                    continue
                if mode == "donor":
                    donor = observations[(episode_index + 1) % len(observations)]
                    current, reference = donor["features"], donor["reference"]
                elif fixed_state is not None:
                    current, reference = fixed_state
                else:
                    current, reference = obs["features"], obs["reference"]
                feature = self.models.agent_features(current, reference, goal_text(task, turn))
                self.model_calls += 1
                raw.append(feature)
            if mode not in ("noop", "scalar"):
                x = torch.tensor(controller.normalize(np.stack(raw)))
                with torch.no_grad():
                    action, z, logp = controller.policy.sample(
                        x, generator, deterministic=not stochastic
                    )
                actions = action.numpy()
                zs = z.numpy()
                logps = logp.numpy()
            futures = [
                pool.submit(step_worker, action) for pool, action in zip(self.pools, actions)
            ]
            after = [future.result(timeout=180) for future in futures]
            self.simulated_windows += 5 * len(tasks)
            for i, (before, result) in enumerate(zip(observations, after)):
                row = {
                    "action": np.asarray(actions[i]).tolist(),
                    "before": before["reward_readout"],
                    "after": result["reward_readout"],
                }
                if mode not in ("noop", "scalar"):
                    row.update(feature=raw[i], z=zs[i], old_log_prob=float(logps[i]))
                trajectories[i]["decisions"].append(row)
                trajectories[i]["final"] = result["reward_readout"]
                trajectories[i]["reward"] = float(result["reward"])
            observations = after
        return trajectories


def metrics(episodes):
    return {
        "episodes": len(episodes),
        "mean_reward": float(np.mean([r["reward"] for r in episodes])),
        "mean_absolute_target_error": float(
            np.mean([abs(r["final"]["valence"] - r["task"]["target"]) for r in episodes])
        ),
        "mean_initial_target_error": float(
            np.mean([abs(r["initial"]["valence"] - r["task"]["target"]) for r in episodes])
        ),
        "low_output_fraction": float(
            np.mean([r["final"]["approach_hz"] + r["final"]["avoid_hz"] < 100 for r in episodes])
        ),
    }


def serializable(episodes):
    """Keep latent vectors in NPZ training records, not duplicated JSON text."""
    return [
        {
            **row,
            "decisions": [
                {k: v for k, v in decision.items() if k not in ("feature", "z")}
                for decision in row["decisions"]
            ],
        }
        for row in episodes
    ]
