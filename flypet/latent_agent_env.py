"""Two-decision neural conditioning task, with hidden initial plasticity.

The agent sees neural observations and its goal. Recipe, hidden history and
measured valence are evaluator metadata, never policy features.
"""

from __future__ import annotations
import numpy as np
from .latent_observation import from_step


ACTION_HIGH = np.array([1.5, 32.0, 32.0], dtype=np.float64)
TEACHER_ACTIONS = np.array(
    [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 4.0, 0.0],
        [1.0, 12.0, 0.0],
        [1.0, 30.0, 0.0],
        [1.0, 0.0, 4.0],
        [1.0, 0.0, 12.0],
        [1.0, 0.0, 30.0],
        [0.5, 12.0, 0.0],
        [0.5, 0.0, 12.0],
    ]
)


def task_spec(index):
    from scripts.latent.collect_calibration import ODORS

    rng = np.random.default_rng(960000 + index)
    return {
        "id": int(index),
        "split": "train" if index < 32 else "validation" if index < 40 else "test",
        "odors": rng.choice(ODORS, 2, replace=False).tolist(),
        "mix": float(rng.uniform(0.2, 0.8)),
        "scale": float(rng.uniform(0.6, 1.4)),
        "history": str(rng.choice(["naive", "reward", "punish"])),
        "history_rate": float(rng.choice([30.0, 60.0, 90.0])),
        "target": round(float(rng.uniform(-0.45, 0.45)), 2),
        "seed": int(970000 + index * 929),
    }


def goal_text(task, turn):
    templates = [
        "Bring the response to the same probe odor close to MBON valence {target:+.2f}.",
        "Condition this fly so that its MBON balance approaches {target:+.2f} when the same odor is probed.",
        "Aim for a neural approach/avoidance balance of {target:+.2f} on the next probes.",
        "Please steer the subsequent response to this odor toward a normalized MBON balance of {target:+.2f}.",
    ]
    template = templates[3 if task["split"] == "test" else task["id"] % 3]
    return (
        template.format(target=task["target"]) + " "
        "Balance is (approach-avoidance)/(approach+avoidance+1). "
        f"{2 - turn} conditioning decisions remain. Choose continuous exposure strength, PAM drive and PPL1 drive. "
        "Use the supplied neural state to decide. Avoid unnecessary stimulation."
    )


def action_cost(action):
    return float(np.asarray(action).astype(np.float64) @ np.reciprocal(ACTION_HIGH) / 3)


class AgentEnvironment:
    def __init__(self, preprocessing):
        from scripts.latent.collect_calibration import build, ODORS
        from .neural_runtime import NeuralRuntime

        brain, memory, door, ports, _ = build()
        self.runtime = NeuralRuntime(brain, input_ids=ports, seed=1, memory=memory)
        self.pre = preprocessing
        actual = np.array([brain.i2flyid[int(i)] for i in preprocessing["neuron_indices"]])
        if not np.array_equal(actual, preprocessing["neuron_root_ids"]):
            raise ValueError("Neural order mismatch")
        self.profiles = {name: self.runtime.drive_from_inputs(door.stim(name)) for name in ODORS}
        self.pam = self.runtime.drive_from_inputs([memory.reward(1)])
        self.ppl = self.runtime.drive_from_inputs([memory.punishment(1)])

    def advance(self, drive, learning):
        self.runtime.set_learning(learning)
        return self.runtime.advance(drive, 250)

    def observation(self, step):
        value = self.runtime.memory.valence(step.result)
        return {
            "features": from_step(step, self.pre, self.runtime.dt_ms),
            "measured": {
                "valence": value.score,
                "approach_hz": value.approach_hz,
                "avoid_hz": value.avoid_hz,
                "spikes": sum(step.result.counts.values()),
                "memory_strength": self.runtime.memory.memory_strength(),
            },
            "reference": self.reference,
            "turn": len(self.actions),
        }

    def reset(self, task):
        self.task = task
        self.actions = []
        self.runtime.reset(seed=task["seed"], reset_memory=True)
        a, b = (self.profiles[n] for n in task["odors"])
        self.probe = np.clip(task["scale"] * (task["mix"] * a + (1 - task["mix"]) * b), 0, 300)
        history = np.zeros_like(self.probe)
        if task["history"] != "naive":
            history = self.probe + task["history_rate"] * (
                self.pam if task["history"] == "reward" else self.ppl
            )
        self.advance(history, task["history"] != "naive")
        self.advance(np.zeros_like(self.probe), False)
        self.reference = None
        obs = self.probe_observation()
        self.reference = obs["features"]
        obs["reference"] = self.reference
        return obs

    def probe_observation(self):
        steps = [self.advance(self.probe, False) for _ in range(3)]
        probes = [self.runtime.memory.valence(step.result) for step in steps]
        ap = float(np.mean([p.approach_hz for p in probes]))
        av = float(np.mean([p.avoid_hz for p in probes]))
        obs = self.observation(steps[-1])
        obs["reward_readout"] = {
            "approach_hz": ap,
            "avoid_hz": av,
            "valence": (ap - av) / (ap + av + 1),
            "duration_ms": 750,
        }
        obs["probe_valences"] = [p.score for p in probes]
        return obs

    def step(self, action):
        action = np.asarray(action, dtype=np.float64)
        if len(self.actions) >= 2:
            raise RuntimeError("Two-decision budget exhausted")
        if (
            action.shape != (3,)
            or not np.isfinite(action).all()
            or np.any(action < 0)
            or np.any(action > ACTION_HIGH)
        ):
            raise ValueError("Action outside the declared continuous bounds")
        drive = (
            np.clip(action[0] * self.probe, 0, 300) + action[1] * self.pam + action[2] * self.ppl
        )
        self.advance(drive, True)
        self.advance(np.zeros_like(self.probe), False)
        self.actions.append(action.tolist())
        obs = self.probe_observation()
        obs["reward"] = -abs(
            obs["reward_readout"]["valence"] - self.task["target"]
        ) - 0.03 * np.mean([action_cost(a) for a in self.actions])
        obs["done"] = len(self.actions) == 2
        return obs

    def replay(self, task, actions):
        obs = self.reset(task)
        for action in actions:
            obs = self.step(action)
        return obs

    def close(self):
        self.runtime.close()


WORKER = None


def initialize(preprocessing_path):
    global WORKER
    with np.load(preprocessing_path, allow_pickle=False) as z:
        pre = {k: z[k] for k in z.files if k not in ("x", "population_x", "auxiliary", "targets")}
    WORKER = AgentEnvironment(pre)


def reset_worker(task):
    return WORKER.reset(task)


def step_worker(action):
    return WORKER.step(action)


def replay_worker(task, actions):
    return WORKER.replay(task, actions)


def teacher_episode(task):
    actions = []
    examples = []
    for turn in range(2):
        observation = WORKER.replay(task, actions)
        results = []
        for action in TEACHER_ACTIONS:
            trials = []
            for future in range(3):
                WORKER.replay(task, actions)
                # Keep exactly the observed state, memory and pending delays;
                # vary only future input noise. Shared seeds across candidates.
                WORKER.runtime.rng = np.random.Generator(
                    np.random.PCG64(29000000 + task["id"] * 101 + turn * 17 + future)
                )
                trials.append(WORKER.step(action.tolist()))
            results.append(trials)
        rewards = [float(np.mean([trial["reward"] for trial in trials])) for trials in results]
        chosen = int(np.argmax(rewards))
        examples.append(
            {
                "observation": observation,
                "action": TEACHER_ACTIONS[chosen].tolist(),
                "candidate_rewards": rewards,
                "candidate_reward_trials": [[r["reward"] for r in trials] for trials in results],
                "candidate_measurements": [
                    [r["reward_readout"] for r in trials] for trials in results
                ],
            }
        )
        actions.append(TEACHER_ACTIONS[chosen].tolist())
    final = WORKER.replay(task, actions)
    return {
        "task": task,
        "examples": examples,
        "final": final["reward_readout"],
        "final_reward": final["reward"],
    }
