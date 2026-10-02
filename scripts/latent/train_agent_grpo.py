#!/usr/bin/env python3
"""Bounded continuous-head GRPO, matched elite-SFT control and frozen evaluation."""

import argparse, copy, json, shutil, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch
from safetensors.torch import save_file
from flypet.latent_inference import LatentModels
from flypet.latent_agent_env import task_spec
from flypet.latent_agent_rollouts import PolicyState, RolloutPool, metrics, serializable
from flypet.latent_policy import ContinuousPolicy, group_advantages, clipped_objective
from flypet.neural_records import file_hash, write_json
from scripts.latent.train_agent_sft import fit


def save_policy(controller, folder, source_config):
    folder.mkdir(parents=True, exist_ok=False)
    save_file(
        {k: v.detach().contiguous() for k, v in controller.policy.state_dict().items()},
        str(folder / "policy.safetensors"),
    )
    np.savez_compressed(folder / "normalization.npz", mean=controller.mean, std=controller.std)
    shutil.copy2(source_config, folder / "config.json")


def cases(ids, seed_base, repeats):
    return [
        {**task_spec(i), "seed": seed_base + repeat * 1000000 + i * 929}
        for repeat in range(repeats)
        for i in ids
    ]


def save_rollouts(folder, episodes):
    folder.mkdir(parents=True, exist_ok=True)
    write_json(folder / "episodes.json", serializable(episodes))
    decisions = [d for e in episodes for d in e["decisions"] if "feature" in d]
    if decisions:
        np.savez_compressed(
            folder / "policy_records.npz",
            features=np.stack([d["feature"] for d in decisions]),
            z=np.stack([d["z"] for d in decisions]),
            actions=np.array([d["action"] for d in decisions]),
            old_log_prob=np.array([d["old_log_prob"] for d in decisions]),
            episode_returns=np.repeat([e["reward"] for e in episodes], 2),
        )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("bundle", "features", "sft", "out"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--max-seconds", type=int, default=6500)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    torch.set_num_threads(4)
    torch.manual_seed(929)

    def deadline():
        if time.monotonic() - start > a.max_seconds:
            raise TimeoutError("Agent experiment deadline")

    models = LatentModels(a.bundle)
    if models.reader_config["kind"] != "paired_dialogue":
        raise ValueError("Agent requires conversational paired reader")
    dataset = json.loads((a.features / "examples.json").read_text())
    with np.load(a.features / "features.npz", allow_pickle=False) as z:
        x = z["x"]
        actions = z["actions"]
        splits = np.array([r["task"]["split"] for r in dataset])
        fixed = (
            {k: z["mean_" + k] for k in ("fine", "population")},
            {k: z["mean_reference_" + k] for k in ("fine", "population")},
        )
    pool = RolloutPool(a.bundle / "reader_preprocessing.npz", models, 8)
    validation = cases(range(32, 40), 70000000, 2)
    report = {
        "status": "running",
        "scope": "frozen Qwen and paired neural reader; continuous action head only",
        "budget": {
            "updates": 12,
            "groups_per_update": 2,
            "attempts_per_group": 8,
            "decisions_per_episode": 2,
            "rl_episodes": 192,
            "rft_episodes": 192,
        },
        "bundle_sha256": file_hash(a.bundle / "manifest.json"),
        "feature_manifest_sha256": file_hash(a.features / "manifest.json"),
    }
    write_json(a.out / "report.json", report)

    def evaluate(controller, tasks, mode="neural", fixed_state=None, gain=32.0):
        rows = []
        for k in range(0, len(tasks), 8):
            deadline()
            rows.extend(
                pool.episodes(
                    tasks[k : k + 8],
                    controller,
                    mode=mode,
                    fixed_state=fixed_state,
                    gain=gain,
                    action_seed=7200 + k,
                )
            )
        return rows

    def choose_sft(folder, destination, mode="neural", fixed_state=None):
        candidates = []
        seen = set()
        for checkpoint in ("policy.safetensors", "policy_last.safetensors"):
            digest = file_hash(folder / checkpoint)
            if digest in seen:
                continue
            seen.add(digest)
            controller = PolicyState(folder, checkpoint)
            rows = evaluate(controller, validation, mode, fixed_state)
            candidates.append((metrics(rows)["mean_reward"], checkpoint, rows))
        selected = max(candidates, key=lambda r: r[0])
        controller = PolicyState(folder, selected[1])
        save_policy(controller, destination, folder / "config.json")
        save_rollouts(destination / "validation", selected[2])
        write_json(
            destination / "selection.json",
            {
                "criterion": "mean validation episode reward",
                "selected": selected[1],
                "candidates": [
                    {"checkpoint": name, **metrics(rows)} for _, name, rows in candidates
                ],
            },
        )
        return PolicyState(destination)

    try:
        sft = choose_sft(a.sft / "sft", a.out / "selected_sft")
        task_only = choose_sft(
            a.sft / "task_only_sft", a.out / "selected_task_only", fixed_state=fixed
        )
        print(json.dumps({"stage": "sft_selected", "wall_s": time.monotonic() - start}), flush=True)
        # Fit a simple strong controller on separate training-only rollouts.
        scalar = []
        for gain in (8.0, 16.0, 32.0, 64.0):
            rows = evaluate(None, cases(range(8), 60000000, 2), mode="scalar", gain=gain)
            scalar.append({"gain": gain, **metrics(rows)})
        gain = max(scalar, key=lambda r: r["mean_reward"])["gain"]
        write_json(a.out / "scalar_training.json", {"candidates": scalar, "chosen_gain": gain})
        rng = np.random.default_rng(929)
        schedule = []
        for update in range(12):
            ids = rng.choice(32, 2, replace=False).tolist()
            schedule.append(
                [
                    {
                        "task": i,
                        "seed": 50000000 + update * 10000 + i * 31,
                        "action_seed": 930000 + update * 101 + j,
                    }
                    for j, i in enumerate(ids)
                ]
            )
        write_json(a.out / "training_schedule.json", schedule)
        current = copy.deepcopy(sft)
        reference = copy.deepcopy(sft.policy).requires_grad_(False)
        optimizer = torch.optim.AdamW(current.policy.parameters(), lr=0.0003, weight_decay=0.01)
        history = []
        rl_candidates = []
        rl_rows = []
        rft_rows = []
        for update, groups in enumerate(schedule, 1):
            deadline()
            batch = []
            returns = []
            for group in groups:
                task = {**task_spec(group["task"]), "seed": group["seed"]}
                tasks = [task.copy() for _ in range(8)]
                rows = pool.episodes(
                    tasks, current, stochastic=True, action_seed=group["action_seed"]
                )
                batch.extend(rows)
                returns.append([r["reward"] for r in rows])
            rewards = torch.tensor(returns)
            adv = group_advantages(rewards, std_floor=0.05).flatten().repeat_interleave(2)
            decisions = [d for row in batch for d in row["decisions"]]
            features = torch.tensor(current.normalize(np.stack([d["feature"] for d in decisions])))
            z = torch.tensor(np.stack([d["z"] for d in decisions]))
            old = torch.tensor([d["old_log_prob"] for d in decisions])
            for _ in range(4):
                optimizer.zero_grad(set_to_none=True)
                loss, diagnostics = clipped_objective(
                    current.policy, reference, features, z, old, adv, clip=0.2, kl_weight=0.01
                )
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite policy objective")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(current.policy.parameters(), 1)
                optimizer.step()
            row = {
                "update": update,
                **metrics(batch),
                "group_reward_std": rewards.std(-1, unbiased=False).tolist(),
                "zero_variance_groups": int((rewards.std(-1, unbiased=False) < 1e-6).sum()),
                "kl": float(diagnostics["kl"]),
                "ratio": float(diagnostics["ratio"]),
                "wall_s": time.monotonic() - start,
            }
            history.append(row)
            rl_rows.extend(batch)
            save_rollouts(a.out / f"rollouts/grpo-{update:02d}", batch)
            if update % 4 == 0:
                folder = a.out / f"grpo-{update:02d}"
                save_policy(current, folder, a.out / "selected_sft/config.json")
                validation_rows = evaluate(current, validation)
                score = metrics(validation_rows)
                save_rollouts(folder / "validation", validation_rows)
                rl_candidates.append({"folder": str(folder), "update": update, **score})
            write_json(a.out / "training.json", history)
            print(json.dumps(row), flush=True)
        selected = max(rl_candidates, key=lambda row: row["mean_reward"])
        grpo = PolicyState(selected["folder"])
        save_policy(grpo, a.out / "selected_grpo", a.out / "selected_sft/config.json")
        write_json(
            a.out / "grpo_selection.json", {"selected": selected, "candidates": rl_candidates}
        )
        # Independent frozen-SFT sampling, exactly matched episode/task/seed budget.
        elites = []
        for update, groups in enumerate(schedule, 1):
            deadline()
            batch = []
            for group in groups:
                task = {**task_spec(group["task"]), "seed": group["seed"]}
                rows = pool.episodes(
                    [task.copy() for _ in range(8)],
                    sft,
                    stochastic=True,
                    action_seed=group["action_seed"],
                )
                batch.extend(rows)
                elites.extend(max(rows, key=lambda row: row["reward"])["decisions"])
            rft_rows.extend(batch)
            save_rollouts(a.out / f"rollouts/rft-{update:02d}", batch)
            print(
                json.dumps(
                    {
                        "stage": "rft_collection",
                        "group_batch": update,
                        "wall_s": time.monotonic() - start,
                    }
                ),
                flush=True,
            )
        augmented_x = np.concatenate([x, np.stack([d["feature"] for d in elites])])
        augmented_actions = np.concatenate([actions, np.array([d["action"] for d in elites])])
        augmented_splits = np.r_[splits, np.full(len(elites), "train")]
        fit(
            augmented_x,
            augmented_actions,
            augmented_splits,
            a.out / "rft_fit",
            epochs=200,
            initial_directory=a.out / "selected_sft",
        )
        rft = choose_sft(a.out / "rft_fit", a.out / "selected_rft")
        random_head = copy.deepcopy(sft)
        torch.manual_seed(930)
        random_head.policy = ContinuousPolicy(x.shape[1], hidden=64).eval()
        save_policy(random_head, a.out / "random_head", a.out / "selected_sft/config.json")
        test = cases(range(40, 64), 90000000, 4)
        methods = {
            "noop": (None, "noop", None),
            "random_head": (random_head, "neural", None),
            "scalar": (None, "scalar", None),
            "task_only_sft": (task_only, "neural", fixed),
            "sft": (sft, "neural", None),
            "sft_donor": (sft, "donor", None),
            "rft": (rft, "neural", None),
            "grpo": (grpo, "neural", None),
        }
        lock = {
            "tasks": test,
            "methods": list(methods),
            "scalar_gain": gain,
            "selected_grpo_update": selected["update"],
            "checkpoint_sha256": {
                str(path.relative_to(a.out)): file_hash(path)
                for path in a.out.rglob("*.safetensors")
            },
            "bundle_sha256": file_hash(a.bundle / "manifest.json"),
            "source_sha256": {
                n: file_hash(n)
                for n in [
                    "flypet/latent_agent_env.py",
                    "flypet/latent_agent_rollouts.py",
                    "flypet/latent_policy.py",
                    "scripts/latent/train_agent_grpo.py",
                ]
            },
        }
        write_json(a.out / "evaluation_lock.json", lock)
        results = {}
        summary = {}
        for name, (controller, mode, state) in methods.items():
            rows = evaluate(controller, test, mode, state, gain)
            results[name] = rows
            summary[name] = metrics(rows)
            save_rollouts(a.out / "evaluation" / name, rows)
            print(json.dumps({"stage": "evaluation", "method": name, **summary[name]}), flush=True)
        bootstrap = np.random.default_rng(930).integers(0, 24, size=(4000, 24))
        paired = {}
        for name, rows in results.items():
            errors = (
                np.array([abs(row["final"]["valence"] - row["task"]["target"]) for row in rows])
                .reshape(4, 24)
                .mean(0)
            )
            base = (
                np.array(
                    [abs(row["final"]["valence"] - row["task"]["target"]) for row in results["sft"]]
                )
                .reshape(4, 24)
                .mean(0)
            )
            delta = errors - base
            interval = np.quantile(delta[bootstrap].mean(1), [0.025, 0.975])
            paired[name] = {
                "target_error_minus_sft": float(delta.mean()),
                "paired_recipe_bootstrap_95": interval.tolist(),
            }
        report.update(
            status="complete",
            metrics=summary,
            paired=paired,
            selected_grpo_update=selected["update"],
            training_episode_counts={"grpo": len(rl_rows), "rft": len(rft_rows)},
            simulated_windows=pool.simulated_windows,
            model_calls=pool.model_calls,
            total_wall_s=time.monotonic() - start,
        )
        write_json(a.out / "report.json", report)
    finally:
        pool.close()


if __name__ == "__main__":
    main()
