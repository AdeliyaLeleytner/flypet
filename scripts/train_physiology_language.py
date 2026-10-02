"""Bounded real-LLM physiological pilot. Fixed graph; history enters only the core."""

import argparse, hashlib, json, math, random, time
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from flypet.physiology_language import PhysiologicalLanguageBridge, MODEL_ID, MODEL_REVISION
from flypet.physiology_memory_task import OBJECTS, LOCATIONS, question_text
from flypet.physiology_torch import PhysiologicalCore


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, obj):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def sync():
    torch.cuda.synchronize()


def state_digest(state):
    h = hashlib.sha256()
    for name, value in sorted(state.items()):
        h.update(name.encode())
        h.update(value.contiguous().numpy().tobytes())
    return h.hexdigest()


def cuda_core_check():
    cpu = PhysiologicalCore(4, [0, 1, 2], [1, 2, 3], [100, 100, 100], surrogate_scale=0.1)
    gpu = PhysiologicalCore(4, [0, 1, 2], [1, 2, 3], [100, 100, 100], surrogate_scale=0.1).cuda()
    x = torch.zeros(64, 2, 1)
    x[::25] = 2000.0
    x.requires_grad_()
    xx = x.detach().cuda().requires_grad_()
    a = cpu(x, [0], [1, 2, 3], checkpoint_steps=17)
    b = gpu(xx, [0], [1, 2, 3], checkpoint_steps=17)
    for k in ["voltage_mv", "synaptic_mv", "rate_hz"]:
        torch.testing.assert_close(a[k], b[k].cpu(), rtol=1e-4, atol=1e-4)
    a["voltage_mv"].square().sum().backward()
    b["voltage_mv"].square().sum().backward()
    torch.testing.assert_close(cpu.raw_gain.grad, gpu.raw_gain.grad.cpu(), rtol=2e-4, atol=2e-4)
    torch.testing.assert_close(x.grad, xx.grad.cpu(), rtol=2e-4, atol=2e-4)
    return {"cpu_cuda_forward_backward": "passed", "gradient_atol": 2e-4, "tiny_graph_only": True}


def long_core_probe(data, checkpoint_steps, surrogate_scale, physics_dtype="float32"):
    graph = np.load(Path(data) / "graph" / "graph.npz")
    event = np.load(Path(data) / "calibration" / "input.npz")
    reference = np.load(Path(data) / "calibration" / "reference_counts.npy")
    core = PhysiologicalCore(
        len(graph["order"]),
        graph["pre"],
        graph["post"],
        graph["contacts"],
        surrogate_scale=surrogate_scale,
    ).cuda()
    if physics_dtype == "float64":
        core.double()
        with torch.no_grad():
            core.raw_gain.fill_(math.log(0.75))
    elif physics_dtype != "float32":
        raise ValueError("Unknown physics precision")
    pulses = torch.from_numpy(event["impulses_mv"][:, None, :]).to(
        device="cuda", dtype=core.raw_gain.dtype
    )
    drive = torch.zeros_like(pulses).requires_grad_()
    torch.cuda.reset_peak_memory_stats()
    sync()
    start = time.monotonic()
    result = core(
        drive,
        event["ports"],
        graph["output_indices"],
        impulses_mv=pulses,
        input_refractory_ms=0,
        checkpoint_steps=checkpoint_steps,
    )
    loss = (result["voltage_mv"] / 7).square().mean()
    sync()
    fwd = time.monotonic() - start
    start = time.monotonic()
    loss.backward()
    sync()
    back = time.monotonic() - start
    counts = result["neuron_spike_counts"][0].detach().cpu().numpy()
    grad = core.raw_gain.grad
    report = {
        "ticks": len(drive),
        "batch": 1,
        "physics_dtype": physics_dtype,
        "forward_seconds": fwd,
        "backward_seconds": back,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(),
        "spikes": int(counts.sum()),
        "brian_spikes": int(reference.sum()),
        "same_neuron_count_fraction": float((counts == reference).mean()),
        "count_mae": float(abs(counts - reference).mean()),
        "core_gradient_finite": bool(torch.isfinite(grad).all()),
        "core_gradient_nonzero": int(torch.count_nonzero(grad)),
        "scope": "Full300ms physiological replay/backward without LLM; synthetic voltage loss, no optimizer step.",
    }
    if report["core_gradient_finite"]:
        report["core_gradient_norm"] = float(grad.double().norm())
    return report


def selection(rows, count):
    # Whole four-row counterfactual/query groups; input-only deterministic IDs.
    if count == 48:
        ids = []
        pairs = [(a, b) for i, a in enumerate(LOCATIONS) for b in LOCATIONS[i + 1 :]]
        for i, pair in enumerate(pairs):
            for last in LOCATIONS:
                if last in pair:
                    continue
                matches = {
                    r["group"]
                    for r in rows
                    if r["events"][0][0] == OBJECTS[i % len(OBJECTS)]
                    and set((r["events"][0][1], r["events"][2][1])) == set(pair)
                    and r["events"][-1][1] == last
                }
                ids.append(sorted(matches)[0])
        return [r for r in rows if r["group"] in ids]
    ids = sorted({r["group"] for r in rows})[: math.ceil(count / 4)]
    return [r for r in rows if r["group"] in ids]


def summarize(records):
    ans = {}
    for name, rr in [
        ("all", records),
        ("earlier_object", [r for r in records if not r["query_is_last_mentioned"]]),
        ("most_recent_object", [r for r in records if r["query_is_last_mentioned"]]),
    ]:
        ans[name] = {
            "n": len(rr),
            "accuracy": sum(r["correct"] for r in rr) / len(rr) if rr else None,
            "valid_fraction": sum(r["valid"] for r in rr) / len(rr) if rr else None,
        }
    paired = {}
    for r in records:
        paired.setdefault((r["group"], r["assignment"]), []).append(r)
    ans["both_queries_correct"] = (
        sum(len(v) == 2 and all(r["correct"] for r in v) for v in paired.values()) / len(paired)
        if paired
        else None
    )
    return ans


def clip_gradients(parameters, maximum=1.0):
    parameters = [p for p in parameters if p.grad is not None]
    if not parameters:
        return torch.tensor(0.0)
    norm = torch.stack([p.grad.detach().double().square().sum() for p in parameters]).sum().sqrt()
    if not torch.isfinite(norm):
        raise FloatingPointError("Nonfinite aggregate gradient norm")
    factor = (maximum / (norm + 1e-12)).clamp(max=1.0)
    for p in parameters:
        p.grad.mul_(factor.to(p.grad.dtype))
    return norm


def clip_bridge_gradients(bridge, train_core, separate=False):
    # Scale disparity upstream of the spiking core must not suppress updates
    # of the direct language decoder through Adam's finite epsilon.
    groups = {
        "encoder": [
            p for n, p in bridge.named_parameters() if n.startswith("encoder.") and p.requires_grad
        ],
        "decoder": [
            p
            for n, p in bridge.named_parameters()
            if not n.startswith(("encoder.", "core.")) and p.requires_grad
        ],
    }
    raw = {}
    for name, pp in groups.items():
        squares = [p.grad.detach().double().square().sum() for p in pp if p.grad is not None]
        raw[name] = float(torch.stack(squares).sum().sqrt()) if squares else 0.0
    interface = math.hypot(raw["encoder"], raw["decoder"])
    if separate:
        for pp in groups.values():
            clip_gradients(pp, 1.0)
    else:
        clip_gradients(groups["encoder"] + groups["decoder"], 1.0)
    core = float(clip_gradients([bridge.core.raw_gain], 1.0)) if train_core else 0.0
    return {
        "interface_norm": interface,
        "encoder_norm": raw["encoder"],
        "decoder_norm": raw["decoder"],
        "core_norm": core,
    }


class Pilot:
    def __init__(self, a):
        self.a = a
        self.out = Path(a.out)
        self.out.mkdir(parents=True, exist_ok=False)
        self.started = time.monotonic()
        self.deadline = self.started + a.max_seconds
        self.report = {
            "status": "running",
            "complete": False,
            "config": vars(a),
            "model": MODEL_ID,
            "revision": MODEL_REVISION,
            "no_history_in_lm_context": True,
            "shared_history_kv_cache": False,
            "pet_state_loaded": False,
            "source_sha256": {
                str(p): sha(p)
                for p in [
                    Path(__file__),
                    Path("flypet/physiology_torch.py"),
                    Path("flypet/physiology_language.py"),
                    Path("flypet/physiology_memory_task.py"),
                ]
            },
        }
        self.persist()
        if not torch.cuda.is_available():
            raise RuntimeError("Actual CUDA GPU required for this pilot")
        self.report["cuda_core_check"] = cuda_core_check()
        self.persist()
        self.report["long_core_probe"] = long_core_probe(
            a.data, a.checkpoint_steps, a.surrogate_scale, a.physics_dtype
        )
        self.persist()
        torch.cuda.empty_cache()
        torch.set_num_threads(4)
        torch.manual_seed(a.seed)
        np.random.seed(a.seed)
        random.seed(a.seed)
        self.data = {
            s: [
                json.loads(x)
                for x in (Path(a.data) / "task" / f"{s}.jsonl").read_text().splitlines()
            ]
            for s in ["train", "validation", "test"]
        }
        self.tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
        self.llm = (
            AutoModelForCausalLM.from_pretrained(
                MODEL_ID, revision=MODEL_REVISION, dtype=torch.bfloat16, attn_implementation="sdpa"
            )
            .to("cuda")
            .eval()
        )
        self.llm.requires_grad_(False)
        self.embedding = self.llm.get_input_embeddings()
        self.std = float(self.embedding.weight.detach().float().std())
        graph = np.load(Path(a.data) / "graph" / "graph.npz")
        self.bridge = PhysiologicalLanguageBridge(
            graph,
            self.embedding.weight.shape[1],
            self.std,
            tokens=a.prefix_tokens,
            hidden=a.hidden,
            ticks_per_event=a.ticks_per_event,
            max_drive_mv=a.max_drive_mv,
            checkpoint_steps=a.checkpoint_steps,
            physics_dtype=a.physics_dtype,
            surrogate_scale=a.surrogate_scale,
        ).to("cuda")
        self.initial = self.bridge.trainable_state()
        self.report["initial_bridge_sha256"] = state_digest(self.initial)
        torch.save(self.initial, self.out / "initial.pt")
        self.events = {}
        with torch.no_grad():
            for obj in OBJECTS:
                for loc in LOCATIONS:
                    text = f"The {obj} moved to the {loc} box."
                    ids = torch.tensor(
                        self.tok(text, add_special_tokens=False)["input_ids"], device="cuda"
                    )
                    self.events[text] = self.embedding(ids).float().mean(0).detach()
        self.questions = {
            obj: self.chat_ids(question_text({"query": f"Where is the {obj} now?"}))
            for obj in OBJECTS
        }
        self.targets = {
            x: self.tok(x + self.tok.eos_token, add_special_tokens=False)["input_ids"]
            for x in LOCATIONS
        }
        self.report.update(
            graph_manifest=json.loads((Path(a.data) / "graph" / "manifest.json").read_text()),
            task_manifest=json.loads((Path(a.data) / "task" / "manifest.json").read_text()),
            bridge_parameters=sum(p.numel() for p in self.bridge.parameters()),
            llm_parameters=sum(p.numel() for p in self.llm.parameters()),
            gpu=torch.cuda.get_device_name(),
            torch_version=torch.__version__,
            embedding_std=self.std,
            llm_dtype=str(self.embedding.weight.dtype),
            old_baselines={"chance": 0.25, "last_location_heuristic": 0.625},
        )
        self.persist()

    def persist(self):
        self.report["elapsed_seconds"] = time.monotonic() - self.started
        save(self.out / "report.json", self.report)

    def check_time(self, reserve=0):
        if time.monotonic() + reserve > self.deadline:
            raise TimeoutError("Declared experiment deadline")

    def chat_ids(self, text):
        return self.tok.apply_chat_template(
            [{"role": "user", "content": text}],
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
            return_dict=False,
        )

    def brain(self, row, intervention="normal"):
        x = torch.stack([self.events[e] for e in row["history"]])[None]
        return self.bridge(x, intervention=intervention)

    def language_inputs(self, row, prefix, answer=False):
        q = self.questions[row["query_object"]]
        tail = q + (self.targets[row["answer"]] if answer else [])
        device = self.embedding.weight.device
        ids = torch.tensor([tail], device=device)
        embedded = self.embedding(ids).detach()
        joined = torch.cat((prefix.to(embedded.dtype), embedded), 1)
        labels = torch.full(joined.shape[:2], -100, device=device, dtype=torch.long)
        if answer:
            labels[0, len(prefix[0]) + len(q) :] = torch.tensor(
                self.targets[row["answer"]], device=device
            )
        return joined, labels

    def batch_language_inputs(self, rows, prefix):
        pieces = [
            self.language_inputs(row, prefix[i : i + 1], answer=True) for i, row in enumerate(rows)
        ]
        longest = max(x.shape[1] for x, y in pieces)
        embedded = prefix.new_zeros(
            (len(rows), longest, prefix.shape[-1]), dtype=self.embedding.weight.dtype
        )
        labels = torch.full((len(rows), longest), -100, device=prefix.device, dtype=torch.long)
        mask = torch.zeros_like(labels)
        # Indexed copy preserves the graph to every prefix; right padding is masked.
        for i, (x, y) in enumerate(pieces):
            embedded[i, : x.shape[1]] = x[0]
            labels[i, : y.shape[1]] = y[0]
            mask[i, : x.shape[1]] = 1
        return embedded, labels, mask

    def loss(self, rows):
        if isinstance(rows, dict):
            rows = [rows]
        x = torch.stack([torch.stack([self.events[e] for e in row["history"]]) for row in rows])
        prefix, stats = self.bridge(x)
        embeddings, labels, mask = self.batch_language_inputs(rows, prefix)
        value = self.llm(
            inputs_embeds=embeddings, attention_mask=mask, labels=labels, use_cache=False
        ).loss
        return value, stats

    def predict(self, row, kind="normal"):
        self.check_time()
        with torch.no_grad():
            if kind in ("raw_history", "question_only"):
                text = question_text(row)
                if kind == "raw_history":
                    text = "\n".join(row["history"]) + "\n" + text
                ids = torch.tensor([self.chat_ids(text)], device="cuda")
                generated = self.llm.generate(
                    input_ids=ids,
                    attention_mask=torch.ones_like(ids),
                    max_new_tokens=8,
                    do_sample=False,
                    pad_token_id=self.tok.eos_token_id,
                    return_dict_in_generate=True,
                    output_scores=True,
                )
                tokens = generated.sequences[0, len(ids[0]) :].tolist()
            else:
                prefix, _ = self.brain(row, kind)
                embedded, _ = self.language_inputs(row, prefix)
                generated = self.llm.generate(
                    inputs_embeds=embedded,
                    attention_mask=torch.ones(embedded.shape[:2], device="cuda", dtype=torch.long),
                    max_new_tokens=8,
                    do_sample=False,
                    pad_token_id=self.tok.eos_token_id,
                    return_dict_in_generate=True,
                    output_scores=True,
                )
                tokens = generated.sequences[0, -len(generated.scores) :].tolist()
            text = self.tok.decode(tokens, skip_special_tokens=True).strip()
            first_nll = float(
                -generated.scores[0].float().log_softmax(-1)[0, self.targets[row["answer"]][0]]
            )
        if not math.isfinite(first_nll):
            raise FloatingPointError("Nonfinite generation score")
        normalized = text.lower().strip().rstrip(".!").strip()
        valid = normalized in LOCATIONS
        return {
            "id": row["id"],
            "group": row["group"],
            "assignment": row["assignment"],
            "query_object": row["query_object"],
            "query_is_last_mentioned": row["query_is_last_mentioned"],
            "truth": row["answer"],
            "prediction": normalized,
            "valid": valid,
            "correct": valid and normalized == row["answer"],
            "generated": text,
            "token_ids": tokens,
            "intervention": kind,
            "first_answer_token_nll": first_nll,
        }

    def evaluate(self, rows, path, kind="normal"):
        records = []
        start = time.monotonic()
        self.bridge.eval()
        for row in rows:
            records.append(self.predict(row, kind))
        save(path, records)
        return {
            "metrics": summarize(records),
            "seconds": time.monotonic() - start,
            "predictions_sha256": sha(path),
            "first_answer_token_nll": float(
                np.mean([r["first_answer_token_nll"] for r in records])
            ),
        }

    def profile(self):
        profile = []
        probe = selection(self.data["train"], 4 * self.a.batch_size)
        for offset in range(0, len(probe), self.a.batch_size):
            rows = probe[offset : offset + self.a.batch_size]
            self.bridge.zero_grad(set_to_none=True)
            torch.cuda.reset_peak_memory_stats()
            sync()
            start = time.monotonic()
            loss, stats = self.loss(rows)
            sync()
            fwd = time.monotonic() - start
            start = time.monotonic()
            loss.backward()
            sync()
            back = time.monotonic() - start
            grad = self.bridge.core.raw_gain.grad
            finite = all(
                p.grad is None or bool(torch.isfinite(p.grad).all())
                for p in self.bridge.parameters()
            )
            if not finite:
                raise FloatingPointError("Nonfinite joint gradient; training not started")
            outside = grad.detach().clone()
            outside[self.bridge.input_indices] = 0
            profile.append(
                {
                    "ids": [row["id"] for row in rows],
                    "batch_size": len(rows),
                    "forward_seconds": fwd,
                    "backward_seconds": back,
                    "loss": float(loss.detach()),
                    "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                    "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                    "core_grad_norm": float(grad.double().norm()),
                    "core_grad_nonzero": int(torch.count_nonzero(grad)),
                    "non_input_core_grad_nonzero": int(torch.count_nonzero(outside)),
                    **{k: float(v) for k, v in stats.items()},
                }
            )
            del loss, grad, outside
        self.bridge.zero_grad(set_to_none=True)
        self.report["joint_profile"] = profile
        self.persist()
        controls = selection(self.data["validation"], 48)
        self.report["positive_control"] = self.evaluate(
            controls, self.out / "raw_history_control.json", "raw_history"
        )
        self.report["question_only_control"] = self.evaluate(
            controls, self.out / "question_only_control.json", "question_only"
        )
        self.persist()
        if self.report["positive_control"]["metrics"]["all"]["accuracy"] < 0.8:
            raise RuntimeError(
                "Language positive control below80%; do not interpret hybrid failure"
            )
        start = time.monotonic()
        for row in controls[:4]:
            self.predict(row)
        self.report["normal_prediction_seconds"] = (time.monotonic() - start) / 4
        self.persist()
        return float(np.median([p["forward_seconds"] + p["backward_seconds"] for p in profile]))

    def train(self, step_seconds):
        # Fix matched exposure BEFORE arm outcomes. Leave time for validation,
        # both tests, zero-state controls and artifact transfer within wall cap.
        remain = self.deadline - time.monotonic()
        forward_seconds = float(
            np.median([p["forward_seconds"] for p in self.report["joint_profile"]])
        )
        steps = (self.a.steps // 25) * 25
        while steps >= 50:
            evaluations = 2 * (
                (1 + math.ceil(steps / self.a.eval_every)) * self.a.validation_size
                + len(self.data["test"])
            )
            zero_seconds = self.report["question_only_control"]["seconds"] / 48
            estimate = (
                2 * steps * step_seconds * 1.35
                + evaluations * self.report["normal_prediction_seconds"] * 1.2
                + 2 * len(self.data["test"]) * zero_seconds * 1.5
                + 240
            )
            if estimate < remain:
                break
            steps -= 25
        if steps < max(50, self.a.decoder_warmup + 50):
            raise TimeoutError("Insufficient budget for two matched arms including joint training")
        self.report["matched_steps_planned"] = steps
        self.report["arms"] = {}
        self.persist()
        val = selection(self.data["validation"], self.a.validation_size)
        self.report["validation_selection_ids"] = [r["id"] for r in val]
        self.persist()
        for name, train_core in [("trainable_core", True), ("frozen_core", False)]:
            self.bridge.load_trainable_state(self.initial)
            self.bridge.core.raw_gain.requires_grad_(train_core)
            self.bridge.zero_grad(set_to_none=True)
            self.bridge.encoder.requires_grad_(True)
            groups = [
                {
                    "params": list(self.bridge.encoder.parameters()),
                    "lr": self.a.encoder_lr or self.a.lr,
                },
                {
                    "params": [
                        p
                        for n, p in self.bridge.named_parameters()
                        if not n.startswith(("encoder.", "core."))
                    ],
                    "lr": self.a.lr,
                },
            ]
            if train_core:
                groups.append({"params": [self.bridge.core.raw_gain], "lr": self.a.lr})
            optimizer = torch.optim.Adam(groups)
            rng = random.Random(self.a.seed)
            order = list(range(len(self.data["train"])))
            rng.shuffle(order)
            arm = {
                "updates": 0,
                "training": [],
                "validation": [],
                "complete": False,
                "initial_bridge_sha256": state_digest(self.bridge.trainable_state()),
            }
            if arm["initial_bridge_sha256"] != self.report["initial_bridge_sha256"]:
                raise RuntimeError("Matched arm initialization differs")
            self.report["arms"][name] = arm
            folder = self.out / name
            folder.mkdir()
            baseline = self.evaluate(val, folder / "validation_0.json")
            arm["validation"].append({"step": 0, **baseline})
            best = baseline["first_answer_token_nll"]
            torch.save(self.bridge.trainable_state(), folder / "best.pt")
            arm["selected_validation_accuracy"] = baseline["metrics"]["all"]["accuracy"]
            arm["best_step"] = 0
            arm["best_validation_nll"] = best
            arm["selection"] = "minimum validation first-answer-token NLL; no test-based selection"
            for step in range(1, steps + 1):
                self.check_time(reserve=180)
                self.bridge.train()
                optimizer.zero_grad(set_to_none=True)
                warmup = step <= self.a.decoder_warmup
                self.bridge.encoder.requires_grad_(not warmup)
                self.bridge.core.raw_gain.requires_grad_(train_core and not warmup)
                rows = [
                    self.data["train"][order[((step - 1) * self.a.batch_size + i) % len(order)]]
                    for i in range(self.a.batch_size)
                ]
                start = time.monotonic()
                loss, stats = self.loss(rows)
                loss.backward()
                if any(
                    p.grad is not None and not torch.isfinite(p.grad).all()
                    for p in self.bridge.parameters()
                ):
                    raise FloatingPointError("Nonfinite training gradient")
                norms = clip_bridge_gradients(
                    self.bridge, train_core and not warmup, self.a.clip_groups == "separate"
                )
                raw_norm = norms["interface_norm"]
                arm.setdefault("gradient_clipping", []).append({"step": step, **norms})
                optimizer.step()
                sync()
                arm["updates"] = step
                if step == 1 or step % 10 == 0:
                    arm["training"].append(
                        {
                            "step": step,
                            "phase": "decoder_warmup" if warmup else "joint",
                            "loss": float(loss.detach()),
                            "gradient_norm_before_clip": float(raw_norm),
                            "seconds": time.monotonic() - start,
                            **{k: float(v) for k, v in stats.items()},
                        }
                    )
                    print(json.dumps({"arm": name, **arm["training"][-1]}), flush=True)
                    self.persist()
                if step % self.a.eval_every == 0 or step == steps:
                    evaluation = self.evaluate(val, folder / f"validation_{step}.json")
                    arm["validation"].append({"step": step, **evaluation})
                    score = evaluation["first_answer_token_nll"]
                    if score < best:
                        best = score
                        arm["selected_validation_accuracy"] = evaluation["metrics"]["all"][
                            "accuracy"
                        ]
                        arm["best_step"] = step
                        arm["best_validation_nll"] = best
                        torch.save(self.bridge.trainable_state(), folder / "best.pt")
                    self.persist()
            arm["training_complete"] = True
            self.persist()
        # Both selected checkpoints are fixed before ANY held-out test score.
        for name in ["trainable_core", "frozen_core"]:
            arm = self.report["arms"][name]
            folder = self.out / name
            self.bridge.load_trainable_state(
                torch.load(folder / "best.pt", map_location="cpu", weights_only=True)
            )
            arm["test"] = self.evaluate(self.data["test"], folder / "test.json")
            self.persist()
        for name in ["trainable_core", "frozen_core"]:
            arm = self.report["arms"][name]
            folder = self.out / name
            self.bridge.load_trainable_state(
                torch.load(folder / "best.pt", map_location="cpu", weights_only=True)
            )
            arm["test_zero_state"] = self.evaluate(
                self.data["test"], folder / "test_zero_state.json", "zero_state"
            )
            records = json.loads((folder / "test.json").read_text())
            lookup = {(r["group"], r["assignment"], r["query_object"]): r for r in records}
            swapped = []
            for row in records:
                donor = lookup[(row["group"], 1 - row["assignment"], row["query_object"])]
                swapped.append(
                    {
                        **row,
                        "prediction": donor["prediction"],
                        "generated": donor["generated"],
                        "correct": donor["valid"] and donor["prediction"] == row["truth"],
                        "valid": donor["valid"],
                        "token_ids": donor["token_ids"],
                        "first_answer_token_nll": None,
                        "donor_id": donor["id"],
                        "intervention": "counterfactual_state_swap_reuse_same_question_prediction",
                    }
                )
            save(folder / "test_state_swap.json", swapped)
            arm["test_state_swap"] = summarize(swapped)
            arm.update(complete=True, best_checkpoint_sha256=sha(folder / "best.pt"))
            self.persist()
        self.report.update(status="complete", complete=True)
        self.persist()


def main(a):
    pilot = None
    try:
        pilot = Pilot(a)
        step = pilot.profile()
        if a.profile_only:
            pilot.report.update(status="profile_complete", complete=True)
            pilot.persist()
        else:
            pilot.train(step)
    except BaseException as exc:
        if pilot is not None:
            pilot.report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            pilot.persist()
        elif (Path(a.out) / "report.json").exists():
            path = Path(a.out) / "report.json"
            r = json.loads(path.read_text())
            r.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            save(path, r)
        raise


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--eval-every", type=int, default=100)
    p.add_argument("--validation-size", type=int, default=48)
    p.add_argument("--seed", type=int, default=713)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--prefix-tokens", type=int, default=8)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--ticks-per-event", type=int, default=128)
    p.add_argument("--checkpoint-steps", type=int, default=64)
    p.add_argument("--physics-dtype", choices=["float32", "float64"], default="float32")
    p.add_argument("--surrogate-scale", type=float, default=0.1)
    p.add_argument("--max-seconds", type=int, default=5400)
    p.add_argument("--encoder-lr", type=float)
    p.add_argument("--decoder-warmup", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument(
        "--clip-groups", choices=["core_interface", "separate"], default="core_interface"
    )
    p.add_argument("--max-drive-mv", type=float, default=160.0)
    p.add_argument("--profile-only", action="store_true")
    main(p.parse_args())
