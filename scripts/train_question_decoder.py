"""Matched cached-state decoder experiment. The only arm difference is query access."""

import argparse, hashlib, json, math, os, random, time
from pathlib import Path
import numpy as np

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from flypet.question_decoder import CachedStateQuestionDecoder, QUERY_OBJECTS
from flypet.physiology_language import PhysiologicalLanguageBridge, MODEL_ID, MODEL_REVISION
from flypet.physiology_memory_task import LOCATIONS, question_text


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def read_rows(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines()]


def sync():
    torch.cuda.synchronize()


def state_hash(state):
    h = hashlib.sha256()
    for name, value in sorted(state.items()):
        h.update(name.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def teacher_inputs(prefix, q, y, tails, answer_tokens):
    tail = tails[q, y]
    x = torch.cat([prefix.to(tail.dtype), tail], 1)
    labels = torch.full(x.shape[:2], -100, device=x.device, dtype=torch.long)
    labels[:, -answer_tokens.shape[1] :] = answer_tokens[y]
    return x, labels


def inference_inputs(prefix, q, questions):
    return torch.cat([prefix.to(questions.dtype), questions[q]], 1)


def summary(rows):
    out = {}
    for name, rr in [
        ("all", rows),
        ("earlier_object", [r for r in rows if not r["query_is_last_mentioned"]]),
        ("most_recent_object", [r for r in rows if r["query_is_last_mentioned"]]),
    ]:
        out[name] = {
            "n": len(rr),
            "accuracy": sum(r["correct"] for r in rr) / len(rr),
            "valid_fraction": sum(r["valid"] for r in rr) / len(rr),
            "nll": float(np.mean([r["first_answer_token_nll"] for r in rr])),
        }
    pairs = {}
    for r in rows:
        pairs.setdefault((r["group"], r["assignment"]), []).append(r)
    out["both_queries_correct"] = sum(all(r["correct"] for r in rr) for rr in pairs.values()) / len(
        pairs
    )
    return out


class Experiment:
    def __init__(self, a):
        self.a = a
        self.data = Path(a.data)
        self.out = Path(a.out)
        self.out.mkdir(parents=True, exist_ok=False)
        self.start = time.monotonic()
        self.deadline = self.start + a.max_seconds
        self.report = {
            "status": "running",
            "complete": False,
            "seed": a.seed,
            "config": vars(a),
            "arms": {},
            "scope": "Exploratory decoder information-ablation on fixed INITIAL physiological states; no core training",
            "test_status": "This split was evaluated by previous pilot models; not fresh confirmation",
            "history_text_in_primary_lm": False,
            "core_and_encoder_trainable": False,
            "data_manifest_sha256": sha(self.data / "manifest.json"),
            "features_sha256": sha(self.data / "initial_features.npz"),
        }
        self.persist()
        manifest = json.loads((self.data / "manifest.json").read_text())
        for name, digest in manifest["files"].items():
            if sha(self.data / name) != digest:
                raise ValueError(f"Data changed: {name}")
        self.source = json.loads((self.data / "source_config.json").read_text())
        self.cache = np.load(self.data / "initial_features.npz")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA required for real-Qwen experiment")
        torch.set_num_threads(4)
        torch.manual_seed(a.seed)
        np.random.seed(a.seed)
        random.seed(a.seed)
        # Explicitly fixed attention kernel prevents backend changes between paired arms.
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.use_deterministic_algorithms(True)
        self.tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
        self.lm = (
            AutoModelForCausalLM.from_pretrained(
                MODEL_ID, revision=MODEL_REVISION, dtype=torch.bfloat16, attn_implementation="sdpa"
            )
            .cuda()
            .eval()
            .requires_grad_(False)
        )
        self.embedding = self.lm.get_input_embeddings()
        self.dim = self.embedding.weight.shape[1]
        self.train = read_rows(self.data / "tasks/train.jsonl")
        self.validation = read_rows(self.data / "tasks/validation.jsonl")
        self.x = {s: torch.from_numpy(self.cache[s + "_x"]).cuda() for s in ["train", "validation"]}
        self.q = {
            s: torch.from_numpy(self.cache[s + "_q"]).long().cuda() for s in ["train", "validation"]
        }
        self.y = {
            s: torch.from_numpy(self.cache[s + "_y"]).long().cuda() for s in ["train", "validation"]
        }
        self.mean = torch.from_numpy(self.cache["train_mean"]).cuda()
        self.std = torch.from_numpy(self.cache["train_std"]).cuda()
        self.question_ids = [
            self.chat_ids(question_text({"query": f"Where is the {q} now?"})) for q in QUERY_OBJECTS
        ]
        answer_ids = [
            self.tok(x + self.tok.eos_token, add_special_tokens=False)["input_ids"]
            for x in LOCATIONS
        ]
        assert len({len(x) for x in self.question_ids}) == 1, (
            "Pinned task requires equal prompt lengths"
        )
        assert {len(x) for x in answer_ids} == {2}, "Pinned color+EOS tokenization changed"
        assert len({x[0] for x in answer_ids}) == 4
        self.answer_tokens = torch.tensor(answer_ids, device="cuda")
        self.color_ids = self.answer_tokens[:, 0]
        with torch.no_grad():
            self.questions = self.embedding(torch.tensor(self.question_ids, device="cuda")).detach()
            tails = torch.tensor(
                [[q + t for t in answer_ids] for q in self.question_ids], device="cuda"
            )
            self.tails = self.embedding(tails).detach()
        self.report.update(
            model=MODEL_ID,
            revision=MODEL_REVISION,
            gpu=torch.cuda.get_device_name(),
            torch_version=torch.__version__,
            llm_dtype=str(self.embedding.weight.dtype),
            deterministic_algorithms=True,
            attention_backend="math",
            graph_sha256=self.source["graph_sha256"],
            initial_brain_sha256=self.source["initial_bridge_sha256"],
        )
        if a.cache_leader:
            # The source cache was exported with legacy sparse-kernel settings.
            # Freeze and share exact exported bytes; enforce deterministic LM fitting afterwards.
            torch.use_deterministic_algorithms(False)
            try:
                self.export_test_cache()
            finally:
                torch.use_deterministic_algorithms(True)
        torch.manual_seed(a.seed)
        initial = self.make_model(False)
        self.initial = {k: v.detach().cpu().clone() for k, v in initial.state_dict().items()}
        self.report["initial_decoder_sha256"] = state_hash(self.initial)
        torch.save(initial.make_checkpoint(), self.out / "initial_decoder.pt")
        second = self.make_model(True)
        second.load_state_dict(self.initial)
        with torch.no_grad():
            torch.testing.assert_close(
                initial(self.x["train"][:4], self.q["train"][:4]),
                second(self.x["train"][:4], self.q["train"][:4]),
                rtol=0,
                atol=0,
            )
        self.report["initial_arm_prefix_equivalence"] = True
        self.report["nominal_decoder_parameters"] = sum(p.numel() for p in initial.parameters())
        self.report["plain_inactive_query_parameters"] = 3 * initial.hidden
        self.report["profile"] = self.profile(initial)
        del initial, second
        torch.cuda.empty_cache()
        self.persist()

    def persist(self):
        self.report["elapsed_seconds"] = time.monotonic() - self.start
        save(self.out / "report.json", self.report)

    def check_time(self, reserve=0):
        if time.monotonic() + reserve >= self.deadline:
            raise TimeoutError("Experiment deadline reached")

    def chat_ids(self, text):
        return self.tok.apply_chat_template(
            [{"role": "user", "content": text}],
            tokenize=True,
            return_dict=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

    def make_model(self, conditioned):
        return CachedStateQuestionDecoder(
            self.x["train"].shape[1],
            self.dim,
            self.source["embedding_std"],
            conditioned=conditioned,
            feature_mean=self.mean,
            feature_std=self.std,
        ).cuda()

    def forward_lm(self, **kwargs):
        return self.lm(**kwargs)

    def loss(self, model, indices):
        q = self.q["train"][indices]
        y = self.y["train"][indices]
        prefix = model(self.x["train"][indices], q)
        inputs, labels = teacher_inputs(prefix, q, y, self.tails, self.answer_tokens)
        return self.forward_lm(
            inputs_embeds=inputs,
            attention_mask=torch.ones_like(labels),
            labels=labels,
            use_cache=False,
        ).loss

    def export_test_cache(self):
        # Sole exporter; no evaluation answers are parsed in this function.
        cfg = self.source["config"]
        g = np.load(self.data / "graph.npz")
        bridge = (
            PhysiologicalLanguageBridge(
                g,
                self.dim,
                self.source["embedding_std"],
                tokens=cfg["prefix_tokens"],
                hidden=cfg["hidden"],
                ticks_per_event=cfg["ticks_per_event"],
                checkpoint_steps=cfg["checkpoint_steps"],
                physics_dtype=cfg["physics_dtype"],
                surrogate_scale=cfg["surrogate_scale"],
                max_drive_mv=cfg["max_drive_mv"],
            )
            .cuda()
            .eval()
            .requires_grad_(False)
        )
        bridge.load_trainable_state(
            torch.load(self.data / "initial.pt", map_location="cpu", weights_only=True)
        )
        events = {}
        with torch.no_grad():
            for row in self.train:
                for text in row["history"]:
                    if text not in events:
                        ids = torch.tensor(
                            self.tok(text, add_special_tokens=False)["input_ids"], device="cuda"
                        )
                        events[text] = self.embedding(ids).float().mean(0).detach()
        features = []
        hook = bridge.decoder[0].register_forward_pre_hook(
            lambda mod, args: features.append(args[0].detach().cpu().numpy().copy())
        )

        def calculate(rows):
            histories = list(dict.fromkeys(tuple(r["history"]) for r in rows))
            result = {}
            with torch.no_grad():
                for offset in range(0, len(histories), 16):
                    self.check_time(240)
                    hh = histories[offset : offset + 16]
                    features.clear()
                    e = torch.stack([torch.stack([events[t] for t in h]) for h in hh])
                    bridge(e)
                    for h, x in zip(hh, features[-1]):
                        result[h] = x
            return np.stack([result[tuple(r["history"])] for r in rows])

        check_rows = []
        seen = set()
        for r in self.train:
            if tuple(r["history"]) not in seen:
                seen.add(tuple(r["history"]))
                check_rows.append(r)
            if len(check_rows) == 16:
                break
        computed = calculate(check_rows)
        lookup = {r["id"]: i for i, r in enumerate(self.train)}
        expected = np.stack([self.cache["train_x"][lookup[r["id"]]] for r in check_rows])
        np.testing.assert_allclose(computed, expected, rtol=1e-4, atol=1e-3)
        check = {
            "n": 16,
            "max_abs": float(np.max(abs(computed - expected))),
            "rtol": 1e-4,
            "atol": 1e-3,
        }
        test_rows = read_rows(self.data / "tasks/test_features.jsonl")
        if any("answer" in r for r in test_rows):
            raise ValueError("Target leaked into feature export input")
        test_x = calculate(test_rows)
        hook.remove()
        path = self.data / "test_features.npz"
        temp = path.with_suffix(".tmp.npz")
        np.savez_compressed(temp, x=test_x, id=np.asarray([r["id"] for r in test_rows]))
        temp.replace(path)
        meta = {
            "sha256": sha(path),
            "graph_sha256": self.source["graph_sha256"],
            "initial_file_sha256": self.source["initial_file_sha256"],
            "test_input_sha256": sha(self.data / "tasks/test_features.jsonl"),
            "train_recompute_check": check,
            "rows": len(test_x),
            "scope": "No supervision-label field consumed; history contains legitimate task facts; exported states shared byte-for-byte",
            "sparse_export_deterministic_algorithms": False,
        }
        save(self.data / "test_cache_ready.json", meta)
        save(self.out / "test_cache_provenance.json", meta)
        self.report["cache_export"] = meta
        self.persist()

    def profile(self, model):
        values = []
        indices = torch.arange(self.a.batch_size, device="cuda")
        for _ in range(3):
            model.zero_grad(set_to_none=True)
            sync()
            torch.cuda.reset_peak_memory_stats()
            start = time.monotonic()
            loss = self.loss(model, indices)
            loss.backward()
            sync()
            if not all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()):
                raise FloatingPointError("Nonfinite profile gradients")
            values.append(
                {
                    "seconds": time.monotonic() - start,
                    "loss": float(loss.detach()),
                    "cuda_peak_bytes": torch.cuda.max_memory_allocated(),
                }
            )
        model.zero_grad(set_to_none=True)
        estimate = 2 * self.a.steps * np.median([x["seconds"] for x in values]) * 1.20 + 150
        if not self.a.require_training_plan and time.monotonic() + estimate >= self.deadline:
            raise TimeoutError(
                f"Fixed two-arm recipe needs about{estimate:.0f}s; lease cannot fit it"
            )
        return {"measurements": values, "fixed_recipe_estimate_seconds": float(estimate)}

    def token_metrics(self, model, split):
        model.eval()
        nll = []
        correct = []
        with torch.no_grad():
            for start in range(0, len(self.x[split]), self.a.batch_size):
                self.check_time(60)
                xx = self.x[split][start : start + self.a.batch_size]
                q = self.q[split][start : start + self.a.batch_size]
                y = self.y[split][start : start + self.a.batch_size]
                inputs = inference_inputs(model(xx, q), q, self.questions)
                logits = (
                    self.forward_lm(
                        inputs_embeds=inputs,
                        attention_mask=torch.ones(
                            inputs.shape[:2], device="cuda", dtype=torch.long
                        ),
                        use_cache=False,
                    )
                    .logits[:, -1]
                    .float()
                )
                lp = logits.log_softmax(-1)
                nll.extend(
                    (-lp[torch.arange(len(q), device="cuda"), self.color_ids[y]]).cpu().tolist()
                )
                correct.extend((logits.argmax(-1) == self.color_ids[y]).cpu().tolist())
        return {
            "n": len(nll),
            "nll": float(np.mean(nll)),
            "next_token_accuracy": float(np.mean(correct)),
            "metric_scope": "Unconstrained first-token accuracy; not full strict generated-answer accuracy",
        }

    def predict(self, model, features, rows, intervention="normal"):
        output = []
        model.eval()
        with torch.no_grad():
            for start in range(0, len(rows), self.a.batch_size):
                self.check_time(20)
                rr = rows[start : start + self.a.batch_size]
                q = torch.tensor(
                    [QUERY_OBJECTS.index(r["query_object"]) for r in rr], device="cuda"
                )
                prefix = model(features[start : start + len(rr)], q)
                inputs = inference_inputs(prefix, q, self.questions)
                result = self.lm.generate(
                    inputs_embeds=inputs,
                    attention_mask=torch.ones(inputs.shape[:2], device="cuda", dtype=torch.long),
                    max_new_tokens=8,
                    do_sample=False,
                    pad_token_id=self.tok.eos_token_id,
                    return_dict_in_generate=True,
                    output_scores=True,
                )
                tokens = result.sequences[:, -len(result.scores) :].cpu().tolist()
                lp = result.scores[0].float().log_softmax(-1)[:, self.color_ids].cpu().tolist()
                for row, ids, prob in zip(rr, tokens, lp):
                    text = self.tok.decode(ids, skip_special_tokens=True).strip()
                    pred = text.lower().rstrip(".!").strip()
                    valid = pred in LOCATIONS
                    output.append(
                        {
                            "id": row["id"],
                            "group": row["group"],
                            "assignment": row["assignment"],
                            "query_object": row["query_object"],
                            "query_is_last_mentioned": row["query_is_last_mentioned"],
                            "truth": row["answer"],
                            "prediction": pred,
                            "valid": valid,
                            "correct": valid and pred == row["answer"],
                            "generated": text,
                            "token_ids": ids,
                            "answer_log_probs": prob,
                            "first_answer_token_nll": -prob[LOCATIONS.index(row["answer"])],
                            "intervention": intervention,
                        }
                    )
        return output

    def fit(self):
        if self.a.require_training_plan:
            plan_path = self.data / "training_plan.json"
            while not plan_path.exists():
                self.check_time(180)
                time.sleep(2)
            plan = json.loads(plan_path.read_text())
            if (
                plan["selection"] != "timing_only"
                or plan["batch_size"] != self.a.batch_size
                or plan["steps"] not in [1000, 1500, 2000]
                or plan["steps"] > self.a.steps
            ):
                raise ValueError("Unexpected shared timing plan")
            self.report["requested_steps"] = self.a.steps
            self.a.steps = plan["steps"]
            self.report["training_plan_sha256"] = sha(plan_path)
            self.report["matched_steps"] = self.a.steps
            save(self.out / "training_plan.json", plan)
            self.persist()
        order = list(range(len(self.train)))
        random.Random(self.a.seed).shuffle(order)
        names = ["plain", "conditioned"] if self.a.seed % 2 else ["conditioned", "plain"]
        self.report["arm_order"] = names
        self.report["training_order_sha256"] = hashlib.sha256(
            np.asarray(order, dtype=np.int64).tobytes()
        ).hexdigest()
        for name in names:
            model = self.make_model(name == "conditioned")
            model.load_state_dict(self.initial)
            if state_hash(model.state_dict()) != self.report["initial_decoder_sha256"]:
                raise RuntimeError("Initial decoder mismatch")
            folder = self.out / name
            folder.mkdir()
            optimizer = torch.optim.Adam(model.parameters(), lr=self.a.lr)
            arm = {
                "updates": 0,
                "training": [],
                "validation": [],
                "complete": False,
                "initial_decoder_sha256": state_hash(model.state_dict()),
            }
            self.report["arms"][name] = arm
            best = float("inf")
            for step in range(self.a.steps + 1):
                if step:
                    self.check_time(90)
                    model.train()
                    optimizer.zero_grad(set_to_none=True)
                    ids = torch.tensor(
                        [
                            order[((step - 1) * self.a.batch_size + i) % len(order)]
                            for i in range(self.a.batch_size)
                        ],
                        device="cuda",
                    )
                    loss = self.loss(model, ids)
                    loss.backward()
                    norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), 1.0, error_if_nonfinite=True
                    )
                    optimizer.step()
                    arm["updates"] = step
                    if step == 1 or step % 100 == 0:
                        arm["training"].append(
                            {
                                "step": step,
                                "loss": float(loss.detach()),
                                "gradient_norm": float(norm),
                            }
                        )
                        print(
                            json.dumps({"seed": self.a.seed, "arm": name, **arm["training"][-1]}),
                            flush=True,
                        )
                        self.persist()
                if step % self.a.eval_every == 0 or step == self.a.steps:
                    metrics = self.token_metrics(model, "validation")
                    arm["validation"].append({"step": step, **metrics})
                    if metrics["nll"] < best:
                        best = metrics["nll"]
                        arm["best_step"] = step
                        arm["best_validation_nll"] = best
                        torch.save(model.make_checkpoint(), folder / "best.pt")
                    self.persist()
            arm["training_complete"] = True
            arm["checkpoint_sha256"] = sha(folder / "best.pt")
            self.persist()
            del model, optimizer
            torch.cuda.empty_cache()
        # All checkpoint choices for this seed are now fixed; no test labels were used above.
        marker = self.data / "test_cache_ready.json"
        while not marker.exists():
            self.check_time(60)
            time.sleep(2)
        meta = json.loads(marker.read_text())
        path = self.data / "test_features.npz"
        if (
            sha(path) != meta["sha256"]
            or meta["graph_sha256"] != self.source["graph_sha256"]
            or meta["initial_file_sha256"] != self.source["initial_file_sha256"]
        ):
            raise ValueError("Shared test feature cache identity mismatch")
        cached = np.load(path)
        rows = read_rows(self.data / "tasks/test_evaluation.jsonl")
        if cached["id"].tolist() != [r["id"] for r in rows]:
            raise ValueError("Test cache order mismatch")
        features = torch.from_numpy(cached["x"]).cuda()
        self.report["test_features_sha256"] = sha(path)
        for name in names:
            folder = self.out / name
            arm = self.report["arms"][name]
            model = (
                CachedStateQuestionDecoder.from_checkpoint(
                    torch.load(folder / "best.pt", map_location="cpu", weights_only=True)
                )
                .cuda()
                .eval()
            )
            arm["train_selected"] = self.token_metrics(model, "train")
            arm["validation_selected"] = self.token_metrics(model, "validation")
            normal = self.predict(model, features, rows)
            save(folder / "test.json", normal)
            arm["test"] = summary(normal)
            # Three unique questions exhaust the deterministic train-mean-state control.
            examples = [next(r for r in rows if r["query_object"] == obj) for obj in QUERY_OBJECTS]
            control = self.predict(model, self.mean[None].expand(3, -1), examples, "mean_state")
            by_query = {r["query_object"]: r for r in control}
            mean_rows = []
            for row in normal:
                donor = by_query[row["query_object"]]
                lp = donor["answer_log_probs"]
                mean_rows.append(
                    {
                        **row,
                        **{
                            k: donor[k]
                            for k in [
                                "prediction",
                                "generated",
                                "token_ids",
                                "valid",
                                "answer_log_probs",
                            ]
                        },
                        "correct": donor["valid"] and donor["prediction"] == row["truth"],
                        "first_answer_token_nll": -lp[LOCATIONS.index(row["truth"])],
                        "intervention": "mean_state",
                    }
                )
            save(folder / "mean_state.json", mean_rows)
            arm["mean_state"] = summary(mean_rows)
            lookup = {(r["group"], r["assignment"], r["query_object"]): r for r in normal}
            swapped = []
            for row in normal:
                donor = lookup[(row["group"], 1 - row["assignment"], row["query_object"])]
                lp = donor["answer_log_probs"]
                swapped.append(
                    {
                        **row,
                        **{
                            k: donor[k]
                            for k in [
                                "prediction",
                                "generated",
                                "token_ids",
                                "valid",
                                "answer_log_probs",
                            ]
                        },
                        "correct": donor["valid"] and donor["prediction"] == row["truth"],
                        "first_answer_token_nll": -lp[LOCATIONS.index(row["truth"])],
                        "donor_id": donor["id"],
                        "intervention": "paired_state_swap",
                    }
                )
            save(folder / "swapped_state.json", swapped)
            arm["swapped_state"] = summary(swapped)
            arm["complete"] = True
            self.persist()
            del model
            torch.cuda.empty_cache()
        self.report.update(status="complete", complete=True)
        self.persist()


def main(a):
    run = None
    try:
        # Math SDPA is slower but deterministic at this short context length.
        from torch.nn.attention import sdpa_kernel, SDPBackend

        with sdpa_kernel(SDPBackend.MATH):
            run = Experiment(a)
            run.fit()
    except BaseException as exc:
        if run is not None:
            run.report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            run.persist()
        elif (Path(a.out) / "report.json").exists():
            p = Path(a.out) / "report.json"
            r = json.loads(p.read_text())
            r.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            save(p, r)
        raise


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--eval-every", type=int, default=250)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--max-seconds", type=int, default=1400)
    p.add_argument("--cache-leader", action="store_true")
    p.add_argument("--require-training-plan", action="store_true")
    args = p.parse_args()
    if min(args.steps, args.batch_size, args.eval_every, args.max_seconds) < 1:
        raise ValueError("Positive experiment bounds required")
    main(args)
