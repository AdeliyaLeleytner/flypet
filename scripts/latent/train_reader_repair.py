#!/usr/bin/env python3
"""Warm-started core-reading repair with signed supervision and factual-token loss."""

import argparse, json, re, time, sys, random
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch
from torch.nn import functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, set_peft_model_state_dict
from safetensors.torch import load_file
from flypet.latent_dialogue import (
    FocusedDialogueReader,
    CodecDialogueReader,
    CORE_SYSTEM,
    prompt_parts,
    comparison_prompt_parts,
    current_prompt_parts,
)
from flypet.latent_reader_data import score, parse_reply
from flypet.neural_records import file_hash, write_json
from flypet.latent_questions import QUESTIONS as PARAPHRASES

MODEL = "Qwen/Qwen3-4B"
REVISION = "1cfa9a7208912126459214e8b04321603b3df60c"


def answer_tokens(tokenizer, text):
    encoded = tokenizer(
        text + tokenizer.eos_token, add_special_tokens=False, return_offsets_mapping=True
    )
    spans = [
        m.span()
        for m in re.finditer(
            r"[+-]?\d+(?:\.\d+)?|\b(?:approach|avoidance|balanced)\b|no resolved change|no change|no MBON output spikes|no approach or avoidance output",
            text,
            re.I,
        )
    ]
    weights = [
        5.0 if any(b > start and a < end for start, end in spans) else 1.0
        for a, b in encoded["offset_mapping"]
    ]
    return list(encoded["input_ids"]), weights


def balanced_error(errors, values):
    groups = [
        np.asarray(errors)[np.asarray(values) < 0],
        np.asarray(errors)[np.asarray(values) >= 0],
    ]
    return float(np.mean([g.mean() for g in groups if len(g)]))


def parse_change(text):
    direct = re.search(r"changed by\s+([+-]?\d+(?:\.\d+)?)", text)
    if direct:
        delta = float(direct.group(1))
        return np.array([0.0, delta]) if np.isfinite(delta) and abs(delta) <= 2 else None
    match = re.search(r"from\s+([+-]?\d+(?:\.\d+)?)\s+to\s+([+-]?\d+(?:\.\d+)?)", text)
    if not match:
        return None
    values = np.array([float(v) for v in match.groups()])
    return values if np.isfinite(values).all() and (abs(values) <= 1).all() else None


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "init", "out"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--extra-mbon-tokens", type=int, choices=[0, 4], default=0)
    p.add_argument("--seed", type=int, default=934)
    p.add_argument("--max-seconds", type=int, default=4000)
    p.add_argument("--codec", type=Path)
    p.add_argument("--paraphrases", action="store_true")
    p.add_argument("--separate-context", action="store_true")
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    torch.set_num_threads(4)
    torch.manual_seed(a.seed)
    random.seed(a.seed)

    def deadline():
        if time.monotonic() - start > a.max_seconds:
            raise TimeoutError("Reader repair deadline")

    manifest = json.loads((a.data / "manifest.json").read_text())
    for name, digest in manifest["files"].items():
        if file_hash(a.data / name) != digest:
            raise ValueError("Dataset hash mismatch")
    initial = json.loads((a.init / "manifest.json").read_text())
    for name, digest in initial["files"].items():
        if file_hash(a.init / name) != digest:
            raise ValueError("Initial bundle hash mismatch")
    states = [json.loads(line) for line in (a.data / "states.jsonl").read_text().splitlines()]
    qa = [
        json.loads(line)
        for line in (a.data / "qa.jsonl").read_text().splitlines()
        if json.loads(line)["kind"] in ("audit", "summary", "change")
    ]
    with np.load(a.data / "features.npz", allow_pickle=False) as z:
        x = z["x"]
        pop = z["population_x"]
        aux = z["auxiliary"]
        targets = z["targets"]
        metadata = z["metadata"]
        pre = {k: z[k] for k in z.files if k not in ("x", "population_x", "auxiliary", "targets")}
    mbon = np.flatnonzero(
        pre["population_ids"][pre["neuron_indices"]] == manifest["population_names"].index("MBON")
    )
    # Align verbal direction with the precision actually displayed to the user.
    for row in qa:
        if row["kind"] == "change":
            index = row["state_index"]
            base = states[index]["baseline_index"]
            before = round(float(targets[base, 2]), 2)
            after = round(float(targets[index, 2]), 2)
            before = 0.0 if before == 0 else before
            after = 0.0 if after == 0 else after
            direction = (
                "shifting toward approach relative to the reference observation"
                if after > before
                else "shifting toward avoidance relative to the reference observation"
                if after < before
                else "with no resolved change at this precision"
            )
            row["answer"] = (
                f"The measured balance changed from {before:+.2f} to {after:+.2f}, {direction}."
            )
            if a.separate_context:
                difference = round(float(targets[index, 2] - targets[base, 2]), 2)
                difference = 0.0 if difference == 0 else difference
                direction = (
                    "toward approach"
                    if difference > 0
                    else "toward avoidance"
                    if difference < 0
                    else "with no resolved change at this precision"
                )
                row["answer"] = f"The neural balance changed by {difference:+.2f}, {direction}."
        elif a.paraphrases and row["kind"] == "summary":
            value = targets[row["state_index"]]
            if value[0] + value[1] == 0:
                row["answer"] = (
                    "There is no approach or avoidance output signal in this observation."
                )
            elif value[2] > 0:
                row["answer"] = "My current neural response leans toward approach."
            elif value[2] < 0:
                row["answer"] = "My current neural response leans toward avoidance."
            else:
                row["answer"] = (
                    "My approach-related and avoidance-related outputs are balanced in this observation."
                )
    if a.paraphrases:
        for row in list(qa):
            if row["kind"] != "change":
                continue
            index = row["state_index"]
            before = round(float(targets[states[index]["baseline_index"], 2]), 2)
            after = round(float(targets[index, 2]), 2)
            if a.separate_context:
                before = 0.0
                after = round(
                    float(targets[index, 2] - targets[states[index]["baseline_index"], 2]), 2
                )
            answer = (
                "Relative to the reference, the neural balance shifted toward approach."
                if after > before
                else "Relative to the reference, the neural balance shifted toward avoidance."
                if after < before
                else "No change in the neural balance is resolved at the reported precision."
            )
            qa.append(
                {
                    **row,
                    "kind": "change_qualitative",
                    "question": PARAPHRASES["change_qualitative"][0],
                    "answer": answer,
                }
            )
    train_states = np.array([r["split"] == "train" for r in states])
    y = np.c_[np.log1p(targets[:, :2]), targets[:, 2]]
    ym = y[train_states].mean(0)
    ys = np.maximum(y[train_states].std(0), 0.1)
    y = (y - ym) / ys
    np.savez_compressed(a.out / "preprocessing.npz", **pre)
    write_json(a.out / "dataset_manifest.json", manifest)
    write_json(a.out / "repair_qa.json", qa)
    tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    tok.pad_token = tok.eos_token
    base = (
        AutoModelForCausalLM.from_pretrained(
            MODEL, revision=REVISION, dtype=torch.bfloat16, attn_implementation="sdpa"
        )
        .cuda()
        .eval()
        .requires_grad_(False)
    )
    llm = PeftModel.from_pretrained(base, a.init / "reader_lora", is_trainable=True).eval()
    emb = llm.get_input_embeddings()
    old = json.loads((a.init / "reader_config.json").read_text())
    fields = [
        "n_neurons",
        "channels",
        "metadata",
        "n_populations",
        "embedding_dim",
        "embedding_std",
        "tokens",
        "width",
        "auxiliary_dim",
    ]
    kwargs = {k: old[k] for k in fields}
    kwargs.update(mbon_positions=mbon.tolist(), extra_mbon_tokens=a.extra_mbon_tokens)
    if a.codec:
        codec_report = json.loads((a.codec / "report.json").read_text())
        codec_config = json.loads((a.codec / "config.json").read_text())
        if codec_report["dataset_manifest_sha256"] != file_hash(a.data / "manifest.json"):
            raise ValueError("Codec trained with different preprocessing")
        if codec_config["mbon_positions"] != mbon.tolist():
            raise ValueError("Codec MBON order differs")
        reader = CodecDialogueReader(
            **kwargs, codec_tokens=codec_config["tokens"], codec_width=codec_config["width"]
        ).cuda()
    else:
        reader = FocusedDialogueReader(**kwargs).cuda()
    transferred = reader.load_state_dict(
        load_file(str(a.init / "reader.safetensors")), strict=False
    )
    if transferred.unexpected_keys:
        raise ValueError("Unexpected warm-start tensors")
    if a.codec:
        reader.codec.load_state_dict(load_file(str(a.codec / "codec.safetensors")))
        if old.get("architecture") != "codec":
            reader.codec_projection.load_state_dict(reader.output.state_dict())
            reader.semantic_projection.load_state_dict(reader.output.state_dict())
            reader.delta_projection.load_state_dict(reader.output.state_dict())
        codec_initial = {k: v.detach().cpu().clone() for k, v in reader.codec.state_dict().items()}
    formatter = comparison_prompt_parts if a.codec else prompt_parts

    def format_question(kind, question):
        if a.separate_context and not kind.startswith("change"):
            return current_prompt_parts(tok, question)
        return formatter(tok, question, CORE_SYSTEM)

    parts = [format_question(r["kind"], r["question"]) for r in qa]
    variants = (
        {
            kind: [format_question(kind, q) for q in questions]
            for kind, questions in PARAPHRASES.items()
        }
        if a.paraphrases
        else {}
    )
    answers = [answer_tokens(tok, r["answer"]) for r in qa]
    splits = {
        s: np.array([i for i, r in enumerate(qa) if r["split"] == s])
        for s in ("train", "validation", "development_test")
    }
    probabilities = np.zeros(len(qa))
    sampling = {}
    kinds = (
        ("audit", "summary", "change", "change_qualitative")
        if a.paraphrases
        else ("audit", "summary", "change")
    )
    for kind in kinds:
        ids = np.array([i for i in splits["train"] if qa[i]["kind"] == kind])
        values = np.array(
            [
                targets[qa[i]["state_index"], 2]
                - (
                    targets[states[qa[i]["state_index"]]["baseline_index"], 2]
                    if kind.startswith("change")
                    else 0
                )
                for i in ids
            ]
        )
        sampling[kind] = {}
        for sign, mask in [("negative", values < 0), ("nonnegative", values >= 0)]:
            selected = ids[mask]
            sampling[kind][sign] = len(selected)
            if len(selected):
                probabilities[selected] = 1 / len(selected)
    probabilities /= probabilities.sum()
    rng = np.random.default_rng(a.seed)

    def encode(indices, donors=None):
        ids = (
            np.array([qa[i]["state_index"] for i in indices])
            if donors is None
            else np.array(donors)
        )
        refs = np.array([states[i]["baseline_index"] for i in ids])
        joined = np.r_[refs, ids]
        features = torch.tensor(x[joined], device="cuda")
        soft, global_aux, readout, mbon_pred, latent = reader.forward_extended(
            features, torch.tensor(pop[joined], device="cuda")
        )
        delta = reader.pair_difference(features) if a.codec else None
        return (
            soft[: len(ids)],
            soft[len(ids) :],
            global_aux,
            readout,
            mbon_pred,
            latent,
            delta,
            joined,
        )

    def loss_batch(indices):
        ref, cur, ga, readout, mbon_pred, latent, delta_token, joined = encode(indices)
        vectors = []
        labels = []
        weights = []
        for j, i in enumerate(indices):
            part = (
                variants[qa[i]["kind"]][int(rng.integers(len(variants[qa[i]["kind"]])))]
                if a.paraphrases
                else parts[i]
            )
            right = part[-1]
            answer, fact_weights = answers[i]
            if len(part) == 2:
                pieces = [emb(torch.tensor(part[0], device="cuda")), cur[j].to(emb.weight.dtype)]
            else:
                pieces = [
                    emb(torch.tensor(part[0], device="cuda")),
                    ref[j].to(emb.weight.dtype),
                    emb(torch.tensor(part[1], device="cuda")),
                    cur[j].to(emb.weight.dtype),
                ]
                if a.codec:
                    pieces.extend(
                        [
                            emb(torch.tensor(part[2], device="cuda")),
                            delta_token[j].to(emb.weight.dtype),
                        ]
                    )
            pieces.append(emb(torch.tensor(right + answer, device="cuda")))
            vector = torch.cat(pieces)
            vectors.append(vector)
            prefix = len(vector) - len(answer)
            labels.append(torch.tensor([-100] * prefix + answer, device="cuda"))
            weights.append(torch.tensor([0.0] * prefix + fact_weights, device="cuda"))
        width = max(map(len, vectors))
        batch = emb(
            torch.full((len(indices), width), tok.pad_token_id, device="cuda", dtype=torch.long)
        )
        mask = torch.zeros((len(indices), width), device="cuda", dtype=torch.long)
        label = torch.full_like(mask, -100)
        weight = torch.zeros_like(mask, dtype=torch.float32)
        for j, (v, l, w) in enumerate(zip(vectors, labels, weights)):
            batch[j, : len(v)] = v
            mask[j, : len(v)] = 1
            label[j, : len(v)] = l
            weight[j, : len(v)] = w
        output = llm(inputs_embeds=batch, attention_mask=mask, use_cache=False)
        ce = F.cross_entropy(
            output.logits[:, :-1].float().reshape(-1, output.logits.shape[-1]),
            label[:, 1:].reshape(-1),
            reduction="none",
        ).reshape(len(indices), -1)
        lm = (ce * weight[:, 1:]).sum() / weight[:, 1:].sum()
        goal = torch.tensor(y[joined], device="cuda", dtype=torch.float32)
        n = len(indices)
        delta = reader.change_head(torch.cat((latent[:n], latent[n:]), dim=1)).squeeze(-1)
        signed = F.mse_loss(readout, goal)
        change = F.mse_loss(delta, goal[n:, 2] - goal[:n, 2])
        global_loss = F.mse_loss(ga, torch.tensor(aux[joined], device="cuda"))
        loss = lm + 0.5 * signed + 0.5 * change + 0.05 * global_loss
        if mbon_pred is not None:
            loss = loss + 0.2 * F.mse_loss(
                mbon_pred, torch.tensor(x[joined][:, mbon, 5], device="cuda")
            )
        return loss, float(lm.detach()), float(signed.detach()), float(change.detach())

    @torch.no_grad()
    def generate(indices, donors=None):
        reader.eval()
        llm.eval()
        texts = []
        for k in range(0, len(indices), a.batch):
            deadline()
            group = indices[k : k + a.batch]
            ref, cur, _, _, _, _, delta_token, _ = encode(
                group, None if donors is None else donors[k : k + len(group)]
            )
            vectors = []
            for j, i in enumerate(group):
                part = parts[i]
                right = part[-1]
                if len(part) == 2:
                    pieces = [
                        emb(torch.tensor(part[0], device="cuda")),
                        cur[j].to(emb.weight.dtype),
                    ]
                else:
                    pieces = [
                        emb(torch.tensor(part[0], device="cuda")),
                        ref[j].to(emb.weight.dtype),
                        emb(torch.tensor(part[1], device="cuda")),
                        cur[j].to(emb.weight.dtype),
                    ]
                    if a.codec:
                        pieces.extend(
                            [
                                emb(torch.tensor(part[2], device="cuda")),
                                delta_token[j].to(emb.weight.dtype),
                            ]
                        )
                pieces.append(emb(torch.tensor(right, device="cuda")))
                vectors.append(torch.cat(pieces))
            width = max(map(len, vectors))
            batch = emb(
                torch.full((len(group), width), tok.pad_token_id, device="cuda", dtype=torch.long)
            )
            mask = torch.zeros(batch.shape[:2], device="cuda", dtype=torch.long)
            for j, v in enumerate(vectors):
                batch[j, width - len(v) :] = v
                mask[j, width - len(v) :] = 1
            out = llm.generate(
                inputs_embeds=batch,
                attention_mask=mask,
                max_new_tokens=100,
                do_sample=False,
                pad_token_id=tok.pad_token_id,
                eos_token_id=tok.eos_token_id,
            )
            texts.extend(tok.batch_decode(out, skip_special_tokens=True))
        return texts

    def fact_selection():
        audit = np.array([i for i in splits["validation"] if qa[i]["kind"] == "audit"])
        change = np.array([i for i in splits["validation"] if qa[i]["kind"] == "change"])
        numeric = generate(audit)
        changes = generate(change)
        value = np.array([targets[qa[i]["state_index"], 2] for i in audit])
        errors = []
        for text, target in zip(numeric, value):
            parsed = parse_reply(text)
            errors.append(abs(parsed[2] - target) if parsed is not None else 2.0)
        delta = np.array(
            [
                targets[qa[i]["state_index"], 2]
                - targets[states[qa[i]["state_index"]]["baseline_index"], 2]
                for i in change
            ]
        )
        delta_errors = []
        for text, target in zip(changes, delta):
            parsed = parse_change(text)
            delta_errors.append(abs(parsed[1] - parsed[0] - target) if parsed is not None else 4.0)
        current = balanced_error(errors, value)
        difference = balanced_error(delta_errors, delta)
        return current + 0.5 * difference, {
            "balanced_current_mae": current,
            "balanced_delta_mae": difference,
            "ordinary_current_mae": float(np.mean(errors)),
            "ordinary_delta_mae": float(np.mean(delta_errors)),
        }

    optimizer = torch.optim.AdamW(
        [
            {"params": reader.parameters(), "lr": 0.0003},
            {"params": [p for p in llm.parameters() if p.requires_grad], "lr": 0.00003},
        ],
        weight_decay=0.01,
    )
    report = {
        "status": "training",
        "model": MODEL,
        "revision": REVISION,
        "initial_bundle_sha256": file_hash(a.init / "manifest.json"),
        "dataset_manifest_sha256": file_hash(a.data / "manifest.json"),
        "repair_qa_sha256": file_hash(a.out / "repair_qa.json"),
        "config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()},
        "sampling_counts": sampling,
        "warm_start_missing_keys": transferred.missing_keys,
        "selection": "validation balanced current MAE + 0.5 balanced change MAE; invalid replies receive maximum error",
        "scope": "core current/changed readout repair; development data reused; no odor or temporal-bin capability claimed",
    }
    if a.paraphrases:
        report["training_question_variants"] = PARAPHRASES
        report["summary_omits_exact_counts"] = True
    write_json(a.out / "report.json", report)
    best = float("inf")
    history = []
    if a.codec:
        report["codec_sha256"] = file_hash(a.codec / "codec.safetensors")
    for epoch in range(a.epochs):
        reader.train()
        llm.eval()
        indices = rng.choice(len(qa), size=len(splits["train"]), replace=True, p=probabilities)
        losses = []
        t = time.monotonic()
        for k in range(0, len(indices), a.batch):
            deadline()
            optimizer.zero_grad(set_to_none=True)
            loss, *diagnostics = loss_batch(indices[k : k + a.batch])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(reader.parameters(), 1)
            torch.nn.utils.clip_grad_norm_([p for p in llm.parameters() if p.requires_grad], 1)
            optimizer.step()
            losses.append(diagnostics)
        criterion, validation = fact_selection()
        row = {
            "epoch": epoch + 1,
            "validation": validation,
            "selection_value": criterion,
            "train_losses": np.mean(losses, axis=0).tolist(),
            "wall_s": time.monotonic() - t,
        }
        history.append(row)
        if criterion < best:
            best = criterion
            selected = epoch + 1
            checkpoint = {k: old[k] for k in fields}
            checkpoint.update(
                state_dict=reader.state_dict(),
                architecture="focused",
                mbon_positions=mbon.tolist(),
                extra_mbon_tokens=a.extra_mbon_tokens,
                system=CORE_SYSTEM,
                epoch=selected,
            )
            if a.codec:
                checkpoint.update(
                    architecture="codec",
                    codec_tokens=codec_config["tokens"],
                    codec_width=codec_config["width"],
                )
            checkpoint["separate_context"] = a.separate_context
            torch.save(checkpoint, a.out / "reader.pt")
            llm.save_pretrained(a.out / "lora")
            write_json(
                a.out / "best_checkpoint.json",
                {
                    "epoch": selected,
                    "reader_sha256": file_hash(a.out / "reader.pt"),
                    "lora_sha256": file_hash(a.out / "lora/adapter_model.safetensors"),
                },
            )
        write_json(a.out / "training.json", history)
        print(json.dumps(row), flush=True)
    checkpoint = torch.load(a.out / "reader.pt", map_location="cuda", weights_only=True)
    reader.load_state_dict(checkpoint["state_dict"])
    set_peft_model_state_dict(llm, load_file(str(a.out / "lora/adapter_model.safetensors")))
    predictions = {}
    metrics = {}
    for split in ("validation", "development_test"):
        audit = np.array([i for i in splits[split] if qa[i]["kind"] == "audit"])
        for mode in ("neural", "donor"):
            donors = None if mode == "neural" else np.roll([qa[i]["state_index"] for i in audit], 1)
            text = generate(audit, donors)
            values = targets[[qa[i]["state_index"] for i in audit]]
            metrics[split + "/" + mode] = score(text, values)
            predictions[split + "/" + mode] = [
                {"id": qa[i]["id"], "kind": "audit", "text": t, "target": v.tolist()}
                for i, t, v in zip(audit, text, values)
            ]
        voice = []
        for kind in (
            ("summary", "change", "change_qualitative") if a.paraphrases else ("summary", "change")
        ):
            choices = [i for i in splits[split] if qa[i]["kind"] == kind]
            voice.extend(
                np.random.default_rng(17017)
                .choice(choices, min(12, len(choices)), replace=False)
                .tolist()
            )
        text = generate(np.array(voice))
        predictions[split + "/voice"] = [
            {
                "id": qa[i]["id"],
                "kind": qa[i]["kind"],
                "question": qa[i]["question"],
                "text": t,
                "target": qa[i]["answer"],
            }
            for i, t in zip(voice, text)
        ]
    report.update(
        status="complete",
        selected_epoch=selected,
        selection_value=best,
        metrics=metrics,
        total_seconds=time.monotonic() - start,
    )
    if a.codec:
        report["codec_unchanged"] = all(
            torch.equal(value.detach().cpu(), codec_initial[key])
            for key, value in reader.codec.state_dict().items()
        )
        if not report["codec_unchanged"]:
            raise ValueError("Frozen neural codec changed during language training")
    write_json(a.out / "report.json", report)
    write_json(a.out / "predictions.json", predictions)
    print(json.dumps({"status": "complete", "metrics": metrics}), flush=True)


if __name__ == "__main__":
    main()
