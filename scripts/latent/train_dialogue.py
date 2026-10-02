#!/usr/bin/env python3
"""Paired neural states -> conversational LLM with optional small LoRA SFT."""

import argparse, json, time, random
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from flypet.latent_dialogue import DialogueReader, prompt_parts
from flypet.latent_reader_data import score
from flypet.neural_records import file_hash, write_json

MODEL = "Qwen/Qwen3-4B"
REVISION = "1cfa9a7208912126459214e8b04321603b3df60c"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--seed", type=int, default=916)
    p.add_argument("--lora-rank", type=int, default=8)
    p.add_argument("--max-seconds", type=int, default=7200)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    torch.set_num_threads(4)
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    random.seed(a.seed)

    def deadline():
        if time.monotonic() - start > a.max_seconds:
            raise TimeoutError("Dialogue training deadline")

    manifest = json.loads((a.data / "manifest.json").read_text())
    for name, digest in manifest["files"].items():
        if file_hash(a.data / name) != digest:
            raise ValueError("Data hash mismatch")
    states = [json.loads(x) for x in (a.data / "states.jsonl").read_text().splitlines()]
    qa = [json.loads(x) for x in (a.data / "qa.jsonl").read_text().splitlines()]
    with np.load(a.data / "features.npz", allow_pickle=False) as z:
        x = z["x"]
        px = z["population_x"]
        aux = z["auxiliary"]
        targets = z["targets"]
        metadata = z["metadata"]
        preprocessing = {
            k: z[k] for k in z.files if k not in ("x", "population_x", "auxiliary", "targets")
        }
    np.savez_compressed(a.out / "preprocessing.npz", **preprocessing)
    write_json(a.out / "dataset_manifest.json", manifest)
    tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    tok.pad_token = tok.eos_token
    llm = (
        AutoModelForCausalLM.from_pretrained(
            MODEL, revision=REVISION, dtype=torch.bfloat16, attn_implementation="sdpa"
        )
        .cuda()
        .eval()
        .requires_grad_(False)
    )
    if a.lora_rank:
        from peft import LoraConfig, get_peft_model

        llm = get_peft_model(
            llm,
            LoraConfig(
                r=a.lora_rank,
                lora_alpha=2 * a.lora_rank,
                lora_dropout=0,
                target_modules=["q_proj", "v_proj"],
                task_type="CAUSAL_LM",
            ),
        )
        llm.eval()
    emb = llm.get_input_embeddings()
    dim = emb.weight.shape[1]
    scale = float(emb.weight.std())
    reader = DialogueReader(
        x.shape[1], x.shape[2], metadata, px.shape[1], dim, scale, auxiliary_dim=aux.shape[1]
    ).cuda()
    parts = [prompt_parts(tok, row["question"]) for row in qa]
    answers = [
        list(tok(row["answer"] + tok.eos_token, add_special_tokens=False)["input_ids"])
        for row in qa
    ]
    splits = {
        s: np.array([i for i, r in enumerate(qa) if r["split"] == s])
        for s in ("train", "validation", "development_test")
    }
    rng = np.random.default_rng(a.seed)
    # Fixed balanced validation panel for checkpoint selection, not the test.
    validation = []
    for kind in manifest["question_types"]:
        group = [i for i in splits["validation"] if qa[i]["kind"] == kind]
        validation.extend(rng.choice(group, min(32, len(group)), replace=False).tolist())
    validation = np.array(sorted(validation))

    def encode(indices, donors=None):
        ids = (
            np.array([qa[i]["state_index"] for i in indices])
            if donors is None
            else np.asarray(donors)
        )
        bases = np.array([states[i]["baseline_index"] for i in ids])
        joined = np.r_[bases, ids]
        tokens, pred = reader(
            torch.tensor(x[joined], device="cuda"), torch.tensor(px[joined], device="cuda")
        )
        return tokens[: len(ids)], tokens[len(ids) :], pred, joined

    def loss_batch(indices):
        reference, current, pred, joined = encode(indices)
        vectors = []
        labels = []
        for j, i in enumerate(indices):
            left, middle, right = parts[i]
            pieces = [
                emb(torch.tensor(left, device="cuda")),
                reference[j].to(emb.weight.dtype),
                emb(torch.tensor(middle, device="cuda")),
                current[j].to(emb.weight.dtype),
                emb(torch.tensor(right + answers[i], device="cuda")),
            ]
            v = torch.cat(pieces)
            vectors.append(v)
            labels.append(
                torch.tensor([-100] * (len(v) - len(answers[i])) + answers[i], device="cuda")
            )
        width = max(map(len, vectors))
        mask = torch.zeros((len(indices), width), device="cuda", dtype=torch.long)
        batch = emb(
            torch.full((len(indices), width), tok.pad_token_id, device="cuda", dtype=torch.long)
        )
        label = torch.full((len(indices), width), -100, device="cuda", dtype=torch.long)
        for j, (v, l) in enumerate(zip(vectors, labels)):
            batch[j, : len(v)] = v
            label[j, : len(v)] = l
            mask[j, : len(v)] = 1
        loss = llm(inputs_embeds=batch, attention_mask=mask, labels=label, use_cache=False).loss
        auxiliary = torch.nn.functional.mse_loss(pred, torch.tensor(aux[joined], device="cuda"))
        return loss + 0.05 * auxiliary, loss.detach(), int((label != -100).sum())

    @torch.no_grad()
    def validation_nll():
        reader.eval()
        llm.eval()
        scores = []
        for kind in manifest["question_types"]:
            subset = [i for i in validation if qa[i]["kind"] == kind]
            total = weight = 0
            for k in range(0, len(subset), a.batch):
                deadline()
                _, loss, n = loss_batch(subset[k : k + a.batch])
                total += float(loss) * n
                weight += n
            if weight:
                scores.append(total / weight)
        return sum(scores) / len(scores)

    @torch.no_grad()
    def generate(indices, donors=None):
        reader.eval()
        llm.eval()
        texts = []
        for offset in range(0, len(indices), a.batch):
            deadline()
            group = indices[offset : offset + a.batch]
            ref, cur, _, _ = encode(
                group, None if donors is None else donors[offset : offset + len(group)]
            )
            vectors = []
            for j, i in enumerate(group):
                left, middle, right = parts[i]
                vectors.append(
                    torch.cat(
                        (
                            emb(torch.tensor(left, device="cuda")),
                            ref[j].to(emb.weight.dtype),
                            emb(torch.tensor(middle, device="cuda")),
                            cur[j].to(emb.weight.dtype),
                            emb(torch.tensor(right, device="cuda")),
                        )
                    )
                )
            width = max(map(len, vectors))
            batch = emb(
                torch.full((len(group), width), tok.pad_token_id, device="cuda", dtype=torch.long)
            )
            mask = torch.zeros((len(group), width), device="cuda", dtype=torch.long)
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

    parameters = [{"params": reader.parameters(), "lr": 0.001}]
    if a.lora_rank:
        parameters.append({"params": [p for p in llm.parameters() if p.requires_grad], "lr": 2e-5})
    optim = torch.optim.AdamW(parameters, weight_decay=0.01)
    report = {
        "status": "training",
        "model": MODEL,
        "revision": REVISION,
        "scope": manifest["scope"],
        "config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()},
        "reader_parameters": sum(p.numel() for p in reader.parameters()),
        "lora_parameters": sum(p.numel() for p in llm.parameters() if p.requires_grad),
        "dataset_manifest_sha256": file_hash(a.data / "manifest.json"),
        "population_features": px.shape[1],
        "n_fine_neurons": x.shape[1],
        "validation_selection_examples": len(validation),
        "selection": "macro mean of token NLL within each question type",
    }
    write_json(a.out / "report.json", report)
    best = float("inf")
    history = []
    for epoch in range(a.epochs):
        reader.train()
        llm.eval()
        order = rng.permutation(splits["train"])
        total = weight = 0
        t = time.monotonic()
        for k in range(0, len(order), a.batch):
            deadline()
            optim.zero_grad(set_to_none=True)
            loss, lm, n = loss_batch(order[k : k + a.batch])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(reader.parameters(), 1)
            if a.lora_rank:
                torch.nn.utils.clip_grad_norm_([p for p in llm.parameters() if p.requires_grad], 1)
            optim.step()
            total += float(lm) * n
            weight += n
        val = validation_nll()
        row = {
            "epoch": epoch + 1,
            "train_nll": total / weight,
            "validation_nll": val,
            "wall_s": time.monotonic() - t,
        }
        history.append(row)
        if val < best:
            best = val
            selected = epoch + 1
            torch.save(
                {
                    "state_dict": reader.state_dict(),
                    "embedding_dim": dim,
                    "embedding_std": scale,
                    "tokens": 16,
                    "width": 128,
                    "n_neurons": x.shape[1],
                    "channels": x.shape[2],
                    "n_populations": px.shape[1],
                    "auxiliary_dim": aux.shape[1],
                    "metadata": metadata,
                    "epoch": selected,
                },
                a.out / "reader.pt",
            )
            if a.lora_rank:
                llm.save_pretrained(a.out / "lora")
            write_json(
                a.out / "best_checkpoint.json",
                {
                    "epoch": selected,
                    "reader_sha256": file_hash(a.out / "reader.pt"),
                    "lora_sha256": file_hash(a.out / "lora/adapter_model.safetensors")
                    if a.lora_rank
                    else None,
                },
            )
        write_json(a.out / "training.json", history)
        print(json.dumps(row), flush=True)
    receipt = json.loads((a.out / "best_checkpoint.json").read_text())
    if file_hash(a.out / "reader.pt") != receipt["reader_sha256"]:
        raise ValueError("Reader checkpoint mismatch")
    saved = torch.load(a.out / "reader.pt", map_location="cuda", weights_only=False)
    reader.load_state_dict(saved["state_dict"])
    if a.lora_rank:
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file

        if file_hash(a.out / "lora/adapter_model.safetensors") != receipt["lora_sha256"]:
            raise ValueError("LoRA checkpoint mismatch")
        set_peft_model_state_dict(llm, load_file(str(a.out / "lora/adapter_model.safetensors")))
    predictions = {}
    metrics = {}
    for split in ("validation", "development_test"):
        audit = np.array([i for i in splits[split] if qa[i]["kind"] == "audit"])
        for mode in ("neural", "donor"):
            donor = None if mode == "neural" else np.roll([qa[i]["state_index"] for i in audit], 1)
            text = generate(audit, donor)
            values = targets[[qa[i]["state_index"] for i in audit]]
            metrics[split + "/" + mode] = score(text, values)
            predictions[split + "/" + mode] = [
                {"id": qa[i]["id"], "kind": "audit", "text": t, "target": v.tolist()}
                for i, t, v in zip(audit, text, values)
            ]
        voice = []
        for kind in manifest["question_types"]:
            if kind == "audit":
                continue
            choices = [i for i in splits[split] if qa[i]["kind"] == kind]
            chooser = np.random.default_rng(17017)
            voice.extend(chooser.choice(choices, min(12, len(choices)), replace=False).tolist())
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
        validation_nll=best,
        metrics=metrics,
        total_seconds=time.monotonic() - start,
    )
    write_json(a.out / "report.json", report)
    write_json(a.out / "predictions.json", predictions)
    print(json.dumps({"status": "complete", "metrics": metrics}), flush=True)


if __name__ == "__main__":
    main()
