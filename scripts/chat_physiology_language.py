"""Run a saved physiological hybrid on one four-event history (no LLM history bypass)."""

import argparse, json, hashlib
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from flypet.physiology_language import PhysiologicalLanguageBridge, MODEL_ID, MODEL_REVISION
from flypet.physiology_memory_task import OBJECTS, LOCATIONS, question_text


def main(a):
    report = json.loads(Path(a.report).read_text())
    config = report["config"]
    digest = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    if digest(a.graph) != report["graph_manifest"]["graph_sha256"]:
        raise ValueError("Wrong anatomical graph/ports")
    if report["model"] != MODEL_ID or report["revision"] != MODEL_REVISION:
        raise ValueError("Wrong language model revision")
    allowed = {
        x["best_checkpoint_sha256"] for x in report.get("arms", {}).values() if x.get("complete")
    }
    if digest(a.checkpoint) not in allowed:
        raise ValueError("Checkpoint is not a completed arm in this report")
    episode = json.loads(Path(a.episode).read_text())
    events = episode["events"]
    obj = episode["query_object"]
    if (
        len(events) != 4
        or obj not in OBJECTS
        or any(len(e) != 2 or e[0] not in OBJECTS or e[1] not in LOCATIONS for e in events)
    ):
        raise ValueError(
            "Expected four [key/coin/ring, red/blue/green/yellow] events and a query_object"
        )
    if not any(e[0] == obj for e in events):
        raise ValueError("Queried object is absent from history")
    device = a.device
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    llm = (
        AutoModelForCausalLM.from_pretrained(
            MODEL_ID, revision=MODEL_REVISION, dtype=dtype, attn_implementation="sdpa"
        )
        .to(device)
        .eval()
        .requires_grad_(False)
    )
    embedding = llm.get_input_embeddings()
    graph = np.load(a.graph)
    bridge = (
        PhysiologicalLanguageBridge(
            graph,
            embedding.weight.shape[1],
            report["embedding_std"],
            tokens=config["prefix_tokens"],
            hidden=config["hidden"],
            ticks_per_event=config["ticks_per_event"],
            checkpoint_steps=config["checkpoint_steps"],
            physics_dtype=config["physics_dtype"],
            surrogate_scale=config["surrogate_scale"],
            max_drive_mv=config["max_drive_mv"],
        )
        .to(device)
        .eval()
    )
    bridge.load_trainable_state(torch.load(a.checkpoint, map_location="cpu", weights_only=True))
    with torch.no_grad():
        encoded = []
        for item, place in events:
            ids = tok(f"The {item} moved to the {place} box.", add_special_tokens=False)[
                "input_ids"
            ]
            encoded.append(embedding(torch.tensor(ids, device=device)).float().mean(0))
        prefix, stats = bridge(torch.stack(encoded)[None], intervention=a.intervention)
        # Only the question, not events, is given to the LLM as text.
        question = question_text({"query": f"Where is the {obj} now?"})
        ids = tok.apply_chat_template(
            [{"role": "user", "content": question}],
            tokenize=True,
            return_dict=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        prompt = embedding(torch.tensor([ids], device=device))
        inputs = torch.cat((prefix.to(prompt.dtype), prompt), 1)
        generated = llm.generate(
            inputs_embeds=inputs,
            attention_mask=torch.ones(inputs.shape[:2], dtype=torch.long, device=device),
            max_new_tokens=8,
            do_sample=False,
            pad_token_id=tok.eos_token_id,
            return_dict_in_generate=True,
            output_scores=True,
        )
        tokens = generated.sequences[0, -len(generated.scores) :]
    print(
        json.dumps(
            {
                "answer": tok.decode(tokens, skip_special_tokens=True).strip(),
                "intervention": a.intervention,
                "stats": {k: float(v) for k, v in stats.items()},
                "scope": "Restricted four-event research pilot; not a general dialogue model.",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--graph", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--report", required=True)
    p.add_argument("--episode", required=True)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    p.add_argument(
        "--intervention", choices=["normal", "zero_state", "reverse_history"], default="normal"
    )
    main(p.parse_args())
