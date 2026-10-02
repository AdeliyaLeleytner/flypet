#!/usr/bin/env python3
"""Does Qwen3-32B know more about the fly than it says? Linear probes on its residual stream at the answer.

Each FlyTalk diagnose episode is played greedily in the same format as grpo_multiturn.py. The hidden states are read
at the last token before the model writes its final ANSWER, after it has seen all of its measurements. Per odour, a
logistic-regression probe predicts the fly's true label (+, -, 0) from those states. Probes are fitted on training
flies (seeds from 1000) and scored on the 60 test flies with the FlyTalk balanced score, next to two references:
  answer        the model's own ANSWER on the same flies;
  measurements  the same probe on what the transcript contains in numeric form: per odour the number of measurements
                and the mean, minimum and maximum valence shift from the untrained fly, as shown (two decimals).
A hidden-state probe above the model's answer means the model represents more than it reports; a probe below the
measurement probe means its representation loses evidence it was shown.

Stages, each resumable, outputs in data/flytalk_rl_20260926/latent_probe_<tag>/:
  episodes  play the training and test flies, save token ids, answer positions, measurements and results (GPU + CPU)
  states    one forward pass per episode, hidden states at the answer position for --layers (GPU)
  probe     fit and score the probes (CPU; runs anywhere once states.npz and episodes.json exist)
"""

from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
LABELS = ["-", "0", "+"]


def answer_position(mask):
    """Index of the last prompt token before the final generated segment (the ANSWER)."""
    end = len(mask) - 1
    while end >= 0 and mask[end] == 0:
        end -= 1
    start = end
    while start > 0 and mask[start - 1] == 1:
        start -= 1
    return start - 1


def stage_episodes(a, out):
    import torch  # noqa: F401  (grpo_multiturn imports it; kept explicit for readers)
    from grpo_multiturn import Agent, rollout, with_brief  # noqa: F401
    from flypet.flytalk import PANEL, Env
    from flypet.teach import FlyPool

    path = out / "episodes.json"
    if path.exists():
        print("episodes: exists, skipping", flush=True)
        return
    env = Env(FlyPool(a.workers), PANEL)
    agent = Agent(a.model, 0)
    if a.adapter:
        from peft import PeftModel

        agent.model = PeftModel.from_pretrained(agent.model, a.adapter).eval()
    specs = [("diagnose", s) for s in range(a.n_test)] + [
        ("diagnose", 1000 + s) for s in range(a.n_train)
    ]
    brief = open(a.brief).read() if a.brief else ""
    t0, eps = time.time(), []
    for c0 in range(0, len(specs), a.batch):
        runs = rollout(
            agent, env, specs[c0 : c0 + a.batch], True, a.max_turns, a.max_new, brief=brief
        )
        for (task, seed), r in zip(specs[c0 : c0 + a.batch], runs):
            ep = env.episodes[r["eid"]]
            smells = [
                (t["action"][6:], round(t["value"], 2)) for t in ep.transcript if "value" in t
            ]
            eps.append(
                {
                    "seed": seed,
                    "split": "test" if seed < a.n_test else "train",
                    "ids": r["ids"],
                    "answer_pos": answer_position(r["mask"]),
                    "smells": smells,
                    "texts": r["texts"],
                    "history": r["history"],
                    "result": r["result"],
                }
            )
        print(f"episodes {len(eps)}/{len(specs)}  {time.time() - t0:.0f} s", flush=True)
    json.dump({"naive": env.naive, "panel": PANEL, "episodes": eps}, open(path, "w"), default=str)


def stage_states(a, out):
    import torch
    from transformers import AutoModelForCausalLM

    path = out / "states.npz"
    if path.exists():
        print("states: exists, skipping", flush=True)
        return
    eps = json.load(open(out / "episodes.json"))["episodes"]
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16).to("cuda").eval()
    if a.adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, a.adapter).eval()
    layers = [int(x) for x in a.layers.split(",")]
    H = {L: [] for L in layers}
    t0 = time.time()
    with torch.no_grad():
        for k, e in enumerate(eps):
            ids = torch.tensor([e["ids"][: e["answer_pos"] + 1]], device="cuda")
            hs = model(input_ids=ids, output_hidden_states=True).hidden_states
            for L in layers:
                H[L].append(hs[L][0, -1].float().cpu().numpy().astype(np.float16))
            if (k + 1) % 50 == 0:
                print(f"states {k + 1}/{len(eps)}  {time.time() - t0:.0f} s", flush=True)
    np.savez_compressed(path, **{f"L{L}": np.stack(H[L]) for L in layers})


def measurement_features(e, panel, naive):
    f = []
    for o in panel:
        v = [x - naive[o] for n, x in e["smells"] if n == o]
        f += [len(v), np.mean(v) if v else 0.0, min(v) if v else 0.0, max(v) if v else 0.0]
    return f


def balanced(pred, truth, panel):
    trained = [o for o in panel if truth[o] != "0"]
    right = {o: pred.get(o) == truth[o] for o in panel}
    ur = np.mean([right[o] for o in panel if o not in trained])
    return 0.5 * (np.mean([right[o] for o in trained]) + ur) if trained else ur, all(right.values())


def stage_probe(a, out):
    from sklearn.decomposition import PCA
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    d = json.load(open(out / "episodes.json"))
    eps, panel, naive = d["episodes"], d["panel"], d["naive"]
    Z = np.load(out / "states.npz")
    tr = np.array([e["split"] == "train" for e in eps])
    te = ~tr
    truth = [e["result"]["truth"] for e in eps]
    Y = {o: np.array([LABELS.index(t[o]) for t in truth]) for o in panel}
    chosen = {}

    def probe(reduce):
        """Standardise, optionally reduce with PCA, then multinomial logistic regression with balanced class weights (most
        odours are untrained, and the FlyTalk score weighs trained and untrained odours equally). For hidden states,
        where features far outnumber flies, the PCA size and the regularisation are chosen by 5-fold CV on the
        training flies only (log-loss)."""
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import GridSearchCV

        pipe = Pipeline(
            [
                ("scale", StandardScaler()),
                ("pca", "passthrough"),
                ("clf", LogisticRegression(class_weight="balanced", max_iter=5000)),
            ]
        )
        grid = {"clf__C": np.logspace(-5, 1, 7)}
        if reduce:
            n = int(tr.sum()) - int(tr.sum()) // 5 - 1
            grid["pca"] = [PCA(n_components=k, random_state=0) for k in (8, 32, 128) if k < n] + [
                "passthrough"
            ]
        return GridSearchCV(pipe, grid, scoring="neg_log_loss", cv=5, n_jobs=-1)

    def fit_predict(X, reduce):
        preds = [{} for _ in range(te.sum())]
        for o in panel:
            clf = probe(reduce).fit(X[tr], Y[o][tr])
            chosen[o] = {
                k: (str(v) if k == "pca" else float(v)) for k, v in clf.best_params_.items()
            }
            for k, lab in enumerate(clf.predict(X[te])):
                preds[k][o] = LABELS[lab]
        return preds

    def score(preds):
        s = [balanced(p, t, panel) for p, t in zip(preds, [t for t, m in zip(truth, te) if m])]
        x = np.array([v for v, _ in s])
        rng = np.random.default_rng(0)
        b = [x[rng.integers(len(x), size=len(x))].mean() for _ in range(5000)]
        return {
            "score": round(float(x.mean()), 4),
            "ci": [round(float(np.percentile(b, 2.5)), 4), round(float(np.percentile(b, 97.5)), 4)],
            "exact": round(float(np.mean([e for _, e in s])), 4),
        }

    res = {
        "n_train": int(tr.sum()),
        "n_test": int(te.sum()),
        "answer": score([e["result"]["pred"] or {} for e, m in zip(eps, te) if m]),
        "measurements": score(
            fit_predict(
                np.array([measurement_features(e, panel, naive) for e in eps], float), False
            )
        ),
    }
    for key in sorted(Z.files, key=lambda k: int(k[1:])):
        res[key] = score(fit_predict(Z[key].astype(np.float32), True))
        res[key]["chosen"] = dict(chosen)
        print(key, {k: v for k, v in res[key].items() if k != "chosen"}, flush=True)
    # sanity check: can the states decode the model's own answer? (should be near perfect at late layers)
    own = {
        o: np.array([LABELS.index((e["result"]["pred"] or {}).get(o, "0")) for e in eps])
        for o in panel
    }
    last = sorted(Z.files, key=lambda k: int(k[1:]))[-1]
    X = Z[last].astype(np.float32)
    agree = []
    for o in panel:
        if len(set(own[o][tr])) < 2:
            continue
        clf = probe(True).fit(X[tr], own[o][tr])
        agree.append(float(np.mean(clf.predict(X[te]) == own[o][te])))
    res["decode_own_answer_" + last] = round(float(np.mean(agree)), 4)
    json.dump(res, open(out / "probe.json", "w"), indent=1)
    print(json.dumps(res, indent=1))


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("stage", choices=["episodes", "states", "probe", "all"])
    ap.add_argument("--model", default="Qwen/Qwen3-32B")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--tag", default="qwen32b_zero")
    ap.add_argument("--brief", default=None)
    ap.add_argument("--n-train", type=int, default=400)
    ap.add_argument("--n-test", type=int, default=60)
    ap.add_argument("--batch", type=int, default=64, help="episodes played together")
    ap.add_argument("--workers", type=int, default=40)
    ap.add_argument("--max-turns", type=int, default=14)
    ap.add_argument("--max-new", type=int, default=96)
    ap.add_argument("--layers", default="8,16,24,32,40,48,56,64")
    a = ap.parse_args()
    out = ROOT / "data/flytalk_rl_20260926" / f"latent_probe_{a.tag}"
    out.mkdir(parents=True, exist_ok=True)
    if a.stage in ("episodes", "all"):
        stage_episodes(a, out)
    if a.stage in ("states", "all"):
        stage_states(a, out)
    if a.stage in ("probe", "all"):
        stage_probe(a, out)


if __name__ == "__main__":
    main()
