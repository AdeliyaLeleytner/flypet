"""Bounded local forward/backward resource probe; no language learning or rental."""

import argparse
import hashlib
import json
import platform
import resource
import time
from pathlib import Path
import numpy as np
import torch
from flypet.physiology_torch import PhysiologicalCore

ROOT = Path(__file__).resolve().parents[1]


def run(args):
    torch.set_num_threads(args.threads)
    torch.manual_seed(20260924)
    source = ROOT / "data/connectivity_783.npz"
    out = Path(args.out)
    if out.exists():
        raise FileExistsError("Choose a fresh profiling output")
    out.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with np.load(source) as z:
        n = len(z["order"])
        pre = z["i_pre"]
        post = z["i_post"]
        weights = z["w"]
    if args.nodes:
        # Induced first-N subset: technical microbenchmark, not representative biology.
        n = min(n, args.nodes)
        mask = (pre < n) & (post < n)
        pre = pre[mask]
        post = post[mask]
        weights = weights[mask]
    inputs = np.arange(min(128, n), dtype=np.int64)
    reached = post[np.isin(pre, inputs)]
    degree = np.bincount(reached, minlength=n)
    degree[inputs] = 0
    outputs = np.flatnonzero(degree > 0)
    if not len(outputs):
        outputs = np.arange(n // 2, n, dtype=np.int64)
    outputs = outputs[np.argsort(degree[outputs], kind="stable")[-min(128, len(outputs)) :]]
    core = PhysiologicalCore(n, pre, post, weights)
    construction = time.perf_counter() - started
    optimizer = torch.optim.Adam(core.parameters(), lr=1e-3)
    result = {
        "status": "running",
        "scope": "CPU-only resource probe. Continuous synthetic drive; no DoOR/task dataset, no LLM, no dopamine plasticity or live-pet changes.",
        "graph_nodes": n,
        "graph_edges": len(pre),
        "full_graph": not bool(args.nodes),
        "graph_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "trainable_parameters": sum(x.numel() for x in core.parameters()),
        "parameterization": "one positive presynaptic gain per neuron, range[0.25,2], existing signs/topology fixed",
        "construction_seconds": construction,
        "torch": torch.__version__,
        "device": "cpu",
        "cpu_threads": args.threads,
        "physiology": vars(core.parameters_spec),
        "runs": [],
        "source_sha256": {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [Path(__file__), ROOT / "flypet/physiology_torch.py"]
        },
    }

    def save():
        out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")

    save()
    for ticks in args.ticks:
        if time.perf_counter() - started > args.max_seconds:
            result["status"] = "partial_time_budget"
            save()
            return
        optimizer.zero_grad(set_to_none=True)
        drive = (
            (80 + 80 * torch.rand(1, args.batch, len(inputs)))
            .expand(ticks, -1, -1)
            .clone()
            .requires_grad_()
        )
        saved = {}
        calls = 0

        def pack(t):
            nonlocal calls
            calls += 1
            if t.layout == torch.strided:
                storage = t.untyped_storage()
                saved[(storage.data_ptr(), storage.nbytes())] = storage.nbytes()
            return t

        start = time.perf_counter()
        with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
            y = core(drive, inputs, outputs)
            # Deliberately synthetic loss: checks backward inside the core, not task quality.
            loss = (y["voltage_mv"] - 1).square().mean() + 1e-6 * y["rate_hz"].square().mean()
        forward = time.perf_counter() - start
        start = time.perf_counter()
        loss.backward()
        backward = time.perf_counter() - start
        grad = core.raw_gain.grad
        before = core.raw_gain.detach().clone()
        start = time.perf_counter()
        optimizer.step()
        update = time.perf_counter() - start
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        rss_bytes = rss if platform.system() == "Darwin" else rss * 1024
        row = {
            "ticks": ticks,
            "simulated_ms": ticks * core.parameters_spec.dt_ms,
            "batch": args.batch,
            "forward_seconds": forward,
            "backward_seconds": backward,
            "optimizer_seconds": update,
            "loss_before_step": float(loss.detach()),
            "total_spikes": float(y["spike_count"]),
            "core_gradient_finite": bool(torch.isfinite(grad).all()),
            "core_gradient_nonzero_neurons": int(torch.count_nonzero(grad)),
            "non_input_core_gradient_nonzero_neurons": int(torch.count_nonzero(grad))
            - int(torch.count_nonzero(grad[inputs])),
            "core_gradient_l2": float(grad.norm()),
            "input_gradient_l2": float(drive.grad.norm()),
            "parameters_changed": int(torch.count_nonzero(core.raw_gain.detach() - before)),
            "process_peak_rss_bytes": rss_bytes,
            "saved_dense_unique_storage_bytes": sum(saved.values()),
            "saved_tensor_hook_calls": calls,
            "memory_caveat": "Process RSS is cumulative peak incl loading/optimizer. Saved dense storage excludes sparse internals and is not CUDA peak VRAM.",
        }
        if not row["core_gradient_finite"] or not row["core_gradient_nonzero_neurons"]:
            raise RuntimeError("No finite usable core gradient")
        result["runs"].append(row)
        save()
        print(json.dumps(row), flush=True)
        del loss, y, drive, before, grad
    result["status"] = "complete"
    result["wall_seconds"] = time.perf_counter() - started
    save()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--nodes", type=int, default=0)
    p.add_argument("--ticks", type=int, nargs="+", default=[32, 64])
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--max-seconds", type=int, default=180)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    if args.batch < 1 or args.batch > 4 or min(args.ticks) < 1 or max(args.ticks) > 128:
        p.error("Local profile is bounded to batch1–4 and1–128ticks")
    run(args)
