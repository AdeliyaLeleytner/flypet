"""Same pre-generated ORN impulses, corrected full graph, Brian2 versus Torch."""

import argparse, json, time
from pathlib import Path
import numpy as np
import torch
from flypet.physiology_torch import PhysiologicalCore


def main(a):
    out = Path(a.out)
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    graph = np.load(Path(a.graph) / "graph.npz")
    n = len(graph["order"])
    from flypet.odor import DoorOdor

    index = {int(x): i for i, x in enumerate(graph["order"])}
    rates = {}
    for s in DoorOdor(rate_max=150).stim("ethyl acetate"):
        for rid in s.ids:
            rates[index[rid]] = s.rate_hz
    ins = np.asarray(sorted(rates))
    hz = np.asarray([rates[i] for i in ins])
    ticks = round(a.duration_ms / 0.1)
    events = np.random.default_rng(20260924).random((ticks, len(ins))) < hz[None, :] * 0.0001
    pulses = events.astype(np.float32) * 68.75
    np.savez_compressed(out / "input.npz", ports=ins, hz=hz, impulses_mv=pulses)
    report = {
        "status": "running",
        "duration_ms": a.duration_ms,
        "nodes": n,
        "edges": len(graph["pre"]),
        "input": "identical pre-generated Bernoulli/Poisson-discretized ORN voltage jumps;150Hz maximum; input refractory0",
        "dopamine_plasticity": False,
        "corrected_graph": True,
    }

    def save():
        (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")

    save()
    import brian2 as b

    b.start_scope()
    b.prefs.codegen.target = "numpy"
    b.defaultclock.dt = 0.1 * b.ms
    stim = b.TimedArray(pulses * b.mV, dt=0.1 * b.ms)
    groups = b.NeuronGroup(
        n,
        "dv/dt=(-v+g)/(20*ms):volt (unless refractory)\n"
        "dg/dt=-g/(5*ms):volt (unless refractory)\nrfc:second\nport:integer (constant)\nstimulated:integer (constant)",
        threshold="v>7*mV",
        reset="v=0*mV;g=0*mV",
        refractory="rfc",
        method="linear",
        namespace={"stim": stim},
    )
    groups.rfc = 2.2 * b.ms
    groups.rfc[ins] = 0 * b.ms
    groups.port = 0
    groups.port[ins] = np.arange(len(ins))
    groups.stimulated = 0
    groups.stimulated[ins] = 1
    # Mask prevents non-input nodes from receiving the dummy port's pulses.
    groups.run_regularly("v += stimulated*stim(t,port)", when="start")
    syn = b.Synapses(groups, groups, "w:volt", on_pre="g_post += w", delay=1.8 * b.ms)
    syn.connect(i=graph["pre"], j=graph["post"])
    syn.w = graph["contacts"] * 0.275 * b.mV
    sm = b.SpikeMonitor(groups, record=False)
    started = time.perf_counter()
    b.Network(groups, syn, sm).run(a.duration_ms * b.ms)
    reference = np.asarray(sm.count[:], np.int64)
    report["brian_seconds"] = time.perf_counter() - started
    np.save(out / "brian_counts.npy", reference)
    save()
    del groups, syn, sm
    torch.set_num_threads(4)
    core = PhysiologicalCore(n, graph["pre"], graph["post"], graph["contacts"])
    impulse = torch.from_numpy(pulses[:, None, :])
    drive = torch.zeros_like(impulse)
    started = time.perf_counter()
    with torch.no_grad():
        y = core(drive, ins, graph["output_indices"], impulses_mv=impulse, input_refractory_ms=0)
    actual = y["neuron_spike_counts"][0].numpy()
    report["torch_seconds"] = time.perf_counter() - started
    np.save(out / "torch_counts.npy", actual)
    report.update(
        status="complete",
        identical_count_fraction=float(np.mean(actual == reference)),
        total_spikes_brian=int(reference.sum()),
        total_spikes_torch=int(actual.sum()),
        count_mae=float(abs(actual - reference).mean()),
        max_count_difference=float(abs(actual - reference).max()),
        populations={},
    )
    for key in graph.files:
        if key.startswith("population_"):
            ids = graph[key]
            report["populations"][key] = {
                "n": len(ids),
                "active_brian": int((reference[ids] > 0).sum()),
                "active_torch": int((actual[ids] > 0).sum()),
                "rate_mae_hz": float(
                    abs(actual[ids] - reference[ids]).mean() / (a.duration_ms / 1000)
                )
                if len(ids)
                else None,
            }
    save()
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--graph", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--duration-ms", type=float, default=300)
    main(p.parse_args())
