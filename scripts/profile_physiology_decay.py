"""Post-stimulus activity on corrected full graph; diagnostic only, no fitting."""

import argparse, json, hashlib, time
from pathlib import Path
import numpy as np
import torch
from flypet.physiology_torch import PhysiologicalCore


def main(a):
    graph = np.load(Path(a.data) / "graph/graph.npz")
    event = np.load(Path(a.data) / "calibration/input.npz")
    core = PhysiologicalCore(
        len(graph["order"]), graph["pre"], graph["post"], graph["contacts"]
    ).to(a.device)
    pulse = torch.from_numpy(event["impulses_mv"][:, None, :]).to(a.device)
    populations = {
        k: torch.as_tensor(graph[k], device=a.device)
        for k in graph.files
        if k.startswith("population_")
    }
    windows = []
    state = None
    start = time.monotonic()
    with torch.no_grad():
        for tick in range(0, 5000, 500):
            impulses = pulse[tick : tick + 500] if tick < 3000 else torch.zeros_like(pulse[:500])
            result = core(
                torch.zeros_like(impulses),
                event["ports"],
                graph["output_indices"],
                impulses_mv=impulses,
                input_refractory_ms=0,
                initial_state=state,
            )
            state = result["state"]
            counts = result["neuron_spike_counts"][0]
            windows.append(
                {
                    "start_ms": tick * 0.1,
                    "end_ms": (tick + 500) * 0.1,
                    "stimulus_on": tick < 3000,
                    "spikes": int(counts.sum()),
                    "active_neurons": int((counts > 0).sum()),
                    "voltage_rms_mv": float(state[0].square().mean().sqrt()),
                    "synaptic_rms_mv": float(state[1].square().mean().sqrt()),
                    "populations": {
                        k: {
                            "n": len(ids),
                            "spikes": int(counts[ids].sum()),
                            "active_neurons": int((counts[ids] > 0).sum()),
                            "mean_rate_hz": float(counts[ids].mean() / 0.05) if len(ids) else None,
                        }
                        for k, ids in populations.items()
                    },
                }
            )
    report = {
        "windows": windows,
        "seconds": time.monotonic() - start,
        "graph": "full corrected fixed gains1",
        "input": "300ms calibration pulses followed by200ms zero input",
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    Path(a.out).write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cuda")
    main(p.parse_args())
