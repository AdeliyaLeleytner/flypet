#!/usr/bin/env python3
from pathlib import Path
import argparse, json, sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from flypet.latent_reader_data import prepare

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--max-neurons", type=int, default=4096)
    a = p.parse_args()
    report = prepare(a.source, a.output, max_neurons=a.max_neurons)
    print(
        json.dumps(
            {k: report[k] for k in ("status", "counts", "n_neurons", "writable_inputs_excluded")}
        )
    )
