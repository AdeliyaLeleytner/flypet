#!/usr/bin/env python3
"""One bounded GPU/CPU job: frozen features, SFT, GRPO and matched evaluation."""

import os, subprocess, sys, shutil, json
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["FLYPET_BRIAN_TARGET"] = "cython"
if sys.version_info < (3, 12):
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--disable-pip-version-check", "uv"], check=True
    )
    subprocess.run(["uv", "venv", "/workspace/fly/.agent-venv", "--python", "3.12"], check=True)
    os.execv(
        "/workspace/fly/.agent-venv/bin/python",
        ["/workspace/fly/.agent-venv/bin/python", __file__, *sys.argv[1:]],
    )
if not shutil.which("g++"):
    subprocess.run(["apt-get", "update", "-qq"], check=True)
    subprocess.run(["apt-get", "install", "-y", "-qq", "build-essential"], check=True)
subprocess.run(
    [
        "uv",
        "pip",
        "install",
        "--python",
        sys.executable,
        "torch==2.6.0",
        "--index-url",
        "https://download.pytorch.org/whl/cu124",
    ],
    check=True,
)
subprocess.run(
    [
        "uv",
        "pip",
        "install",
        "--python",
        sys.executable,
        "brian2==2.10.1",
        "cython==3.1.3",
        "numpy==2.5.3",
        "pandas==3.0.5",
        "scipy==1.18.1",
        "pyarrow==25.0.1",
        "setuptools",
        "transformers==5.17.0",
        "peft==0.21.0",
        "accelerate",
        "safetensors",
    ],
    check=True,
)
subprocess.run(
    ["uv", "pip", "freeze", "--python", sys.executable],
    stdout=open("environment.txt", "w"),
    check=True,
)
subprocess.run(
    [
        sys.executable,
        "-m",
        "unittest",
        "tests.test_latent_policy",
        "tests.test_latent_agent_features",
        "tests.test_latent_agent_rollouts",
        "tests.test_latent_dialogue",
    ],
    check=True,
)
# The caller must supply a completed, empirically state-dependent paired reader.
reader = json.loads(Path("reader_report.json").read_text())
if reader["status"] != "complete":
    raise ValueError("Reader training is incomplete")
neural = reader["metrics"]["validation/neural"]
donor = reader["metrics"]["validation/donor"]
if neural["valid_fraction"] < 0.95 or donor["mae_valid"] is None or neural["mae_valid"] is None:
    raise ValueError("Reader numeric validation is incomplete or unreliable")
if donor["mae_valid"]["valence"] <= neural["mae_valid"]["valence"]:
    raise ValueError("Reader has not shown a neural-channel benefit")
subprocess.run(
    [
        sys.executable,
        "-u",
        "scripts/latent/evaluate_sealed_reader.py",
        "--data",
        "data/sealed_reader",
        "--bundle",
        "bundle",
        "--selection",
        "selection.json",
        "--out",
        "results/fresh_reader",
    ],
    check=True,
)
subprocess.run(
    [
        sys.executable,
        "-u",
        "scripts/latent/check_live_twins.py",
        "--bundle",
        "bundle",
        "--selection",
        "selection.json",
        "--out",
        "results/fresh_twins",
    ],
    check=True,
)
panel = json.loads(Path("results/fresh_reader/report.json").read_text())
twins = json.loads(Path("results/fresh_twins/report.json").read_text())
qualified = panel["panel_gate_pass"] and twins["gate_pass"]
Path("results/reader_gate.json").write_text(
    json.dumps(
        {
            "qualified_for_agent_experiment": qualified,
            "panel_gates": panel["gates"],
            "fresh_twins_correct": twins["correct_sign"],
        },
        indent=2,
    )
    + "\n"
)
if not qualified:
    print(
        "Fresh reader evaluation completed; agent training deferred because a declared reader gate failed.",
        flush=True,
    )
    raise SystemExit(0)
if "--evaluation-only" in sys.argv:
    print(
        "Reader qualification complete. Agent training was not requested in this evaluation-only lease.",
        flush=True,
    )
    raise SystemExit(0)
commands = [
    [
        "scripts/latent/prepare_agent_features.py",
        "--teacher",
        "data/teacher",
        "--bundle",
        "bundle",
        "--out",
        "results/features",
    ],
    ["scripts/latent/train_agent_sft.py", "--data", "results/features", "--out", "results/sft"],
    [
        "scripts/latent/train_agent_grpo.py",
        "--features",
        "results/features",
        "--bundle",
        "bundle",
        "--sft",
        "results/sft",
        "--out",
        "results/agent",
        "--max-seconds",
        "6500",
    ],
]
for command in commands:
    subprocess.run([sys.executable, "-u", *command], check=True)
