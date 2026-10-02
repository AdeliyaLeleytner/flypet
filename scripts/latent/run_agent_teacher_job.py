#!/usr/bin/env python3
"""Install simulator dependencies on the bounded worker, then collect teachers."""

import os, subprocess, sys, shutil

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["FLYPET_BRIAN_TARGET"] = "cython"
if sys.version_info < (3, 12):
    # The base CUDA image ships 3.11; Brian2 2.10 requires Python >=3.12.
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--disable-pip-version-check", "uv"], check=True
    )
    subprocess.run(["uv", "venv", "/workspace/fly/.sim-venv", "--python", "3.12"], check=True)
    os.execv(
        "/workspace/fly/.sim-venv/bin/python",
        ["/workspace/fly/.sim-venv/bin/python", __file__, *sys.argv[1:]],
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
        "brian2==2.10.1",
        "cython==3.1.3",
        "numpy==2.5.3",
        "pandas==3.0.5",
        "scipy==1.18.1",
        "pyarrow==25.0.1",
        "setuptools",
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
        "scripts/latent/check_agent_env.py",
        "--preprocessing",
        "data/agent_preprocessing.npz",
        "--out",
        "results/agent_environment_check.json",
    ],
    check=True,
)
subprocess.run(
    [sys.executable, "-u", "scripts/latent/collect_agent_teacher.py", *sys.argv[1:]], check=True
)
