#!/usr/bin/env python3
"""Exercise the exact installed PEFT/Transformers API before loading 4B weights."""

import subprocess, sys

subprocess.run(
    [
        sys.executable,
        "-m",
        "unittest",
        "discover",
        "-s",
        "tests",
        "-p",
        "test_latent_dialogue.py",
        "-q",
    ],
    check=True,
)
subprocess.run(
    [sys.executable, "-u", "scripts/latent/train_dialogue.py", *sys.argv[1:]], check=True
)
