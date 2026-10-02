#!/usr/bin/env python3
import subprocess, sys

subprocess.run([sys.executable, "-m", "unittest", "tests.test_latent_dialogue"], check=True)
subprocess.run(
    [sys.executable, "-u", "scripts/latent/train_reader_repair.py", *sys.argv[1:]], check=True
)
