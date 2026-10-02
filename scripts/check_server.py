#!/usr/bin/env python3
"""Exercise a running garden with real stimuli, without language-model calls."""

import argparse
import json
import time
import urllib.error
import urllib.request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8765")
    parser.add_argument("--timeout", type=float, default=180)
    args = parser.parse_args()
    base = args.url.rstrip("/")

    def request(path, body=None):
        headers = {"Content-Type": "application/json"} if body is not None else {}
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(base + path, data=data, headers=headers)
        with urllib.request.urlopen(req, timeout=args.timeout) as response:
            return response.read()

    deadline = time.monotonic() + args.timeout
    while True:
        try:
            health = json.loads(request("/healthz"))
            if health.get("ok") and health.get("neurons") == 138639:
                break
        except (OSError, ValueError):
            pass
        if time.monotonic() >= deadline:
            raise TimeoutError("Garden did not become ready")
        time.sleep(1)

    for path in ("/garden", "/static/garden-brain.js", "/static/garden-controls.js"):
        if not request(path):
            raise AssertionError(f"Empty response: {path}")
    options = json.loads(request("/api/garden/options"))
    if not options:
        raise AssertionError("No available garden controls")
    results = {}
    for name, payload in (
        ("baseline", {"action": "baseline", "seed": 11}),
        ("sugar", {"action": "stimuli", "stimuli": [{"key": "sugar"}], "seed": 11}),
    ):
        result = json.loads(request("/api/garden", payload))
        if not isinstance(result.get("n_spikes_total"), int):
            raise AssertionError(f"{name} did not return measured spikes")
        results[name] = {"spikes": result["n_spikes_total"], "source": result.get("source")}
    if results["sugar"]["spikes"] <= results["baseline"]["spikes"]:
        raise AssertionError("Sugar failed to produce a response above the no-input control")
    print(json.dumps({"health": health, "trials": results}, indent=2))


if __name__ == "__main__":
    main()
