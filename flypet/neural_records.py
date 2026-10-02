"""Array-only helpers for auditable latent-interface trajectory records."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def verify_bundle(directory):
    """Verify every declared artifact and reject unlisted files and escaping paths."""
    directory = Path(directory).resolve()
    manifest = json.loads((directory / "manifest.json").read_text())
    if any(p.is_symlink() for p in directory.rglob("*")):
        raise ValueError("Unsafe model bundle symlink")
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("Model bundle has no file manifest")
    for name, digest in files.items():
        relative = Path(name)
        target = directory / relative
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or not target.resolve().is_relative_to(directory)
        ):
            raise ValueError("Unsafe model bundle path")
        if not target.is_file() or file_hash(target) != digest:
            raise ValueError(f"Model bundle hash mismatch: {name}")
    actual = {p.relative_to(directory).as_posix() for p in directory.rglob("*") if p.is_file()}
    if actual != set(files) | {"manifest.json"}:
        raise ValueError("Model bundle contains unlisted or missing files")
    return manifest


def array_hash(value):
    a = np.ascontiguousarray(value)
    h = hashlib.sha256()
    h.update(str(a.dtype).encode())
    h.update(json.dumps(list(a.shape)).encode())
    h.update(a.tobytes())
    return h.hexdigest()


def spike_arrays(results, dt_ms):
    """Canonical time-major sparse spikes; model indices map through root_ids.npy."""
    indices, ticks, offset = [], [], 0
    for result in results:
        for i, times in result.trains.items():
            indices.append(np.full(len(times), i, dtype=np.int32))
            ticks.append(np.rint(times * 1000 / dt_ms).astype(np.int64) + offset)
        offset += int(round(result.duration_s * 1000 / dt_ms))
    if not indices:
        return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int64)
    indices, ticks = np.concatenate(indices), np.concatenate(ticks)
    order = np.lexsort((indices, ticks))
    return indices[order], ticks[order]


def write_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def compare_arrays(expected, actual):
    """Exact spikes/memory, float32 observation tolerance of 1e-6 in saved units."""
    out = {}
    if set(expected) != set(actual):
        return {"ok": False, "error": "array names differ"}
    for name in expected:
        x, y = np.asarray(expected[name]), np.asarray(actual[name])
        same_shape = x.shape == y.shape
        if not same_shape:
            out[name] = {"ok": False, "shape": [list(x.shape), list(y.shape)]}
            continue
        exact = bool(np.array_equal(x, y))
        tolerance = 1e-6 if name in ("voltage_mv", "synaptic_mv", "refractory_remaining_ms") else 0
        ok = bool(np.allclose(x, y, rtol=0, atol=tolerance))
        difference = float(np.max(np.abs(x.astype(np.float64) - y))) if x.size else 0.0
        out[name] = {"ok": ok, "exact": exact, "max_abs_difference": difference, "atol": tolerance}
    return {"ok": all(x["ok"] for x in out.values()), "arrays": out}
