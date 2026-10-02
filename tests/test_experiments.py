"""Protocol and artifact checks that do not build a connectome network."""

import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import numpy as np

from flypet.experiments import (
    CaseRunner,
    array_sha256,
    effective_inputs,
    expected_counts,
    load_protocol,
    memory_schedule,
    protocol_sha256,
    serialize_inputs,
    summarize_run,
    verify_artifacts,
    write_json,
)


class ExperimentTests(unittest.TestCase):
    def test_memory_histories_match_exposures_but_remove_pairing(self):
        p = load_protocol()
        steps = memory_schedule(p, 11)
        training = {
            h: [s for s in steps if s["action"] == "train" and s["history"] == h]
            for h in ("A", "B", "unpaired")
        }
        self.assertEqual([len(training[h]) for h in training], [6, 6, 12])
        self.assertTrue(
            all(s["compound"] is None or s["reinforcement"] == "none" for s in training["unpaired"])
        )
        for a, b in zip(training["A"], training["B"]):
            self.assertEqual((a["compound"], a["seed"]), (b["compound"], b["seed"]))
            self.assertNotEqual(a["reinforcement"], b["reinforcement"])
        odor_unpaired = [s for s in training["unpaired"] if s["compound"] is not None]
        self.assertEqual(
            [(s["compound"], s["seed"]) for s in training["A"]],
            [(s["compound"], s["seed"]) for s in odor_unpaired],
        )
        self.assertEqual(sum(expected_counts(p, ["A", "B", "C", "GF"], p["seeds"]).values()), 156)

    def test_explicit_rates_preserve_zero_and_engine_overwrite(self):
        brain = SimpleNamespace(
            flyid2i={101: 0, 102: 1, 103: 2}, i2flyid={0: 101, 1: 102, 2: 103}, n_stim=2
        )
        inputs = [
            SimpleNamespace(ids=[101, 102, 999], rate_hz=10.0, key="first"),
            SimpleNamespace(ids=[101], rate_hz=0.0, key="zero"),
            SimpleNamespace(ids=[102, 103], rate_hz=20.0, key="last"),
        ]
        self.assertEqual(serialize_inputs(inputs)[1]["rate_hz"], 0.0)
        self.assertEqual(
            effective_inputs(brain, inputs),
            [
                {"model_index": 0, "root_id": 101, "rate_hz": 10.0},
                {"model_index": 1, "root_id": 102, "rate_hz": 20.0},
            ],
        )

    def test_isolated_memory_snapshots_are_content_addressed(self):
        with tempfile.TemporaryDirectory() as temporary:
            mb = SimpleNamespace(syn_idx=np.array([4, 8]), mem=np.ones(2, dtype=np.float32), log=[])
            mb.memory_strength = lambda: float(1 - mb.mem.mean())
            runner = CaseRunner(temporary, load_protocol(), mb=mb)
            before = runner._memory_snapshot()
            self.assertEqual(before, runner._memory_snapshot())
            mb.mem[0] = 0.5
            after = runner._memory_snapshot()
            self.assertNotEqual(before["path"], after["path"])
            np.testing.assert_array_equal(
                np.load(Path(temporary) / before["path"])["mem"], [1.0, 1.0]
            )
            self.assertEqual(len(list((Path(temporary) / "memory").glob("*.npz"))), 2)

    def test_integrity_check_detects_tampered_memory_and_duplicate_record(self):
        with tempfile.TemporaryDirectory() as temporary:
            p = Path(temporary)
            idx, mem = np.array([1]), np.array([0.5], dtype=np.float32)
            np.savez(p / "memory.npz", syn_idx=idx, mem=mem)
            record = {
                "record_id": "one",
                "memory_before": {"path": "memory.npz", "array_sha256": array_sha256(idx, mem)},
            }
            (p / "records.jsonl").write_text(json.dumps(record) + "\n")
            self.assertTrue(verify_artifacts(p)["ok"])
            np.savez(p / "memory.npz", syn_idx=idx, mem=mem * 0)
            (p / "records.jsonl").write_text((json.dumps(record) + "\n") * 2)
            errors = verify_artifacts(p)["errors"]
            self.assertEqual(len(errors), 2)

    def test_protocol_fingerprint_ignores_key_order_but_not_seed_change(self):
        p = load_protocol()
        reversed_keys = {k: p[k] for k in reversed(list(p))}
        self.assertEqual(protocol_sha256(p), protocol_sha256(reversed_keys))
        self.assertNotEqual(protocol_sha256(p), protocol_sha256(p | {"seeds": [12, 23, 47]}))

    def test_refuse_overwrite_and_nonfinite_json(self):
        with tempfile.TemporaryDirectory() as temporary:
            p = Path(temporary)
            (p / "records.jsonl").write_text("")
            with self.assertRaises(FileExistsError):
                CaseRunner(p, load_protocol())
            with self.assertRaises(ValueError):
                write_json(p / "bad.json", {"value": float("nan")})


if __name__ == "__main__":
    unittest.main()
