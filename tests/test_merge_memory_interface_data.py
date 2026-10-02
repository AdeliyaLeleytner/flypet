"""Small end-to-end checks of six-shard recovery merging and audit failures."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from flypet.experiments import ROOT, array_sha256, protocol_sha256, sha256_file, write_json
from scripts.merge_memory_interface_data import merge


class MergeRecoveryTests(unittest.TestCase):
    def fixture(self, root):
        labels = [f"part{j}_{kind}" for j in range(3) for kind in ("completed", "remaining")]
        protocol = {"seeds": [101], "odors": ["odor"]}
        histories = [{"history_id": f"h{i}"} for i in range(6)]
        plan = {
            "protocol": protocol,
            "protocol_sha256": protocol_sha256(protocol),
            "histories": histories,
            "expected_histories": 6,
            "expected_examples": 6,
            "expected_records": 7,
            "heldout_compound": "acid",
            "probe_seed_rule": {"split_namespaces": {"train": 0}},
        }
        write_json(root / "plan.json", plan)
        source = "scripts/gen_memory_interface_data.py"
        for i, label in enumerate(labels):
            d = root / "shards" / label
            d.mkdir(parents=True)
            (d / "raw").mkdir()
            write_json(
                d / "plan.json",
                dict(
                    plan,
                    histories=[histories[i]],
                    expected_histories=1,
                    expected_examples=1,
                    expected_records=2,
                ),
            )
            write_json(
                d / "manifest.json",
                {
                    "status": "complete",
                    "pet_unchanged": True,
                    "pet_before": {},
                    "files": {source: {"sha256": sha256_file(ROOT / source)}},
                    "elapsed_s": 1,
                    "actual_records": 2,
                    "actual_histories": 1,
                    "actual_examples": 1,
                },
            )
            write_json(d / "feature_schema.json", {"mbon_signs": [1, -1]})
            write_json(d / "input_profiles.json", {"odor": [1]})
            write_json(d / "validation.json", {"ok": True})
            np.save(d / "brain_root_ids.npy", np.arange(13))
            records = []
            for phase in ("naive", "post"):
                rid = phase + "-1"
                relative = f"raw/{rid}.npz"
                idx, counts = np.array([10, 11]), np.array([2, 1])
                np.savez(
                    d / relative, spike_times_s=np.array([0.1, 0.2, 0.3]), idx=idx, counts=counts
                )
                raw = {
                    "path": relative,
                    "sha256": sha256_file(d / relative),
                    "count_sha256": array_sha256(idx, counts),
                }
                records.append(
                    {
                        "record_id": rid,
                        "phase": phase,
                        "compound": "odor",
                        "seed": 101,
                        "metadata": {"history_split": "train"},
                        "raw": raw,
                    }
                )
            (d / "records.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
            example = {
                "example_id": f"h{i}-probe0",
                "record_id": "post-1",
                "state_npz": records[1]["raw"]["path"],
                "raw": records[1]["raw"],
                "checkpoint": None,
                "naive_reference": {"record_id": "naive-1", "raw": records[0]["raw"]},
                "split_group_id": f"g{i}",
                "split": "train",
                "history_split": "train",
                "probe_compound": "odor",
                "training_compounds": ["odor", "other"],
                "target": {"approach_hz": 2, "avoid_hz": 1, "valence": 0.25},
            }
            (d / "examples.jsonl").write_text(json.dumps(example) + "\n")
            np.savez(
                d / "features.npz",
                rates=np.array([[2, 1, i]], dtype=np.float32),
                neuron_indices=np.array([10, 11, 12]),
                example_ids=np.array([example["example_id"]]),
                mbon_rates=np.array([[2, 1]], dtype=np.float32),
                mbon_indices=np.array([10, 11]),
            )
        return labels

    @patch("scripts.merge_memory_interface_data.pet_fingerprints", return_value={})
    def test_six_shards_preserve_history_order_and_all_naive_replicas(self, _):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            labels = self.fixture(root)
            result = merge(root, labels)
            self.assertEqual(result["actual_records"], 12)
            self.assertEqual(result["actual_histories"], 6)
            self.assertEqual(result["actual_examples"], 6)
            audit = json.loads((root / "naive_reproducibility.json").read_text())
            self.assertEqual(audit["n_replicas_per_group"], 6)
            self.assertEqual(audit["n_comparisons_to_first"], 5)
            manifest = json.loads((root / "manifest.json").read_text())
            self.assertEqual(manifest["duplicate_naive_reference_records"], 5)
            with np.load(root / "features.npz") as z:
                self.assertEqual(z["example_ids"].tolist(), [f"h{i}-probe0" for i in range(6)])
                self.assertEqual(z["rates"][:, 2].tolist(), list(range(6)))

    @patch("scripts.merge_memory_interface_data.pet_fingerprints", return_value={})
    def test_rejects_history_order_change(self, _):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            labels = self.fixture(root)
            with self.assertRaisesRegex(AssertionError, "Histories missing"):
                merge(root, list(reversed(labels)))

    @patch("scripts.merge_memory_interface_data.pet_fingerprints", return_value={})
    def test_rejects_wrong_target_readout(self, _):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            labels = self.fixture(root)
            path = root / "shards" / labels[-1] / "examples.jsonl"
            example = json.loads(path.read_text())
            example["target"]["valence"] = 0.8
            path.write_text(json.dumps(example) + "\n")
            with self.assertRaisesRegex(AssertionError, "MBON readout"):
                merge(root, labels)


if __name__ == "__main__":
    unittest.main()
