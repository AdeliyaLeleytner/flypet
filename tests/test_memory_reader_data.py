"""Scientific split, preprocessing and intervention tests for memory reader."""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from flypet.memory_reader_data import (
    load_dataset,
    load_profiles,
    parse_output,
    prepare_dataset,
    score_outputs,
    swap_pairs,
)


def fixture():
    rows = []
    for group, split in enumerate(("train", "train", "validation", "test", "chemical_test")):
        for assignment in (0, 1):
            rows.append(
                {
                    "example_id": f"{group}-{assignment}",
                    "history_id": f"{group}-{assignment}",
                    "split_group_id": str(group),
                    "split": split,
                    "history_split": split,
                    "probe_compound": "a",
                    "probe_seed": 101,
                    "probe_duration_ms": 250,
                    "probe_rate_hz": 150,
                    "training_compounds": ["a", "b"],
                    "pairing_trials": 1,
                    "history_seed": group,
                    "assignment": assignment,
                    "history": [
                        {
                            "chemical": "a",
                            "rate_hz": 150,
                            "duration_ms": 250,
                            "dan_mode": "reward" if assignment else "punish",
                        }
                    ],
                    "target": {"approach_hz": 12.0, "avoid_hz": 8.0, "valence": 4 / 21},
                }
            )
    rates = np.column_stack(
        [np.arange(len(rows)) + 1.0, np.ones(len(rows)), np.zeros(len(rows))]
    ).astype(np.float32)
    profiles = {"a": [1.0, 0.0], "b": [0.0, 1.0]}
    return rows, rates, np.array([100, 200, 300]), profiles


class MemoryReaderDataTests(unittest.TestCase):
    def test_train_only_selection_normalization_and_unseen_features(self):
        rows, rates, indices, profiles = fixture()
        baseline = prepare_dataset(rows, rates, indices, chemical_profiles=profiles)
        changed = rates.copy()
        changed[4:, 0] = 10000
        changed[4:, 2] = 30000
        other = prepare_dataset(rows, changed, indices, chemical_profiles=profiles)
        for key in ("neural_mu", "neural_sd", "neuron_indices", "history_mu", "history_sd"):
            np.testing.assert_array_equal(baseline[key], other[key])
        self.assertNotIn(300, other["neuron_indices"])
        np.testing.assert_allclose(other["neural"][:4].mean(0), 0, atol=1e-5)
        self.assertLess(
            other["manifest"]["audit"]["feature_coverage"]["test"]["fraction_rate_mass_retained"], 1
        )

    def test_history_split_leakage_refused_but_query_repeat_explicit(self):
        rows, rates, indices, profiles = fixture()
        bad = deepcopy(rows)
        bad[4]["split_group_id"] = "0"
        with self.assertRaisesRegex(ValueError, "crosses splits"):
            prepare_dataset(bad, rates, indices, chemical_profiles=profiles)
        extra = deepcopy(rows[0])
        extra.update(example_id="query", split="query_chemical_test", probe_compound="c")
        prepared = prepare_dataset(
            rows + [extra],
            np.vstack((rates, rates[0])),
            indices,
            chemical_profiles=profiles | {"c": [0.3, 0.4]},
        )
        self.assertNotIn("c", prepared["history_vocabulary"])
        self.assertEqual(prepared["manifest"]["counts"]["query_chemical_test"], 1)

    def test_swaps_follow_assignment_and_ignore_targets(self):
        rows, _, _, _ = fixture()
        first = swap_pairs(rows, range(len(rows)))
        self.assertEqual(first[:2], [(0, 1), (1, 0)])
        for i, row in enumerate(rows):
            row["target"]["valence"] = i / 100
        self.assertEqual(first, swap_pairs(rows, range(len(rows))))
        rows[1]["probe_seed"] += 1
        self.assertNotIn((0, 1), swap_pairs(rows, range(len(rows))))

    def test_join_by_unique_example_id_and_reject_nonfinite_rates(self):
        rows, rates, indices, _ = fixture()
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            (directory / "examples.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
            np.savez(
                directory / "features.npz",
                example_ids=[r["example_id"] for r in reversed(rows)],
                rates=rates[::-1],
                neuron_indices=indices,
            )
            _, loaded, _, _ = load_dataset(directory)
            np.testing.assert_array_equal(loaded, rates)
            rates[0, 0] = np.nan
            np.savez(
                directory / "features.npz",
                example_ids=[r["example_id"] for r in rows],
                rates=rates,
                neuron_indices=indices,
            )
            with self.assertRaisesRegex(ValueError, "finite"):
                load_dataset(directory)

    def test_missing_profile_values_have_explicit_mask(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "profiles.json"
            path.write_text(
                json.dumps(
                    {
                        "glomeruli": ["x", "y"],
                        "compounds": {"a": {"glomerular_profile": {"x": None, "y": 0.0}}},
                    }
                )
            )
            profiles, metadata = load_profiles(path)
            self.assertEqual(profiles["a"], [0.0, 0.0, 0.0, 1.0])
            self.assertIn("observed mask", metadata["representation"])

    def test_mbon_feature_order_must_match_rows_and_neural_columns(self):
        rows, rates, indices, _ = fixture()
        for row, rate in zip(rows, rates):
            row["mbon_rates_hz"] = rate[:2].tolist()
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            (directory / "examples.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
            np.savez(
                directory / "features.npz",
                example_ids=[r["example_id"] for r in rows],
                rates=rates,
                neuron_indices=indices,
                mbon_rates=rates[:, :2],
                mbon_indices=indices[:2],
            )
            load_dataset(directory)
            np.savez(
                directory / "features.npz",
                example_ids=[r["example_id"] for r in rows],
                rates=rates,
                neuron_indices=indices,
                mbon_rates=rates[:, :2],
                mbon_indices=indices[:2][::-1],
            )
            with self.assertRaisesRegex(ValueError, "MBON rates/order"):
                load_dataset(directory)

    def test_scores_separate_invalid_output_from_conditional_mae(self):
        rows, _, _, _ = fixture()
        text = '{"approach_hz":12.0,"avoid_hz":8.0,"valence":0.190}'
        self.assertIsNotNone(parse_output(text))
        self.assertIsNone(parse_output('{"approach_hz":true,"avoid_hz":8,"valence":0.2}'))
        score = score_outputs([text, "not JSON"], rows[:2])
        self.assertEqual(score["parse_rate"], 0.5)
        self.assertEqual(score["direction_accuracy_all"], 0.5)
        self.assertEqual(score["mae_valid"]["approach_hz"], 0)


if __name__ == "__main__":
    unittest.main()
