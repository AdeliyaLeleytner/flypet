"""Target isolation, split ownership and scoring checks for language forecasts."""

from copy import deepcopy
import unittest

import numpy as np

from flypet.memory_forecast_data import (
    DIAGNOSTICS,
    parse_output,
    prepare_dataset,
    prospective_rows,
    score_outputs,
    swap_pairs,
)


def fixture():
    rows = []
    for group, split in enumerate(("train", "validation", "test")):
        for assignment in (0, 1):
            for probe, seed in zip(
                (*DIAGNOSTICS, "ethyl acetate", "acetic acid"), (10, 20, 30, 40)
            ):
                rows.append(
                    {
                        "example_id": f"{group}-{assignment}-{seed}",
                        "history_id": f"{group}-{assignment}",
                        "split_group_id": str(group),
                        "history_split": split,
                        "split": "query_chemical_test" if probe == "acetic acid" else split,
                        "probe_compound": probe,
                        "probe_seed": seed,
                        "probe_rate_hz": 150,
                        "probe_duration_ms": 250,
                        "training_compounds": ["methyl acetate", "ethyl acetate"],
                        "pairing_trials": 1,
                        "history_seed": group + 100,
                        "assignment": assignment,
                        "checkpoint_hash": f"checkpoint-{group}-{assignment}",
                        "history": [
                            {
                                "chemical": "ethyl acetate",
                                "rate_hz": 150,
                                "duration_ms": 250,
                                "dan_mode": "reward" if assignment else "punish",
                            }
                        ],
                        "mbon_rates_hz": [float(assignment + 1), float(seed)],
                        "target": {"valence": -0.8 if assignment == 0 else 0.8},
                        "naive_reference": {"valence": {"score": 0.3}},
                    }
                )
    profiles = {
        name: [i / 10, 1.0] for i, name in enumerate((*DIAGNOSTICS, "ethyl acetate", "acetic acid"))
    }
    return rows, profiles


class ForecastDataTests(unittest.TestCase):
    def test_target_response_cannot_change_predictor_inputs(self):
        source, profiles = fixture()
        rows, neural, contexts = prospective_rows(source, profiles)
        changed = deepcopy(source)
        for r in changed:
            if r["probe_compound"] not in DIAGNOSTICS:
                r["mbon_rates_hz"] = [9999.0, 9999.0]
                r["target"]["valence"] = 0.1
        other, other_neural, other_contexts = prospective_rows(changed, profiles)
        np.testing.assert_array_equal(neural, other_neural)
        self.assertEqual(contexts, other_contexts)
        self.assertNotEqual(rows[0]["target"], other[0]["target"])
        self.assertFalse(any(r["probe_compound"] in DIAGNOSTICS for r in rows))

    def test_same_checkpoint_and_independent_query_rng_required(self):
        source, profiles = fixture()
        source[0]["checkpoint_hash"] = "wrong"
        with self.assertRaisesRegex(ValueError, "checkpoint"):
            prospective_rows(source, profiles)
        source, profiles = fixture()
        source[0]["probe_seed"] = source[2]["probe_seed"]
        with self.assertRaisesRegex(ValueError, "RNG"):
            prospective_rows(source, profiles)

    def test_scaling_uses_training_only_and_constant_scale_is_one(self):
        source, profiles = fixture()
        rows, neural, _ = prospective_rows(source, profiles)
        history = np.c_[np.ones(len(rows)), np.arange(len(rows))].astype(np.float32)
        baseline = prepare_dataset(rows, neural, history)
        changed, hchanged = neural.copy(), history.copy()
        held = [i for i, r in enumerate(rows) if r["split"] != "train"]
        changed[held] = 10000
        hchanged[held] = 20000
        other = prepare_dataset(rows, changed, hchanged)
        for key in ("neural_mu", "neural_sd", "history_mu", "history_sd"):
            np.testing.assert_array_equal(baseline[key], other[key])
        self.assertEqual(baseline["history_sd"][0], 1.0)
        np.testing.assert_array_equal(baseline["null"], np.zeros_like(neural))

    def test_split_and_acid_exclusion_enforced(self):
        source, profiles = fixture()
        rows, neural, _ = prospective_rows(source, profiles)
        history = np.ones((len(rows), 2))
        changed = deepcopy(rows)
        changed[4]["split_group_id"] = changed[0]["split_group_id"]
        with self.assertRaisesRegex(ValueError, "crosses splits"):
            prepare_dataset(changed, neural, history)
        changed = deepcopy(rows)
        changed[1]["split"] = "train"
        with self.assertRaisesRegex(ValueError, "Excluded chemical"):
            prepare_dataset(changed, neural, history)

    def test_signed_delta_bounds_strict_schema_and_partial_coverage(self):
        self.assertEqual(parse_output('{"delta_valence":-1.5}'), {"delta_valence": -1.5})
        for invalid in (
            '{"delta_valence":true}',
            '{"delta_valence":2.1}',
            '{"delta_valence":NaN}',
            '{"delta_valence":0,"valence":0}',
            "invalid",
        ):
            self.assertIsNone(parse_output(invalid))
        source, profiles = fixture()
        rows, _, _ = prospective_rows(source, profiles)
        pair = [rows[0], rows[2]]
        result = score_outputs(['{"delta_valence":-1.1}', "invalid"], pair)
        self.assertEqual(result["parse_rate"], 0.5)
        self.assertEqual(result["mae_valid"]["delta_valence"], 0.0)
        self.assertEqual(result["matched_assignment_contrast"]["coverage"], 0.0)
        result = score_outputs(['{"delta_valence":-1.1}', '{"delta_valence":0.5}'], pair)
        self.assertEqual(result["matched_assignment_contrast"]["mae_valid"], 0.0)
        self.assertAlmostEqual(result["no_change_mae"], 0.8)

    def test_swap_matching_ignores_target_and_excludes_rng_mismatch(self):
        source, profiles = fixture()
        rows, _, _ = prospective_rows(source, profiles)
        before = swap_pairs(rows, range(len(rows)))
        self.assertIn((0, 2), before)
        for r in rows:
            r["target"]["delta_valence"] = 0.0
        self.assertEqual(before, swap_pairs(rows, range(len(rows))))
        rows[2]["probe_seed"] += 1
        self.assertNotIn((0, 2), swap_pairs(rows, range(len(rows))))


if __name__ == "__main__":
    unittest.main()
