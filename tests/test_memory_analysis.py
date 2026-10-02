import copy
import importlib.util
from pathlib import Path
import unittest

import numpy as np

spec = importlib.util.spec_from_file_location(
    "memory_analysis", Path(__file__).resolve().parents[1] / "scripts/analyze_memory_interface.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class ProspectiveIsolationTests(unittest.TestCase):
    def rows(self):
        return [
            {
                "history_id": "h1",
                "probe_compound": compound,
                "example_id": str(i),
                "checkpoint_hash": "same",
                "mbon_rates_hz": [10 * i, 2 + i],
                "target": {"valence": i},
            }
            for i, compound in enumerate([*module.DIAGNOSTICS, "ethyl acetate"])
        ]

    def test_changing_target_response_cannot_change_features(self):
        rows = self.rows()
        prof = {"ethyl acetate": np.asarray([0.5, 1.0])}
        selected, before, context = module.prospective_view(rows, prof)
        rows[-1]["mbon_rates_hz"] = [999, 999]
        rows[-1]["target"]["valence"] = -999
        _, after, _ = module.prospective_view(rows, prof)
        np.testing.assert_array_equal(before, after)
        self.assertEqual(context, [{"target": "2", "sources": ["0", "1"]}])
        self.assertEqual(len(selected), 1)

    def test_other_memory_checkpoint_is_rejected(self):
        rows = self.rows()
        rows[0]["checkpoint_hash"] = "different"
        with self.assertRaisesRegex(ValueError, "same frozen checkpoint"):
            module.prospective_view(rows, {"ethyl acetate": np.zeros(2)})

    def test_duplicate_probe_is_rejected(self):
        rows = self.rows()
        rows.append(copy.deepcopy(rows[0]))
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            module.prospective_view(rows, {"ethyl acetate": np.zeros(2)})

    def test_delta_bounds_preserve_negative_rate_changes(self):
        actual = module.bounded(np.array([[-60.0, 30.0, -1.5]]), "delta")
        np.testing.assert_array_equal(actual, [[-60, 30, -1.5]])

    def test_standardization_fits_training_only(self):
        x = np.array([[1.0, 1.0], [3.0, 1.0], [100.0, 999.0]])
        z, mean, sd = module.standardize(x, np.array([0, 1]))
        np.testing.assert_allclose(mean, [2, 1])
        np.testing.assert_allclose(sd, [1, 1])
        np.testing.assert_allclose(z[:2, 0], [-1, 1])


if __name__ == "__main__":
    unittest.main()
