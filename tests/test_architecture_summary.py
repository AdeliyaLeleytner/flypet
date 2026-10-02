import importlib.util
from pathlib import Path
import unittest

path = Path(__file__).resolve().parents[1] / "scripts/summarize_architecture_reader.py"
spec = importlib.util.spec_from_file_location("architecture_summary", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class ArchitectureSummaryTests(unittest.TestCase):
    def test_out_of_range_numbers_remain_in_diagnostic_error(self):
        text = '{"approach_hz":120,"avoid_hz":180,"valence":3.0}'
        self.assertIsNone(module.parse_output(text))
        self.assertEqual(module.finite_numeric_output(text)["valence"], 3.0)
        self.assertIsNone(
            module.finite_numeric_output('{"approach_hz":true,"avoid_hz":1,"valence":0}')
        )
        self.assertIsNone(
            module.finite_numeric_output('{"approach_hz":NaN,"avoid_hz":1,"valence":0}')
        )

    def test_opposite_history_contrast_and_missing_output_coverage(self):
        rows = {}
        predictions = []
        for seed in (1, 2):
            for assignment in (0, 1):
                ident = f"{seed}-{assignment}"
                rows[ident] = {
                    "split_group_id": "family",
                    "history_seed": seed,
                    "pairing_trials": 3,
                    "probe_compound": "odor",
                    "assignment": assignment,
                    "target": {"valence": -0.4 if assignment == 0 else 0.6},
                }
                generated = (
                    '{"approach_hz":10,"avoid_hz":5,"valence":'
                    + str(-0.3 if assignment == 0 else 0.5)
                    + "}"
                )
                if seed == 2 and assignment == 1:
                    generated = "invalid"
                predictions.append(
                    {"mode": "graph", "split": "test", "example_id": ident, "generated": generated}
                )
        result = module.contrasts(predictions, rows)["graph/test"]
        self.assertEqual(result["total_pairs"], 2)
        self.assertEqual(result["valid_pairs"], 1)
        self.assertEqual(result["coverage"], 0.5)
        self.assertAlmostEqual(result["mae_valid"], 0.2)
        self.assertAlmostEqual(result["zero_contrast_mae_valid"], 1.0)


if __name__ == "__main__":
    unittest.main()
