"""Language-panel checks use file fixtures and fakes, never live models."""

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "run_language_cases", ROOT / "scripts/run_language_cases.py"
)
language = importlib.util.module_from_spec(spec)
spec.loader.exec_module(language)


class LanguageCaseTests(unittest.TestCase):
    def test_fixed_panel_and_clamp_preserve_original_planner_and_blind_history(self):
        panel = language.load_panel()
        self.assertEqual(len(panel["cases"]), 18)
        self.assertEqual(sum(c["kind"] == "controlled_memory_query" for c in panel["cases"]), 6)
        original = {
            "is_stimulus": False,
            "duration_ms": 500,
            "interpretation": "chemical",
            "stimuli": [{"key": "sugar", "rate_hz": 150}],
            "channels": [],
            "odor_text": "apple",
            "reinforcement": "reward",
        }
        saved = deepcopy(original)
        memory_case = next(c for c in panel["cases"] if c.get("history") == "A")
        accepted, interventions = language.accept_plan(original, memory_case, panel)
        self.assertEqual(original, saved)
        self.assertEqual(accepted["odor_compound"], "ethyl acetate")
        self.assertEqual(accepted["reinforcement"], "none")
        self.assertEqual(accepted["stimuli"], [])
        self.assertNotIn("history", accepted)
        self.assertEqual(accepted["duration_ms"], 250)
        self.assertTrue(any("not planner accuracy" in x["reason"] for x in interventions))
        natural, adjustments = language.accept_plan(original, panel["cases"][0], panel)
        self.assertEqual(natural["stimuli"], original["stimuli"])
        self.assertEqual([x["field"] for x in adjustments], ["duration_ms"])

    def test_trace_preserves_raw_response_failure_and_budget(self):
        class FakeLLM:
            name, model = "fake", "fixture"

            def complete(self, *args, **kwargs):
                self.last_model_ids = ["actual-fixture"]
                self.last_usage = {"input_tokens": 5, "cost_usd": 0.0}
                if args[1] == "failure":
                    raise RuntimeError("offline")
                return {"stimuli": [{"key": "sugar"}]}

        with tempfile.TemporaryDirectory() as directory:
            trace = language.TraceLLM(FakeLLM(), directory, max_calls=2)
            trace.case_id, trace.stage = "fixture", "planner"
            result = trace.complete("system", "prompt", schema={"type": "object"})
            result["stimuli"].clear()
            with self.assertRaisesRegex(RuntimeError, "offline"):
                trace.complete("system", "failure")
            with self.assertRaisesRegex(RuntimeError, "ceiling"):
                trace.complete("system", "third")
            self.assertEqual(trace.count, 2)
            saved = json.loads((Path(directory) / trace.calls[0]["path"]).read_text())
            self.assertEqual(saved["response"]["stimuli"], [{"key": "sugar"}])
            self.assertEqual(saved["actual_model_ids"], ["actual-fixture"])
            self.assertEqual(trace.calls[1]["status"], "error")

    def test_checkpoint_selection_checks_history_seed_and_content(self):
        panel = language.load_panel()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records = []
            for history, factor in (("naive", 1.0), ("A", 0.8), ("B", 0.6)):
                idx, mem = np.array([1, 2]), np.array([factor, factor], dtype=np.float32)
                digest = language.array_sha256(idx, mem)
                np.savez(root / f"{history}.npz", syn_idx=idx, mem=mem)
                for compound in ("ethyl acetate", "pyrrolidine"):
                    records.append(
                        {
                            "case": "C",
                            "history": history,
                            "phase": "probe" if history == "naive" else "post",
                            "compound": compound,
                            "seed": 11,
                            "duration_ms": 250,
                            "record_id": f"{history}-{compound}",
                            "inputs": [],
                            "memory_before": {"path": f"{history}.npz", "array_sha256": digest},
                            "raw": {"count_sha256": "fixture"},
                        }
                    )
            (root / "records.jsonl").write_text("\n".join(json.dumps(r) for r in records))
            sources = language.source_checkpoints(root, panel)
            self.assertEqual(len(sources), 6)
            self.assertEqual(sources["memory_A_ethyl_acetate"]["record_id"], "A-ethyl acetate")
            np.savez(root / "A.npz", syn_idx=np.array([1, 2]), mem=np.array([0.0, 0.0]))
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                language.source_checkpoints(root, panel)

    def test_raw_spikes_and_memory_are_complete_and_isolated(self):
        result = SimpleNamespace(
            counts={7: 2, 3: 1},
            trains={7: np.array([0.01, 0.03]), 3: np.array([0.02])},
            duration_s=0.25,
        )
        mb = SimpleNamespace(
            syn_idx=np.array([5, 8]),
            mem=np.array([0.7, 0.9], dtype=np.float32),
            log=[],
            memory_strength=lambda: 0.2,
        )
        with tempfile.TemporaryDirectory() as directory:
            raw = language.save_brain_result(result, directory, Path("trial/primary.npz"))
            with np.load(Path(directory) / raw["path"]) as z:
                np.testing.assert_array_equal(z["idx"], [3, 7])
                np.testing.assert_array_equal(z["counts"], [1, 2])
                np.testing.assert_array_equal(z["offsets"], [0, 1, 3])
                np.testing.assert_allclose(z["spike_times_s"], [0.02, 0.01, 0.03])
            snapshot = language.memory_snapshot(mb, directory)
            self.assertTrue((Path(directory) / snapshot["path"]).exists())
            self.assertEqual(snapshot["array_sha256"], language.array_sha256(mb.syn_idx, mb.mem))


if __name__ == "__main__":
    unittest.main()
