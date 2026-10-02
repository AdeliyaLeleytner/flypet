"""Regression tests for data leakage; run with python -m unittest discover -s tests."""

from copy import deepcopy
import tempfile
from pathlib import Path
import unittest

import numpy as np

from flypet.projector_data import (
    Sample,
    canonical_state,
    generation_rows,
    prepare_samples,
    state_hash,
    write_prepared,
)


def label(stimuli=(), odor=None):
    return {
        "stimuli": [{"key": key, "side": "both", "rate_hz": 150} for key in stimuli],
        "odor": odor,
        "dopamine": None,
        "behaviour": {
            key: 0.0
            for key in (
                "walk_forward",
                "walk_backward",
                "turn_left",
                "turn_right",
                "proboscis",
                "escape",
                "grooming",
            )
        },
    }


def fixture(n=30):
    return [
        Sample(np.array([0, 5]), np.array([i + 1.0, 2.0]), label(), {"shard": "fixture", "row": i})
        for i in range(n)
    ]


class ProjectorDataTests(unittest.TestCase):
    def test_duplicates_and_conflicting_targets_stay_together(self):
        samples = fixture()
        duplicate = deepcopy(samples[0])
        duplicate.indices = duplicate.indices[::-1]
        duplicate.rates = duplicate.rates[::-1]
        duplicate.source = {"shard": "duplicate", "row": 0}
        conflict = deepcopy(samples[0])
        conflict.label["odor"] = "conflicting label"
        samples.extend([duplicate, conflict])
        prepared = prepare_samples(samples)
        manifest = prepared["manifest"]
        self.assertEqual(manifest["counts"]["duplicates_removed"], 1)
        self.assertEqual(manifest["counts"]["conflicting_target_state_groups"], 1)
        self.assertFalse(any(manifest["audit"]["state_group_overlap"].values()))
        self.assertFalse(any(manifest["audit"]["state_target_overlap"].values()))
        replay = prepare_samples(list(reversed(samples)))
        self.assertEqual(manifest["split_fingerprint"], replay["manifest"]["split_fingerprint"])
        np.testing.assert_array_equal(prepared["X"], replay["X"])

    def test_sugar_bitter_reserves_the_entire_state_group(self):
        samples = fixture()
        interaction = deepcopy(samples[2])
        interaction.label = label(("sugar", "bitter"))
        samples.append(interaction)
        prepared = prepare_samples(samples)
        manifest = prepared["manifest"]
        self.assertEqual(manifest["counts"]["interaction_metric_rows"], 1)
        self.assertEqual(manifest["counts"]["interaction_quarantined_non_sb_rows"], 1)
        for split in ("train", "validation", "test"):
            self.assertEqual(manifest["audit"][f"sugar_bitter_{split}_rows"], 0)

    def test_fit_uses_train_only_and_unknown_odor_bucket(self):
        # min_index removes upstream identity; feature99 and novel odor appear
        # exclusively in the reserved interaction set, so cannot change fit.
        samples = fixture()
        extra = Sample(
            np.array([0, 5, 99]),
            np.array([300.0, 500.0, 1.0]),
            label(("sugar", "bitter"), "held-out odor"),
            {"shard": "extra", "row": 0},
        )
        a = prepare_samples(samples, min_frequency=1)
        b = prepare_samples(samples + [extra], min_frequency=1)
        np.testing.assert_array_equal(a["feats"], b["feats"])
        np.testing.assert_allclose(a["mu"], b["mu"])
        np.testing.assert_allclose(a["sd"], b["sd"])
        self.assertNotIn(99, b["feats"])
        self.assertNotIn("held-out odor", b["words"])
        index = b["interaction_rows"][0]
        unknown = b["manifest"]["preprocessing"]["unknown_odor_column"]
        self.assertEqual(b["D"][index, unknown], 1)
        train = b["splits"]["train"]
        np.testing.assert_allclose(b["X"][train].mean(0), 0, atol=1e-5)

    def test_canonical_state_and_downstream_alias_groups(self):
        a = canonical_state(np.array([5, 0, 9]), np.array([2.0, 8.0, 0.0]), min_index=1)
        b = canonical_state(np.array([5]), np.array([2.0]), min_index=1)
        self.assertEqual(state_hash(*a), state_hash(*b))
        samples = fixture()
        for i, sample in enumerate(samples):
            sample.rates[1] = i + 1
        aliased = deepcopy(samples[0])
        aliased.rates[0] = 999  # different afferent state, identical downstream state
        samples.append(aliased)
        prepared = prepare_samples(samples, min_index=1)
        self.assertEqual(prepared["manifest"]["counts"]["duplicates_removed"], 1)
        with self.assertRaises(ValueError):
            canonical_state(np.array([1, 1]), np.array([1.0, 2.0]))
        with self.assertRaises(ValueError):
            canonical_state(np.array([1]), np.array([np.nan]))

    def test_generation_is_seeded_sample_and_artifacts_are_not_overwritten(self):
        rows = np.arange(1000)
        a = generation_rows(rows, 100, 1)
        np.testing.assert_array_equal(a, generation_rows(rows, 100, 1))
        self.assertFalse(np.array_equal(a, rows[:100]))
        self.assertEqual(len(np.unique(a)), 100)
        prepared = prepare_samples(fixture())
        with tempfile.TemporaryDirectory() as directory:
            summary = write_prepared(prepared, directory)
            self.assertEqual(summary["counts"]["unique_rows"], 30)
            self.assertTrue((Path(directory) / "manifest.json").is_file())
            with np.load(Path(directory) / "prepared.npz", allow_pickle=False) as z:
                np.testing.assert_array_equal(z["train_rows"], prepared["splits"]["train"])
            with self.assertRaises(FileExistsError):
                write_prepared(prepared, directory)


if __name__ == "__main__":
    unittest.main()
