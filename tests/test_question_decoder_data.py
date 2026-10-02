"""Meaningful failure checks for the fixed-state decoder data boundary."""

import copy
import unittest

import numpy as np

from flypet.physiology_memory_task import generate
from scripts.prepare_question_decoder_data import (
    LOCATIONS,
    OBJECTS,
    audit_dataset,
    audit_probe_alignment,
    split_test_payload,
    train_normalization,
)


class QuestionDecoderDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = generate()

    def make_cache(self):
        arrays = {}
        for split in ("train", "validation"):
            rows = self.rows[split]
            history_values = {
                history: value
                for value, history in enumerate(
                    dict.fromkeys(tuple(row["history"]) for row in rows)
                )
            }
            arrays[split + "_x"] = np.asarray(
                [[history_values[tuple(row["history"])] + 0.1, 2.0] for row in rows],
                dtype=np.float32,
            )
            for key, values in {
                "y": [LOCATIONS.index(row["answer"]) for row in rows],
                "q": [OBJECTS.index(row["query_object"]) for row in rows],
                "last": [row["query_is_last_mentioned"] for row in rows],
                "group": [row["group"] for row in rows],
            }.items():
                arrays[split + "_" + key] = np.asarray(values)
        return arrays

    def test_disjoint_balanced_truth_and_group_partition_failure(self):
        audit = audit_dataset(self.rows)
        self.assertEqual(
            [audit[s]["rows"] for s in ("train", "validation", "test")], [1728, 288, 288]
        )
        broken = copy.deepcopy(self.rows)
        copied = copy.deepcopy(broken["train"][0])
        copied["split"] = "test"
        broken["test"].append(copied)
        with self.assertRaisesRegex(ValueError, "cross partitions"):
            audit_dataset(broken)
        broken = copy.deepcopy(self.rows)
        broken["train"][0]["answer"] = "unrelated"
        with self.assertRaisesRegex(ValueError, "incorrect target"):
            audit_dataset(broken)

    def test_row_alignment_and_distractor_assignment_ambiguity(self):
        cache = self.make_cache()
        audit_probe_alignment(cache, self.rows)
        wrong = {key: value.copy() for key, value in cache.items()}
        wrong["train_q"][0] = (wrong["train_q"][0] + 1) % len(OBJECTS)
        with self.assertRaisesRegex(ValueError, "train_q is not aligned"):
            audit_probe_alignment(wrong, self.rows)
        # Distractor targets/question/group are unchanged across the paired
        # histories. The paired-query check must still catch a state swap.
        distractor = next(
            i for i, row in enumerate(self.rows["train"]) if row["query_is_last_mentioned"]
        )
        row = self.rows["train"][distractor]
        opposite = next(
            i
            for i, other in enumerate(self.rows["train"])
            if other["group"] == row["group"]
            and other["assignment"] != row["assignment"]
            and other["query_object"] == row["query_object"]
        )
        wrong = {key: value.copy() for key, value in cache.items()}
        wrong["train_x"][[distractor, opposite]] = wrong["train_x"][[opposite, distractor]]
        with self.assertRaisesRegex(ValueError, "misaligned physiological features"):
            audit_probe_alignment(wrong, self.rows)

    def test_no_test_cache_or_labels_in_feature_export_payload(self):
        cache = self.make_cache()
        cache["test_y"] = np.array([0])
        with self.assertRaisesRegex(ValueError, "only known train/validation"):
            audit_probe_alignment(cache, self.rows)
        rows = copy.deepcopy(self.rows["test"])
        rows[0]["future_target_metadata"] = "must not leak"
        features, evaluation = split_test_payload(rows)
        self.assertEqual(len(features), len(evaluation))
        self.assertEqual([r["id"] for r in features], [r["id"] for r in evaluation])
        self.assertEqual(set(features[0]), {"id", "history", "query", "query_object"})
        self.assertTrue(all("events" not in row for row in features))
        self.assertTrue(
            all("answer" not in r and "future_target_metadata" not in r for r in features)
        )
        self.assertTrue(all("history" not in r and "events" not in r for r in evaluation))

    def test_normalization_depends_only_on_training_and_floors_constant_features(self):
        cache = self.make_cache()
        train_before = cache["train_x"].copy()
        mean, std = train_normalization(cache["train_x"])
        cache["validation_x"] *= 1e6
        cache["validation_y"][:] = 3
        second_mean, second_std = train_normalization(cache["train_x"])
        np.testing.assert_array_equal(mean, second_mean)
        np.testing.assert_array_equal(std, second_std)
        np.testing.assert_array_equal(train_before, cache["train_x"])
        self.assertAlmostEqual(float(std[1]), 0.01)
        self.assertAlmostEqual(float(mean[1]), 2.0)
        extreme = np.array([[1e9, -1e9]])
        clipped = np.clip((extreme - mean) / std, -10, 10)
        np.testing.assert_array_equal(clipped, np.array([[10.0, -10.0]]))


if __name__ == "__main__":
    unittest.main()
