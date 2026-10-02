"""Scientific split and exposure invariants; no large network construction."""

import unittest
from collections import defaultdict

from flypet.memory_interface_data import (
    PROBE_SPLIT_NAMESPACES,
    _probe_seed,
    check_example_splits,
    make_plan,
    pair_id,
)


class MemoryInterfaceDataTests(unittest.TestCase):
    def test_sign_counterparts_and_repeats_cannot_cross_splits(self):
        p = make_plan()
        groups = defaultdict(list)
        for h in p["histories"]:
            groups[h["split_group_id"]].append(h)
        self.assertEqual(len(groups), 15)
        self.assertTrue(all(len(rows) == 24 for rows in groups.values()))
        self.assertTrue(all(len({h["split"] for h in rows}) == 1 for rows in groups.values()))
        self.assertEqual(p["expected_histories"], 360)
        self.assertEqual(p["expected_examples"], 2160)
        self.assertEqual(p["expected_records"], 3744)
        self.assertEqual(
            {
                s: list(p["split_by_group"].values()).count(s)
                for s in ("train", "validation", "test", "chemical_test")
            },
            {"train": 6, "validation": 2, "test": 2, "chemical_test": 5},
        )

    def test_counterparts_have_identical_exposures_and_opposite_dopamine(self):
        histories = make_plan()["histories"]
        for a, b in zip(histories[::2], histories[1::2]):
            self.assertEqual(a["split_group_id"], b["split_group_id"])
            for sa, sb in zip(a["steps"], b["steps"]):
                self.assertEqual(
                    (sa["chemical"], sa["seed"], sa["duration_ms"], sa["rate_hz"]),
                    (sb["chemical"], sb["seed"], sb["duration_ms"], sb["rate_hz"]),
                )
                self.assertNotEqual(sa["dan_mode"], sb["dan_mode"])

    def test_plan_deterministic_and_pair_identity_order_independent(self):
        self.assertEqual(make_plan(), make_plan())
        self.assertEqual(pair_id(["a", "b"]), pair_id(["b", "a"]))
        self.assertNotEqual(
            make_plan()["split_by_group"], make_plan(split_seed=17)["split_by_group"]
        )

    def test_reject_split_leakage(self):
        rows = [
            {"split_group_id": "one", "split": "train"},
            {"split_group_id": "one", "split": "test"},
        ]
        with self.assertRaisesRegex(ValueError, "Split leakage"):
            check_example_splits(rows)

    def test_heldout_compound_never_conditions_training_or_validation(self):
        p = make_plan()
        for h in p["histories"]:
            if h["split"] in ("train", "validation", "test"):
                self.assertNotIn(p["heldout_compound"], h["training_compounds"])
            else:
                self.assertIn(p["heldout_compound"], h["training_compounds"])

    def test_reject_invalid_exposure_plan(self):
        for options in ({"seeds": [1, 1]}, {"seeds": []}, {"pairing_trials": 0}, {"max_pairs": 0}):
            with self.assertRaises(ValueError):
                make_plan(**options)

    def test_probe_seed_pools_are_disjoint_across_splits_and_compounds(self):
        p = make_plan()
        pools = [
            {_probe_seed(seed, j, split) for seed in p["protocol"]["seeds"]}
            for split in PROBE_SPLIT_NAMESPACES
            for j in range(6)
        ]
        self.assertEqual(len(set.union(*pools)), sum(len(s) for s in pools))
        for a, b in zip(p["histories"][::2], p["histories"][1::2]):
            for j in range(6):
                self.assertEqual(
                    _probe_seed(a["history_seed"], j, a["split"]),
                    _probe_seed(b["history_seed"], j, b["split"]),
                )

    def test_invalid_probe_namespace_rejected(self):
        with self.assertRaises(ValueError):
            _probe_seed(100000, 0, "train")
        with self.assertRaises(ValueError):
            _probe_seed(101, 100, "train")


if __name__ == "__main__":
    unittest.main()
