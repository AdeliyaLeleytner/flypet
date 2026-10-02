"""Paired-estimator and alignment regressions for the question-decoder study."""

import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from flypet.physiology_memory_task import generate
from scripts.analyze_question_decoder import (
    ARMS,
    LABELS,
    METRICS,
    SEEDS,
    analyze,
    bootstrap_contrast,
    group_metrics,
    validate_alignment,
    validate_interventions,
    validate_records,
)


def prediction(row, answer):
    probabilities = [0.7 if label == answer else 0.1 for label in LABELS]
    lp = np.log(probabilities).tolist()
    return {
        **{
            key: row[key]
            for key in ("id", "group", "assignment", "query_object", "query_is_last_mentioned")
        },
        "truth": row["answer"],
        "prediction": answer,
        "valid": True,
        "correct": answer == row["answer"],
        "token_ids": [LABELS.index(answer)],
        "generated": answer,
        "answer_log_probs": lp,
        "first_answer_token_nll": -lp[LABELS.index(row["answer"])],
    }


def swapped(records):
    by_id = {row["id"]: row for row in records}
    result = []
    for row in records:
        donor = by_id[f"{row['group']}-a{1 - row['assignment']}-{row['query_object']}"]
        result.append(
            {
                **row,
                **{
                    key: donor[key]
                    for key in ("prediction", "valid", "generated", "token_ids", "answer_log_probs")
                },
                "correct": donor["prediction"] == row["truth"],
                "first_answer_token_nll": -donor["answer_log_probs"][LABELS.index(row["truth"])],
            }
        )
    return result


class QuestionDecoderAnalysisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = generate()["test"]

    def test_known_paired_metrics_and_constant_bootstrap(self):
        perfect = validate_records([prediction(row, row["answer"]) for row in self.rows])
        constant = validate_records([prediction(row, "blue") for row in self.rows])
        group_ids, a = group_metrics(perfect)
        _, b = group_metrics(constant)
        self.assertEqual(len(group_ids), 72)
        self.assertAlmostEqual(a[:, METRICS.index("accuracy")].mean(), 1.0)
        self.assertAlmostEqual(b[:, METRICS.index("accuracy")].mean(), 0.25)
        self.assertAlmostEqual((a - b)[:, METRICS.index("accuracy")].mean(), 0.75)
        self.assertAlmostEqual(a[:, METRICS.index("both_queries_correct")].mean(), 1.0)
        all_fixed = np.full((3, 72, len(METRICS)), 0.375)
        interval = bootstrap_contrast(all_fixed, repetitions=100)
        for metric in interval["metrics"].values():
            self.assertEqual(metric["difference"], 0.375)
            self.assertEqual(metric["group_bootstrap_ci95"], [0.375, 0.375])
            self.assertEqual(metric["seed_and_group_bootstrap_ci95"], [0.375, 0.375])

    def test_alignment_and_intervention_evidence_are_checked(self):
        records = [prediction(row, row["answer"]) for row in self.rows]
        baseline = validate_records(records)
        intervention = validate_records(swapped(records))
        validate_interventions(baseline, intervention, "swapped_state")
        _, values = group_metrics(intervention)
        self.assertEqual(values[:, METRICS.index("earlier_object_accuracy")].mean(), 0.0)
        self.assertEqual(values[:, METRICS.index("most_recent_object_accuracy")].mean(), 1.0)
        wrong = copy.deepcopy(baseline)
        wrong.pop(next(iter(wrong)))
        with self.assertRaisesRegex(ValueError, "row IDs do not match"):
            validate_alignment(baseline, wrong, "bad")
        with self.assertRaisesRegex(ValueError, "matched donor prediction"):
            validate_interventions(baseline, baseline, "swapped_state")
        constant = validate_records([prediction(row, "red") for row in self.rows])
        validate_interventions(baseline, constant, "mean_state")
        wrong = copy.deepcopy(records)
        wrong[0]["first_answer_token_nll"] += 1
        with self.assertRaisesRegex(ValueError, "NLL does not match"):
            validate_records(wrong)

    def make_complete_run(self, directory):
        data = directory / "data"
        (data / "tasks").mkdir(parents=True)
        evaluation = data / "tasks/test_evaluation.jsonl"
        evaluation.write_text("".join(json.dumps(row) + "\n" for row in self.rows))
        manifest = data / "manifest.json"
        manifest.write_text(
            json.dumps(
                {
                    "files": {
                        "tasks/test_evaluation.jsonl": hashlib.sha256(
                            evaluation.read_bytes()
                        ).hexdigest()
                    }
                }
            )
        )
        data_hash = hashlib.sha256(manifest.read_bytes()).hexdigest()
        results = directory / "results/run"
        for seed in SEEDS:
            seed_dir = results / f"seed{seed}"
            seed_dir.mkdir(parents=True)
            report = {
                "status": "complete",
                "complete": True,
                "seed": seed,
                "data_manifest_sha256": data_hash,
                "features_sha256": "a" * 64,
                "test_features_sha256": "b" * 64,
                "model": "frozen-model",
                "revision": "fixed-revision",
                "config": {"seed": seed, "steps": 2000, "batch_size": 16},
                "arms": {},
            }
            for arm in ARMS:
                path = seed_dir / arm
                path.mkdir()
                records = [
                    prediction(row, row["answer"] if arm == "conditioned" else "blue")
                    for row in self.rows
                ]
                (path / "test.json").write_text(json.dumps(records))
                (path / "mean_state.json").write_text(
                    json.dumps([prediction(row, "blue") for row in self.rows])
                )
                (path / "swapped_state.json").write_text(json.dumps(swapped(records)))
                report["arms"][arm] = {
                    "complete": True,
                    "updates": 2000,
                    "best_step": 1000,
                    "validation": [{"step": 1000, "nll": 0.2}],
                    "train_selected": {"nll": 0.1},
                    "validation_selected": {"nll": 0.2},
                    "test": {"complete": True},
                }
            (seed_dir / "report.json").write_text(json.dumps(report))
        return results

    def test_full_aggregation_and_incomplete_or_hash_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            results = self.make_complete_run(Path(temp))
            report = analyze(results, repetitions=25)
            self.assertTrue(report["all_six_models_complete"])
            self.assertEqual(
                report["conditioned_minus_plain"]["metrics"]["accuracy"]["difference"], 0.75
            )
            self.assertEqual(report["aggregate"]["conditioned"]["test"]["accuracy"], 1.0)
            self.assertEqual(len(report["source_files_sha256"]), 23)
            path = results / "seed714/report.json"
            raw = json.loads(path.read_text())
            raw["test_features_sha256"] = "c" * 64
            path.write_text(json.dumps(raw))
            with self.assertRaisesRegex(ValueError, "differs across seeds"):
                analyze(results, repetitions=25)
            raw["test_features_sha256"] = "b" * 64
            raw["complete"] = False
            path.write_text(json.dumps(raw))
            with self.assertRaisesRegex(ValueError, "incomplete"):
                analyze(results, repetitions=25)

    def test_recovery_accepts_1000_updates_on_one_gpu_but_not_mixed_campaigns(self):
        with tempfile.TemporaryDirectory() as temp:
            results = self.make_complete_run(Path(temp))
            for seed in SEEDS:
                path = results / f"seed{seed}/report.json"
                report = json.loads(path.read_text())
                report["config"]["steps"] = 1000
                report["gpu"] = "same test GPU for all three seeds"
                for arm in ARMS:
                    report["arms"][arm]["updates"] = 1000
                path.write_text(json.dumps(report))
            summary = analyze(results, repetitions=25)
            self.assertTrue(summary["all_six_models_complete"])
            self.assertTrue(
                all(
                    summary["selected_checkpoints"][str(seed)][arm]["updates"] == 1000
                    for seed in SEEDS
                    for arm in ARMS
                )
            )
            # A broader results directory containing an aborted campaign must
            # be rejected; callers must identify the recovery campaign itself.
            old = results.parent / "failed_campaign/seed713"
            old.mkdir(parents=True)
            (old / "report.json").write_text(json.dumps({"status": "aborted", "complete": False}))
            with self.assertRaisesRegex(ValueError, "exactly one report"):
                analyze(results.parent, repetitions=25)


if __name__ == "__main__":
    unittest.main()
