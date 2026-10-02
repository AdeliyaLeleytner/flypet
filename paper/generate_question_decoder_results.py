"""Generate anonymous manuscript results only from all six completed real fits.

Run the analysis script first. This generator rechecks source hashes and repeats
the paired analysis before writing any table or prose. It never supplies missing
results or substitutes a successful subset of runs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.analyze_question_decoder import ARMS, SEEDS, analyze, require

LABELS = {"plain": "State-only decoder", "conditioned": "Question-conditioned decoder"}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def anonymous_source_name(path: Path) -> str:
    parts = path.parts
    for i, part in enumerate(parts):
        if re.fullmatch(r"seed(?:713|714|715)", part):
            return "/".join(parts[i:])
    if path.name == "manifest.json":
        return "data/manifest.json"
    if path.name == "test_evaluation.jsonl":
        return "data/tasks/test_evaluation.jsonl"
    raise ValueError(f"Unrecognized manuscript source kind: {path.name}")


def load_verified(analysis_path: Path) -> tuple[dict, dict, dict]:
    analysis = json.loads(analysis_path.read_text())
    require(
        analysis.get("status") == "complete" and analysis.get("all_six_models_complete") is True,
        "paper output requires all six completed fits",
    )
    require(analysis.get("seeds") == list(SEEDS), "expected all three fixed seeds")
    report_paths, anonymous_hashes, data_dir = {}, {}, None
    for raw, expected in analysis["source_files_sha256"].items():
        path = Path(raw)
        require(
            path.is_file() and digest(path) == expected,
            f"source file missing or changed: {path.name}",
        )
        name = anonymous_source_name(path)
        require(name not in anonymous_hashes, "ambiguous anonymous source identity")
        anonymous_hashes[name] = expected
        if path.name == "report.json":
            report_paths[int(path.parent.name.removeprefix("seed"))] = path
        if name == "data/manifest.json":
            data_dir = path.parent
    require(
        set(report_paths) == set(SEEDS) and data_dir is not None,
        "missing fit reports or canonical data",
    )
    roots = {path.parent.parent for path in report_paths.values()}
    require(len(roots) == 1, "fit reports belong to different experiment directories")
    fresh = analyze(next(iter(roots)), data_dir=data_dir, repetitions=10000)
    # Exact deterministic recomputation protects both point estimates and intervals.
    for key, value in fresh.items():
        require(analysis.get(key) == value, f"analysis no longer matches raw results: {key}")
    reports = {seed: json.loads(path.read_text()) for seed, path in report_paths.items()}
    for report in reports.values():
        require(
            report.get("history_text_in_primary_lm") is False
            and report.get("core_and_encoder_trainable") is False,
            "these manuscript templates describe frozen initial states without history text in the LM",
        )
        require(
            report.get("initial_arm_prefix_equivalence") is True,
            "matched initial prefixes were not verified",
        )
        for arm in ARMS:
            require(
                report["arms"][arm]["initial_decoder_sha256"] == report["initial_decoder_sha256"],
                "decoder arms did not share initialization",
            )
    controls = ("steps", "batch_size", "lr", "eval_every")
    require(
        len({tuple(report["config"][key] for key in controls) for report in reports.values()}) == 1,
        "manuscript recipe differs across seeds",
    )
    dataset = json.loads((data_dir / "manifest.json").read_text())["dataset_audit"]
    receipt = {
        "analysis_sha256": digest(analysis_path),
        "sources": anonymous_hashes,
        "source_fingerprints": analysis["source_fingerprints"],
        "source_verification": "All raw file hashes checked; point estimates and 10000-draw paired intervals recomputed exactly.",
        "seeds": list(SEEDS),
        "n_models": 6,
        "dataset_counts": {key: value["rows"] for key, value in dataset.items()},
        "scope": "Exploratory question-access comparison with fixed initial physiological states; prior test split reused.",
    }
    return analysis, reports, receipt


def tex_escape(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(c, c) for c in text)


def load_ridge_verified(path: Path, analysis: dict) -> tuple[dict, dict]:
    # Keep Torch optional: the plotting runtime only needs NumPy/Matplotlib.
    from scripts.evaluate_question_decoder_ridge import (
        LABELS as RIDGE_LABELS,
        summarize as summarize_ridge,
    )

    ridge = json.loads(path.read_text())
    require(
        ridge.get("status") == "complete" and ridge.get("regression", {}).get("passed") is True,
        "ridge reference is incomplete or failed regression",
    )
    require(
        ridge.get("gpu_used") is False
        and ridge["recipe"]["lambda"] == 10
        and ridge["recipe"]["hyperparameter_search"] is False,
        "unexpected ridge reference recipe",
    )
    require(ridge.get("score_label_order") == list(RIDGE_LABELS), "ridge score labels differ")
    require(
        set(ridge["source_file_paths"]) == set(ridge["source_hashes"]),
        "ridge source pointers are incomplete",
    )
    for name, raw in ridge["source_file_paths"].items():
        source = Path(raw)
        require(
            source.is_file() and digest(source) == ridge["source_hashes"][name],
            f"ridge source missing or changed: {name}",
        )
    for name, key in (
        ("data/manifest.json", "data_manifest_sha256"),
        ("data/initial_features.npz", "features_sha256"),
        ("data/test_features.npz", "test_features_sha256"),
    ):
        require(
            ridge["source_hashes"][name] == analysis["source_fingerprints"][key],
            "ridge and primary state/data identity differs",
        )
    for split, filename in (
        ("validation", "data/tasks/validation.jsonl"),
        ("test", "data/tasks/test_evaluation.jsonl"),
    ):
        rows = [
            json.loads(line)
            for line in Path(ridge["source_file_paths"][filename]).read_text().splitlines()
            if line.strip()
        ]
        canonical = {row["id"]: row for row in rows}
        require(len(canonical) == 288, "ridge reference must score all 288 canonical rows")
        base = None
        for condition in ("intact", "mean_state", "swapped_state"):
            value = ridge[split][condition]
            records = value["records"]
            require(
                len(records) == len(canonical)
                and len({row["id"] for row in records}) == len(canonical),
                "ridge record IDs are incomplete or duplicated",
            )
            require(
                {row["id"] for row in records} == set(canonical),
                "ridge and canonical row IDs differ",
            )
            for row in records:
                truth = canonical[row["id"]]
                require(
                    all(
                        row[key] == truth[key]
                        for key in (
                            "group",
                            "assignment",
                            "query_object",
                            "query_is_last_mentioned",
                        )
                    )
                    and row["truth"] == truth["answer"],
                    "ridge truth/question metadata mismatch",
                )
                scores = np.asarray(row["ridge_scores"], dtype=float)
                require(scores.shape == (4,) and np.isfinite(scores).all(), "invalid ridge scores")
                predicted = RIDGE_LABELS[int(scores.argmax())]
                require(
                    row["prediction"] == predicted
                    and isinstance(row["correct"], bool)
                    and row["correct"] == (predicted == row["truth"]),
                    "ridge stored correctness is inconsistent",
                )
            require(
                summarize_ridge(records) == value["metrics"],
                "ridge aggregate metrics do not match its records",
            )
            if condition == "intact":
                base = {row["id"]: row for row in records}
            elif condition == "mean_state":
                by_question = {}
                for row in records:
                    question = row["query_object"]
                    if question in by_question:
                        require(
                            row["ridge_scores"] == by_question[question],
                            "ridge mean-state control varies within question",
                        )
                    by_question[question] = row["ridge_scores"]
            else:
                for row in records:
                    donor_id = f"{row['group']}-a{1 - row['assignment']}-{row['query_object']}"
                    require(
                        row["donor_id"] == donor_id
                        and row["ridge_scores"] == base[donor_id]["ridge_scores"],
                        "ridge swap does not use the paired donor's scores",
                    )
    receipt = {
        "report_sha256": digest(path),
        "source_hashes": ridge["source_hashes"],
        "recipe": ridge["recipe"],
        "decision_timing": ridge["decision_timing"],
        "scope": ridge["scope"],
        "test_status": ridge["test_status"],
        "normalizer_rounding_relative_to_softprefix_cache": ridge[
            "normalizer_rounding_relative_to_softprefix_cache"
        ],
        "verification": "Prior-recipe regression passed; raw file hashes and primary cache identity verified; every validation/test record, control donor and aggregate correctness recomputed.",
    }
    return ridge, receipt


def seed_values(analysis: dict, arm: str, condition: str, metric: str) -> np.ndarray:
    return np.asarray([analysis["per_seed"][str(seed)][arm][condition][metric] for seed in SEEDS])


def mean_sd(values: np.ndarray, *, scale: float = 1.0, digits: int = 1) -> str:
    scaled = np.asarray(values) * scale
    return f"{scaled.mean():.{digits}f} $\\pm$ {scaled.std(ddof=1):.{digits}f}"


def interval(values: list[float], *, scale: float = 1.0, digits: int = 2) -> str:
    return f"$[{values[0] * scale:.{digits}f}, {values[1] * scale:.{digits}f}]$"


def make_table(analysis: dict) -> str:
    rows = []
    for arm in ARMS:
        for condition, label in (
            ("test", "Intact"),
            ("mean_state", "Train mean"),
            ("swapped_state", "Paired swap"),
        ):
            numbers = [
                mean_sd(seed_values(analysis, arm, condition, metric), scale=100.0)
                for metric in (
                    "accuracy",
                    "earlier_object_accuracy",
                    "most_recent_object_accuracy",
                    "both_queries_correct",
                )
            ]
            numbers.append(mean_sd(seed_values(analysis, arm, condition, "nll"), digits=2))
            name = "State only" if arm == "plain" else "State + question"
            rows.append(" & ".join([name, label, *numbers]) + r" \\")
        if arm == "plain":
            rows.append(r"\midrule")
    return "\n".join(
        [
            r"\begin{table}[t]",
            r"\centering\small",
            r"\setlength{\tabcolsep}{3.5pt}",
            r"\caption{Question access at the decoder, with the encoder and physiological states held fixed. Values are mean $\pm$ sample standard deviation over three initializations; this is not a confidence interval. Accuracy columns are percentages on the same 288 examples. Earlier/latest identify which object's location is queried; Both requires correct answers to both questions about one history. NLL is the first-answer-token negative log-likelihood in nats. Invalid generated answers count as errors. Chance accuracy is 25\%. The test split was used by the previous pilot and is reused here for exploration.}",
            r"\label{tab:question-decoder}",
            r"\begin{tabular}{@{}llrrrrr@{}}",
            r"\toprule",
            r"Decoder & State & Overall & Earlier & Latest & Both & NLL $\downarrow$ \\",
            r"\midrule",
            *rows,
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            "",
        ]
    )


def make_contrast_table(analysis: dict) -> str:
    metrics = analysis["conditioned_minus_plain"]["metrics"]
    rows = []
    for key, label, scale in (
        ("accuracy", "Overall accuracy (pp)", 100),
        ("earlier_object_accuracy", "Earlier-object accuracy (pp)", 100),
        ("most_recent_object_accuracy", "Latest-object accuracy (pp)", 100),
        ("both_queries_correct", "Both questions correct (pp)", 100),
        ("counterfactual_both_correct", "Both paired histories correct (pp)", 100),
        ("nll", "Answer-token NLL (nats)", 1),
    ):
        metric = metrics[key]
        rounded_difference = round(metric["difference"] * scale, 2)
        difference_text = f"{rounded_difference:+.2f}" if rounded_difference else "0.00"
        rows.append(
            " & ".join(
                [
                    label,
                    difference_text,
                    interval(metric["group_bootstrap_ci95"], scale=scale),
                    interval(metric["seed_and_group_bootstrap_ci95"], scale=scale),
                ]
            )
            + r" \\"
        )
    return "\n".join(
        [
            r"\begin{table}[t]",
            r"\centering\small",
            r"\caption{Paired differences, question-conditioned minus state-only decoder. Positive accuracy differences and negative NLL differences favor conditioning. Intervals use 10,000 bootstrap samples of the 72 whole counterfactual groups; the final column additionally resamples the three initialization indices. These intervals describe this exploratory reused split and three seeds, not population-level architecture superiority. Both paired histories correct requires correct answers for the same queried object under both event orders.}",
            r"\label{tab:question-decoder-contrasts}",
            r"\begin{tabular}{@{}lrrr@{}}",
            r"\toprule",
            r"Metric & Mean difference & Group interval & Seed + group interval \\",
            r"\midrule",
            *rows,
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            "",
        ]
    )


def make_section(analysis: dict, reports: dict, receipt: dict, ridge: dict | None = None) -> str:
    config = reports[SEEDS[0]]["config"]
    counts = receipt["dataset_counts"]
    plain = analysis["aggregate"]["plain"]["test"]
    conditioned = analysis["aggregate"]["conditioned"]["test"]
    delta = analysis["conditioned_minus_plain"]["metrics"]
    per_seed = [
        analysis["per_seed_conditioned_minus_plain"][str(seed)]["accuracy"] * 100 for seed in SEEDS
    ]
    warm = [(reports[seed]["arms"][arm]["best_step"]) for seed in SEEDS for arm in ARMS]
    inactive = reports[SEEDS[0]]["plain_inactive_query_parameters"]
    nll_change = (
        "less than 0.001 nats in magnitude"
        if round(delta["nll"]["difference"], 3) == 0
        else f"{delta['nll']['difference']:+.3f} nats"
    )
    paragraphs = [
        r"\section{Question access when decoding fixed physiological states}",
        r"\label{app:question-decoder}",
        "The physiological pilot leaves open whether the readout can retrieve the location of the queried object from a shared neural state. We isolate this interface by holding the initial encoder and physiological core fixed and varying whether the decoder receives the question identity. The frozen language model receives the natural-language question in both conditions.",
        r"\paragraph{Matched comparison.}",
        "Both decoders map the same cached 768-dimensional final voltage and synaptic-state vector to eight soft tokens. One decoder receives three zero channels; the other receives a one-hot identity for the queried object (key, coin, or ring). This identity contains no location label. Layer sizes, initialization, minibatch order, and optimizer settings match within each seed. Query-column weights start at zero, giving exactly equal initial prefixes. The state-only decoder has "
        + str(inactive)
        + " inactive query-column coefficients, so nominal parameter counts match while the additional channel changes which coefficients can learn. Normalization uses training-state means and sample standard deviations, floored at 0.01, with standardized values clipped to $[-10,10]$.",
        f"The comparison uses {counts['train']:,} training, {counts['validation']:,} validation, and {counts['test']:,} test examples. Each of three initializations receives {config['steps']:,} updates with batch size {config['batch_size']}, Adam learning rate {config['lr']:g}, and full-validation evaluation every {config['eval_every']} updates. Validation first-answer-token NLL selects the checkpoint independently for each arm; selected checkpoints range from update {min(warm)} to {max(warm)}. Both checkpoint choices within each seed are fixed before that seed's test evaluation. At evaluation, the language model receives only the decoded state and question. History sentences supply the legitimate event facts to the frozen physiological encoder; history text is never supplied to the primary language-model arms. Training uses answer-token supervision, and evaluation targets are kept separate from feature export and prediction.",
        r"\input{tables/question_decoder_results}",
        r"\paragraph{Observed decoding performance.}",
        f"The question-conditioned decoder obtains {100 * conditioned['accuracy']:.2f}\\% mean strict generated-answer accuracy, compared with {100 * plain['accuracy']:.2f}\\% for the state-only decoder. Their paired difference is {100 * delta['accuracy']['difference']:+.2f} percentage points, with a whole-group bootstrap interval of {interval(delta['accuracy']['group_bootstrap_ci95'], scale=100)} and a seed-and-group interval of {interval(delta['accuracy']['seed_and_group_bootstrap_ci95'], scale=100)}. The individual seed differences are "
        + ", ".join(f"${v:+.2f}$" for v in per_seed)
        + " percentage points. Mean answer-token NLL is "
        + f"{conditioned['nll']:.3f} versus {plain['nll']:.3f} nats; the conditioned-minus-state-only difference is {nll_change}. Table~\\ref{{tab:question-decoder-contrasts}} also reports joint-answer and paired-history criteria.",
        r"\paragraph{Dependence on the supplied state.}",
        "We replace every state with the training-state mean, and separately substitute the state from the opposite-order history while keeping the queried object unchanged. Replacing a state with its training mean removes history-specific variation but can also move the decoder away from the distribution of states seen during training. The paired swap uses an actual history state and preserves the multiset of events while changing which of the target object's moves occurred last; the distractor object's answer remains unchanged. Control outputs are scored against each recipient's original answer.",
    ]
    for arm in ARMS:
        intact = analysis["aggregate"][arm]["test"]["accuracy"] * 100
        mean = analysis["aggregate"][arm]["mean_state"]["accuracy"] * 100
        swap = analysis["aggregate"][arm]["swapped_state"]["accuracy"] * 100
        earlier_intact = analysis["aggregate"][arm]["test"]["earlier_object_accuracy"] * 100
        earlier_swap = analysis["aggregate"][arm]["swapped_state"]["earlier_object_accuracy"] * 100
        effect = analysis["state_effects"][arm]["swapped_state"]["metrics"][
            "earlier_object_accuracy"
        ]
        paragraphs.append(
            f"For the {LABELS[arm].lower()}, intact, mean-state, and swapped-state overall accuracies are {intact:.2f}\\%, {mean:.2f}\\%, and {swap:.2f}\\%, respectively. On earlier-object questions, swapping changes accuracy from {earlier_intact:.2f}\\% to {earlier_swap:.2f}\\%; the intact-minus-swapped difference has group interval {interval(effect['group_bootstrap_ci95'], scale=100)} percentage points."
        )
    mean_valid = [analysis["aggregate"][arm]["mean_state"]["valid_fraction"] * 100 for arm in ARMS]
    intact_valid = [analysis["aggregate"][arm]["test"]["valid_fraction"] * 100 for arm in ARMS]
    all_invalid = [
        f"seed {seed}, {LABELS[arm].lower()}"
        for seed in SEEDS
        for arm in ARMS
        if analysis["per_seed"][str(seed)][arm]["mean_state"]["valid_fraction"] == 0
    ]
    caveat = f"Valid-answer fractions on intact states are {intact_valid[0]:.2f}\\% and {intact_valid[1]:.2f}\\% for the state-only and question-conditioned decoders. With the training-mean state these become {mean_valid[0]:.2f}\\% and {mean_valid[1]:.2f}\\%."
    if all_invalid:
        caveat += " All mean-state outputs are invalid for " + "; ".join(all_invalid) + "."
    if min(mean_valid) < 100:
        caveat += " The mean-state accuracy decrease therefore mixes changed input information with response-format failures under an altered state distribution; it is not an isolated estimate of the causal contribution of history information."
    paragraphs.append(caveat)
    paragraphs += [r"\input{tables/question_decoder_contrasts}"]
    if ridge is not None:
        test = ridge["test"]["intact"]["metrics"]
        validation = ridge["validation"]["intact"]["metrics"]
        mean = ridge["test"]["mean_state"]["metrics"]["all"]["accuracy"] * 100
        swap = ridge["test"]["swapped_state"]["metrics"]["all"]["accuracy"] * 100
        paragraphs += [
            r"\paragraph{Post-hoc numerical reference.}",
            f"After inspecting the first seed's language-model test result, we applied the already-used ridge-regression recipe with fixed $\\lambda=10$, one four-score head per queried object, and its original Torch float32 train-only normalization. No hyperparameters were tuned or models selected using the new test results. The reference exactly reproduces the previous validation predictions ({100 * validation['all']['accuracy']:.2f}\\% accuracy). On the shared test states it obtains {100 * test['all']['accuracy']:.2f}\\% accuracy: {100 * test['earlier_object']['accuracy']:.2f}\\% for the earlier object and {100 * test['most_recent_object']['accuracy']:.2f}\\% for the latest object. Test accuracy is {mean:.2f}\\% with the training-mean state and {swap:.2f}\\% with the paired-swapped state. This is a post-hoc numerical classifier reference, not language generation, an exposure-matched architecture comparison, or a ceiling on recoverable information. Its four-score cross-entropy is not compared with the language model's full-vocabulary token NLL.",
        ]
    paragraphs += [
        r"\paragraph{Scope.}",
        "This experiment measures retrieval through a particular decoder under a fixed physiological representation. It does not train the connectome or compare biological connectivity with a rewired or conventional recurrent core. The same test split was already evaluated in the preceding pilot; the present analysis is exploratory reuse rather than fresh confirmation. Bootstrap intervals preserve whole counterfactual groups, but three initialization seeds provide only limited evidence about optimization variability. A decoder effect therefore cannot establish a new memory mechanism or a general advantage of connectome-based architectures.",
        r"\begin{figure}[t]",
        r"\centering",
        r"\includegraphics[width=\linewidth]{figures/question_decoder_results.pdf}",
        r"\caption{Decoding the same fixed physiological states with and without question identity at the decoder. (a) Full-validation first-answer-token NLL; thin curves show three initializations and thick curves their mean. The dotted line marks $\log 4$, a uniform distribution over four color tokens. (b) Strict test accuracy; connected points pair initialization seeds, and black marks show means. (c) Accuracy by the queried object's position in the history, with individual seeds overlaid. (d) Overall accuracy under intact, training-mean, and paired-swapped states. Accuracy panels use a common 0--100\% scale; dotted lines mark 25\%. The plot retains all three seeds and both decoder arms.}",
        r"\label{fig:question-decoder}",
        r"\end{figure}",
        "",
    ]
    return "\n\n".join(paragraphs)


def generate(analysis_path: Path, paper_dir: Path, ridge_reference: Path | None = None) -> dict:
    analysis, reports, receipt = load_verified(analysis_path)
    ridge = None
    if ridge_reference is not None:
        ridge, ridge_receipt = load_ridge_verified(ridge_reference, analysis)
        receipt["ridge_reference"] = ridge_receipt
    outputs = {
        "tables/question_decoder_results.tex": make_table(analysis),
        "tables/question_decoder_contrasts.tex": make_contrast_table(analysis),
        "sections/question_decoder_results.tex": make_section(analysis, reports, receipt, ridge),
    }
    for name, text in outputs.items():
        require(
            "/Users/" not in text and "/home/" not in text and "/workspace/" not in text,
            "private path in manuscript",
        )
        path = paper_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    receipt["generator_sha256"] = digest(Path(__file__))
    receipt["outputs"] = {name: digest(paper_dir / name) for name in outputs}
    path = paper_dir / "analysis/question_decoder_sources.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--paper-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument(
        "--ridge-reference", type=Path, help="Optional completed fixed-recipe numerical reference."
    )
    args = parser.parse_args()
    receipt = generate(args.analysis, args.paper_dir, args.ridge_reference)
    print(
        json.dumps(
            {"outputs": receipt["outputs"], "analysis_sha256": receipt["analysis_sha256"]}, indent=2
        )
    )
