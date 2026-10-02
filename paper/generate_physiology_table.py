"""Generate the exploratory appendix table from a completed, verified run."""

import argparse, hashlib, json
from pathlib import Path


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main(a):
    run = Path(a.run)
    out = Path(__file__).resolve().parent
    report_path = run / "results/20260924T152840Z-2c5587/pilot/report.json"
    analysis_path = report_path.parent / "analysis.json"
    report = json.loads(report_path.read_text())
    analysis = json.loads(analysis_path.read_text())
    assert report["complete"] and analysis["report_sha256"] == digest(report_path)
    rows = []
    for name, label in [("trainable_core", "Trainable core"), ("frozen_core", "Frozen core")]:
        arm = report["arms"][name]
        assert arm["updates"] == 1000 and arm["test"]["metrics"]["all"]["n"] == 288
        values = [
            arm["test"]["metrics"]["all"]["accuracy"],
            arm["test_zero_state"]["metrics"]["all"]["accuracy"],
            arm["test_state_swap"]["all"]["accuracy"],
        ]
        rows.append(label + " & " + " & ".join(f"{100 * v:.2f}" for v in values) + r" \\")
    tex = "\n".join(
        [
            r"\begin{table}[t]",
            r"\centering",
            r"\caption{Exploratory physiological-core language pilot. Accuracy (\%) on 288 held-out examples from 72 paired groups. A constant answer obtains 25\%; the last-mentioned-location heuristic obtains 62.5\%.}",
            r"\label{tab:physiology-pilot}",
            r"\begin{tabular}{lrrr}",
            r"\toprule",
            r"Variant & Normal & Zero state & State swap \\",
            r"\midrule",
            *rows,
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            "",
        ]
    )
    (out / "tables/physiology_pilot.tex").write_text(tex)
    # No local paths, account metadata or provider IDs in the manuscript receipt.
    receipt = {
        "report_sha256": digest(report_path),
        "analysis_sha256": digest(analysis_path),
        "graph_sha256": report["graph_manifest"]["graph_sha256"],
        "model": report["model"],
        "revision": report["revision"],
        "scope": "exploratory appendix; no architecture advantage established",
        "table_sha256": digest(out / "tables/physiology_pilot.tex"),
    }
    (out / "analysis/physiology_pilot_sources.json").write_text(
        json.dumps(receipt, indent=2) + "\n"
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    main(p.parse_args())
