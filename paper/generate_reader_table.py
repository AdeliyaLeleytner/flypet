"""Generate the paper's reader table from the verified result summary (no experiments)."""

from pathlib import Path
import json, hashlib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
source = ROOT / "data/projector_clean_20260924_gpu/verified_summary.json"
if not source.exists():
    source = ROOT / "paper/reader_results/verified_summary.json"
if not source.exists():
    raise FileNotFoundError(
        "Reader summary missing: export with deploy/build_paper_bundle.py --prepare-reader-results in the source checkout"
    )
z = json.loads(source.read_text())
assert z["all_local_sha256_verified"] and z["shared_split_and_generation_rows"]
runs = [z["seed_reports"][str(i)] for i in range(3)]


def mean_sd(vals, scale=1, digits=3):
    vals = np.array(vals) * scale
    return f"{vals.mean():.{digits}f} $\\pm$ {vals.std(ddof=1):.{digits}f}"


rows = []
for name, key in [
    ("Neural-state reader", "brain"),
    ("Direct stimulus input", "direct"),
    ("Reader, shuffled states", "brain_shuffled_states"),
]:
    nll = mean_sd([r["nll_test"][key] for r in runs])
    if key == "brain_shuffled_states":
        acc = interaction = "--"
    else:
        acc = mean_sd([r["gen_test"][key]["keys_exact"] for r in runs], 100, 1)
        interaction = mean_sd(
            [r["interaction_sugar_bitter"][key]["proboscis"] for r in runs], 100, 1
        )
    rows.append(f"{name} & {nll} & {acc} & {interaction} \\\\")
text = (
    r"""\begin{table}[t]
\caption{Reader characterization on the corrected split. NLL uses 578 test rows; stimulus-set accuracy uses the same 100 generated captions; interaction accuracy uses 27 sugar--bitter rows. Values are mean $\pm$ sample standard deviation over three initializations of one fixed split. Interaction majority-class accuracy is 81.5\%. Shuffled-state generation was not evaluated.}
\label{tab:reader}
\centering\small
\begin{tabular}{@{}lrrr@{}}
\toprule
Input & Test NLL $\downarrow$ & Stimuli exact (\%) & Interaction (\%) \\
\midrule
"""
    + "\n".join(rows)
    + r"""
\bottomrule
\end{tabular}
\end{table}
"""
)
(ROOT / "paper/tables/reader_results.tex").write_text(text)
(ROOT / "paper/tables/reader_source.json").write_text(
    json.dumps(
        {
            "source": str(source.relative_to(ROOT)),
            "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "seeds": [0, 1, 2],
            "statistics": "mean and sample SD, ddof=1",
            "not_a_confidence_interval": True,
        },
        indent=2,
    )
    + "\n"
)
