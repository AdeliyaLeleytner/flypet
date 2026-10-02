#!/usr/bin/env python3
"""Export only static garden assets, never connectome data, sessions or secrets."""

import argparse
import json
from pathlib import Path
import re
import shutil
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]


def export(output, api_base, local_preview=False, standalone=False):
    url = urlsplit(api_base)
    local = local_preview and url.scheme == "http" and url.hostname in ("localhost", "127.0.0.1")
    if (
        (url.scheme != "https" and not local)
        or not url.hostname
        or url.username
        or url.password
        or url.query
        or url.fragment
        or not re.fullmatch(r"[/a-zA-Z0-9_-]*", url.path)
    ):
        raise ValueError(
            "Use a credential-free HTTPS API origin (local HTTP requires --local-preview)."
        )
    output = Path(output)
    if output.exists():
        raise FileExistsError(
            "Choose a new output directory; existing files are never overwritten."
        )
    (output / "static").mkdir(parents=True)
    html = (
        (ROOT / "flypet/static/garden.html").read_text().replace('src="/static/', 'src="./static/')
    )
    (output / "index.html").write_text(html)
    for name in ("garden-brain.js", "garden-controls.js"):
        shutil.copyfile(ROOT / "flypet/static" / name, output / "static" / name)
    (output / "static/garden-config.js").write_text(
        "window.FLYPET_API_BASE = " + json.dumps(api_base.rstrip("/")) + ";\n"
    )
    (output / "README.md").write_text(
        "---\ntitle: Fly's Garden\nemoji: 🪰\ncolorFrom: yellow\ncolorTo: green\nsdk: static\napp_file: index.html\npinned: false\n---\n\n"
        "Interactive garden frontend. Neural simulation runs on a separately configured server.\n"
        "The API must allow this Space's exact https://<owner>-<space>.hf.space origin.\n"
        "No server credentials, model weights, private memory or connectome datasets are included.\n\n"
        "## What you can do\n\n"
        "Offer 27 sensory stimuli and mixtures, adjust eight physical channels, pair measured odor proxies with reinforcement, and inspect neural readouts. "
        "Each browser gets a temporary isolated memory; sessions expire after one hour of inactivity or a backend restart. "
        "The canvas illustrates neural outputs, not body mechanics or an animal's subjective experience.\n\n"
        "English readout descriptions arrive with the simulation. Optional Qwen3-0.6B reflections run separately; wording may be cached for identical descriptive evidence. "
        "Neural trials themselves are computed live. Concurrent simulations are bounded by the small shared server.\n\n"
        "## Scientific sources and attribution\n\n"
        "- [Shiu et al. brain model](https://github.com/philshiu/Drosophila_brain_model): MIT model code.\n"
        "- [FlyWire v783](https://home.flywire.ai/guidelines): CC BY-NC 4.0; see the source citation credits.\n"
        "- [DoOR 2.0](https://github.com/ropensci/DoOR.data): CC BY-SA 4.0; Daniel Muench, C. Giovanni Galizia and contributing measurements.\n"
        "- [MaleCNS v1.0](https://male-cns.janelia.org/download/): CC BY 4.0; FlyEM/HHMI Janelia, Cambridge, MRC LMB and Google Research.\n"
        "- [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B): Apache 2.0 base model.\n\n"
        "The female FlyWire brain and male VNC are connected by a one-way type-based bridge. "
        "Fruit labels select single chemical proxies, not measured complete fruit bouquets. "
        "Source terms remain applicable to displayed derived data; this frontend does not relicense upstream assets.\n"
    )
    if standalone:
        html = (output / "index.html").read_text()
        for name in ("garden-config.js", "garden-brain.js", "garden-controls.js"):
            script = (output / "static" / name).read_text().replace("</script", "<\\/script")
            html = html.replace(
                f'<script src="./static/{name}"></script>', "<script>\n" + script + "\n</script>"
            )
        (output / "index.html").write_text(html)
    return {
        "output": str(output.resolve()),
        "api_base": api_base,
        "local_preview": local_preview,
        "files": sorted(p.relative_to(output).as_posix() for p in output.rglob("*") if p.is_file()),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-base", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--local-preview", action="store_true")
    parser.add_argument(
        "--standalone", action="store_true", help="Inline scripts for a two-file browser upload."
    )
    args = parser.parse_args()
    print(
        json.dumps(
            export(args.output, args.api_base, args.local_preview, args.standalone), indent=2
        )
    )
