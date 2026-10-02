"""Turn spike trains into a behaviour readout the LLM (and a human) can read."""

from __future__ import annotations
import numpy as np
from . import connectome as C
from .catalog import READOUTS, STIMULI
from .engine import RunResult


def neuron_rates(res: RunResult, exclude_stimulated: bool = True) -> list[dict]:
    flyid2i, i2flyid = C.id_maps()
    stim = set()
    for ids in res.stimulated.values():
        stim.update(ids)
    rows = []
    for i, t in res.trains.items():
        fid = i2flyid[i]
        if exclude_stimulated and fid in stim:
            continue
        d = C.describe(fid)
        d["rate_hz"] = round(len(t) / res.duration_s, 1)
        d["first_spike_ms"] = round(float(t[0]) * 1000, 1)
        rows.append(d)
    rows.sort(key=lambda r: -r["rate_hz"])
    return rows


def readout_table(res: RunResult) -> dict:
    """Per-behaviour readout: which of the catalog's output neurons fired, and how fast."""
    out = {}
    for key, r in READOUTS.items():
        ids = r.resolve()
        rates = {fid: res.rate(fid) for fid in ids}
        active = [(fid, rt) for fid, rt in rates.items() if rt > 0]
        entry = {
            "label_ru": r.label_ru,
            "n_neurons": len(ids),
            "n_active": len(active),
            "mean_rate_hz": round(float(np.mean(list(rates.values()))) if rates else 0.0, 1),
            "max_rate_hz": round(max(rates.values()) if rates else 0.0, 1),
        }
        if key in ("other_motor", "descending_all"):
            # summarise by cell type instead of listing 1300 neurons
            by_type = {}
            for fid, rt in active:
                d = C.describe(fid)
                by_type.setdefault(d["cell_type"] or "?", []).append((d["side"], round(rt, 1)))
            entry["active_types"] = dict(
                sorted(by_type.items(), key=lambda kv: -max(x[1] for x in kv[1]))[:25]
            )
        else:
            entry["neurons"] = [
                dict(C.describe(fid), rate_hz=round(rt, 1))
                for fid, rt in sorted(active, key=lambda x: -x[1])
            ]
        out[key] = entry
    return out


def summarize(res: RunResult, top_n: int = 20) -> dict:
    rows = neuron_rates(res)
    by_super = {}
    for r in rows:
        by_super[r["super_class"] or "?"] = by_super.get(r["super_class"] or "?", 0) + 1
    stim_summary = {}
    for key, ids in res.stimulated.items():
        rates = [res.rate(fid) for fid in ids]
        stim_summary[key] = {
            "n": len(ids),
            "mean_rate_hz": round(float(np.mean(rates)), 1) if rates else 0,
        }
    return {
        "duration_ms": int(res.duration_s * 1000),
        "wall_s": round(res.wall_s, 1),
        "stimulated": stim_summary,
        "n_active_downstream": len(rows),
        "n_spikes_total": int(sum(res.counts.values())),
        "active_by_super_class": by_super,
        "behaviours": readout_table(res),
        "top_downstream": [
            {
                k: r[k]
                for k in (
                    "cell_type",
                    "super_class",
                    "cell_class",
                    "side",
                    "nt",
                    "rate_hz",
                    "first_spike_ms",
                )
            }
            for r in rows[:top_n]
        ],
    }


def behaviours_text(summary: dict) -> str:
    """Compact human-readable line per behaviour (used in the CLI and as LLM input)."""
    lines = []
    for key, b in summary["behaviours"].items():
        if key in ("other_motor", "descending_all"):
            n_types = len(b.get("active_types", {}))
            lines.append(
                f"{key} [{b['label_ru']}]: {b['n_active']}/{b['n_neurons']} active, {n_types} types, max {b['max_rate_hz']} Hz"
            )
        else:
            desc = (
                ", ".join(
                    f"{n['cell_type']}/{n['side'][:1] or '?'} {n['rate_hz']}Hz"
                    for n in b["neurons"]
                )
                or "silent"
            )
            lines.append(
                f"{key} [{b['label_ru']}]: {b['n_active']}/{b['n_neurons']} active — {desc}"
            )
    if "valence" in summary:
        v = summary["valence"]
        lines.append(
            f"valence [память о запахе '{v['odor']}']: naive {v['naive_score']:+.2f} → with memory {v['score']:+.2f} (Δ {v['delta']:+.2f}); approach {v['approach_hz']} Hz, avoid {v['avoid_hz']} Hz"
        )
    if "learning" in summary:
        l = summary["learning"]
        lines.append(
            f"learning: {l['reinforcement']}, KC→MBON synapses changed {l['synapses_changed']}, memory strength {l['memory_strength']}"
        )
    if summary.get("brain_reading"):
        lines.append(
            f"brain reading [проектор состояния → Qwen3-0.6B, без меток]: {summary['brain_reading']}"
        )
    return "\n".join(lines)


def behaviour_vector(summary: dict) -> dict:
    """Map readout rates to 0..1 drive levels for the pet's body animation."""
    b = summary["behaviours"]

    def lvl(key, scale):
        return round(min(1.0, b[key]["max_rate_hz"] / scale), 2) if key in b else 0.0

    def side_rates(key):
        l = r = 0.0
        for n in b.get(key, {}).get("neurons", []):
            if n["side"] == "left":
                l = max(l, n["rate_hz"])
            elif n["side"] == "right":
                r = max(r, n["rate_hz"])
        return l, r

    tl, tr = side_rates("turn")
    return {
        "proboscis": lvl("proboscis_extension", 60),
        "escape": lvl("escape_takeoff", 150),
        "grooming": lvl("antennal_grooming", 40),
        "walk_forward": lvl("walk_forward", 40),
        "walk_backward": lvl("walk_backward", 30),
        "turn_left": round(min(1.0, tl / 40), 2),
        "turn_right": round(min(1.0, tr / 40), 2),
        "head": lvl("head_neck_movement", 30),
        "arousal": round(min(1.0, summary["n_active_downstream"] / 800), 2),
    }
