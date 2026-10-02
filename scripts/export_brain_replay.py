#!/usr/bin/env python3
"""Export a self-contained 3D spike-replay page from recorded paper trials (no new simulation).

Neuron positions are FlyWire v783 annotation points (voxels 4x4x40 nm); spikes are the stored
spike trains of seed-11 trials in paper/results/cases-v1. Output: one static HTML file.
"""

from __future__ import annotations
import argparse
import base64
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "paper/results/cases-v1"
TRIALS = [
    ("0024-B-looming-11", "Looming shadow"),
    ("0023-B-antenna_touch-11", "Antenna touch"),
    ("0001-A-odor1-11", "Ethyl acetate (fruity odour)"),
    ("0020-B-sugar-11", "Sugar taste"),
]
LABELS = {
    "proboscis_extension": "Proboscis extension",
    "antennal_grooming": "Antennal grooming",
    "escape_takeoff": "Escape / take-off",
    "walk_forward": "Walk forward",
    "walk_backward": "Walk backward",
    "turn": "Turn",
    "head_neck_movement": "Head movement",
    "other_motor": "Other brain motor neurons",
    "descending_all": "All descending neurons",
}


def b64(a: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, default=ROOT / "output/brain-replay/index.html")
    args = ap.parse_args(argv)
    root_ids = np.load(RUN / "brain_root_ids.npy")
    ann = (
        pd.read_csv(
            ROOT / "data/flywire_annotations_783.tsv",
            sep="\t",
            low_memory=False,
            usecols=[
                "root_id",
                "pos_x",
                "pos_y",
                "pos_z",
                "super_class",
                "cell_class",
                "cell_type",
                "side",
            ],
        )
        .drop_duplicates("root_id")
        .set_index("root_id")
        .reindex(root_ids)
    )
    xyz = np.stack([ann.pos_x * 4e-3, ann.pos_y * 4e-3, ann.pos_z * 40e-3], 1)  # micrometres
    lo, hi = np.nanmin(xyz, 0), np.nanmax(xyz, 0)
    q = np.round((np.nan_to_num(xyz, nan=0) - lo) / (hi - lo) * 65535).astype("<u2")
    classes = sorted(ann.super_class.fillna("unknown").astype(str).unique())
    cls = (
        ann.super_class.fillna("unknown")
        .astype(str)
        .map({c: i for i, c in enumerate(classes)})
        .to_numpy()
        .astype(np.uint8)
    )
    records = {}
    for line in open(RUN / "records.jsonl"):
        r = json.loads(line)
        if r["record_id"] in dict(TRIALS):
            records[r["record_id"]] = r
    trials = []
    for rid, title in TRIALS:
        r = records[rid]
        z = np.load(RUN / r["raw"]["path"])
        neuron = np.repeat(z["idx"], z["counts"]).astype("<u4")
        t = z["spike_times_s"]
        order = np.argsort(t, kind="stable")
        neuron, t = neuron[order], t[order]
        t01 = np.clip(np.round(t * 1e4), 0, 65535).astype("<u2")  # 0.1 ms units
        active, counts = np.unique(neuron, return_counts=True)
        info = [
            [
                int(i),
                str(ann.cell_type.iloc[i])
                if pd.notna(ann.cell_type.iloc[i])
                else str(ann.cell_class.iloc[i]),
                str(ann.super_class.iloc[i]),
                str(ann.side.iloc[i]),
                int(c),
            ]
            for i, c in zip(active, counts)
        ]
        rate = np.histogram(t * 1e3, bins=np.arange(0, r["duration_ms"] + 1, 1.0))[0]
        beh = r["summary"]["behaviours"]
        readout = [
            [
                LABELS.get(k, k),
                round(float(v["max_rate_hz"]), 1),
                int(v["n_active"]),
                int(v["n_neurons"]),
            ]
            for k, v in beh.items()
            if isinstance(v, dict) and v.get("max_rate_hz", 0) > 0
        ]
        inputs = sorted({k.split(":")[0].split("/")[0] for k in r["stimulated"]})
        trials.append(
            {
                "id": rid,
                "title": title,
                "duration_ms": r["duration_ms"],
                "inputs": inputs,
                "n_spikes": int(len(t)),
                "n_active": int(len(active)),
                "neuron": b64(neuron),
                "t": b64(t01),
                "active": info,
                "rate": rate.tolist(),
                "readout": readout,
            }
        )
    data = {
        "n": int(len(root_ids)),
        "lo": lo.tolist(),
        "hi": hi.tolist(),
        "pos": b64(q),
        "cls": b64(cls),
        "classes": classes,
        "trials": trials,
    }
    html = TEMPLATE.replace("__DATA__", json.dumps(data, separators=(",", ":")))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(html)
    print(
        f"wrote {args.output} ({args.output.stat().st_size / 1e6:.1f} MB), trials:",
        [(x["title"], x["n_spikes"], x["n_active"]) for x in trials],
    )


TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Fly Brain Replay</title>
<style>
:root{--bg:#07090d;--panel:rgba(14,18,26,.86);--ink:#e8edf5;--dim:#8b97a8;--line:#243044;--accent:#ffd27a}
*{box-sizing:border-box}html,body{margin:0;height:100%;background:var(--bg);color:var(--ink);font:14px/1.45 system-ui,-apple-system,Segoe UI,sans-serif;overflow:hidden}
#view{position:fixed;inset:0}
.panel{position:fixed;background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:12px 14px;backdrop-filter:blur(6px)}
#top{left:16px;top:16px;max-width:380px}
#top h1{font-size:17px;margin:0 0 4px}#top p{margin:0;color:var(--dim);font-size:12.5px}
#trials{display:flex;flex-wrap:wrap;gap:6px;margin-top:10px}
button{background:#141b27;color:var(--ink);border:1px solid var(--line);border-radius:7px;padding:6px 10px;font:inherit;font-size:13px;cursor:pointer}
button.on{border-color:var(--accent);color:var(--accent)}
#side{right:16px;top:16px;width:300px;max-height:calc(100% - 200px);overflow:auto}
#side h2{font-size:13px;margin:0 0 6px;color:var(--dim);font-weight:600;text-transform:uppercase;letter-spacing:.04em}
.row{display:flex;justify-content:space-between;gap:8px;font-size:13px;padding:2px 0}.row b{color:var(--accent);font-weight:600}
#pick{margin-top:10px;font-size:13px;color:var(--dim)}#pick b{color:var(--ink)}
#legend{display:flex;flex-wrap:wrap;gap:4px 10px;margin-top:10px;font-size:12px;color:var(--dim)}
#legend span i{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:5px;vertical-align:-1px}
#bottom{left:16px;right:16px;bottom:16px;display:flex;gap:12px;align-items:center}
#rate{flex:1;height:64px;width:100%}
#clock{font-variant-numeric:tabular-nums;min-width:120px;text-align:right;color:var(--dim)}
@media (max-width:760px){#side{display:none}#top{right:16px;max-width:none}}
</style></head><body>
<div id="view"></div>
<div id="top" class="panel"><h1>A fly brain, spike by spike</h1>
<p>138,639 neurons of the FlyWire connectome in a leaky integrate-and-fire model. Each flash is a recorded model spike; 250&nbsp;ms of model time is slowed about 30&times;. Model activity, not a recording from a real fly.</p>
<div id="trials"></div></div>
<div id="side" class="panel"><h2>Stimulus</h2><div id="inputs"></div><h2 style="margin-top:10px">Output groups (max rate)</h2><div id="readout"></div>
<div id="pick">Click a lit neuron to identify it.</div><div id="legend"></div></div>
<div id="bottom" class="panel"><button id="play">Pause</button><canvas id="rate"></canvas><div id="clock"></div></div>
<script id="data" type="application/json">__DATA__</script>
<script type="importmap">{"imports":{"three":"https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js","three/addons/":"https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/"}}</script>
<script type="module">
import * as THREE from 'three';
import {OrbitControls} from 'three/addons/controls/OrbitControls.js';
const D = JSON.parse(document.getElementById('data').textContent);
const buf = s => {const b = atob(s), u = new Uint8Array(b.length); for (let i = 0; i < b.length; i++) u[i] = b.charCodeAt(i); return u.buffer;};
const PAL = {central:'#3d5a80', optic:'#2a4a5a', sensory:'#2ec4b6', visual_projection:'#4d908e', visual_centrifugal:'#577590',
  ascending:'#90be6d', descending:'#f8961e', motor:'#f94144', endocrine:'#b5838d', sensory_ascending:'#43aa8b', unknown:'#555'};
const n = D.n, q = new Uint16Array(buf(D.pos)), cls = new Uint8Array(buf(D.cls));
const span = D.hi.map((h, i) => h - D.lo[i]), scale = 2 / Math.max(...span);
const P = new Float32Array(n * 3);
for (let i = 0; i < n; i++) for (let k = 0; k < 3; k++) {
  const v = D.lo[k] + q[i*3+k] / 65535 * span[k];
  P[i*3+k] = (v - D.lo[k] - span[k] / 2) * scale * (k === 1 ? -1 : 1) * (k === 2 ? -1 : 1);
}
const color = c => new THREE.Color(PAL[D.classes[c]] || PAL.unknown);
const C = new Float32Array(n * 3);
for (let i = 0; i < n; i++) { const c = color(cls[i]); C[i*3] = c.r; C[i*3+1] = c.g; C[i*3+2] = c.b; }
const scene = new THREE.Scene(), view = document.getElementById('view');
const cam = new THREE.PerspectiveCamera(40, innerWidth / innerHeight, 0.01, 50); cam.position.set(0, 0.15, 2.6);
const ren = new THREE.WebGLRenderer({antialias: true}); ren.setPixelRatio(Math.min(devicePixelRatio, 2)); ren.setSize(innerWidth, innerHeight);
view.appendChild(ren.domElement);
const ctl = new OrbitControls(cam, ren.domElement); ctl.enableDamping = true; ctl.autoRotate = true; ctl.autoRotateSpeed = 0.35;
const bg = new THREE.BufferGeometry(); bg.setAttribute('position', new THREE.BufferAttribute(P, 3)); bg.setAttribute('color', new THREE.BufferAttribute(C, 3));
scene.add(new THREE.Points(bg, new THREE.PointsMaterial({size: 1.3, sizeAttenuation: false, vertexColors: true, transparent: true, opacity: 0.22, depthWrite: false, blending: THREE.AdditiveBlending})));
const mat = new THREE.ShaderMaterial({transparent: true, depthWrite: false, blending: THREE.AdditiveBlending,
  uniforms: {px: {value: ren.getPixelRatio()}},
  vertexShader: `attribute float glow; attribute vec3 tint; varying float g; varying vec3 c; uniform float px;
    void main(){ g = glow; c = tint; vec4 mv = modelViewMatrix * vec4(position, 1.0); gl_Position = projectionMatrix * mv; gl_PointSize = px * (2.0 + 9.0 * g); }`,
  fragmentShader: `varying float g; varying vec3 c; void main(){ vec2 d = gl_PointCoord - 0.5; float r = dot(d, d); if (r > 0.25) discard;
    float edge = smoothstep(0.25, 0.0, r); vec3 hot = mix(c, vec3(1.0, 0.93, 0.75), g * 0.8); gl_FragColor = vec4(hot, edge * (0.18 + 0.82 * g)); }`});
let act = null, T = null, glow = null, local = new Int32Array(n), ptr = 0, tSim = 0, playing = true, cur = null;
const tr = D.trials.map(t => ({...t, nn: new Uint32Array(buf(t.neuron)), tt: new Uint16Array(buf(t.t))}));
function load(k) {
  cur = tr[k]; ptr = 0; tSim = 0; local.fill(-1);
  if (act) { scene.remove(act); act.geometry.dispose(); }
  const m = cur.active.length, ap = new Float32Array(m * 3), tint = new Float32Array(m * 3); glow = new Float32Array(m);
  cur.active.forEach(([i], j) => { local[i] = j; for (let a = 0; a < 3; a++) { ap[j*3+a] = P[i*3+a]; tint[j*3+a] = C[i*3+a]; } });
  const g = new THREE.BufferGeometry(); g.setAttribute('position', new THREE.BufferAttribute(ap, 3));
  g.setAttribute('tint', new THREE.BufferAttribute(tint, 3)); g.setAttribute('glow', new THREE.BufferAttribute(glow, 1));
  act = new THREE.Points(g, mat); scene.add(act);
  document.querySelectorAll('#trials button').forEach((b, j) => b.classList.toggle('on', j === k));
  document.getElementById('inputs').innerHTML = `<div class="row"><span>${cur.inputs.join(', ')}</span></div><div class="row"><span>${cur.n_active.toLocaleString()} neurons fired</span><span>${cur.n_spikes.toLocaleString()} spikes</span></div>`;
  document.getElementById('readout').innerHTML = cur.readout.length ? cur.readout.map(([l, hz, a, nn]) => `<div class="row"><span>${l}</span><b>${hz} Hz</b></div>`).join('') : '<div class="row"><span>none</span></div>';
  document.getElementById('pick').innerHTML = 'Click a lit neuron to identify it.';
}
D.trials.forEach((t, k) => { const b = document.createElement('button'); b.textContent = t.title; b.onclick = () => load(k); document.getElementById('trials').appendChild(b); });
document.getElementById('legend').innerHTML = D.classes.filter(c => c !== 'unknown').map(c => `<span><i style="background:${PAL[c] || PAL.unknown}"></i>${c.replace('_', ' ')}</span>`).join('');
document.getElementById('play').onclick = e => { playing = !playing; e.target.textContent = playing ? 'Pause' : 'Play'; };
const cv = document.getElementById('rate'), cx = cv.getContext('2d');
function drawRate() {
  const w = cv.width = cv.clientWidth * devicePixelRatio, h = cv.height = cv.clientHeight * devicePixelRatio, r = cur.rate, mx = Math.max(1, ...r);
  cx.clearRect(0, 0, w, h); cx.fillStyle = '#3d5a80';
  r.forEach((v, i) => { const bh = v / mx * (h - 4); cx.fillRect(i / r.length * w, h - bh, Math.max(1, w / r.length - 1), bh); });
  cx.fillStyle = '#ffd27a'; cx.fillRect(tSim / cur.duration_ms * w, 0, 2 * devicePixelRatio, h);
}
const ray = new THREE.Raycaster(); ray.params.Points.threshold = 0.012;
ren.domElement.addEventListener('pointerdown', e => {
  const m = new THREE.Vector2(e.clientX / innerWidth * 2 - 1, -e.clientY / innerHeight * 2 + 1); ray.setFromCamera(m, cam);
  const hit = act && ray.intersectObject(act)[0]; if (!hit) return;
  const [i, type, sc, side, cnt] = cur.active[hit.index];
  document.getElementById('pick').innerHTML = `<b>${type}</b> &middot; ${sc.replace('_', ' ')} &middot; ${side}<br>${cnt} spikes in ${cur.duration_ms} ms`;
});
const SLOW = 8000 / 250; let last = performance.now();
function tick(now) {
  const dt = Math.min(50, now - last); last = now;
  if (playing && cur) {
    const dSim = dt / SLOW; tSim += dSim;
    const decay = Math.exp(-dSim / 6);
    for (let j = 0; j < glow.length; j++) glow[j] *= decay;
    while (ptr < cur.tt.length && cur.tt[ptr] / 10 <= tSim) { const j = local[cur.nn[ptr]]; if (j >= 0) glow[j] = 1; ptr++; }
    act.geometry.attributes.glow.needsUpdate = true;
    if (tSim > cur.duration_ms + 40) { tSim = 0; ptr = 0; glow.fill(0); }
    document.getElementById('clock').textContent = `${Math.min(tSim, cur.duration_ms).toFixed(1)} / ${cur.duration_ms} ms`;
  }
  if (cur) drawRate();
  ctl.update(); ren.render(scene, cam); requestAnimationFrame(tick);
}
addEventListener('resize', () => { cam.aspect = innerWidth / innerHeight; cam.updateProjectionMatrix(); ren.setSize(innerWidth, innerHeight); });
load(0); requestAnimationFrame(tick);
</script></body></html>
"""

if __name__ == "__main__":
    main()
