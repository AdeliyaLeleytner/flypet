"""The ventral nerve cord, bolted onto the brain model.

FlyWire v783 ends at the neck. Our brain model can therefore say that DNp01 fires at 240 Hz but not
what that command does to the body. MaleCNS v1.0 contains the VNC, and its cell-type names are the
same ones FlyWire uses, so we can take the descending-neuron rates the brain model produces and push
them into the MaleCNS VNC by cell type, then read the motor neurons that leave through the leg, wing,
haltere, neck and abdominal nerves.

What this is not: the two connectomes are different animals (FlyWire is a female brain, MaleCNS a male
CNS), so this is a bridge by cell type, not one continuous circuit. Descending neurons enter the VNC
network as Poisson sources firing at the rate measured in the brain, which means DN -> DN synapses
inside the VNC are dropped and the VNC cannot feed back into the brain.

Assumptions:
  w_syn         same 0.275 mV as the brain model (Shiu et al. fitted it on FlyWire; nothing equivalent
                has been fitted on MaleCNS, so this is inherited, not measured)
  sign rule     acetylcholine +1, GABA / glutamate / histamine -1, monoamines 0 (see scripts/build_vnc.py)
  proprioception VNC sensory neurons exist in the graph but are never driven - there is no body yet
"""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import os, time
import numpy as np
import pandas as pd
from brian2 import (
    NeuronGroup,
    PoissonGroup,
    Synapses,
    SpikeMonitor,
    Network,
    defaultclock,
    mV,
    ms,
    Hz,
    second,
)
from .engine import EQS, DEFAULT_PARAMS, RunResult
from . import connectome as C

ROOT = Path(os.environ.get("FLYPET_ROOT", Path(__file__).resolve().parent.parent))
PATH_NPZ = ROOT / "data" / "vnc_malecns.npz"
PATH_NODES = ROOT / "data" / "vnc_nodes.parquet"

MUSCLE_ORDER = [
    "передняя нога",
    "средняя нога",
    "задняя нога",
    "прыжок",
    "крыло",
    "жужжальце",
    "шея",
    "брюшко",
    "брюшко (LB)",
    "хоботок",
    "прочее",
]


def _require_data():
    if not (PATH_NPZ.exists() and PATH_NODES.exists()):
        raise FileNotFoundError(f"нет {PATH_NPZ.name}: сначала запусти scripts/build_vnc.py")


@dataclass
class VncResult:
    duration_s: float
    rates: np.ndarray  # Hz for every VNC neuron (drivers excluded)
    driver_rates: np.ndarray  # Hz that was pushed into each descending neuron
    nodes: pd.DataFrame  # VNC rows only, aligned with `rates`
    wall_s: float = 0.0
    driven: dict[str, float] = field(
        default_factory=dict
    )  # cell type -> Hz, what came from the brain

    def by_muscle(self, min_hz: float = 0.5) -> pd.DataFrame:
        mn = self.nodes.superclass.to_numpy() == "vnc_motor"
        d = pd.DataFrame(
            {
                "muscle": self.nodes.muscle.to_numpy()[mn],
                "side": self.nodes.side.to_numpy()[mn],
                "hz": self.rates[mn],
            }
        )
        g = d.groupby(["muscle", "side"], observed=True)["hz"].agg(
            активных=lambda s: int((s > min_hz).sum()), всего="size", макс="max", средн="mean"
        )
        g = g[g.активных > 0].sort_values("макс", ascending=False)
        return g.round(1)

    def top_motor(self, k: int = 12, min_hz: float = 0.5) -> pd.DataFrame:
        mn = self.nodes.superclass.to_numpy() == "vnc_motor"
        d = pd.DataFrame(
            {
                "тип": self.nodes.type.to_numpy()[mn],
                "мышца": self.nodes.muscle.to_numpy()[mn],
                "сторона": self.nodes.side.to_numpy()[mn],
                "нерв": self.nodes.nerve.to_numpy()[mn],
                "Гц": self.rates[mn],
            }
        )
        d = d[d["Гц"] > min_hz].sort_values("Гц", ascending=False)
        return d.head(k).round(1)

    def brief(self, k: int = 6, min_hz: float = 0.5) -> dict:
        """Compact form for the narrator prompt."""
        t = self.top_motor(k, min_hz)
        return {
            "мышцы": self.summary(min_hz),
            "мотонейроны": [
                f"{r['тип']} {r['сторона']} {r['Гц']:.0f} Гц ({r['мышца']})"
                for _, r in t.iterrows()
            ],
            "нисходящие": dict(sorted(self.driven.items(), key=lambda x: -x[1])[:6]),
        }

    def summary(self, min_hz: float = 0.5) -> str:
        g = self.by_muscle(min_hz)
        if g.empty:
            return "мотонейроны молчат"
        parts = []
        for (muscle, side), r in g.iterrows():
            parts.append(
                f"{muscle} {side}: {int(r.активных)}/{int(r.всего)} нейронов, до {r.макс:.0f} Гц"
            )
        return "; ".join(parts)


class Vnc:
    """MaleCNS VNC as a second LIF network, driven by descending-neuron rates from the brain model."""

    def __init__(self, params: dict | None = None):
        _require_data()
        self.params = p = dict(DEFAULT_PARAMS, **(params or {}))
        p["w_stim"] = p["w_syn"] * p["f_poi"]
        z = np.load(PATH_NPZ)
        self.n_drv = int(z["n_drivers"])
        nodes = pd.read_parquet(PATH_NODES)
        self.drivers = nodes.iloc[: self.n_drv].reset_index(drop=True)
        self.nodes = nodes.iloc[self.n_drv :].reset_index(drop=True)
        self.n = len(self.nodes)
        i_pre, i_post, w = z["i_pre"], z["i_post"], z["w"]
        dn = i_pre < self.n_drv
        self.d_pre, self.d_post, self.d_w = (
            i_pre[dn],
            i_post[dn] - self.n_drv,
            w[dn].astype(np.float32),
        )
        self.v_pre = i_pre[~dn] - self.n_drv
        self.v_post = i_post[~dn] - self.n_drv
        self.v_w = w[~dn].astype(np.float32)
        self._index_types()
        self._build()

    def _index_types(self):
        """cell type (and FlyWire alias) -> driver rows, so brain rates can be matched by name."""
        self.by_type: dict[tuple[str, str], np.ndarray] = {}
        d = self.drivers
        for col in ("type", "flywire_type"):
            for (t, s), g in d.groupby([d[col], d.side], observed=True):
                if t:
                    self.by_type.setdefault((t, s), g.index.to_numpy())

    def _build(self):
        p = self.params
        t0 = time.time()
        self.drv = PoissonGroup(self.n_drv, rates=np.zeros(self.n_drv) * Hz, name="descending")
        self.neu = NeuronGroup(
            self.n,
            EQS,
            method="linear",
            threshold="v > v_th",
            reset="v = v_rst; g = 0 * mV",
            refractory="rfc",
            namespace=p,
            name="vnc",
        )
        self.stim_op = self.neu.run_regularly(
            "v += int(rand() < r_stim * dt) * w_stim",
            dt=defaultclock.dt,
            when="start",
            name="vnc_stim",
        )
        self.stim_op.active = False
        self.syn_d = Synapses(
            self.drv, self.neu, "w : volt", on_pre="g += w", delay=p["t_dly"], name="dn_syn"
        )
        self.syn_d.connect(i=self.d_pre, j=self.d_post)
        self.syn_d.w = self.d_w * p["w_syn"]
        self.syn_v = Synapses(
            self.neu, self.neu, "w : volt", on_pre="g += w", delay=p["t_dly"], name="vnc_syn"
        )
        self.syn_v.connect(i=self.v_pre, j=self.v_post)
        self.syn_v.w = self.v_w * p["w_syn"]
        self.mon = SpikeMonitor(self.neu, name="vnc_spikes")
        self.net = Network(self.drv, self.neu, self.syn_d, self.syn_v, self.mon, self.stim_op)
        self._reset_state()
        self.build_s = round(time.time() - t0, 1)

    def _reset_state(self):
        p = self.params
        self.neu.v = p["v_0"]
        self.neu.g = 0 * mV
        self.neu.rfc = p["t_rfc"]
        self.neu.r_stim = 0 * Hz
        self.drv.rates = np.zeros(self.n_drv) * Hz

    def _flush(self):
        self.stim_op.active = False
        self.drv.rates = np.zeros(self.n_drv) * Hz
        self.neu.thresholder["spike"].active = False
        self.net.run(2 * self.params["t_dly"])
        self.neu.thresholder["spike"].active = True
        self._reset_state()

    # -- the bridge ------------------------------------------------------------------------------
    def rates_from_brain(
        self, result: RunResult, min_hz: float = 1.0
    ) -> tuple[np.ndarray, dict[str, float]]:
        """Average the brain model's descending neurons by (cell type, side) and map onto MaleCNS drivers."""
        ann = C.annotations()
        dn = ann[ann.super_class.astype(str) == "descending"]
        rates = np.zeros(self.n_drv, dtype=np.float32)
        driven: dict[str, float] = {}
        ct = dn.cell_type.astype(str).to_numpy()
        sides = dn.side.astype(str).to_numpy()
        hz = np.array([result.rate(int(r)) for r in dn.index], dtype=np.float32)
        key = pd.MultiIndex.from_arrays([ct, sides])
        for (t, s), pos in pd.Series(np.arange(len(hz)), index=key).groupby(level=[0, 1]):
            f = float(hz[pos.to_numpy()].mean())
            if f < min_hz or not t:
                continue
            tgt = self.by_type.get((t, {"left": "L", "right": "R", "center": "M"}.get(s, "")))
            if tgt is None:
                tgt = self.by_type.get((t, "L"))
                tgt = (
                    np.concatenate([x for x in (tgt, self.by_type.get((t, "R"))) if x is not None])
                    if (tgt is not None or self.by_type.get((t, "R")) is not None)
                    else None
                )
            if tgt is None or len(tgt) == 0:
                continue
            rates[tgt] = f
            driven[f"{t} {s}"] = round(f, 1)
        return rates, driven

    def run(
        self,
        driver_rates: np.ndarray,
        duration_ms: float = 300,
        seed: int | None = None,
        driven: dict[str, float] | None = None,
    ) -> VncResult:
        if seed is not None:
            from brian2 import seed as b2seed

            b2seed(seed)
        t0 = time.time()
        self._flush()
        self.drv.rates = np.asarray(driver_rates, dtype=float) * Hz
        n0 = int(self.mon.num_spikes)
        self.net.run(duration_ms * ms)
        idx = np.asarray(self.mon.i[:])[n0:]
        counts = np.bincount(idx, minlength=self.n).astype(np.float32)
        self.drv.rates = np.zeros(self.n_drv) * Hz
        return VncResult(
            duration_s=duration_ms / 1000,
            rates=counts / (duration_ms / 1000),
            driver_rates=np.asarray(driver_rates, dtype=np.float32),
            nodes=self.nodes,
            wall_s=time.time() - t0,
            driven=driven or {},
        )

    def run_from_brain(
        self, result: RunResult, duration_ms: float = 300, seed: int | None = None
    ) -> VncResult:
        rates, driven = self.rates_from_brain(result)
        return self.run(rates, duration_ms=duration_ms, seed=seed, driven=driven)

    # -- coverage --------------------------------------------------------------------------------
    def coverage(self) -> dict:
        ann = C.annotations()
        fw = {
            str(t)
            for t in ann[ann.super_class.astype(str) == "descending"].cell_type.unique()
            if str(t)
        }
        have = {t for (t, _) in self.by_type}
        return {
            "типов DN во FlyWire": len(fw),
            "из них есть в MaleCNS": len(fw & have),
            "нисходящих в MaleCNS": self.n_drv,
            "нейронов ВНЦ": self.n,
            "моторных": int((self.nodes.superclass == "vnc_motor").sum()),
            "связей DN→ВНЦ": len(self.d_w),
            "связей внутри ВНЦ": len(self.v_w),
        }
