"""Brian2 engine: the Shiu et al. 2024 leaky integrate-and-fire model on the FlyWire v783 connectome.

Equations and constants are copied from vendor/Drosophila_brain_model/model.py (MIT, Shiu et al.).
Differences from the original:
  * the network is built once; before each run the delay queue is flushed with spiking disabled
    and the neuron state is reset by hand (no Network.store(), which would hold a second copy
    of the 15M synapses);
  * external input is a per-neuron Poisson rate `r_stim`, drawn each step only for the smallest
    tier of stimulable neurons that covers the current stimulus (see connectome.neuron_order),
    with the same event weight as the original PoissonInput (one event = one spike);
  * any number of stimulus groups with different rates can be combined in one run;
  * synaptic weights can be scaled persistently (`set_weights`, kept in `w_scale`) - the hook for learning.
"""

from __future__ import annotations
from dataclasses import dataclass, field
import os, time
import numpy as np
from brian2 import (
    NeuronGroup,
    Synapses,
    SpikeMonitor,
    Network,
    prefs,
    defaultclock,
    mV,
    ms,
    Hz,
    second,
)
from . import connectome as C

prefs.codegen.target = os.environ.get("FLYPET_BRIAN_TARGET", "auto")

EQS = """
dv/dt = (v_0 - v + g) / t_mbr : volt (unless refractory)
dg/dt = -g / tau               : volt (unless refractory)
rfc                            : second
r_stim                         : Hz
"""
DEFAULT_PARAMS = dict(
    v_0=-52 * mV,
    v_rst=-52 * mV,
    v_th=-45 * mV,
    t_mbr=20 * ms,  # Kakaria & de Bivort 2017
    tau=5 * ms,  # Jürgensen et al. 2021
    t_rfc=2.2 * ms,  # Lazar et al. 2021
    t_dly=1.8 * ms,  # Paul et al. 2015
    w_syn=0.275 * mV,  # free parameter (Shiu et al.)
    f_poi=250,  # Poisson weight scale: one event = one spike
)
MAX_MONITOR_SPIKES = 3_000_000  # rebuild the network objects when the spike monitor grows past this


@dataclass
class StimInput:
    ids: list[int]  # FlyWire root IDs
    rate_hz: float
    key: str = ""


@dataclass
class RunResult:
    duration_s: float
    trains: dict[int, np.ndarray]  # model index -> spike times (s, relative to run start)
    stimulated: dict[str, list[int]]  # key -> root ids
    wall_s: float
    counts: dict[int, int] = field(default_factory=dict)

    def rate(self, root_id: int) -> float:
        flyid2i, _ = C.id_maps()
        i = flyid2i.get(root_id)
        return 0.0 if i is None else len(self.trains.get(i, ())) / self.duration_s

    def rate_vector(self, n: int) -> np.ndarray:
        """Firing rate (Hz) of every neuron in model index order."""
        v = np.zeros(n, dtype=np.float32)
        for i, c in self.counts.items():
            v[i] = c / self.duration_s
        return v


class Brain:
    def __init__(self, params: dict | None = None, conn=None):
        """conn: a connectome module with id_maps, neuron_order and connections; FlyWire (flypet.connectome) by
        default, flypet.malecns for the male central nervous system."""
        self.params = p = dict(DEFAULT_PARAMS, **(params or {}))
        p["w_stim"] = p["w_syn"] * p["f_poi"]
        self.conn = conn or C
        self.flyid2i, self.i2flyid = self.conn.id_maps()
        order, self.tier_ends = self.conn.neuron_order()
        self.n = len(order)
        self.n_stim = self.tier_ends[-1]
        i_pre, i_post, w = self.conn.connections()
        self.n_syn = len(w)
        self.i_pre = i_pre.copy()  # kept for weight edits (60 MB)
        self.w0 = w.copy()  # original signed synapse counts (int16)
        self._i_post = i_post.copy()
        self.conn.connections.cache_clear()
        self.w_scale = np.ones(
            self.n_syn, dtype=np.float32
        )  # persistent per-synapse factor (learning)
        self.n_runs = 0
        self._build()

    def _build(self):
        if getattr(self, "_continuous_active", False):
            raise RuntimeError("Close the NeuralRuntime before rebuilding this brain")
        p = self.params
        t0 = time.time()
        self._gen = None  # spike-generator input for run_timed, added to the network on first use
        self.neu = NeuronGroup(
            self.n,
            EQS,
            method="linear",
            threshold="v > v_th",
            reset="v = v_rst; g = 0 * mV",
            refractory="rfc",
            namespace=p,
            name="neurons",
        )
        self.stim_ops = []
        for k, end in enumerate(self.tier_ends):
            op = self.neu[:end].run_regularly(
                "v += int(rand() < r_stim * dt) * w_stim",
                dt=defaultclock.dt,
                when="start",
                name=f"stim_op{k}",
            )
            op.active = False
            self.stim_ops.append(op)
        self.syn = Synapses(
            self.neu, self.neu, "w : volt", on_pre="g += w", delay=p["t_dly"], name="synapses"
        )
        self.syn.connect(i=self.i_pre, j=self._i_post)
        self.syn.w = self.w0 * self.w_scale * p["w_syn"]
        self.mon = SpikeMonitor(
            self.neu, name="spikes"
        )  # accumulates across runs; runs read slices
        self.net = Network(self.neu, self.syn, self.mon, *self.stim_ops)
        self._reset_state()
        self.build_s = round(time.time() - t0, 1)

    # -- state handling -------------------------------------------------------------------------
    def _reset_state(self):
        p = self.params
        self.neu.v = p["v_0"]
        self.neu.g = 0 * mV
        self.neu.rfc = p["t_rfc"]
        self.neu.r_stim = 0 * Hz

    def _flush(self):
        """Deliver whatever is still in the delay queue with spiking disabled, then zero the state."""
        if getattr(self, "_continuous_active", False):
            raise RuntimeError("Use NeuralRuntime.advance, or close it before a resetting trial")
        for op in self.stim_ops:
            op.active = False
        self.neu.thresholder["spike"].active = False
        self.net.run(2 * self.params["t_dly"])
        self.neu.thresholder["spike"].active = True
        self._reset_state()

    def set_weights(self, syn_idx: np.ndarray, factor: float | np.ndarray, persistent: bool = True):
        """Scale the weights of the given synapse indices (positions in connectome.connections order)."""
        if persistent:
            self.w_scale[syn_idx] = factor
        self.syn.w[syn_idx] = (
            self.w0[syn_idx]
            * self.w_scale[syn_idx]
            * (factor if not persistent else 1.0)
            * self.params["w_syn"]
        )

    def synapses_from(self, pre_root_ids: list[int]) -> np.ndarray:
        pre = np.array([self.flyid2i[i] for i in pre_root_ids if i in self.flyid2i], dtype=np.int32)
        return np.flatnonzero(np.isin(self.i_pre, pre))

    def synapses_between(self, pre_root_ids: list[int], post_root_ids: list[int]) -> np.ndarray:
        pre = np.array([self.flyid2i[i] for i in pre_root_ids if i in self.flyid2i], dtype=np.int32)
        post = np.array(
            [self.flyid2i[i] for i in post_root_ids if i in self.flyid2i], dtype=np.int32
        )
        return np.flatnonzero(np.isin(self.i_pre, pre) & np.isin(self._i_post, post))

    # -- simulation ------------------------------------------------------------------------------
    def run(
        self,
        inputs: list[StimInput],
        duration_ms: float = 300,
        seed: int | None = None,
        silence: list[int] | None = None,
    ) -> RunResult:
        if getattr(self, "_continuous_active", False):
            raise RuntimeError("Close NeuralRuntime before a resetting trial")
        if seed is not None:
            from brian2 import seed as b2seed

            b2seed(seed)
        t0 = time.time()
        if int(self.mon.num_spikes) > MAX_MONITOR_SPIKES:
            self._build()
        self._flush()
        stimulated, max_idx = {}, -1
        for k, inp in enumerate(inputs):
            idx = np.array(
                sorted({self.flyid2i[i] for i in inp.ids if i in self.flyid2i}), dtype=int
            )
            idx = idx[idx < self.n_stim]
            if len(idx) == 0 or inp.rate_hz <= 0:
                continue
            self.neu.r_stim[idx] = inp.rate_hz * Hz
            self.neu.rfc[idx] = 0 * ms  # as in Shiu et al.: stimulated neurons follow the input
            max_idx = max(max_idx, int(idx.max()))
            stimulated[inp.key or f"input_{k}"] = [int(self.i2flyid[i]) for i in idx]
        if max_idx >= 0:
            tier = next(k for k, end in enumerate(self.tier_ends) if max_idx < end)
            self.stim_ops[tier].active = True
        silenced = None
        if silence:
            silenced = self.synapses_from(silence)
            self.set_weights(silenced, 0.0, persistent=False)
        n0 = int(self.mon.num_spikes)
        t_start = float(self.net.t / second)
        self.net.run(duration_ms * ms)
        trains = self._trains_since(n0, t_start)
        if silenced is not None:
            self.set_weights(silenced, 1.0, persistent=False)
        for op in self.stim_ops:
            op.active = False
        self.n_runs += 1
        return RunResult(
            duration_s=duration_ms / 1000,
            trains=trains,
            stimulated=stimulated,
            wall_s=time.time() - t0,
            counts={i: len(t) for i, t in trains.items()},
        )

    def _trains_since(self, n0: int, t_start: float) -> dict[int, np.ndarray]:
        """Spike trains recorded after monitor position n0, times relative to t_start (s)."""
        idx = np.asarray(self.mon.i[:])[n0:]
        ts = np.asarray(self.mon.t[:] / second)[n0:] - t_start
        if not len(idx):
            return {}
        srt = np.argsort(idx, kind="stable")
        ts_sorted = ts[srt]
        uniq, starts = np.unique(idx[srt], return_index=True)
        ends = np.append(starts[1:], len(idx))
        return {int(u): ts_sorted[a:b].copy() for u, a, b in zip(uniq, starts, ends)}

    def run_timed(
        self,
        inputs: list[tuple[list[int], np.ndarray]],
        duration_ms: float,
        seed: int | None = None,
    ) -> RunResult:
        """Input with any time course in one run: inputs [(root ids, rate_hz per 1 ms bin)], e.g. a pulse train of
        courtship song. Poisson spikes are drawn in numpy (seeded) and delivered through a SpikeGeneratorGroup with the
        same event weight as the Poisson input of run(); the driven neurons follow the input (no refractoriness).
        Much faster than run_schedule for pulse trains, but not spike-for-spike comparable with run() for a seed."""
        if getattr(self, "_continuous_active", False):
            raise RuntimeError("Close NeuralRuntime before a resetting trial")
        from brian2 import SpikeGeneratorGroup

        t0 = time.time()
        if int(self.mon.num_spikes) > MAX_MONITOR_SPIKES:
            self._build()
        if self._gen is None:
            self._gen = SpikeGeneratorGroup(
                self.n_stim, np.zeros(0, int), np.zeros(0) * ms, name="generator"
            )
            self._gen_syn = Synapses(
                self._gen,
                self.neu,
                on_pre="v += w_stim",
                namespace=self.params,
                name="generator_syn",
            )
            self._gen_syn.connect(j="i")
            self.net.add(self._gen, self._gen_syn)
        self._flush()
        rng = np.random.default_rng(seed)
        dt = float(defaultclock.dt / ms)
        idx_all, t_all, stimulated, followers = [], [], {}, set()
        for k, (ids, rates) in enumerate(inputs):
            idx = np.array(sorted({self.flyid2i[i] for i in ids if i in self.flyid2i}), dtype=int)
            idx = idx[idx < self.n_stim]
            rates = np.asarray(rates, float)[: int(duration_ms)]
            if not len(idx) or not rates.any():
                continue
            counts = rng.poisson(
                rates[None, :] / 1000.0, size=(len(idx), len(rates))
            )  # spikes per neuron per ms bin
            n_i, n_b = np.nonzero(counts)
            reps = counts[n_i, n_b]
            n_i, n_b = np.repeat(n_i, reps), np.repeat(n_b, reps)
            times = n_b + rng.random(len(n_b))
            idx_all.append(idx[n_i])
            t_all.append(times)
            followers.update(idx.tolist())
            stimulated[f"input_{k}"] = [int(self.i2flyid[i]) for i in idx]
        t_start = float(self.net.t / second)
        if idx_all:
            gi, gt = np.concatenate(idx_all), np.concatenate(t_all)
            bins = np.floor(gt / dt).astype(np.int64)
            _, keep = np.unique(
                gi.astype(np.int64) * 10_000_000 + bins, return_index=True
            )  # one spike per neuron per step
            self.neu.rfc[np.array(sorted(followers))] = 0 * ms
            self._gen.set_spikes(gi[keep], (t_start * 1000 + dt + bins[keep] * dt) * ms)
        else:
            self._gen.set_spikes(np.zeros(0, int), np.zeros(0) * ms)
        n0 = int(self.mon.num_spikes)
        self.net.run(duration_ms * ms)
        trains = self._trains_since(n0, t_start)
        self.n_runs += 1
        return RunResult(
            duration_s=duration_ms / 1000,
            trains=trains,
            stimulated=stimulated,
            wall_s=time.time() - t0,
            counts={i: len(t) for i, t in trains.items()},
        )

    def run_schedule(
        self, segments: list[tuple[float, list[StimInput]]], seed: int | None = None
    ) -> RunResult:
        """Time-varying input, e.g. a pulse train of courtship song: segments [(duration_ms, inputs)] run back to back.
        The network starts from rest once; between segments only the input rates change, so activity carries over."""
        if getattr(self, "_continuous_active", False):
            raise RuntimeError("Close NeuralRuntime before a resetting trial")
        if seed is not None:
            from brian2 import seed as b2seed

            b2seed(seed)
        t0 = time.time()
        if int(self.mon.num_spikes) > MAX_MONITOR_SPIKES:
            self._build()
        self._flush()
        plan, stimulated, followers = [], {}, set()
        for dur, inputs in segments:
            seg = []
            for k, inp in enumerate(inputs):
                idx = np.array(
                    sorted({self.flyid2i[i] for i in inp.ids if i in self.flyid2i}), dtype=int
                )
                idx = idx[idx < self.n_stim]
                if len(idx) and inp.rate_hz > 0:
                    seg.append((idx, inp.rate_hz))
                    followers.update(idx.tolist())
                    stimulated.setdefault(
                        inp.key or f"input_{k}", [int(self.i2flyid[i]) for i in idx]
                    )
            plan.append((dur, seg))
        if followers:
            self.neu.rfc[np.array(sorted(followers))] = (
                0 * ms
            )  # as in run(): stimulated neurons follow the input
        n0 = int(self.mon.num_spikes)
        t_start = float(self.net.t / second)
        for dur, seg in plan:
            self.neu.r_stim[: self.n_stim] = 0 * Hz
            for op in self.stim_ops:
                op.active = False
            for idx, rate in seg:
                self.neu.r_stim[idx] = rate * Hz
            if seg:
                top = max(int(idx.max()) for idx, _ in seg)
                self.stim_ops[
                    next(k for k, end in enumerate(self.tier_ends) if top < end)
                ].active = True
            self.net.run(dur * ms)
        trains = self._trains_since(n0, t_start)
        for op in self.stim_ops:
            op.active = False
        self.n_runs += 1
        total_s = sum(d for d, _ in segments) / 1000
        return RunResult(
            duration_s=total_s,
            trains=trains,
            stimulated=stimulated,
            wall_s=time.time() - t0,
            counts={i: len(t) for i, t in trains.items()},
        )
