"""Fast surrogate of the teaching environment in flypet.teach, fitted to and checked against the Brian2 fly.

Assumptions, each checked by validation against real episodes:
- Kenyon-cell and dopamine-neuron rates in a training trial depend on the odour, reinforcement and seed but
  not on the current memory (dopamine outputs are non-synaptic; MBON->DAN feedback is the ignored route).
- A probe's MBON rate changes linearly with its Kenyon-cell drive from the naive value, one slope per MBON,
  rectified at zero. Drive = sum over KC->MBON synapses of base weight x memory factor x presynaptic KC rate.
The learning rule itself is copied exactly from MushroomBody.learn.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class MBSpec:
    """The parts of a MushroomBody the surrogate needs (exported once from a real one)."""

    syn_pre: np.ndarray  # KC model index per KC->MBON synapse row
    syn_post: np.ndarray  # MBON model index per row
    base_w: np.ndarray  # w0 * base_scale per row (signed synaptic weight before memory)
    mbon: np.ndarray  # MBON model indices, order used for rate vectors
    sign: np.ndarray  # valence sign per MBON in that order (+1 approach, -1 avoid, 0 excluded)
    eta: float
    floor: float
    kc_scale_hz: float

    @classmethod
    def from_mb(cls, mb):
        w0 = mb.b.w0[mb.syn_idx].astype(np.float64)
        return cls(
            mb.syn_pre.copy(),
            mb.syn_post.copy(),
            w0 * mb.base_scale.astype(np.float64),
            np.asarray(mb.MBON),
            np.array([mb.sign[int(i)] for i in mb.MBON]),
            float(mb.eta),
            float(mb.floor),
            float(mb.kc_scale_hz),
        )

    def save(self, path, **extra):
        np.savez_compressed(
            path,
            syn_pre=self.syn_pre,
            syn_post=self.syn_post,
            base_w=self.base_w,
            mbon=self.mbon,
            sign=self.sign,
            eta=self.eta,
            floor=self.floor,
            kc_scale_hz=self.kc_scale_hz,
            **extra,
        )

    @classmethod
    def load(cls, path):
        z = np.load(path)
        return cls(
            z["syn_pre"],
            z["syn_post"],
            z["base_w"],
            z["mbon"],
            z["sign"],
            float(z["eta"]),
            float(z["floor"]),
            float(z["kc_scale_hz"]),
        ), z


class Surrogate:
    """trial_kc[(odour, reinf, seed)] = KC rates (model-index-aligned vector over KCs listed in kc_index);
    trial_dop[(odour, reinf, seed)] = compartment dopamine per MBON (thresholded, as in learn);
    probe_kc[(odour, seed)], probe_mbon[(odour, seed)] = KC and MBON rates of the naive probe."""

    def __init__(
        self,
        spec: MBSpec,
        kc_index: np.ndarray,
        trial_kc: dict,
        trial_dop: dict,
        probe_kc: dict,
        probe_mbon: dict,
        slopes: np.ndarray | None = None,
    ):
        self.s = spec
        pos = np.full(int(max(kc_index.max(), spec.syn_pre.max())) + 1, -1)
        pos[kc_index] = np.arange(len(kc_index))
        self.pre_pos = pos[spec.syn_pre]
        assert (self.pre_pos >= 0).all(), (
            "a KC->MBON synapse has a presynaptic cell outside kc_index"
        )
        mpos = np.full(int(spec.mbon.max()) + 1, -1)
        mpos[spec.mbon] = np.arange(len(spec.mbon))
        self.post_pos = mpos[spec.syn_post]
        self.trial_kc, self.trial_dop, self.probe_kc, self.probe_mbon = (
            trial_kc,
            trial_dop,
            probe_kc,
            probe_mbon,
        )
        self.slopes = np.zeros(len(spec.mbon)) if slopes is None else slopes
        self._change = {}

    # --- learning, copied from MushroomBody.learn
    def change(self, odour, reinf, seed):
        key = (odour, reinf, seed)
        if key not in self._change:
            a = np.minimum(1.0, self.trial_kc[key][self.pre_pos] / self.s.kc_scale_hz)
            d = self.trial_dop[key][self.post_pos]
            self._change[key] = self.s.eta * a * d
        return self._change[key]

    def train(self, mem, odour, reinf, seed):
        c = self.change(odour, reinf, seed)
        t = c > 0
        mem = mem.copy()
        mem[t] = np.maximum(self.s.floor, mem[t] * (1.0 - c[t]))
        return mem

    # --- readout
    def drive(self, mem, odour, seed):
        kc = self.probe_kc[(odour, seed)][self.pre_pos]
        return np.bincount(
            self.post_pos, weights=self.s.base_w * mem * kc, minlength=len(self.s.mbon)
        )

    def mbon_rates(self, mem, odour, seed):
        base = self.probe_mbon[(odour, seed)]
        dx = self.drive(mem, odour, seed) - self.drive(np.ones_like(mem), odour, seed)
        return np.maximum(0.0, base + self.slopes * dx)

    def valence_from_rates(self, r):
        ap, av = r[self.s.sign > 0].sum(), r[self.s.sign < 0].sum()
        return float((ap - av) / (ap + av + 1.0))

    def episode(self, actions, panel, train_seed=5000, eval_seeds=(101, 102, 103)):
        mem = np.ones(len(self.s.syn_pre))
        for step, (o, r) in enumerate(actions):
            mem = self.train(mem, o, r, train_seed + step)
        return {
            o: float(
                np.mean([self.valence_from_rates(self.mbon_rates(mem, o, s)) for s in eval_seeds])
            )
            for o in panel
        }, mem

    def fit_slopes(self, episodes: list, panel, eval_seeds=(101, 102, 103)):
        """episodes: list of (mem, {(odour, seed): true MBON rate vector}). Least squares through the origin
        of rate change on drive change, per MBON, over probes where the MBON is active either way."""
        num, den = np.zeros(len(self.s.mbon)), np.zeros(len(self.s.mbon))
        for mem, rates in episodes:
            for o in panel:
                for s in eval_seeds:
                    dx = self.drive(mem, o, s) - self.drive(np.ones_like(mem), o, s)
                    dr = rates[(o, s)] - self.probe_mbon[(o, s)]
                    live = (rates[(o, s)] > 0) | (self.probe_mbon[(o, s)] > 0)
                    num += np.where(live, dx * dr, 0.0)
                    den += np.where(live, dx * dx, 0.0)
        self.slopes = np.where(den > 0, num / np.maximum(den, 1e-12), 0.0)
        return self.slopes
