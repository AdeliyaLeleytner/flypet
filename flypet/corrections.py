"""Documented deviations from the Shiu et al. 2024 weight rule, applied as persistent weight scales.

Each correction is justified by anatomy/physiology, is local (a named synapse class), and was checked not to
change the three validated pathways (sugar -> MN9, JO-CE -> antennal grooming, looming -> giant fibre); see
scripts/al_fix.py and scripts/kc_gain.py for the measurements.

  al_ln_inhibitory : antennal-lobe local neurons whose predicted transmitter is a monoamine (dopamine /
                     serotonin / octopamine) are treated as inhibitory (sign flipped), and ACh-predicted LN
                     outputs are removed. Almost all AL LNs are GABAergic (Chou et al. 2010); with the Shiu
                     rule (monoamines excitatory) one glomerulus recruited all 330 uniglomerular PNs.
  dan_modulatory   : dopaminergic neurons have no fast synaptic effect; dopamine acts only through the
                     plasticity rule in flypet.mb. (The Shiu rule makes dopamine excitatory.)
  pn_kc_gain       : PN -> Kenyon cell synapses x2, so that ~7% of KCs respond to an odour (Turner et al. 2008
                     report ~5-10%); with x1 fewer than 3% respond at physiological PN rates.
  kc_mbon_gain     : KC -> MBON synapses x3, so that both glutamatergic (avoidance) and cholinergic (approach)
                     MBONs respond to odours (Hige et al. 2015: most MBONs respond to most odours in naive flies);
                     with x1 only three GABAergic gamma-lobe MBONs fire. Measured in scripts/mbon_readout.py.
"""

from __future__ import annotations
import numpy as np
from . import connectome as C
from .engine import Brain

DEFAULT = ("al_ln_inhibitory", "dan_modulatory", "pn_kc_gain", "kc_mbon_gain")


def masks(brain: Brain) -> dict[str, list[tuple[np.ndarray, float]]]:
    ann = C.annotations()
    flyid2i = brain.flyid2i
    ct = ann.cell_type.astype(str)
    nt = ann.top_nt.astype(str)
    cc = ann.cell_class.astype(str)
    idx = lambda mask: np.array([flyid2i[int(x)] for x in ann.index[mask]], dtype=np.int64)
    pre, post, w0 = brain.i_pre, brain._i_post, brain.w0
    alln_ach = idx((cc == "ALLN") & (nt == "acetylcholine"))
    alln_mono = idx((cc == "ALLN") & nt.isin(["dopamine", "serotonin", "octopamine"]))
    dan = idx(cc == "DAN")
    pn = idx(cc == "ALPN")
    kc = idx(cc == "Kenyon_Cell")
    mbon = idx(cc == "MBON")
    return {
        "al_ln_inhibitory": [
            (np.isin(pre, alln_ach) & (w0 > 0), 0.0),
            (np.isin(pre, alln_mono) & (w0 > 0), -1.0),
        ],
        "dan_modulatory": [(np.isin(pre, dan), 0.0)],
        "pn_kc_gain": [(np.isin(pre, pn) & np.isin(post, kc), 2.0)],
        "kc_mbon_gain": [(np.isin(pre, kc) & np.isin(post, mbon), 3.0)],
    }


def apply(brain: Brain, names=DEFAULT) -> Brain:
    """Multiply the persistent weight scale by each correction (idempotent per Brain: resets to 1 first)."""
    brain.w_scale[:] = 1.0
    m = masks(brain)
    for name in names:
        for mask, factor in m[name]:
            brain.w_scale[mask] *= factor
    brain.syn.w[:] = brain.w0 * brain.w_scale * brain.params["w_syn"]
    brain.corrections = tuple(names)
    return brain
