"""Drive the MaleCNS ventral nerve cord from the brain model and read the muscles.

For each stimulus: run the FlyWire brain, average the descending neurons by cell type, push those
rates into the MaleCNS VNC, report which motor nerves fire.
"""

from __future__ import annotations

_ROOT = str(__import__("pathlib").Path(__file__).resolve().parents[1])  # repository root
import sys

sys.path.insert(0, _ROOT + "")
import sys
import numpy as np
from flypet.engine import Brain, StimInput
from flypet import corrections, catalog
from flypet.vnc import Vnc

CASES = ["looming", "sugar", "bitter", "antenna_touch", "smell_vinegar", "heat"]


def main(argv):
    cases = argv[1:] or CASES
    brain = corrections.apply(Brain())
    vnc = Vnc()
    print(
        f"мозг: {brain.n} нейронов, {brain.n_syn} синапсов | ВНЦ: {vnc.n} нейронов, "
        f"{len(vnc.v_w)} внутренних + {len(vnc.d_w)} нисходящих связей\n"
    )
    for key in cases:
        st = catalog.STIMULI[key]
        ids = st.resolve()
        r = brain.run([StimInput(ids, st.default_rate_hz, key)], duration_ms=300, seed=7)
        vr = vnc.run_from_brain(r, duration_ms=300, seed=7)
        dn_hz = {k: v for k, v in sorted(vr.driven.items(), key=lambda x: -x[1])[:6]}
        print(f"=== {key} ({st.label_ru}), {len(ids)} нейронов на входе")
        print(f"    мозг {r.wall_s:.1f} с, ВНЦ {vr.wall_s:.1f} с")
        print(f"    нисходящие (топ): {dn_hz}")
        print(f"    мышцы: {vr.summary()}")
        top = vr.top_motor(6)
        if not top.empty:
            print(top.to_string(index=False).replace("\n", "\n    ").rjust(4))
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
