#!/usr/bin/env bash
# Second Vast box: Brian2 venv matching the Mac, transformers+peft in the image's torch, warm Brian2.
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq >/dev/null && apt-get install -y -qq build-essential >/dev/null
pip install -q uv transformers peft 2>&1 | grep -v -i warning | tail -1 || true
uv venv -q -p 3.12 /workspace/venv
uv pip install -q --python /workspace/venv/bin/python -r /workspace/fly/requirements-lock.txt
cd /workspace/fly && /workspace/venv/bin/python - <<'PY'
import sys, time
sys.path.insert(0, '.')
from flypet.engine import Brain, StimInput
from flypet import corrections
from flypet.catalog import STIMULI
b = corrections.apply(Brain())
st = STIMULI['looming']
for k in range(2):
    t = time.time(); r = b.run([StimInput(st.resolve(), 200, key='looming')], duration_ms=250, seed=11)
    print(f"sim {k}: {time.time() - t:.1f} s, active {len(r.counts)}, spikes {sum(r.counts.values())}", flush=True)
PY
python -c "import torch, transformers, peft; print('torch', torch.__version__, torch.cuda.get_device_name(0), 'transformers', transformers.__version__, 'peft', peft.__version__)"
