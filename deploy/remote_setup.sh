#!/usr/bin/env bash
# Prepare a fresh Vast box for CPU simulations: compiler for Brian2, Python deps, warm Brian2 cache.
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq >/dev/null && apt-get install -y -qq build-essential >/dev/null
pip install -q brian2 cython pandas pyarrow scipy scikit-learn 2>&1 | tail -1 || true
cd /workspace/fly
python - <<'PY'
import time, sys
sys.path.insert(0, '.')
t = time.time()
from flypet.engine import Brain
from flypet import corrections
from flypet.catalog import STIMULI
from flypet.engine import StimInput
b = corrections.apply(Brain()); print('built', round(time.time() - t, 1), 's', flush=True)
st = STIMULI['looming']; t = time.time()
r = b.run([StimInput(st.resolve(), 200, key='looming')], duration_ms=250, seed=11)
print('first sim (includes compile)', round(time.time() - t, 1), 's, active', len(r.counts), flush=True)
t = time.time(); r = b.run([StimInput(st.resolve(), 200, key='looming')], duration_ms=250, seed=11)
print('second sim', round(time.time() - t, 2), 's, active', len(r.counts), flush=True)
PY
nproc; free -g | head -2
