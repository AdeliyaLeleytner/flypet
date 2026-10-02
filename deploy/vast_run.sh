#!/usr/bin/env bash
# Run projector training variants on a vast.ai GPU box.
# Usage: deploy/vast_run.sh <ssh-host> <ssh-port> [variants...]
set -euo pipefail
HOST="$1"; PORT="$2"; shift 2
SSH="ssh -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=30 -p $PORT root@$HOST"
here="$(cd "$(dirname "$0")/.." && pwd)"
if [[ "${SKIP_SETUP:-0}" == "1" ]]; then
  echo "== copy scripts only"; tar czf - -C "$here" flypet scripts/train_projector.py | $SSH 'tar xzf - -C /workspace/fly'
else
echo "== copy code + data"
tar czf - -C "$here" flypet scripts/train_projector.py data/flywire_annotations_783.tsv vendor/Drosophila_brain_model/Completeness_783.csv \
  data/state_dataset data/state_dataset_b data/state_dataset_c data/state_pairs 2>/dev/null | $SSH 'mkdir -p /workspace/fly && tar xzf - -C /workspace/fly'
echo "== deps + models"
$SSH 'cd /workspace/fly && pip install -q transformers scikit-learn accelerate safetensors huggingface_hub pandas 2>&1 | tail -1; python -c "
from huggingface_hub import snapshot_download
for m in [\"Qwen/Qwen3-0.6B\", \"Qwen/Qwen3-4B\"]: print(snapshot_download(m))"'
fi
echo "== train"
for v in "$@"; do
  echo "--- variant: $v"
  $SSH "cd /workspace/fly && FLYPET_ROOT=/workspace/fly FLYPET_DATA=/workspace/fly/data python scripts/train_projector.py $v 2>&1 | grep -v -i 'warning\|it/s\]' | tail -40"
done
echo "== fetch results"
mkdir -p "$here/data/projector_gpu"
$SSH 'cd /workspace/fly/data/projector && tar czf - report*.json' | tar xzf - -C "$here/data/projector_gpu"
ls -la "$here/data/projector_gpu"
