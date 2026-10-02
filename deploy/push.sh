#!/usr/bin/env bash
# Push code + connectome data to the server and (re)start the service.
# Usage: deploy/push.sh user@host [--install [--with-ollama]] [-- ssh options]
set -euo pipefail
[[ $# -ge 1 ]] || { echo "Usage: deploy/push.sh user@host [--install [--with-ollama]]" >&2; exit 1; }
target="$1"; shift
install_args=(); do_install=0; stage_only=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --install) do_install=1 ;;
    --stage-only) stage_only=1 ;;
    --with-ollama|--model|--serve-port) install_args+=("$1"); [[ "$1" != "--with-ollama" ]] && { install_args+=("$2"); shift; } ;;
    --) shift; break ;;
    *) echo "unknown option $1" >&2; exit 1 ;;
  esac; shift
done
ssh_opts=("$@")
here="$(cd "$(dirname "$0")/.." && pwd)"
V=vendor/Drosophila_brain_model
ssh "${ssh_opts[@]}" "$target" 'sudo -n install -d -m 0755 /opt/flypet'
rsync -az --no-owner --no-group --rsync-path='sudo -n rsync' \
  --exclude '.venv' --exclude '.python' --exclude '__pycache__' --exclude '.git' --exclude 'data/sessions' \
  --exclude 'data/pet_state.json' --exclude 'data/demo_stimuli_results.json' --exclude '.claude' \
  -e "ssh ${ssh_opts[*]}" \
  "$here/flypet" "$here/scripts" "$here/deploy" "$here/requirements.txt" "$here/README.md" \
  "$target:/opt/flypet/"
rsync -az --no-owner --no-group --rsync-path='sudo -n rsync' -e "ssh ${ssh_opts[*]}" --relative \
  "$here/./data/flywire_annotations_783.tsv" "$here/./data/connectivity_783.npz" \
  "$here/./data/door/door_response_matrix.csv" "$here/./data/door/odor.csv" "$here/./data/door/door_mappings.csv" \
  "$here/./data/vnc_malecns.npz" "$here/./data/vnc_nodes.parquet" \
  "$here/./$V/Completeness_783.csv" \
  "$here/./$V/LICENSE" \
  "$target:/opt/flypet/"
if [[ $stage_only -eq 1 ]]; then
  echo 'Code and data staged; no service changed.'
elif [[ $do_install -eq 1 ]]; then
  ssh "${ssh_opts[@]}" "$target" "sudo bash /opt/flypet/deploy/install.sh ${install_args[*]:-}"
else
  ssh "${ssh_opts[@]}" "$target" 'sudo chown -R root:root /opt/flypet && sudo chmod -R a+rX /opt/flypet && sudo systemctl restart flypet.service && sleep 3 && systemctl is-active flypet.service && for i in $(seq 1 300); do curl -fsS http://127.0.0.1:8766/healthz 2>/dev/null && break; sleep 2; done; echo'
fi
