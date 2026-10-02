#!/usr/bin/env bash
# Server-side install (run as root on the VDS): python env, systemd unit, optional Ollama + Qwen, Tailscale serve.
# Usage: sudo deploy/install.sh [--with-ollama] [--model qwen3:4b] [--serve-port 8766]
set -euo pipefail
[[ ${EUID} -eq 0 ]] || { echo "Run with sudo." >&2; exit 1; }

APP_DIR=/opt/flypet
STATE_DIR=/var/lib/flypet
CONFIG_DIR=/etc/flypet
WITH_OLLAMA=0; MODEL=qwen3:4b; SERVE_PORT=8766
while [[ $# -gt 0 ]]; do
  case "$1" in
    --with-ollama) WITH_OLLAMA=1 ;;
    --model) MODEL="$2"; shift ;;
    --serve-port) SERVE_PORT="$2"; shift ;;
    *) echo "unknown option $1" >&2; exit 1 ;;
  esac; shift
done

echo "== system packages (compiler for Brian2/Cython)"
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq && apt-get install -y -qq build-essential rsync curl >/dev/null

if ! getent passwd flypet >/dev/null; then
  useradd --system --home-dir "$STATE_DIR" --shell /usr/sbin/nologin flypet
fi
install -d -m 0755 -o root -g root "$APP_DIR"
install -d -m 0750 -o root -g flypet "$CONFIG_DIR"
install -d -m 0750 -o flypet -g flypet "$STATE_DIR"
[[ -f "$CONFIG_DIR/app.env" ]] || { printf 'FLYPET_LLM=ollama\nFLYPET_NARRATOR_BACKEND=ollama\nFLYPET_OLLAMA_MODEL=%s\nFLYPET_OLLAMA_KEEP_ALIVE=30s\nFLYPET_READER=0\n# FLYPET_ALLOWED_ORIGINS=https://owner-space.hf.space\n' "$MODEL" > "$CONFIG_DIR/app.env"; chown root:flypet "$CONFIG_DIR/app.env"; chmod 0640 "$CONFIG_DIR/app.env"; }

if [[ $(swapon --show --noheadings | wc -l) -eq 0 ]] && [[ ! -f /swapfile ]]; then
  echo "== 2 GB swapfile (the VDS has 3.3 GB RAM; the brain model needs ~1 GB)"
  fallocate -l 2G /swapfile && chmod 600 /swapfile && mkswap /swapfile >/dev/null && swapon /swapfile
  grep -q '^/swapfile' /etc/fstab || echo '/swapfile none swap sw 0 0' >> /etc/fstab
fi

echo "== python 3.12 + venv via uv"
UV=$(command -v uv || ls /root/.local/bin/uv "/home/${SUDO_USER:-nobody}/.local/bin/uv" 2>/dev/null | head -1 || true)
[[ -n "$UV" ]] || { echo "uv not found; install it first (https://docs.astral.sh/uv/)" >&2; exit 1; }
cd "$APP_DIR"
export UV_PYTHON_INSTALL_DIR="$APP_DIR/.python"
"$UV" python install 3.12
PY=$(ls -d "$APP_DIR"/.python/cpython-3.12*/bin/python3.12 | head -1)
[[ -x "$APP_DIR/.venv/bin/python" ]] || "$UV" venv --python "$PY" "$APP_DIR/.venv"
"$UV" pip install --python "$APP_DIR/.venv/bin/python" -r "$APP_DIR/requirements.txt"
chmod -R a+rX "$APP_DIR"

if [[ $WITH_OLLAMA -eq 1 ]]; then
  echo "== ollama + $MODEL"
  command -v ollama >/dev/null || curl -fsSL https://ollama.com/install.sh | sh
  systemctl enable --now ollama
  ollama pull "$MODEL"
fi

echo "== systemd"
install -m 0644 "$APP_DIR/deploy/flypet.service" /etc/systemd/system/flypet.service
systemctl daemon-reload
systemctl enable --now flypet.service
sleep 3
systemctl is-active flypet.service

if command -v tailscale >/dev/null; then
  echo "== tailscale serve: https://$(tailscale status --self --json | python3 -c 'import sys,json;print(json.load(sys.stdin)["Self"]["DNSName"].rstrip("."))'):$SERVE_PORT"
  tailscale serve --bg --https="$SERVE_PORT" http://127.0.0.1:8766 || echo "tailscale serve failed (HTTPS certs / MagicDNS must be enabled in the tailnet)"
fi
echo "== waiting for the brain to load"
for i in $(seq 1 300); do curl -fsS http://127.0.0.1:8766/healthz 2>/dev/null && break; sleep 2; done; echo
free -h | sed -n 2p; systemctl status flypet.service --no-pager | head -5
