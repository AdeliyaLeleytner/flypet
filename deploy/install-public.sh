#!/usr/bin/env bash
# Fresh standalone Ubuntu host only: all services are local except restricted HTTPS API.
set -euo pipefail
[[ $EUID -eq 0 ]] || { echo 'Run as root.' >&2; exit 1; }
[[ $# -eq 2 && $1 =~ ^[a-zA-Z0-9.-]+$ && $2 =~ ^https://[a-zA-Z0-9.-]+$ ]] || { echo 'Usage: install-public.sh api-hostname https://frontend-hostname' >&2; exit 1; }
API_HOST=$1
FRONTEND_ORIGIN=$2
APP_DIR=/opt/flypet
STATE_DIR=/var/lib/flypet
CONFIG_DIR=/etc/flypet
[[ -f "$APP_DIR/requirements.txt" && -d "$APP_DIR/flypet" ]] || { echo 'Stage the application with deploy/push.sh --stage-only first.' >&2; exit 1; }
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq build-essential curl ca-certificates rsync caddy ufw >/dev/null
command -v uv >/dev/null || { curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/usr/local/bin sh; }
if ! id flypet >/dev/null 2>&1; then
    useradd --system --home-dir "$STATE_DIR" --shell /usr/sbin/nologin flypet
fi
install -d -m 0755 "$APP_DIR"
install -d -m 0750 -o flypet -g flypet "$STATE_DIR"
install -d -m 0750 -o root -g flypet "$CONFIG_DIR"
if [[ ! -f "$CONFIG_DIR/app.env" ]]; then
    cat > "$CONFIG_DIR/app.env" <<ENV
FLYPET_LLM=ollama
FLYPET_NARRATOR_BACKEND=ollama
FLYPET_OLLAMA_MODEL=qwen3:0.6b
FLYPET_OLLAMA_KEEP_ALIVE=5m
FLYPET_OLLAMA_NUM_THREAD=2
FLYPET_NARRATOR_CONTEXT=1024
FLYPET_READER=0
FLYPET_ALLOWED_ORIGINS=${FRONTEND_ORIGIN}
ENV
    chown root:flypet "$CONFIG_DIR/app.env"
    chmod 0640 "$CONFIG_DIR/app.env"
fi
if [[ -z $(swapon --show --noheadings) && ! -e /swapfile ]]; then
    fallocate -l 2G /swapfile
    chmod 600 /swapfile
    mkswap /swapfile >/dev/null
    swapon /swapfile
    echo '/swapfile none swap sw 0 0' >> /etc/fstab
fi
cd "$APP_DIR"
export UV_PYTHON_INSTALL_DIR="$APP_DIR/.python"
uv python install 3.12
uv venv --python 3.12 --allow-existing "$APP_DIR/.venv"
uv pip install --python "$APP_DIR/.venv/bin/python" -r requirements.txt
uv pip freeze --python "$APP_DIR/.venv/bin/python" > "$APP_DIR/deploy/installed-versions.txt"
chmod -R a+rX "$APP_DIR"
command -v ollama >/dev/null || { curl -fsSL https://ollama.com/install.sh | sh; }
install -d /etc/systemd/system/ollama.service.d
cat > /etc/systemd/system/ollama.service.d/fly-garden.conf <<'UNIT'
[Service]
Environment=OLLAMA_HOST=127.0.0.1:11434
Environment=OLLAMA_MAX_LOADED_MODELS=1
Environment=OLLAMA_NUM_PARALLEL=1
UNIT
systemctl daemon-reload
systemctl enable --now ollama
systemctl restart ollama
for attempt in $(seq 1 30); do
    curl -fsS http://127.0.0.1:11434/api/tags >/dev/null 2>&1 && break
    sleep 1
done
ollama pull qwen3:0.6b
install -m 0644 deploy/flypet.service /etc/systemd/system/flypet.service
systemctl daemon-reload
systemctl enable --now flypet
systemctl restart flypet
for attempt in $(seq 1 90); do
    curl -fsS http://127.0.0.1:8766/healthz >/dev/null 2>&1 && break
    sleep 1
done
curl --fail --max-time 10 http://127.0.0.1:8766/healthz
"$APP_DIR/.venv/bin/python" - <<'PY'
import json, urllib.request
with urllib.request.urlopen('http://127.0.0.1:8766/healthz',timeout=10) as response:
    health=json.load(response)
assert health.get('ok') and health.get('neurons') == 138639, health
PY
# Preserve the new host's packaged default Caddy configuration for rollback.
if [[ -e /etc/caddy/Caddyfile && ! -e /etc/caddy/Caddyfile.before-flypet ]]; then
    cp -p /etc/caddy/Caddyfile /etc/caddy/Caddyfile.before-flypet
fi
cat > /etc/caddy/Caddyfile <<CADDY
${API_HOST} {
    request_body {
        max_size 16KB
    }
    @garden path /healthz /api/garden /api/garden/session /api/garden/options /api/garden/thought /api/garden/reset
    handle @garden {
        reverse_proxy 127.0.0.1:8766 {
            header_up X-Flypet-Public 1
        }
    }
    handle {
        respond "Not found" 404
    }
}
CADDY
caddy validate --config /etc/caddy/Caddyfile --adapter caddyfile
ufw allow 22/tcp >/dev/null
ufw allow 80/tcp >/dev/null
ufw allow 443/tcp >/dev/null
ufw --force enable
systemctl enable --now caddy
systemctl reload caddy
systemctl is-active flypet ollama caddy
echo "Services configured for https://${API_HOST}; verify certificate and public trials externally before switching HF."
