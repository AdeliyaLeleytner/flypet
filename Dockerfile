FROM python:3.12-slim-bookworm AS simulation

RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY . .
RUN python -m pip install --no-cache-dir -c requirements-lock.txt . \
    && useradd --create-home --uid 1000 researcher \
    && mkdir -p /var/lib/flypet /var/cache/flypet/huggingface \
    && chown -R researcher:researcher /app /var/lib/flypet /var/cache/flypet
USER researcher
ENV PYTHONUNBUFFERED=1 FLYPET_ROOT=/app FLYPET_STATE_DIR=/var/lib/flypet \
    FLYPET_BRIAN_TARGET=cython OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
RUN python scripts/fetch_data.py brain

FROM simulation AS garden
ENV FLYPET_READER=0 FLYPET_THOUGHTS=0
EXPOSE 8765
HEALTHCHECK --interval=10s --timeout=5s --start-period=120s --retries=12 \
    CMD ["python", "-c", "import json,urllib.request; assert json.load(urllib.request.urlopen('http://127.0.0.1:8765/healthz',timeout=4))['ok']"]
CMD ["python", "-m", "flypet.pet", "--host", "0.0.0.0", "--port", "8765"]

FROM simulation AS neural-link
USER root
RUN python -m pip install --no-cache-dir torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu \
    && python -m pip install --no-cache-dir -c requirements-lock.txt -r requirements-llm.txt
USER researcher
ENV HF_HOME=/var/cache/flypet/huggingface HF_HUB_DISABLE_IMPLICIT_TOKEN=1 TOKENIZERS_PARALLELISM=false \
    FLYPET_LATENT_MAX_SESSIONS=2
RUN python scripts/fetch_data.py neural-link
EXPOSE 8775
CMD ["python", "-m", "flypet.latent_web", "--bundle", "output/neural-link", "--device", "cpu", "--host", "0.0.0.0", "--port", "8775"]
