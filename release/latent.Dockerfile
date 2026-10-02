FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
COPY requirements-space.txt /tmp/requirements-space.txt
RUN pip install --no-cache-dir -r /tmp/requirements-space.txt
RUN useradd -m -u 1000 user
USER user
ENV PATH=/home/user/.local/bin:$PATH HF_HOME=/home/user/.cache/huggingface
ENV HF_HUB_DISABLE_IMPLICIT_TOKEN=1 TOKENIZERS_PARALLELISM=false FLYPET_BRIAN_TARGET=cython OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
ENV FLYPET_LATENT_MAX_SESSIONS=8 FLYPET_LATENT_TTL_SECONDS=300
WORKDIR /home/user/app
COPY --chown=user . .
RUN python scripts/fetch_data.py brain
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3-4B', revision='1cfa9a7208912126459214e8b04321603b3df60c', allow_patterns=['*.json', '*.safetensors', '*.txt', '*.jinja'])"
EXPOSE 7860
CMD ["python", "-m", "flypet.latent_web", "--bundle", "bundle", "--host", "0.0.0.0", "--port", "7860", "--device", "cuda"]
