# Flypet

A research platform for a connectome-based fly simulation and language interfaces.
The current Neural Link experiment connects a continuing Brian2 simulation to Qwen3-4B
through a learned continuous writer and a neural-token reader. Earlier experiments cover
FlyTalk, odor representations, conditioning, and a brain-to-VNC bridge.

This is a developer handoff and experimental source release. It includes runnable code,
tests, both manuscript drafts, and downloadable recorded results and interface weights.
It is not a publication-ready scientific claim or a production service.

## Run from a fresh clone

The repository is not a hosted website. The commands below create a server on the
researcher's own machine; they do not connect to the author's computer. No account,
API key, local Claude installation, private files or pretrained interface weights are
needed for the sensory garden.

Install Docker with Docker Compose, allocate at least 4 GB RAM to it, then run:

```bash
git clone https://github.com/AdeliyaLeleytner/flypet.git
cd flypet
docker compose up --build garden
```

After the container is healthy, open `/garden` on port `8765` of the machine running
Docker (on that same machine: `http://localhost:8765/garden`). The first build downloads
about 136 MB of checksum-pinned public brain data and installs the compiler and Python
dependencies. The first simulation also compiles Brian2 code. Subsequent starts reuse
the image. `docker compose down` stops the app; learned garden memory remains in its
named volume. `docker compose down --volumes` deletes that saved memory.

This mode runs the real neural simulator, sensory controls and measured readouts.
Natural-language chat/narration is optional and requires an LLM backend. It is not
part of this no-credentials garden mode. The learned Neural Link interface is below.

GitHub Actions builds this same container from the committed source, downloads the
public data, starts the server, checks the page and static files, runs baseline and
sugar stimuli through the live API, and checks FlyTalk against its reference outputs.

### Neural Link

For the experimental learned reader/writer on CPU, allow at least 24 GB RAM and
several GB of free disk space for Qwen3-4B. From the same checkout:

```bash
docker compose --profile neural up --build neural-link
```

Once the server starts, open port `8775` on the machine running Docker
(`http://localhost:8775` on that machine). This target downloads the public, hash-checked
interface bundle and the pinned Qwen3-4B base model; the base-model cache persists in a
named volume. No API subscription is needed. CPU inference can be slow. Full Qwen
inference is not exercised by the small GitHub Actions runner; the garden container and
synthetic neural tests are the CI-verified paths. The reader's known qualification
failure is recorded below.

For CUDA or Apple MPS, use the native installation below and select `--device cuda`
or `--device mps`. Nothing in either launch command publishes a server on the internet.

### Native installation and development

Use Python 3.12 and a C/C++ compiler for Brian2. Node.js 22 is needed only for frontend
tests. Start in a clone of this repository:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -c requirements-lock.txt -e '.[dev]'
python scripts/fetch_data.py brain
make test
make smoke
FLYPET_READER=0 FLYPET_THOUGHTS=0 python -m flypet.pet
```

The native garden uses port `8765`. Optional chat uses `FLYPET_LLM=ollama` with a
running local Ollama server, or `FLYPET_LLM=anthropic` with `ANTHROPIC_API_KEY`.
The legacy default chat backend is the local Claude CLI; sensory simulation does not
invoke it. Configuration comes directly from environment variables.

For native Neural Link, first install the PyTorch build suitable for your machine:

```bash
python -m pip install -c requirements-lock.txt -r requirements-llm.txt
python scripts/fetch_data.py brain neural-link
python -m flypet.latent_web --bundle output/neural-link --device cpu
```

Every downloaded artifact is checked against `data/manifest.json`. `make smoke`
compares actual simulator outputs with committed FlyTalk reference values. Allow a
few GB of RAM per brain and additional memory for the language model.

## Code map

| Area | Entry points |
|---|---|
| Connectome, independent trials, input corrections | `flypet/connectome.py`, `engine.py`, `corrections.py` |
| Continuing neural state and replay | `flypet/neural_runtime.py`, `neural_records.py`, `scripts/latent/replay_session.py` |
| Sensory inputs and plastic mushroom body | `flypet/odor.py`, `senses.py`, `mb.py` |
| Neural reader/writer and live sessions | `flypet/latent_inference.py`, `latent_observation.py`, `latent_sessions.py`, `latent_web.py` |
| Neural Link collection, training and evaluation | `scripts/latent/` |
| Continuous-agent SFT/GRPO scaffolding | `flypet/latent_agent_env.py`, `latent_policy.py`, `scripts/latent/train_agent_*.py` |
| FlyTalk benchmark and GRPO | `flypet/flytalk.py`, `scripts/flytalk/`, `scripts/flytalk_rl/` |
| Odor representations and conditioning | `scripts/rosetta/`, `scripts/rosetta_pilot/`, `scripts/teach/` |
| Historical physiology and memory experiments | `flypet/physiology_*.py`, `scripts/train_*`, `scripts/analyze_*` |
| Garden and server | `flypet/pet.py`, `garden.py`, `thoughts.py`, `static/` |
| Manuscript drafts | `paper/main.tex` (platform), `paper_iclr27/main.tex` (agent experiments) |
| Export and deployment tooling | `release/`, `deploy/` |

`Brain.run`, `run_timed`, and `run_schedule` reset trial state. `NeuralRuntime.advance`
preserves membrane, synaptic, delay and private RNG state. Do not interchange these paths.
Recorded observations are reader features, not restart checkpoints: replay uses the
initial memory, seed, input-port order and recorded actions on the same backend.

The reader gets neural observations and a question. Current-response and comparison
questions use different context layouts. The writer disables the reader's LoRA when
encoding instructions; preserve that isolation. The garden's optional narrator also sees
structured descriptions and is a separate interface from the neural-token reader.

## Results and publication work

```bash
python scripts/fetch_data.py results paper-results
python scripts/fetch_data.py brain neural-link results paper-results --check
python -m pip install -e '.[analysis]'  # plotting and molecular fingerprints
```

The initial selected results and inference bundle remain in release `v0.1.0`.
Release `v0.2.0` adds the complete local experiment families: Neural Link datasets and
historically sealed panels, all reader/writer versions, memory-interface and forecast,
projector/architecture/decoder runs, FlyTalk GRPO checkpoints, courtship, and remaining
teaching/Rosetta/physiology artifacts. Failed, partial and superseded runs are preserved.

```bash
python scripts/fetch_data.py research --no-cache
python scripts/fetch_data.py research --check
```

`research` expands to all experiment groups, including the earlier release. To work
selectively, use groups such as `neural-history`, `memory-interface`, `memory-forecast`,
`projector-history`, `architecture-reader`, `question-decoder`, `flytalk-checkpoints`,
or `courtship`; some shared summaries are in `results`. Simulation inputs are separate:
`python scripts/fetch_data.py brain vnc`. Allow at least 25 GB of free disk for the full
research download with `--no-cache`; keeping transport archives requires more space.

The machine-readable index is `data/manifest.json`: each file names its archive and
released SHA-256, and new files also record their source SHA-256. `research_coverage`
accounts for every file in the selected original directories, including exact exclusions.
Private user-session state, selected infrastructure/billing receipts, bytecode and process
locks are excluded. Historical worker source snapshots and result archives are retained.

Personal paths, host addresses and credentials are removed from metadata. Nested archive member
ownership and timestamps are normalized. Numeric arrays and tensor storage are unchanged;
original pickle serialization formats are retained. Embedded historical receipts still
refer to original bytes: use the release manifest to verify the downloaded files. These
are recorded experiments, not new training or evaluations performed during packaging.
The formerly sealed panels are now public historical evaluation data; do not treat them
as unseen tests for future model selection. The aborted `latent_runtime_20260927_v1`
is preserved for diagnosis and must not be used as the corrected training set.

Three old GPU launchers depended on an external private experiment farm and remain
omitted; their training/evaluation programs and available project artifacts are included.

Before scientific publication:

1. Repair and requalify the conversational reader. Its recorded fresh panel has current
   valence MAE 0.096 versus 0.386 with donor observations, but exact-neutral classification
   is 0/9 and `panel_gate_pass` is false. Live twins passed 8/8 in a separate recorded check.
   Preserve failed cases and donor controls; the bundle is explicitly experimental.
2. Lock the intended question and evaluation before further model selection. Continuous-agent
   SFT/GRPO code is scaffolding, not evidence of a trained successful agent. The older scalar
   FlyTalk task and the Neural Link experiment have different interfaces and evidence.
3. Select and reconcile the manuscript draft. Both `.tex` trees are working research
   artifacts with existing figures and tables; neither represents a verified submission.
   With a TeX distribution, run `latexmk -pdf main.tex` inside the chosen directory.
   The full project archive is available, but older external experiment-farm directories
   are not part of this checkout. Packaging does not establish end-to-end reproduction
   of every draft claim.
4. Validate the selected inference bundle and actual deployment: resource limits, expired
   sessions, concurrent visitors, model failures, and restart/replay behavior. Unit tests use
   small synthetic networks and fake API models; they do not qualify a public GPU service.

Historical projector experiments had train/test duplication and preprocessing leakage.
Use the clean-projector audit and generation scripts, not legacy checkpoint metrics.
The simulator is based on a female FlyWire brain; the optional male MaleCNS VNC is a
one-way type-matched bridge. This is neither a complete embodied fly nor a validated model
of subjective experience or biological sex differences.

## Development and export

```bash
make check test
python release/check.py
python release/build_latent_space.py --bundle output/neural-link --out dist/neural-preview
```

CI exercises both the simulation-only environment and CPU PyTorch tests. Ruff formats the
Python code and checks syntax/undefined names. `release/check.py` scans tracked files (or
an explicit export directory) for common credentials, personal paths and private hosts.
A bundle must match its manifest and contain no unlisted files. Keep generated artifacts
in the ignored `data/`, `output/`, or `dist/` directories.

`FLYPET_ROOT` points at an alternate checkout/data root; `FLYPET_STATE_DIR` isolates garden
state. Neural Link uses separate in-memory workers per session; the default is at most four
sessions with a 30-minute idle expiry. `FLYPET_LATENT_MAX_SESSIONS` and
`FLYPET_LATENT_TTL_SECONDS` control those bounds.

`deploy/install.sh` and `deploy/push.sh` target an explicitly supplied Linux host.
`deploy/install-public.sh api-hostname https://frontend-hostname` configures a fresh host
with a restricted garden API; it modifies system services and firewall rules. Nothing is
hosted automatically by installing this repository. GPU launchers perform paid actions only
when explicitly invoked; historical experiment-specific budgets must be reviewed for a new run.

## Licenses

Project code is MIT. FlyWire-derived data and simulation outputs retain CC BY-NC 4.0;
DoOR responses retain CC BY-SA 4.0; MaleCNS data retain CC BY 4.0. Neural tensor weights
and adapters use Apache 2.0, while anatomical/preprocessing components keep their source
terms. Bundled LaTeX support files retain their own notices. See
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for sources and attribution.
