# Flypet

A research platform for a connectome-based fly simulation and language interfaces.
The current Neural Link experiment connects a continuing Brian2 simulation to Qwen3-4B
through a learned continuous writer and a neural-token reader. Earlier experiments cover
FlyTalk, odor representations, conditioning, and a brain-to-VNC bridge.

This is a developer handoff and experimental source release. It includes runnable code,
tests, both manuscript drafts, and downloadable recorded results and interface weights.
It is not a publication-ready scientific claim or a production service.

## Install and run

Use Python 3.12, a C/C++ compiler for Brian2, and Node.js 22 for the frontend tests.
Run commands from this checkout. The default interfaces bind to localhost.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -c requirements-lock.txt -e '.[dev]'
python scripts/fetch_data.py brain
make test
make smoke
```

The brain download is about 136 MB and comes directly from pinned upstream revisions.
Every file is checked against `data/manifest.json`; a derived connectivity cache is built
on first use. Allow a few GB of RAM per brain and extra time for the first Cython compilation.
`make smoke` checks FlyTalk simulator outputs against committed reference values.

Run the sensory garden without a language-model service:

```bash
FLYPET_READER=0 FLYPET_THOUGHTS=0 python -m flypet.pet
```

Open <http://127.0.0.1:8765/garden>. Stimuli and measured readouts work without an LLM;
chat requires a backend. `FLYPET_LLM=ollama` uses a running local Ollama server;
`FLYPET_LLM=anthropic` uses `ANTHROPIC_API_KEY`. The legacy default is the local Claude CLI.
Configuration is read directly from the environment; no environment template is needed.

For Neural Link, install the PyTorch build suitable for your machine, then:

```bash
python -m pip install -c requirements-lock.txt -r requirements-llm.txt
python scripts/fetch_data.py neural-link
python -m flypet.latent_web --bundle output/neural-link
```

Open <http://127.0.0.1:8775>. This downloads a roughly 30 MB interface bundle; the
Qwen3-4B base model is downloaded separately from Hugging Face at the revision recorded
in the bundle. `--device cuda`, `--device mps`, or `--device cpu` selects the backend.
Full-model latency and memory use must be checked on the deployment machine.

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

Artifacts are attached to the `v0.1.0` GitHub release. `results` restores the published
manifest's historical experiment outputs; `paper-results` restores the principal platform
cases, language cases, reader summaries and analysis tables. `neural-link` also restores
its reader/writer and fresh evaluation reports in `output/neural-link-reports/`.
These are recorded results, not experiments rerun during this cleanup.

Author-machine paths in a few metadata files were made relative. The transport manifest
records both the released hash and `source_sha256` for changed files. Historical receipts
inside those files still refer to original bytes. Raw numeric arrays and scientific values
were not changed. Raw session memory, billing records, abandoned exports and caches are not
part of the release. Three old GPU launchers depended on an external, private experiment
farm and were omitted; their actual training/evaluation programs remain in `scripts/`.

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
   Some historical physiology/decoder training inputs remain outside the released data;
   the shipped results do not establish full end-to-end reproduction of every draft claim.
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
