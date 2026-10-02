# Third-party code and data

The project's own code is MIT (see `LICENSE`). Raw source datasets are not stored in
the git repository. `scripts/fetch_data.py` downloads it from the original sources
at pinned versions, Recorded outputs and neural interface weights are separate GitHub release assets. Each file
keeps its own licence; using our code does not change the terms of the data.

## Code

**Shiu et al. whole-brain model** — <https://github.com/philshiu/Drosophila_brain_model>,
commit `91bdd1e7dcf193f3e7ca5a8933497fcef63b7960`. The leaky integrate-and-fire
equations and constants in `flypet/engine.py` are adapted from its `model.py`.

```
MIT License

Copyright (c) 2023 Philip Shiu and Nico Spiller

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

Shiu, P. K., Sterne, G. R., Spiller, N. et al. A *Drosophila* computational brain
model reveals sensorimotor processing. *Nature* 634, 210–219 (2024).
<https://doi.org/10.1038/s41586-024-07763-9>

## Data

| Files | Source (pinned) | Licence | Cite |
|---|---|---|---|
| `vendor/Drosophila_brain_model/Connectivity_783.parquet`, `Completeness_783.csv` | FlyWire public release 783, as packaged in the Shiu et al. repository at the commit above | [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/), per the [FlyWire guidelines](https://home.flywire.ai/guidelines) | Dorkenwald et al. 2024; Schlegel et al. 2024; Shiu et al. 2024 |
| `data/flywire_annotations_783.tsv` | [flyconnectome/flywire_annotations](https://github.com/flyconnectome/flywire_annotations) commit `8587524c1748ce5ef2080822a2fc890fc03bf597`, `supplemental_files/Supplemental_file1_neuron_annotations.tsv` | CC BY-NC 4.0 (FlyWire data terms) | Schlegel et al. 2024 |
| `data/door/door_response_matrix.csv`, `odor.csv`, `door_mappings.csv` | [ropensci/DoOR.data](https://github.com/ropensci/DoOR.data) commit `db323a496577c4b4a72b5c2fcd1859e07521ffb5`, `data/` | [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/), per the package `DESCRIPTION` | Münch & Galizia 2016 |
| `data/malecns/*.feather` (optional, VNC only) | [MaleCNS v1.0](https://male-cns.janelia.org/download/) flat connectome, `gs://flyem-male-cns/v1.0/connectome-data/flat-connectome/` | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) | Janelia FlyEM Male CNS connectome v1.0; credit FlyEM (HHMI Janelia), University of Cambridge, MRC LMB and Google Research as the download page asks |

Derived files carry the terms of what they are derived from:

- `data/connectivity_783.npz` is the FlyWire connectivity re-indexed into model order (CC BY-NC 4.0).
- Glomerular odour profiles built from DoOR (spontaneous firing subtracted, receptors averaged per glomerulus) are an adaptation of DoOR and stay CC BY-SA 4.0.
- `data/vnc_malecns.npz` and `data/vnc_nodes.parquet` are a thresholded, signed MaleCNS subgraph built by `scripts/build_vnc.py` (CC BY 4.0, modified).
- Simulation outputs included in the release (spike rates, episodes, evaluation logs) contain FlyWire neuron IDs and annotation labels. We release them for non-commercial use under CC BY-NC 4.0 so that they never carry fewer restrictions than the connectome they came from.

References:

- Dorkenwald, S. et al. Neuronal wiring diagram of an adult brain. *Nature* 634, 124–138 (2024). <https://doi.org/10.1038/s41586-024-07558-y>
- Schlegel, P. et al. Whole-brain annotation and multi-connectome cell typing of *Drosophila*. *Nature* 634, 139–152 (2024). <https://doi.org/10.1038/s41586-024-07686-5>
- Münch, D. & Galizia, C. G. DoOR 2.0 — comprehensive mapping of *Drosophila melanogaster* odorant responses. *Sci. Rep.* 6, 21841 (2016). <https://doi.org/10.1038/srep21841>

## Models

Language models are downloaded separately from Hugging Face and keep their own
licences. Qwen3 models (0.6B, 4B, 32B) are Apache 2.0. The FlyTalk LoRA adapters
we publish are trained on top of `Qwen/Qwen3-32B` and are released under Apache 2.0
as well, to match the base model.

Neural Link reader, writer and LoRA tensor parameters are Apache 2.0. Their anatomical
identifiers and simulation-derived preprocessing retain FlyWire CC BY-NC 4.0 terms;
DoOR-derived odor profiles retain CC BY-SA 4.0. Component-specific terms accompany the
model download as `output/neural-link-LICENSES.md`. This does not relicense source datasets.

## Manuscript support files

Bundled `natbib.sty`, `fancyhdr.sty`, and the ICLR style and bibliography files retain
the copyright and licensing notices in their headers. They are not relicensed as
project-authored MIT code. Figures and tables reporting FlyWire-derived simulation
results are subject to the source-data terms above.
