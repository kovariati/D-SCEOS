# Swarm-inspired peer-to-peer control for resource-aware flexibility sharing in cyber–physical energy systems

[![DOI](https://img.shields.io/badge/DOI-10.1016%2Fj.ecmx.2026.102212-blue)](https://doi.org/10.1016/j.ecmx.2026.102212)
[![Journal](https://img.shields.io/badge/Energy%20Conversion%20and%20Management%3A%20X-Volume%2031-0b7285)](https://www.sciencedirect.com/science/article/pii/S2590174526006951)
[![Article license](https://img.shields.io/badge/article-CC%20BY%204.0-green)](https://creativecommons.org/licenses/by/4.0/)
[![Code license](https://img.shields.io/badge/code-MIT-green)](LICENSE)
[![Release](https://img.shields.io/badge/release-v1.0.0-blueviolet)](https://github.com/kovariati/D-SCEOS/releases/tag/v1.0.0)

**Official code and reproducibility repository for the published D-SCEOS article.**

> **Attila Kovari (2026).** *Swarm-inspired peer-to-peer control for resource-aware flexibility sharing in cyber–physical energy systems*. **Energy Conversion and Management: X, 31**, 102212. https://doi.org/10.1016/j.ecmx.2026.102212

**Publisher:** Elsevier · **Volume:** 31 · **Issue date:** September 2026 · **Article number:** 102212 · **Available online:** 18 August 2026 · **ISSN:** 2590-1745 · **Open access:** CC BY 4.0

- **Canonical DOI:** https://doi.org/10.1016/j.ecmx.2026.102212
- **ScienceDirect article:** https://www.sciencedirect.com/science/article/pii/S2590174526006951
- **Canonical code repository:** https://github.com/kovariati/D-SCEOS
- **Version corresponding to the published article:** [v1.0.0](https://github.com/kovariati/D-SCEOS/releases/tag/v1.0.0)
- **Preferred citation target:** the peer-reviewed journal article at https://doi.org/10.1016/j.ecmx.2026.102212
- **Published numerical artefacts:** [D-SCEOS v1.0.0 results and artefacts](https://github.com/kovariati/D-SCEOS/releases/download/v1.0.0/D-SCEOS-v1.0.0-results-and-artifacts.zip)

If this repository, its implementation, experiments, or released results contribute to a publication, comparison, review, or derivative study, please cite the journal article above. Machine-readable citation files are provided in [`CITATION.cff`](CITATION.cff), [`CITATION.bib`](CITATION.bib), and [`CITATION.ris`](CITATION.ris).

## What is D-SCEOS?

**D-SCEOS** is a gateway-injected, peer-to-peer control and reproducibility framework for **resource-aware flexibility sharing in cyber-physical energy systems (CPES)**. It combines dynamic aggregate and target estimation, a graph-local optimization-guided gradient, capacity-normalized utilization sharing, and a local **control Lyapunov function / high-order control barrier function (CLF/HOCBF) QCQP** operating-envelope step.

The published study evaluates D-SCEOS on heterogeneous virtual-power-plant configurations and controlled stress tests. Each agent retains a fixed-size local decision and exchanges a **nine-scalar payload per neighbor**. Under the equal-effort comparator protocol reported in the article, D-SCEOS has the lower integrated dimensionless objective in five of six configurations and a higher value in one; one of the five reductions lies inside the declared 1% practical-equivalence band. Component ablation identifies the size-consistent tracking and capacity-normalized sharing terms as the main performance contributors. No capacity violation is observed at the recorded sampling instants within the tested hard-feasible envelope. The paper explicitly limits the stability claim to a reduced continuous-time surrogate and does not claim a sampled-data or detailed multi-energy plant-model guarantee.

This repository contains the controller implementation, simulation harness, comparator implementations, experiment runners, validation logic, and machine-readable reference results required to reproduce and audit the published numerical study.

## Research areas and search terms

**Paper keywords:** cyber–physical energy systems; distributed control; flexibility; smart grids; energy storage; control barrier functions; swarm coordination.

**Related indexing terms:** peer-to-peer control; decentralized control; distributed energy resources; DER flexibility; flexibility aggregation; virtual power plant; VPP control; multi-agent systems; swarm-inspired control; swarm intelligence; resource-aware control; capacity-normalized flexibility sharing; dynamic average consensus; distributed optimization; control Lyapunov functions; CLF; high-order control barrier functions; HOCBF; safe control; operating-envelope constraints; demand response; energy storage coordination; smart-grid flexibility; cyber-physical energy system control; reproducible energy-systems simulation.

## Published-paper highlights reproduced here

- Peer-to-peer flexibility sharing without a per-step central allocator.
- Fixed-size local decision at each agent and a nine-scalar neighbor message.
- Capacity-normalized sharing across heterogeneous flexibility resources.
- Local CLF/HOCBF-QCQP operating-envelope certification.
- Equal-effort comparator protocol with per-case retuning and high-effort projected-gradient runs.
- Paired Monte Carlo campaigns, component ablation, sampling-time sensitivity, and robust paired inference.
- Independent conic-solver cross-checks and centralized-reference KKT diagnostics.
- Communication-overlay degradation, gateway sensitivity, and supervised agent-loss diagnostics.

## Repository contents

```text
D-SCEOS/
├── code/                 Controller core, safety filter, scenarios, graph configuration, simulation harness, and figure generators
├── reruns/               One script per reported experiment or robustness analysis
├── results/              Small generated JSON/CSV summaries used for validation
├── run_all.py            Experiment manifest and reproduction orchestrator
├── validate_results.py   Fail-closed validator for released artefacts
├── expected_results.json Reference values checked by the validator
├── objective_main_figure_summary.json  Compact N15 figure/result synchronization manifest
├── test_*.py             Regression and source-integrity tests
├── requirements.txt      Working Python dependency set
├── requirements-lock.txt Fully pinned reproduction environment
├── REPRODUCTION_LEVELS.md
├── README_RUNNING.md
├── CITATION.cff          GitHub/CFF machine-readable citation metadata
├── CITATION.bib          BibTeX citation for the published article
├── CITATION.ris          RIS citation for reference managers
├── codemeta.json         CodeMeta software metadata linked to the article
├── ARTICLE_METADATA.json Schema.org/JSON-LD metadata for the article
├── llms.txt              Plain-text project and citation index for machine/AI retrieval
└── LICENSE               MIT license for the source code
```

The public repository intentionally focuses on the **analysis, simulation, and reproducibility software layer**. The canonical journal article remains the DOI-linked Elsevier publication; manuscript source files and large raw trajectory archives are not duplicated here.

## Environment

For the released v1.0.0 reproduction environment, use **Python 3.11 or 3.12**. Python 3.13 is not part of the pinned validation environment.

`requirements.txt` specifies the working dependency set. `requirements-lock.txt` pins the complete released environment for exact reproduction. The main stack includes NumPy, SciPy, pandas, matplotlib, pytest, pyflakes, and CVXPY; CVXPY is required only for the conic cross-check scripts.

### Windows PowerShell

```powershell
py -3.11 -m venv .venv
Set-ExecutionPolicy -Scope Process Bypass -Force
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements-lock.txt
```

### Linux or macOS

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements-lock.txt
```

Small last-digit differences can occur under different BLAS/LAPACK builds. The validator therefore checks the substantive orderings, win rates, confidence-interval signs, capacity-violation conditions, provenance, and exact-gradient identities in a stack-robust way.

## Quick start

Run the regression suite (**33 tests** in v1.0.0):

```bash
python -m pytest -q
```

Run a complete cold reproduction:

```bash
python run_all.py --cold
```

Validate an existing set of generated artefacts:

```bash
python validate_results.py
```

List the experiment manifest or execute a subset:

```bash
python run_all.py --list
python run_all.py "Stress ladder"
```

See [`REPRODUCTION_LEVELS.md`](REPRODUCTION_LEVELS.md) for the smoke, shipped-artefact-validation, and cold-regeneration levels, and [`README_RUNNING.md`](README_RUNNING.md) for platform-specific notes.

## Selected experiments

```bash
# main technology-representative study
python reruns/batch_realistic.py

# stress ladder
python reruns/ladder_rerun.py

# paired Monte Carlo studies
python reruns/monte_carlo.py

# component ablation
python reruns/ablation.py --configs N15_a,N60_a,N60_c

# independent QCQP/conic solver cross-check
python reruns/qcqp_crosscheck_run.py --only ladder

# centralized-reference diagnostics
python reruns/reference_kkt_diagnostics.py

# equal-effort comparator protocol
python reruns/comparator_protocol.py

# sampling-time sensitivity
python reruns/sampling_time_study.py

# robust paired inference
python reruns/robust_inference.py

# dense-grid and objective-indexing checks
python reruns/intersample_check.py
python reruns/objective_indexing_check.py

# communication-overlay degradation
python reruns/graph_decoupling.py

# in-loop conic recomputation
python reruns/conic_recompute.py
```

The Monte Carlo campaigns and larger-fleet configurations are the most computationally demanding entries.

## Reproducibility policy

The released workflow separates three levels:

1. **Smoke / regression testing** — analytic identities, source integrity, graph reproduction, and table/result consistency.
2. **Validation of shipped artefacts** — checks that released result artefacts satisfy the conclusions asserted by the repository.
3. **Cold regeneration** — deletes resumable artefacts/checkpoints, re-runs the complete experiment manifest, and validates the regenerated outputs.

A fresh clone is expected to fail result validation when required generated trajectories are absent. This **fail-closed** behavior is intentional: missing artefacts are not interpreted as successful validation.

## Data and artefact availability

No external dataset is required for the numerical study. The evidence is generated from the simulation models, controller definitions, experiment configurations, and fixed/randomized seeds included in this repository.

The complete numerical artefact archive associated with the published `v1.0.0` validation is available as a GitHub Release asset:

- **Results and artefacts:** https://github.com/kovariati/D-SCEOS/releases/download/v1.0.0/D-SCEOS-v1.0.0-results-and-artifacts.zip
- **SHA-256 checksum:** https://github.com/kovariati/D-SCEOS/releases/download/v1.0.0/D-SCEOS-v1.0.0-results-and-artifacts.zip.sha256
- **Release page:** https://github.com/kovariati/D-SCEOS/releases/tag/v1.0.0

The archive contains the full trajectories and per-step metrics for the primary published runs together with seed-level Monte Carlo outputs, comparator results, sensitivity and ablation studies, topology and communication diagnostics, solver cross-checks, scalability results, a machine-readable manifest, and per-file SHA-256 checksums.

Large generated trajectories are intentionally excluded from Git history. They can either be obtained from the archived `v1.0.0` Release asset or regenerated locally from the released code and experiment definitions.

When these artefacts, the software, or the experimental protocol are used in scientific work, please cite the associated journal article:

**Kovari, A. (2026).** *Swarm-inspired peer-to-peer control for resource-aware flexibility sharing in cyber–physical energy systems*. **Energy Conversion and Management: X, 31**, 102212. https://doi.org/10.1016/j.ecmx.2026.102212

## How to cite

### Recommended journal citation

Kovari, A. (2026). Swarm-inspired peer-to-peer control for resource-aware flexibility sharing in cyber–physical energy systems. *Energy Conversion and Management: X, 31*, 102212. https://doi.org/10.1016/j.ecmx.2026.102212

### BibTeX

```bibtex
@article{Kovari2026DSCEOS,
  author  = {Kovari, Attila},
  title   = {Swarm-inspired peer-to-peer control for resource-aware flexibility sharing in cyber--physical energy systems},
  journal = {Energy Conversion and Management: X},
  volume  = {31},
  pages   = {102212},
  year    = {2026},
  month   = sep,
  issn    = {2590-1745},
  doi     = {10.1016/j.ecmx.2026.102212},
  url     = {https://doi.org/10.1016/j.ecmx.2026.102212}
}
```

For automated citation discovery, GitHub can read [`CITATION.cff`](CITATION.cff). BibTeX and RIS exports are included for direct import into reference managers.

## Machine-readable metadata

The repository exposes the publication and software relationship in several complementary formats:

- [`CITATION.cff`](CITATION.cff) — Citation File Format with the journal article as the preferred citation.
- [`CITATION.bib`](CITATION.bib) — BibTeX record for the published article.
- [`CITATION.ris`](CITATION.ris) — RIS record for Zotero, Mendeley, EndNote, and other reference managers.
- [`codemeta.json`](codemeta.json) — CodeMeta software metadata with the DOI-linked article as the reference publication.
- [`ARTICLE_METADATA.json`](ARTICLE_METADATA.json) — Schema.org JSON-LD scholarly-article metadata including DOI, PII, journal, volume, publication date, keywords, and license.
- [`llms.txt`](llms.txt) — concise plain-text project, terminology, canonical-link, and citation index intended to make the repository easier for machine and AI retrieval systems to interpret. It is supplementary metadata and does not replace the canonical DOI or citation files.

## Open access and licenses

The **journal article** is published open access by Elsevier under **Creative Commons Attribution 4.0 International (CC BY 4.0)**.

The **source code in this repository** is separately released under the **MIT License**. See [`LICENSE`](LICENSE). The article license and the software license apply to different research objects and should not be conflated.

## Persistent identifiers

- DOI: **10.1016/j.ecmx.2026.102212**
- ScienceDirect PII: **S2590174526006951**
- Journal: **Energy Conversion and Management: X**
- Volume: **31 (September 2026)**
- Article number: **102212**
- Online ISSN: **2590-1745**
- Repository: **https://github.com/kovariati/D-SCEOS**
- Release: **https://github.com/kovariati/D-SCEOS/releases/tag/v1.0.0**

## About

**D-SCEOS — official code and reproducibility package for:** *Swarm-inspired peer-to-peer control for resource-aware flexibility sharing in cyber–physical energy systems*, Energy Conversion and Management: X 31 (2026) 102212, DOI 10.1016/j.ecmx.2026.102212.
