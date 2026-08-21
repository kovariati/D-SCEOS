# Reproduction levels

This document describes the reproduction levels for **D-SCEOS v1.0.0**, the software release associated with the published article DOI **10.1016/j.ecmx.2026.102212**.

The full reproduction is computationally heavy, so it is split into three explicit levels. Each level
is self-contained and states what it does and does not establish.

## Level 1 — smoke (seconds, no simulation)

    python -m pytest -q                        # 33 tests

Establishes: the analytic gradient matches numerical differentiation, the CLF derivation is correct,
graph reproduction from the seed is exact, and the constraint-handling and source-integrity guards
behave as specified. The full 33-test suite requires the dependencies in `requirements-lock.txt`,
including `pyflakes`.

The optional manuscript/table audit can also be run when a LaTeX manuscript source is available:

    python reruns/validate_paper_tables.py --paper /path/to/paper.tex

The public code repository intentionally does not ship the manuscript source, so this optional command
is not part of a source-only Level-1 PASS.

Does not establish: that the stored artefacts can be regenerated.

## Level 2 — validate a complete generated artefact set (minutes)

    python validate_results.py

Establishes: the available result artefacts are internally consistent and satisfy the asserted
conclusions (orderings, confidence-interval signs, win rates, zero capacity violation, solver
residuals, provenance hashes). It also checks the compact `objective_main_figure_summary.json`
against the released N15 objective summaries so the publication-facing objective values stay synchronized.

A source-only fresh clone does **not** contain the large `state_history.npz` trajectories needed by
the centralized-reference/R2 recomputation. The complete numerical artefact set corresponding to the
published `v1.0.0` release can be downloaded from:

https://github.com/kovariati/D-SCEOS/releases/download/v1.0.0/D-SCEOS-v1.0.0-results-and-artifacts.zip

with its SHA-256 checksum available at:

https://github.com/kovariati/D-SCEOS/releases/download/v1.0.0/D-SCEOS-v1.0.0-results-and-artifacts.zip.sha256

Without these released artefacts or locally regenerated trajectories, `validate_results.py` is expected
to fail. This fail-closed behavior is deliberate. Level 3 independently regenerates the complete
artefact set from the released code and then re-runs the validator.

Does not establish by itself that the artefacts were produced from an empty state by the current code.
Level 3 does that.

## Level 3 — cold regeneration (hours)

    python run_all.py --cold

`--cold` first deletes every artefact a run could be resumed from (Monte Carlo `*.partial.json`
checkpoints, stored trajectories, agent-loss outputs and their provenance sidecars) and then runs the
full manifest. Plain `--all` does NOT do this: the Monte Carlo scripts reuse any checkpoint they find,
so `--all` on a shipped package resumes rather than regenerates. Use `--cold` for a genuine cold
reproduction.

Runs every experiment from an empty output directory and then re-runs the validator. This is the only
level that demonstrates end-to-end reproducibility. The Monte Carlo campaigns and the N=60
configurations dominate the runtime; `reruns/monte_carlo.py` and
`reruns/headroom_normalisation.py` checkpoint incrementally and can be interrupted and resumed.

## Dependency notes

Only `reruns/qcqp_crosscheck_run.py` and `reruns/conic_recompute.py` need `cvxpy` (which brings
CLARABEL). Everything else runs without it; `reruns/qcqp_crosscheck.py` imports it lazily so that
scripts merely reusing its configuration builders do not require it.

`pyflakes` is required for the complete 33-test regression/source-hygiene suite and is included in the
pinned environment. If it is absent, the two independent Pyflakes cross-check tests fail for an
environment reason rather than an algorithmic one.
