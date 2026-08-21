# Running the scripts

These instructions apply to **D-SCEOS v1.0.0**, the reproducibility release for DOI **10.1016/j.ecmx.2026.102212**.

## Windows (PowerShell)

Windows has no `python3` command. If you type `python3`, Windows intercepts it with the Microsoft
Store stub and reports that Python was not found. Use `python` (or the launcher `py -3`) instead:

    python reruns/headroom_normalisation.py --configs N15_a,N15_b,N15_c
    python reruns/headroom_normalisation.py --configs N60_a,N60_b,N60_c

`run_all.py` is unaffected: it rewrites a leading `python3` in each manifest entry to the interpreter
that is actually running it, which is why `python run_all.py --all` works on Windows even though the
printed commands say `python3`.

Check the environment first:

    python --version
    python -c "import numpy, scipy; print(numpy.__version__, scipy.__version__)"

The pinned stack is `numpy==2.4.4`, `scipy==1.17.1` (see `requirements.txt`). Different builds may move
the last digits of solver-sensitive means; the conclusions asserted by `validate_results.py` are checked
in a stack-robust way.

## Optional dependencies

Only two scripts need the conic solver:

  * `reruns/qcqp_crosscheck_run.py`
  * `reruns/conic_recompute.py`

They require `cvxpy` (which brings CLARABEL). Everything else -- including
`reruns/headroom_normalisation.py`, `reruns/graph_decoupling.py`, `reruns/intersample_check.py`,
`reruns/objective_indexing_check.py`, `reruns/validate_paper_tables.py` and `validate_results.py` -- runs
without it. `reruns/qcqp_crosscheck.py` imports cvxpy lazily for exactly this reason.

`pyflakes` is required only by the source-hygiene test; without it two tests fail for an environment
reason, not an algorithmic one.

## Long runs

`reruns/headroom_normalisation.py` and `reruns/monte_carlo.py` write incrementally and re-read their
output on start, so they can be interrupted and resumed. Split the work by configuration, e.g. run the
N15 group first and the N60 group afterwards.
