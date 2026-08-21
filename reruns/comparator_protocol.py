#!/usr/bin/env python3
"""
Equal-effort comparator protocol with a bracketed step-size search.

Motivation
----------
The published comparison gives the projected-gradient comparator ONE inner optimisation step per
sampling instant, with a step size selected once on the N15/A configuration. Two objections follow:
the comparator may simply be under-optimised, and its step size may be mistuned at other fleet sizes.
This script removes both objections by construction.

For every configuration and for each inner-effort level (1 substep = as published, 20 substeps =
inner problem driven to convergence) the step size is swept over a pre-specified grid and, if the
best point lands on a grid edge, the grid is EXTENDED in that direction until the optimum is
bracketed by strictly worse neighbours. The reported comparator value is then the best bracketed
point for that configuration and effort level -- i.e. the comparator is given its own best setting
per case, which is deliberately generous.

Reported per configuration:
  * D-SCEOS at the published setting,
  * comparator at its best bracketed 1-substep setting,
  * comparator at its best bracketed 20-substep (converged) setting,
  * the resulting gaps, and whether each optimum was bracketed.

The 20-substep variant uses ~20x the local optimisation work and the same number of communication
rounds, so it is NOT communication-matched in the strict sense; the extra effort is granted to the
comparator, again deliberately.

Output: comparator_protocol.json
"""
from __future__ import annotations
import os, sys, json, math, time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "code"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import dsceos_validation as dv
from dsceos_validation import run_simulation
from qcqp_crosscheck_run import build_realistic_cfg

BASE_GRID = [0.005, 0.01, 0.02, 0.04, 0.08, 0.16, 0.32]
MIN_STEP, MAX_STEP = 1e-4, 5.12
MAX_EXTENSIONS = 6

CFG = {"N15_a": ("realistic_15", "a"), "N15_b": ("realistic_15", "b"), "N15_c": ("realistic_15", "c"),
       "N60_a": ("realistic_60", "a"), "N60_b": ("realistic_60", "b"), "N60_c": ("realistic_60", "c")}


def run_dsceos(cl, sk):
    s = build_realistic_cfg(cl, sk)
    s = s.__class__(**{**s.__dict__, "controller": "dsceos"})
    r = run_simulation(s).summary
    return float(r["integrated_objective_value"]), float(r["max_capacity_violation"])


def run_dpg(cl, sk, step, substeps):
    orig = dv.DistributedProjectedGradientController

    class _Sub(orig):
        def __init__(self, problem, config=None, step_size=0.06, **k):
            super().__init__(problem, config, step_size=step, optimizer_substeps=substeps)

    dv.DistributedProjectedGradientController = _Sub
    try:
        s = build_realistic_cfg(cl, sk)
        s = s.__class__(**{**s.__dict__, "controller": "projected_gradient_hocbf"})
        r = run_simulation(s).summary
    finally:
        dv.DistributedProjectedGradientController = orig
    J = float(r["integrated_objective_value"])
    if not math.isfinite(J):
        J = float("inf")
    return J, float(r["max_capacity_violation"])


def bracketed_search(cl, sk, substeps, log=print):
    """Sweep the grid; extend it while the best point sits on an edge."""
    grid = list(BASE_GRID)
    vals = {}
    extensions = 0
    while True:
        for st in grid:
            if st in vals:
                continue
            J, cap = run_dpg(cl, sk, st, substeps)
            vals[st] = dict(J_T=J, cap=cap)
            log(f"      step={st:<8g} J_T={J:12.5f}")
        ordered = sorted(vals)
        best = min(ordered, key=lambda s: vals[s]["J_T"])
        lo, hi = ordered[0], ordered[-1]
        if best == lo and lo > MIN_STEP and extensions < MAX_EXTENSIONS:
            grid = [lo / 2.0]; extensions += 1; continue
        if best == hi and hi < MAX_STEP and extensions < MAX_EXTENSIONS:
            grid = [hi * 2.0]; extensions += 1; continue
        break
    ordered = sorted(vals)
    best = min(ordered, key=lambda s: vals[s]["J_T"])
    i = ordered.index(best)
    bracketed = (0 < i < len(ordered) - 1
                 and vals[ordered[i - 1]]["J_T"] > vals[best]["J_T"]
                 and vals[ordered[i + 1]]["J_T"] > vals[best]["J_T"])
    return dict(best_step=best, best_J_T=vals[best]["J_T"], best_cap=vals[best]["cap"],
                bracketed=bool(bracketed), n_evaluations=len(vals),
                grid={str(k): vals[k]["J_T"] for k in ordered})


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", default=",".join(CFG))
    ap.add_argument("--substeps", default="1,20")
    args = ap.parse_args()

    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    path = os.path.join(root, "comparator_protocol.json")
    out = {}
    if os.path.exists(path):
        try: out = json.load(open(path))
        except Exception: out = {}

    for name in [c.strip() for c in args.configs.split(",")]:
        cl, sk = CFG[name]
        out.setdefault(name, {})
        if "dsceos" not in out[name]:
            J, cap = run_dsceos(cl, sk)
            out[name]["dsceos"] = dict(J_T=J, cap=cap)
            print(f"[{name}] D-SCEOS J_T={J:.5f}", flush=True)
            json.dump(out, open(path, "w"), indent=2)
        for sub in [int(x) for x in args.substeps.split(",")]:
            key = f"dpg_substeps_{sub}"
            if key in out[name]:
                continue
            print(f"[{name}] comparator, {sub} inner substep(s):", flush=True)
            t0 = time.time()
            res = bracketed_search(cl, sk, sub, log=lambda m: print(m, flush=True))
            res["wall_s"] = round(time.time() - t0, 1)
            d = out[name]["dsceos"]["J_T"]
            res["gap_pct_dsceos_vs_best"] = 100.0 * (res["best_J_T"] - d) / max(abs(d), 1e-12)
            out[name][key] = res
            print(f"[{name}] {sub} substep(s): best step={res['best_step']} J_T={res['best_J_T']:.5f} "
                  f"bracketed={res['bracketed']} gap={res['gap_pct_dsceos_vs_best']:+.2f}% "
                  f"({res['wall_s']}s)", flush=True)
            json.dump(out, open(path, "w"), indent=2)
    print("saved comparator_protocol.json")


if __name__ == "__main__":
    main()
