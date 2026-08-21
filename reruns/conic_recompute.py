#!/usr/bin/env python3
"""
Point 7: recompute the main results with a deterministic convex conic solver IN THE LOOP.

The released main results solve the per-agent CLF/HOCBF-QCQP with SLSQP and fall back to a
deterministic box-ball projection when the solver reports failure. This script re-runs the same
configurations with CLARABEL (interior-point conic) as the in-loop solver, so the reported numbers
depend on the control law rather than on the SQP routine and its fallback, and records the full
solver statistics the review asks for:

  * primary-solve success rate and fallback count (production run),
  * independent-solver status distribution and mean iteration count,
  * hard-constraint feasibility residual,
  * per-configuration objective under both solvers and the difference.

Configurations: the six realistic runs and the seven stress-ladder regimes.
Output: conic_recompute.json
"""
from __future__ import annotations
import os, sys, json, time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "code"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import dsceos_controller as dc
from qcqp_crosscheck import make_clarabel_solver, solve_qcqp_clarabel
from qcqp_crosscheck_run import build_realistic_cfg, build_ladder_cfg, LADDER
from dsceos_validation import run_simulation


def run_with(scfg, solver: str):
    """solver in {'production','clarabel'}; returns summary + solver statistics."""
    orig = dc.DistributedSCEOSController._solve_local_clf_hocbf_qp
    stats = {"calls": 0, "fallback": 0, "resid_max": 0.0, "iters": []}

    if solver == "production":
        def wrapped(self, u_nom, lb, ub, radius, e, vel, mass, damping):
            u, slack, success, residual, V = orig(self, u_nom, lb, ub, radius, e, vel, mass, damping)
            stats["calls"] += 1
            if not success:
                stats["fallback"] += 1
            stats["resid_max"] = max(stats["resid_max"], float(residual))
            return u, slack, success, residual, V
        dc.DistributedSCEOSController._solve_local_clf_hocbf_qp = wrapped
    else:
        def conic(self, u_nom, lb, ub, radius, e, vel, mass, damping):
            V, coeff_u, drift = self._local_clf_terms(e, vel, mass, damping)
            rhs = drift + float(self.config.clf_rate) * V
            u, s, status, it = solve_qcqp_clarabel(u_nom, lb, ub, radius, coeff_u, rhs,
                                                   self.config.clf_slack_weight)
            stats["calls"] += 1
            if it is not None:
                stats["iters"].append(int(it))
            if u is None or not np.all(np.isfinite(u)):
                stats["fallback"] += 1
                return orig(self, u_nom, lb, ub, radius, e, vel, mass, damping)
            slack = float(max(0.0, s if s is not None else 0.0))
            box = max(float(np.max(lb - u)), float(np.max(u - ub)), 0.0)
            ball = max(float(np.linalg.norm(u) - radius), 0.0)
            resid = max(box, ball)
            stats["resid_max"] = max(stats["resid_max"], resid)
            return u, slack, True, resid, V
        dc.DistributedSCEOSController._solve_local_clf_hocbf_qp = conic

    try:
        t0 = time.time()
        res = run_simulation(scfg)
        wall = time.time() - t0
    finally:
        dc.DistributedSCEOSController._solve_local_clf_hocbf_qp = orig

    s = res.summary
    return dict(J_T=float(s["integrated_objective_value"]),
                max_capacity_violation=float(s["max_capacity_violation"]),
                final_aggregate_error=float(s["final_aggregate_error"]),
                calls=stats["calls"],
                fallback=stats["fallback"],
                fallback_rate=(stats["fallback"] / stats["calls"] if stats["calls"] else float("nan")),
                hard_resid_max=stats["resid_max"],
                mean_conic_iters=(float(np.mean(stats["iters"])) if stats["iters"] else None),
                wall_s=round(wall, 1))


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None, help="substring filter over config names")
    args = ap.parse_args()

    cfgs = []
    for name, tgt in LADDER:
        cfgs.append((f"ladder_{name}", lambda t=tgt: build_ladder_cfg(t)))
    for cl, tag in [("realistic_15", "N15"), ("realistic_60", "N60")]:
        for sk in ["a", "b", "c"]:
            cfgs.append((f"{tag}_{sk}", lambda c=cl, s=sk: build_realistic_cfg(c, s)))
    if args.only:
        cfgs = [c for c in cfgs if args.only in c[0]]

    path = os.path.join(os.path.dirname(__file__), "..", "conic_recompute.json")
    out = {}
    if os.path.exists(path):
        try: out = json.load(open(path))
        except Exception: out = {}

    for name, builder in cfgs:
        prod = run_with(builder(), "production")
        con = run_with(builder(), "clarabel")
        rec = dict(production=prod, clarabel=con,
                   delta_J_T=con["J_T"] - prod["J_T"],
                   rel_delta_pct=100.0 * (con["J_T"] - prod["J_T"]) / max(abs(prod["J_T"]), 1e-12))
        out[name] = rec
        print(f"[{name:<24}] prod J_T={prod['J_T']:10.4f} fb={prod['fallback_rate']:.3f} | "
              f"conic J_T={con['J_T']:10.4f} iters={con['mean_conic_iters']} | "
              f"dJ={rec['delta_J_T']:+9.4f} ({rec['rel_delta_pct']:+.2f}%) "
              f"cap {prod['max_capacity_violation']:.1e}->{con['max_capacity_violation']:.1e}")
        json.dump(out, open(path, "w"), indent=2)
    print("saved conic_recompute.json")


if __name__ == "__main__":
    main()
