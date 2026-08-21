#!/usr/bin/env python3
"""
Point 8: direction-dependent (headroom) capacity normalisation.

CORRECTNESS NOTE (read before comparing against any earlier output of this script)
---------------------------------------------------------------------------------
An earlier version of this script was defective in two ways and its results must be discarded:
  1. it selected the denominator from the sign of `rest`, which is identically zero in the released
     catalogues, so it always used the up-regulation headroom instead of the activation direction;
  2. it patched only the D-SCEOS gradient, so the comparators kept optimising the OLD objective while
     being scored under the NEW one, which inflated their objective by up to ~44x.
Both are fixed here: the denominator is selected by the sign of each agent's own activation at the
current state, and the objective, the D-SCEOS gradient and the DPG/DPD gradients all use it.

The released utilisation uses a symmetric denominator,

    rho_i = g_i^T p_i / max(|p_lower_i,P|, |p_upper_i,P|),

which does not reflect direction-dependent reserve when a unit is asymmetric: a unit with large
down-regulation but small up-regulation headroom looks lightly loaded under positive activation.
This script re-runs the study with a direction-aware denominator,

    rho_i = g_i^T p_i / ( p_upper_i,P  if g_i^T p_i >= 0 else |p_lower_i,P| ),

so that utilisation is measured against the headroom actually being consumed.

IMPORTANT: this changes the objective, so J_T under the two conventions is NOT directly comparable
across conventions. The script therefore reports, for each configuration and controller, the objective
evaluated under its OWN convention plus the two comparisons that remain meaningful:
  * the controller ordering within each convention,
  * the fairness spread of realised utilisation (how evenly headroom is actually consumed),
  * capacity violations.

Runtime: the full grid is 6 configurations x 3 controllers x 2 conventions = 36 simulations, of which
the N60 ones dominate. Expect well over an hour on a single core; use --configs to split the work.

Output: headroom_normalisation.json
"""
from __future__ import annotations
import os, sys, json
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "code"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import dsceos_validation as dv
from dsceos_validation import run_simulation
from qcqp_crosscheck_run import build_realistic_cfg

CTRLS = ["dsceos", "projected_gradient_hocbf", "independent_tracking"]
LABEL = {"dsceos": "D-SCEOS", "projected_gradient_hocbf": "DPG-HOCBF",
         "independent_tracking": "PD baseline"}


def directional_capacity_arrays(arrays, p):
    """Sign-selected headroom: the reserve actually being consumed by the current activation."""
    up = np.abs(arrays["upper"][:, 0])
    dn = np.abs(arrays["lower"][:, 0])
    return np.maximum(np.where(np.asarray(p)[:, 0] >= 0.0, up, dn), 1.0e-9)


def utilisation_spread(arrays, p):
    """Spread of realised sign-selected utilisation (lower = headroom shared more evenly)."""
    cap = directional_capacity_arrays(arrays, p)
    rho = np.asarray(p)[:, 0] / cap
    return float(np.max(rho) - np.min(rho))


def _install_directional(arrays_ref):
    """Patch every place the utilisation denominator is used so that it is selected by the SIGN of
    the agent's own activation at the current state, rather than by max(|lower|,|upper|).

    The denominator is frozen at the current sign within each evaluation, so the sharing gradient
    keeps its existing algebraic form; only the scalar denominator changes. Three sites are patched:
    the objective's utilisation, the controller's utilisation, and the controller's sharing gradient.
    """
    import dsceos_controller as dc

    up = np.abs(arrays_ref["upper"][:, 0])
    dn = np.abs(arrays_ref["lower"][:, 0])

    def cap_for(p):
        return np.maximum(np.where(np.asarray(p)[:, 0] >= 0.0, up, dn), 1.0e-9)

    import distributed_projected_gradient_controller as dpg
    import distributed_primal_dual_controller as dpd

    orig_util_val = dv.utilization
    orig_util_ctrl = dc.DistributedSCEOSController.utilization
    orig_grad = dc.DistributedSCEOSController.local_gradients
    orig_dpg = dpg.DistributedProjectedGradientController._gradient
    orig_dpd = dpd.DistributedPrimalDualController._primal_gradient

    def util_val(p, arrays):
        return np.asarray(p)[:, 0] / cap_for(p)

    def util_ctrl(self, p):
        return np.sum(self.problem.service_selector * p, axis=1) / cap_for(p)

    def grad_patched(self, p, *a, **k):
        pr = self.problem
        saved = pr.service_capacity
        try:
            object.__setattr__(pr, "service_capacity", cap_for(p))
            return orig_grad(self, p, *a, **k)
        finally:
            object.__setattr__(pr, "service_capacity", saved)

    def _wrap_comparator(orig_fn, state_attr):
        # The comparators evaluate their sharing gradient on their own iterate; freeze the
        # denominator at that iterate's sign so every controller optimises the SAME objective.
        def wrapped(self, *a, **k):
            pr = self.problem
            saved_cap = pr.service_capacity
            z = getattr(self, state_attr, None)
            try:
                if z is not None:
                    object.__setattr__(pr, "service_capacity", cap_for(z))
                return orig_fn(self, *a, **k)
            finally:
                object.__setattr__(pr, "service_capacity", saved_cap)
        return wrapped

    dv.utilization = util_val
    dc.DistributedSCEOSController.utilization = util_ctrl
    dc.DistributedSCEOSController.local_gradients = grad_patched
    dpg.DistributedProjectedGradientController._gradient = _wrap_comparator(orig_dpg, "z")
    dpd.DistributedPrimalDualController._primal_gradient = _wrap_comparator(orig_dpd, "z")
    return (orig_util_val, orig_util_ctrl, orig_grad, orig_dpg, orig_dpd)


def _restore_directional(saved):
    import dsceos_controller as dc
    import distributed_projected_gradient_controller as dpg
    import distributed_primal_dual_controller as dpd
    (dv.utilization, dc.DistributedSCEOSController.utilization,
     dc.DistributedSCEOSController.local_gradients,
     dpg.DistributedProjectedGradientController._gradient,
     dpd.DistributedPrimalDualController._primal_gradient) = saved


def run_one(cluster, sk, controller, directional):
    scfg = build_realistic_cfg(cluster, sk)
    scfg = scfg.__class__(**{**scfg.__dict__, "controller": controller})
    arrays = orig_arrays(cluster)
    saved = _install_directional(arrays) if directional else None
    try:
        res = run_simulation(scfg)
    finally:
        if saved is not None:
            _restore_directional(saved)
    s = res.summary
    return dict(J_T_own_convention=float(s["integrated_objective_value"]),
                final_aggregate_error=float(s["final_aggregate_error"]),
                max_capacity_violation=float(s["max_capacity_violation"]),
                final_utilisation_spread=utilisation_spread(arrays, np.asarray(res.positions[-1])))


_ARR_CACHE = {}


def orig_arrays(cluster):
    if cluster not in _ARR_CACHE:
        import realistic_cpes_catalog as cat
        from dsceos_validation import arrays_from_units
        units, _ = cat.build_realistic_units(cluster)
        _ARR_CACHE[cluster] = arrays_from_units(units)
    return _ARR_CACHE[cluster]


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", default="N15_a,N15_b,N15_c,N60_a,N60_b,N60_c")
    ap.add_argument("--controllers", default=",".join(CTRLS))
    args = ap.parse_args()

    cfg_map = {"N15_a": ("realistic_15", "a"), "N15_b": ("realistic_15", "b"),
               "N15_c": ("realistic_15", "c"), "N60_a": ("realistic_60", "a"),
               "N60_b": ("realistic_60", "b"), "N60_c": ("realistic_60", "c")}
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    path = os.path.join(root, "headroom_normalisation.json")
    out = {}
    if os.path.exists(path):
        try: out = json.load(open(path))
        except Exception: out = {}

    print(f"{'config':<8} {'controller':<12} {'conv':<12} {'J_T(own)':>11} {'util spread':>12} {'cap':>9}")
    for name in [c.strip() for c in args.configs.split(",")]:
        cl, sk = cfg_map[name]
        out.setdefault(name, {})
        for ctrl in [c.strip() for c in args.controllers.split(",")]:
            out[name].setdefault(LABEL[ctrl], {})
            for directional, tag in ((False, "symmetric"), (True, "directional")):
                r = run_one(cl, sk, ctrl, directional)
                out[name][LABEL[ctrl]][tag] = r
                print(f"{name:<8} {LABEL[ctrl]:<12} {tag:<12} {r['J_T_own_convention']:11.5f} "
                      f"{r['final_utilisation_spread']:12.5f} {r['max_capacity_violation']:9.1e}")
                json.dump(out, open(path, "w"), indent=2)
    print("saved headroom_normalisation.json")
    print("NOTE: J_T is comparable ACROSS CONTROLLERS within a convention, not across conventions.")


if __name__ == "__main__":
    main()
