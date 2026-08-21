#!/usr/bin/env python3
"""
Closed-loop sampling-time sensitivity (discretisation convergence).

The dense intersample replay checks the plant trajectory BETWEEN already-computed held inputs. It
does not check whether the control decisions themselves are converged in the sampling time. This
script re-runs the FULL closed loop at dt, dt/2 and dt/4 -- so the controller, the estimators, the
QCQP and the safety layer all execute at the finer rate -- and reports how the integrated objective,
the terminal aggregate error and the controller ordering move.

The objective is compared on a common basis: J_T is a time integral, so the values are directly
comparable across sampling rates.

Output: sampling_time_study.json
"""
from __future__ import annotations
import os, sys, json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "code"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dsceos_validation import run_simulation
from qcqp_crosscheck_run import build_realistic_cfg

CFG = {"N15_a": ("realistic_15", "a"), "N15_b": ("realistic_15", "b"), "N15_c": ("realistic_15", "c"),
       "N60_a": ("realistic_60", "a"), "N60_b": ("realistic_60", "b"), "N60_c": ("realistic_60", "c")}
CTRLS = [("dsceos", "D-SCEOS"), ("projected_gradient_hocbf", "DPG-HOCBF"),
         ("independent_tracking", "PD baseline")]


def run(cl, sk, ctrl, dt):
    s = build_realistic_cfg(cl, sk)
    s = s.__class__(**{**s.__dict__, "controller": ctrl, "dt": dt})
    r = run_simulation(s).summary
    return dict(J_T=float(r["integrated_objective_value"]),
                final_aggregate_error=float(r["final_aggregate_error"]),
                max_capacity_violation=float(r["max_capacity_violation"]))


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", default="N15_a,N15_b,N15_c")
    ap.add_argument("--levels", default="1,2,4", help="dt divisors")
    args = ap.parse_args()

    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    path = os.path.join(root, "sampling_time_study.json")
    out = {}
    if os.path.exists(path):
        try: out = json.load(open(path))
        except Exception: out = {}

    divisors = [int(x) for x in args.levels.split(",")]
    print(f"{'config':<8} {'controller':<12} {'dt':>8} {'J_T':>11} {'rel vs dt':>10} {'cap':>9}")
    for name in [c.strip() for c in args.configs.split(",")]:
        cl, sk = CFG[name]
        out.setdefault(name, {})
        for ctrl, label in CTRLS:
            out[name].setdefault(label, {})
            base = None
            for dv_ in divisors:
                dt = 0.05 / dv_
                key = f"dt_over_{dv_}"
                if key in out[name][label]:
                    if dv_ == 1:
                        base = out[name][label][key]["J_T"]
                    continue
                r = run(cl, sk, ctrl, dt)
                if dv_ == 1:
                    base = r["J_T"]
                r["rel_vs_dt_pct"] = (100.0 * (r["J_T"] - base) / max(abs(base), 1e-12)) if base else 0.0
                out[name][label][key] = r
                print(f"{name:<8} {label:<12} {dt:8.4f} {r['J_T']:11.5f} "
                      f"{r['rel_vs_dt_pct']:+10.3f} {r['max_capacity_violation']:9.1e}", flush=True)
                json.dump(out, open(path, "w"), indent=2)

    # ordering preserved at every level?
    ok = True
    for cfg, per in out.items():
        for lvl in {k for lab in per for k in per[lab]}:
            vals = {lab: per[lab][lvl]["J_T"] for lab in per if lvl in per[lab]}
            if len(vals) == 3 and not (vals["D-SCEOS"] < vals["DPG-HOCBF"] < vals["PD baseline"]):
                ok = False
    print(f"\nordering D-SCEOS < DPG < PD preserved at every sampling rate: {ok}")
    print("saved sampling_time_study.json")


if __name__ == "__main__":
    main()
