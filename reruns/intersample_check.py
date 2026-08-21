#!/usr/bin/env python3
"""
Point 3: intersample constraint check on a dense internal grid.

The released runs enforce and record the operating-envelope constraints at the sampling instants
only, while the applied force is held constant over each interval. Zero sampled-point violation
therefore does not by itself establish that the continuous plant stays inside the envelope between
samples. This script measures that directly.

For every stored step k it takes the sampled state (p_k, v_k) and the force u_k that was actually
applied, and integrates the continuous unit dynamics

    m_i \\dot v_i = u_i - mu_i v_i ,      \\dot p_i = v_i

over [t_k, t_{k+1}] on a dense sub-grid (default 200 sub-steps per sampling interval), recording the
worst envelope violation anywhere on that sub-grid. It re-uses the recorded controls, so no control
decision is recomputed and no simulation is re-run.

Reported per run:
  * max sampled-point violation (as published),
  * max intersample violation on the dense grid,
  * the sub-interval position of the worst case,
  * a sampling-time sensitivity sweep over sub-grid resolution.

Output: intersample_check.json
"""
from __future__ import annotations
import os, sys, json
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "code"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dsceos_validation import arrays_from_units
import realistic_cpes_catalog as cat

CTRL = {"dsceos": "D-SCEOS", "projected_gradient_hocbf": "DPG-HOCBF",
        "independent_tracking": "PD baseline"}


def worst_intersample(pos, vel, ctrl, time, masses, dampings, lower, upper, sub):
    """Integrate the continuous plant with held inputs on a dense sub-grid; return worst violation."""
    dt = float(time[1] - time[0])
    h = dt / sub
    worst = 0.0
    worst_frac = 0.0
    m = masses[:, None]
    mu = dampings[:, None]
    for k in range(ctrl.shape[0]):
        p = pos[k].copy()
        v = vel[k].copy()
        u = ctrl[k]
        for j in range(sub):
            # explicit sub-stepping of the continuous ODE with the held input
            v = v + h * (u - mu * v) / m
            p = p + h * v
            viol = float(max(np.max(lower - p), np.max(p - upper), 0.0))
            if viol > worst:
                worst = viol
                worst_frac = (j + 1) / sub
    return worst, worst_frac


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--sub", type=int, default=200, help="sub-steps per sampling interval")
    ap.add_argument("--sweep", default="10,50,200", help="resolutions for the sensitivity sweep")
    args = ap.parse_args()

    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    out = {}
    print(f"{'config':<10} {'controller':<12} {'sampled':>11} {'intersample':>13} {'at frac':>8}")
    for cluster, tag in [("realistic_15", "N15"), ("realistic_60", "N60")]:
        units, _ = cat.build_realistic_units(cluster)
        a = arrays_from_units(units)
        for sk in ["a", "b", "c"]:
            for ctrl_key, label in CTRL.items():
                npz = os.path.join(root, "results", f"{tag}_{sk}_{ctrl_key}", "state_history.npz")
                if not os.path.exists(npz):
                    continue
                d = np.load(npz)
                pos, vel, ctl, time = d["positions"], d["velocities"], d["controls"], d["time"]
                sampled = float(max(np.max(a["lower"] - pos), np.max(pos - a["upper"]), 0.0))
                inter, frac = worst_intersample(pos, vel, ctl, time, a["masses"], a["dampings"],
                                                a["lower"], a["upper"], args.sub)
                key = f"{tag}_{sk}/{label}"
                out[key] = dict(max_sampled_point_violation=sampled,
                                max_intersample_violation=inter,
                                worst_at_interval_fraction=frac,
                                sub_steps=args.sub)
                print(f"{tag+'_'+sk:<10} {label:<12} {sampled:11.3e} {inter:13.3e} {frac:8.2f}")

    # resolution sensitivity on the single worst case
    worst_key = max(out, key=lambda k: out[k]["max_intersample_violation"]) if out else None
    sweep = {}
    if worst_key:
        tag_sk, label = worst_key.split("/")
        tag, sk = tag_sk.split("_")
        ctrl_key = [k for k, v in CTRL.items() if v == label][0]
        cluster = "realistic_15" if tag == "N15" else "realistic_60"
        units, _ = cat.build_realistic_units(cluster)
        a = arrays_from_units(units)
        d = np.load(os.path.join(root, "results", f"{tag}_{sk}_{ctrl_key}", "state_history.npz"))
        for s in [int(x) for x in args.sweep.split(",")]:
            v, _f = worst_intersample(d["positions"], d["velocities"], d["controls"], d["time"],
                                      a["masses"], a["dampings"], a["lower"], a["upper"], s)
            sweep[str(s)] = v
        print(f"\nresolution sweep on worst case ({worst_key}): {sweep}")

    summary = dict(method=("continuous plant re-integrated with the recorded held inputs on a dense "
                           "sub-grid; controls are replayed, not recomputed"),
                   sub_steps=args.sub, resolution_sweep_worst_case=sweep,
                   worst_case_run=worst_key,
                   max_intersample_violation_overall=(max(v["max_intersample_violation"] for v in out.values())
                                                      if out else None),
                   per_run=out)
    json.dump(summary, open(os.path.join(root, "intersample_check.json"), "w"), indent=2)
    print("saved intersample_check.json")


if __name__ == "__main__":
    main()
