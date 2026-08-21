"""acceptability_matrix.py
================
Reproducible R1--R4 diagnostic matrix for the realistic study, computed from the released
trajectories in results/ and the authoritative graph/objective configuration.

The four study-specific thresholds (Section 7):
    R1 (delivery):               ||y(T) - y_T(T)||   <= 5% ||y_T(T)||
    R2 (aggregate optimality):   ||y(T) - y*||       <= 5% ||y_T(T)||
    R3 (objective):              J_T                 <= 2x the best J_T in that config
    R4 (actuation):              integrated actuation<= 3x the best in that config

y* is the centralized benchmark optimum of the SAME full CPES objective, recomputed here per
scenario and fleet size (L-BFGS-B on the box). All distances use the PHYSICAL fleet-sum aggregate
(N * internal aggregate), consistent with the scalability table.

This regenerates acceptability_matrix.json, which validate_results.py asserts. Comparator trajectories are the
source-integrity best-tested tuning (DPG step 0.10, PD kp=0.75/kd=2.5); D-SCEOS is tuning-independent.
"""
import json
import os
import sys

import numpy as np
from scipy.optimize import minimize

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_ROOT, os.path.join(_ROOT, "code")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from benchmark_objective import obj_and_grad                      # noqa: E402
from graph_config import (authoritative_cluster_kwargs,             # noqa: E402
                          AUTHORITATIVE_COMMUNICATION_RADIUS,
                          AUTHORITATIVE_NEIGHBOUR_COUNT)
from realistic_cpes_catalog import build_realistic_units          # noqa: E402
import dsceos_validation as dv                                    # noqa: E402
from dsceos_controller import DSCEOSConfig                        # noqa: E402

# Same objective weights as the scalability benchmark (size-consistent form).
CFG = DSCEOSConfig(aggregate_tracking_weight=2.0 / 3.0, loss_weight_scale=0.03,
                   sharing_weight=0.15, internal_weight=0.03)
CTRL = [("dsceos", "DS"), ("projected_gradient_hocbf", "DPG"), ("independent_tracking", "PD")]

# results/ directory naming: N15 uses lower-case scenario letters, N60 upper-case.
SIZES = [("N15", "realistic_15", "abc"), ("N60", "realistic_60", "abc")]


def cluster_for(cluster_name, n):
    n5 = n // 5
    ccfg = dv.ClusterConfig(n_thermal=n5, n_storage=n5, n_hydrogen=n5, n_emobility=n5,
                            n_industrial=n5, initial_spread=0.0, initial_speed_scale=0.0,
                            **authoritative_cluster_kwargs())
    return ccfg


def result_dir(tag, sc):
    # both N15 and N60 result dirs use lower-case scenario letters (matches batch_realistic)
    return f"results/{tag}_{sc.lower()}"


rows = []
for tag, cluster_name, scens in SIZES:
    units, _ = build_realistic_units(cluster_name)
    n = len(units)
    ccfg = cluster_for(cluster_name, n)
    W = dv.make_fixed_local_graph(dv.make_physical_layout(n, ccfg),
                                  AUTHORITATIVE_COMMUNICATION_RADIUS, AUTHORITATIVE_NEIGHBOUR_COUNT)
    arrays = dv.arrays_from_units(units)
    Ablk = dv.make_aggregate_blocks(arrays["aggregate_weight"])
    lo, hi = arrays["lower"], arrays["upper"]
    for sc in scens:
        rd = result_dir(tag, sc)
        s0 = json.load(open(f"{rd}_dsceos/summary.json"))
        target = np.array([s0["_final_target_internal_P"], s0["_final_target_internal_Q"]])
        yT_phys = np.linalg.norm(target * n)          # physical ||y_T||
        res = minimize(lambda x: obj_and_grad(x, arrays, W, Ablk, target, CFG, arrays["rest"].shape),
                       arrays["rest"].reshape(-1), jac=True, method="L-BFGS-B",
                       bounds=list(zip(lo.reshape(-1), hi.reshape(-1))),
                       options=dict(maxiter=20000, ftol=1e-16, gtol=1e-14))
        pstar = res.x.reshape(arrays["rest"].shape)
        ystar = dv.aggregate_output(Ablk, pstar)
        # gather per-controller metrics
        JT, ENER, dyT, dys = {}, {}, {}, {}
        for c, lab in CTRL:
            s = json.load(open(f"{rd}_{c}/summary.json"))
            p = np.load(f"{rd}_{c}/state_history.npz")["positions"][-1]
            y = dv.aggregate_output(Ablk, p)
            JT[lab] = float(s["integrated_objective_value"])
            ENER[lab] = float(s["total_control_energy"])
            dyT[lab] = float(np.linalg.norm((y - target) * n))
            dys[lab] = float(np.linalg.norm((y - ystar) * n))
        best_JT = min(JT.values())
        best_E = min(ENER.values())
        r1 = [lab for _, lab in CTRL if dyT[lab] <= 0.05 * yT_phys]
        r2 = [lab for _, lab in CTRL if dys[lab] <= 0.05 * yT_phys]
        r3 = [lab for _, lab in CTRL if JT[lab] <= 2.0 * best_JT + 1e-12]
        r4 = [lab for _, lab in CTRL if ENER[lab] <= 3.0 * best_E + 1e-12]
        rows.append(dict(config=f"{tag}-{sc.upper()}",
                         yT_phys=round(yT_phys, 6),
                         J_T={lab: round(JT[lab], 4) for _, lab in CTRL},
                         err_yT={lab: round(dyT[lab], 6) for _, lab in CTRL},
                         err_ystar={lab: round(dys[lab], 6) for _, lab in CTRL},
                         energy={lab: round(ENER[lab], 4) for _, lab in CTRL},
                         R1=r1, R2=r2, R3=r3, R4=r4))
        print(f"{tag}-{sc.upper()}: R1={r1} R2={r2} R3={r3} R4={r4}")

json.dump(rows, open("acceptability_matrix.json", "w"), indent=1)
print("\n-> acceptability_matrix.json")
# LaTeX table body
print("\n% LaTeX matrix body:")
for r in rows:
    def cell(x):
        return ",".join(x) if x else "---"
    print(f"{r['config']} & {cell(r['R1'])} & {cell(r['R2'])} & {cell(r['R3'])} & {cell(r['R4'])} \\\\")
