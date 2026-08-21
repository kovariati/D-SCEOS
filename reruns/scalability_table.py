"""Single-source generator for the N=60 scalability table of the manuscript.

Reads the released trajectories in results/ and the authoritative graph/objective configuration,
recomputes the centralized static box-constrained allocation reference p* of Eq. (pstar) under the
CURRENT objective definition, and emits both a machine-readable JSON and the LaTeX table body.
Having one generator prevents the manuscript table and the reproducibility package from drifting
apart (the source-integrity bundle had no such generator for this table).

Run from the package root:  python3 reruns/scalability_table.py
"""
import json
import os
import sys

import numpy as np
from scipy.optimize import minimize

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_ROOT, os.path.join(_ROOT, "code")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.chdir(_ROOT)
sys.dont_write_bytecode = True

from benchmark_objective import obj_and_grad                      # noqa: E402
from graph_config import authoritative_cluster_kwargs             # noqa: E402
from realistic_cpes_catalog import build_realistic_units          # noqa: E402
import dsceos_validation as dv                                    # noqa: E402
from dsceos_controller import DSCEOSConfig                        # noqa: E402

CFG = DSCEOSConfig(aggregate_tracking_weight=2.0 / 3.0, loss_weight_scale=0.03,
                   sharing_weight=0.15, internal_weight=0.03)
CTRL = [("dsceos", "D-SCEOS"), ("projected_gradient_hocbf", "DPG-HOCBF"),
        ("independent_tracking", "PD baseline")]

units, _ = build_realistic_units("realistic_60")
n = len(units)
ccfg = dv.ClusterConfig(n_thermal=3, n_storage=3, n_hydrogen=3, n_emobility=3, n_industrial=3,
                        initial_spread=0.0, initial_speed_scale=0.0,
                        **authoritative_cluster_kwargs())
W = dv.make_fixed_local_graph(dv.make_physical_layout(n, ccfg),
                              ccfg.communication_radius, ccfg.neighbour_count)
arrays = dv.arrays_from_units(units)
Ablk = dv.make_aggregate_blocks(arrays["aggregate_weight"])
lo, hi = arrays["lower"], arrays["upper"]

rows, tex = [], []
for sc in "abc":
    s0 = json.load(open(f"results/N60_{sc}_dsceos/summary.json"))
    target = np.array([s0["_final_target_internal_P"], s0["_final_target_internal_Q"]])
    res = minimize(lambda x: obj_and_grad(x, arrays, W, Ablk, target, CFG, arrays["rest"].shape),
                   arrays["rest"].reshape(-1), jac=True, method="L-BFGS-B",
                   bounds=list(zip(lo.reshape(-1), hi.reshape(-1))),
                   options=dict(maxiter=20000, ftol=1e-16, gtol=1e-14))
    pstar = res.x.reshape(arrays["rest"].shape)
    ystar = dv.aggregate_output(Ablk, pstar)
    bests = [json.load(open(f"results/N60_{sc}_{c}/summary.json"))["integrated_objective_value"]
             for c, _ in CTRL]
    for k, (c, label) in enumerate(CTRL):
        s = json.load(open(f"results/N60_{sc}_{c}/summary.json"))
        p = np.load(f"results/N60_{sc}_{c}/state_history.npz")["positions"][-1]
        y = dv.aggregate_output(Ablk, p)
        # the manuscript reports the PHYSICAL fleet-sum deviation, i.e. N * internal aggregate
        # the design rationale: report the two physical components SEPARATELY (GW and GVAR are
        # different physical quantities and must not be summed inside one Euclidean norm).
        dyT = (y - target) * n
        dys = (y - ystar) * n
        row = dict(scenario=sc.upper(), controller=label,
                   J_T=round(float(s["integrated_objective_value"]), 4),
                   err_yT_P_GW=round(float(dyT[0]), 4), err_yT_Q_GVAR=round(float(dyT[1]), 4),
                   err_ystar_P_GW=round(float(dys[0]), 4), err_ystar_Q_GVAR=round(float(dys[1]), 4),
                   err_yT=round(float(np.linalg.norm(dyT)), 4),
                   err_ystar=round(float(np.linalg.norm(dys)), 4),
                   alloc_rms=round(float(np.sqrt(((p - pstar) ** 2).mean())), 5),
                   energy=round(float(s["total_control_energy"]), 4))
        rows.append(row)
        jt = (f"\\textbf{{{row['J_T']:.3f}}}"
              if abs(row["J_T"] - min(round(b, 4) for b in bests)) < 1e-9 else f"{row['J_T']:.3f}")
        head = f"{sc.upper()} & " if k == 0 else "  & "
        tex.append(f"{head}{label} & {jt} & {row['err_yT_P_GW']:+.3f} & {row['err_yT_Q_GVAR']:+.3f}"
                   f" & {row['err_ystar_P_GW']:+.3f} & {row['err_ystar_Q_GVAR']:+.3f}"
                   f" & {row['energy']:.3f} \\\\")
    if sc != "c":
        tex.append("\\addlinespace[2pt]")

json.dump(rows, open("scalability_N60.json", "w"), indent=1)
print("\n".join(tex))
print("\n-> scalability_N60.json")
ds = [r for r in rows if r["controller"] == "D-SCEOS"]
print(f"D-SCEOS ||y-y*|| range: {min(r['err_ystar'] for r in ds):.3f}--{max(r['err_ystar'] for r in ds):.3f}")
print(f"D-SCEOS per-unit allocation RMS: " + ", ".join(f"{r['alloc_rms']:.5f}" for r in ds))
for lbl in ("DPG-HOCBF", "PD baseline"):
    rt = [next(r for r in rows if r["scenario"] == sc.upper() and r["controller"] == lbl)["J_T"]
          / next(r for r in rows if r["scenario"] == sc.upper() and r["controller"] == "D-SCEOS")["J_T"]
          for sc in "abc"]
    print(f"{lbl} / D-SCEOS ratio: {min(rt):.2f}--{max(rt):.2f}x")
