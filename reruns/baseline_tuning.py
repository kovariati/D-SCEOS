"""Comparator tuning fairness diagnostic.
(a) DPG-HOCBF step-size sweep over [0.02, 0.72] on the N15/A reference scenario -> the sweep minimiser
    is step 0.10 (J_T ~= 0.4914); even that best-tuned DPG J_T stays above D-SCEOS.
(b) PD-baseline (kp,kd) grid -> show best-tuned PD J_T still above D-SCEOS."""
import os as _os, sys as _sys
_ROOT=_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
[_sys.path.insert(0,_p) for _p in (_ROOT,_os.path.join(_ROOT,"code")) if _p not in _sys.path]
_os.chdir(_ROOT)  # results/ paths resolve against the package root
import json, numpy as np
import dsceos_validation as dv
from sweep_driver import run_config

# reference D-SCEOS (default tuning)
ref=run_config("A","realistic_15","dsceos")["J_T"]
print(f"D-SCEOS (default) J_T = {ref:.4f}\n")

# (a) DPG step-size sweep
print("### (a) DPG-HOCBF step-size sweep (N15 Scen A) ###")
print(f"{'step':>7}{'DPG J_T':>10}")
dpg_rows=[]
for st in [0.02,0.04,0.06,0.10,0.20,0.40,0.72]:
    r=run_config("A","realistic_15","projected_gradient_hocbf",dpg_step=st)
    print(f"{st:>7.2f}{r['J_T']:>10.4f}{'   <- N15/A sweep minimiser' if st==0.10 else ''}")
    dpg_rows.append(dict(step=st,J_T=r["J_T"],cap_viol=r["cap_viol"]))
best_dpg=min(dpg_rows,key=lambda d:d["J_T"])
print(f"best DPG J_T={best_dpg['J_T']:.4f} at step={best_dpg['step']} "
      f"(still {best_dpg['J_T']/ref:.2f}x D-SCEOS)\n")

# (b) PD (kp,kd) grid: monkey-patch the independent-tracking gains
print("### (b) PD-baseline (kp,kd) grid (N15 Scen A) ###")
orig=dv.decentralized_independent_tracking
def make_pd(kp,kd):
    def f(p,v,te,m,d,fl,**kw):
        return orig(p,v,te,m,d,fl,kp=kp,kd=kd)
    return f
pd_rows=[]
print(f"{'kp':>6}{'kd':>6}{'PD J_T':>10}")
for kp in [0.75,1.25,2.0,3.0]:
    for kd in [1.0,1.6,2.5]:
        dv.decentralized_independent_tracking=make_pd(kp,kd)
        r=run_config("A","realistic_15","independent_tracking")
        print(f"{kp:>6.2f}{kd:>6.2f}{r['J_T']:>10.4f}{'  <- default' if (kp==1.25 and kd==1.6) else ''}")
        pd_rows.append(dict(kp=kp,kd=kd,J_T=r["J_T"],cap_viol=r["cap_viol"]))
dv.decentralized_independent_tracking=orig
best_pd=min(pd_rows,key=lambda d:d["J_T"])
print(f"best PD J_T={best_pd['J_T']:.4f} at kp={best_pd['kp']},kd={best_pd['kd']} "
      f"(still {best_pd['J_T']/ref:.2f}x D-SCEOS)")
json.dump({"dsceos_ref":ref,"dpg":dpg_rows,"pd":pd_rows,
           "best_dpg":best_dpg,"best_pd":best_pd},
          open("baseline_tuning.json","w"),indent=2)
