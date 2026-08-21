"""One-at-a-time parameter sensitivity sweep on N15 / Scenario A.
For each parameter value, run all three controllers and check whether
D-SCEOS keeps the lowest integrated objective J_T."""
import os as _os, sys as _sys
_ROOT=_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
[_sys.path.insert(0,_p) for _p in (_ROOT,_os.path.join(_ROOT,"code")) if _p not in _sys.path]
_os.chdir(_ROOT)  # results/ paths resolve against the package root
import json, time
from sweep_driver import run_config

CL="realistic_15"; SK="A"
CTRLS=["dsceos","projected_gradient_hocbf","independent_tracking"]

def sweep(name, key, values, all_ctrls=True):
    print(f"\n### sweep {name}  (default marked *) ###")
    print(f"{name:>8} | {'D-SCEOS':>10} {'DPG-HOCBF':>10} {'PD':>10} | D-SCEOS lowest?")
    rows=[]
    # baselines that do not depend on this param can be cached
    for v in values:
        res={}
        for c in CTRLS:
            res[c]=run_config(SK,CL,c,**{key:v})["J_T"]
        lowest = (res["dsceos"] == min(res.values()))
        print(f"{v:>8.3f} | {res['dsceos']:>10.4f} {res['projected_gradient_hocbf']:>10.4f} "
              f"{res['independent_tracking']:>10.4f} | {'YES' if lowest else 'NO'}")
        rows.append(dict(param=name,value=v,**{k:res[k] for k in res},dsceos_lowest=bool(lowest)))
    return rows

t0=time.time(); allrows=[]
allrows+=sweep("w_bar","w_bar",[0.10,0.333,0.667,1.0,1.5,2.0])
allrows+=sweep("lam_s","lam_s",[0.01,0.05,0.15,0.30,0.50])
allrows+=sweep("lam_int","lam_int",[0.001,0.01,0.03,0.06,0.10])
allrows+=sweep("alpha0","alpha0",[0.5,1.0,2.0,4.0])
allrows+=sweep("alpha1","alpha1",[1.0,2.0,3.0,4.0])
allrows+=sweep("clf_rate","clf_rate",[0.4,0.8,1.2,2.0])
print(f"\nTotal sweep wall time: {time.time()-t0:.0f} s")
n_yes=sum(r["dsceos_lowest"] for r in allrows); n=len(allrows)
print(f"D-SCEOS lowest J_T in {n_yes}/{n} sweep points")
json.dump(allrows, open("parameter_sensitivity_N15A.json","w"), indent=2)
