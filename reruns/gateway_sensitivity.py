"""Gateway-node sensitivity. Vary (a) the gateway FRACTION (number of
gateways) and (b) gateway PLACEMENT (evenly spread vs clustered), and report
J_T, final aggregate error, estimator behaviour and capacity feasibility."""
import os as _os, sys as _sys
_ROOT=_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
[_sys.path.insert(0,_p) for _p in (_ROOT,_os.path.join(_ROOT,"code")) if _p not in _sys.path]
_os.chdir(_ROOT)  # results/ paths resolve against the package root
import json, numpy as np
import dsceos_validation as dv
from sweep_driver import run_config

# (a) gateway-fraction sweep on N15 Scenario A and B
print("### (a) gateway-fraction sweep (N15) ###")
print(f"{'scen':>5}{'gw_frac':>9}{'#gw':>5}{'J_T':>10}{'agg_err':>10}{'cap_viol':>10}")
rows=[]
for sk in ["A","B"]:
    for gf in [0.07,0.15,0.30,0.50,1.00]:
        r=run_config(sk,"realistic_15","dsceos",gateway_fraction=gf)
        ngw=max(1,int(np.ceil(gf*15)))
        print(f"{sk:>5}{gf:>9.2f}{ngw:>5}{r['J_T']:>10.4f}{r['agg_err']:>10.5f}{r['cap_viol']:>10.1e}")
        rows.append(dict(kind="fraction",scenario=sk,gw_fraction=gf,n_gateways=ngw,
                         J_T=r["J_T"],agg_err=r["agg_err"],cap_viol=r["cap_viol"]))

# (b) placement: evenly-spread (default linspace) vs clustered (first-k contiguous)
print("\n### (b) gateway PLACEMENT at fixed count (N15, Scen A, ceil(0.15*15)=3 gateways) ###")
orig_gwmask=dv.gateway_mask
def clustered_mask(n,fraction):
    k=max(1,int(np.ceil(max(0.0,min(1.0,fraction))*n)))
    mask=np.zeros(n,dtype=bool); mask[:k]=True   # contiguous cluster at graph corner
    return mask
for label,maskfn in [("spread(default)",orig_gwmask),("clustered",clustered_mask)]:
    dv.gateway_mask=maskfn
    r=run_config("A","realistic_15","dsceos",gateway_fraction=0.15)
    print(f"  {label:<16}: J_T={r['J_T']:.4f}  agg_err={r['agg_err']:.5f}  cap_viol={r['cap_viol']:.1e}")
    rows.append(dict(kind="placement",placement=label,J_T=r["J_T"],agg_err=r["agg_err"],cap_viol=r["cap_viol"]))
dv.gateway_mask=orig_gwmask
json.dump(rows,open("gateway_sensitivity.json","w"),indent=2)
