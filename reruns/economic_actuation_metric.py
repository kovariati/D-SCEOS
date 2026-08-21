"""Economic actuation diagnostic (C-04): recompute raw and cost-weighted actuation from
the bundled N=60 trajectories. Reproduces the economic-actuation numbers in
Section 7. Run from the package root: python3 reruns/economic_actuation_metric.py"""
import os as _os, sys as _sys
_ROOT=_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
[_sys.path.insert(0,_p) for _p in (_ROOT,_os.path.join(_ROOT,"code")) if _p not in _sys.path]
_os.chdir(_ROOT)
import json, numpy as np
from realistic_cpes_catalog import build_realistic_units
from dsceos_validation import arrays_from_units
units,_=build_realistic_units('realistic_60'); arr=arrays_from_units(units)
c=arr['loss_weight'][:,0]  # c_i/c_n ranking coefficients
CTRL=['dsceos','projected_gradient_hocbf','independent_tracking']
rows=[]
print(f"{'scen':>5}{'ctrl':>10}{'raw_||u||^2':>12}{'cost_wtd':>10}")
for sc,SC in [('a','A'),('b','B'),('c','C')]:
    for cc in CTRL:
        d=np.load(f'results/N60_{sc}_{cc}/state_history.npz'); u=d['controls']; dt=float(d['time'][1]-d['time'][0])
        Eu=float(np.sum(u*u)*dt); Euc=float(np.sum(dt*np.sum(u*u,axis=2)*c[None,:]))
        print(f"{SC:>5}{cc[:9]:>10}{Eu:>12.3f}{Euc:>10.3f}")
        rows.append(dict(scenario=SC,controller=cc,raw_actuation=Eu,cost_weighted=Euc))
json.dump(rows,open('reruns/economic_actuation_reference.json','w'),indent=2)
print("\nWrote reruns/economic_actuation_reference.json")
