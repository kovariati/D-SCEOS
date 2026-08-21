"""Consensus-gain comparison: re-run D-SCEOS with the contraction-optimal consensus gain and
compare J_T to the Gershgorin baseline (both under the fixed aggregate)."""
import os as _os, sys as _sys
_ROOT=_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
[_sys.path.insert(0,_p) for _p in (_ROOT,_os.path.join(_ROOT,"code")) if _p not in _sys.path]
_os.chdir(_ROOT)  # results/ paths resolve against the package root
import json, numpy as np
from sweep_driver import run_config
OPT={"realistic_15":0.435908,"realistic_60":0.244689}
# Gershgorin baseline J_T from the fixed core re-run:
G=json.load(open("technology_representative_summary.json"))
gersh={(r["cluster"],r["scenario"]):r["J_T"] for r in G if r["controller"]=="dsceos"}
print(f"{'cfg':<8}{'J_T Gershgorin':>16}{'J_T opt-gain':>14}{'delta%':>9}{'cap_viol':>10}")
rows=[]
for cluster,tag in [("realistic_15","N15"),("realistic_60","N60")]:
    for sk in ["A","B","C"]:
        r=run_config(sk,cluster,"dsceos",adaptive_gain=False,fixed_consensus_gain=OPT[cluster])
        jg=gersh[(tag,sk)]; jo=r["J_T"]; d=100*(jo-jg)/jg
        print(f"{tag+'/'+sk:<8}{jg:>16.4f}{jo:>14.4f}{d:>+9.2f}{r['cap_viol']:>10.1e}")
        rows.append(dict(cfg=f"{tag}/{sk}",JT_gersh=jg,JT_opt=jo,delta_pct=d))
json.dump(rows,open("consensus_gain_comparison.json","w"),indent=2)
