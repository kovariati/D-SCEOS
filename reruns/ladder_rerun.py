"""Re-run the N=30 stress ladder with the physically-exact aggregate fix (uniform A_i=I)."""
import os as _os, sys as _sys
_ROOT=_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
[_sys.path.insert(0,_p) for _p in (_ROOT,_os.path.join(_ROOT,"code")) if _p not in _sys.path]
_os.chdir(_ROOT)  # results/ paths resolve against the package root
import json, numpy as np, time
from dsceos_validation import ClusterConfig, SimulationConfig, run_simulation
from dsceos_controller import DSCEOSConfig
from graph_config import LADDER_UNIT_MIX, ladder_cluster_kwargs  # single ladder source of truth
LADDER=[("soft",(0.18,0.02)),("feasible",(0.30,0.02)),("borderline-low",(0.42,0.02)),
        ("borderline-mid",(0.54,0.02)),("borderline-high",(0.70,0.05)),
        ("hard-stress",(0.75,0.20)),("extreme-infeasible",(0.85,0.35))]
def run(controller,target):
    cc=ClusterConfig(**LADDER_UNIT_MIX, **ladder_cluster_kwargs(),
        initial_spread=0.0,initial_speed_scale=0.0)
    ccfg=DSCEOSConfig(aggregate_tracking_weight=10.0,loss_weight_scale=0.03,sharing_weight=0.15,
        internal_weight=0.03,aggregate_consensus_gain=0.10,adaptive_consensus_gain=False)
    cfg=SimulationConfig(cluster=cc,controller=controller,scenario="static_request",
        dsceos_config=ccfg,horizon=12.0,dt=0.04,gateway_fraction=0.15,target_consensus_gain=0.18,
        target_gateway_gain=0.85,compute_reference_optimum=False,target_override=target,safety_filter=True,
        dpg_step_size=0.10,pd_kp=0.75,pd_kd=2.5)  # best-tested comparator tuning (N15/A sweep)
    return run_simulation(cfg)
t0=time.time(); rows=[]
print(f"{'regime':<20}{'D-SCEOS':>10}{'DPG-HOCBF':>11}{'PD':>10}   lowest?")
for name,tgt in LADDER:
    r={c:run(c,tgt).summary for c in ["dsceos","projected_gradient_hocbf","independent_tracking"]}
    d=r['dsceos']['integrated_objective_value']; g=r['projected_gradient_hocbf']['integrated_objective_value']; p=r['independent_tracking']['integrated_objective_value']
    low="YES" if d==min(d,g,p) else "NO"
    print(f"{name:<20}{d:>10.3f}{g:>11.3f}{p:>10.3f}   {low}")
    rows.append(dict(regime=name,dsceos=d,dpg=g,pd=p,
        yT_dsceos=float(r['dsceos'].get('final_aggregate_error',float('nan'))),
        yT_dpg=float(r['projected_gradient_hocbf'].get('final_aggregate_error',float('nan'))),
        yT_pd=float(r['independent_tracking'].get('final_aggregate_error',float('nan'))),
        cap_viol=r['dsceos']['max_capacity_violation'],
        qp_success=r['dsceos']['mean_local_qp_success_rate']))
print(f"\n(ladder re-run in {time.time()-t0:.0f}s)")
json.dump(rows,open("stress_ladder_summary.json","w"),indent=2)
