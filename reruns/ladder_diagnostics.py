"""C-01/C-02: authoritative stress-ladder SAFETY diagnostics using the EXACT box-ball intersection
test, on the same configuration as ladder_rerun.py. For every agent-and-time step it records the
EXACT emptiness of the HOCBF box intersected with the actuator ball:
    intersection non-empty  <=>  box_consistent (lb <= ub, all coords)  AND
                                 ||clip(0, lb, ub)||_2 <= F_i,           margin = F_i - ||clip(0,lb,ub)||_2.
This is computed by wrapping the controller's own per-agent QP method
DistributedSCEOSController._solve_local_clf_hocbf_qp (no controller-core edit), so the recorder sees
the actual per-step box bounds lb, ub and actuator limit F_i AND the FINAL applied force (the SLSQP
solution actually commanded), not the projection initialization. Per regime x controller it reports:
exact empty_intersection_steps (time steps with any empty agent), empty_agent_steps,
min_intersection_margin, max exact box-ball residual of the FINAL applied force, max CLF slack,
fallback time steps, and max capacity violation. The recorder instruments only the D-SCEOS local QP,
so the comparator rows carry null safety fields with safety_diagnostics_measured=false.
Writes ladder_diagnostics.json at the package root."""
import os as _os, sys as _sys
_ROOT=_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
[_sys.path.insert(0,_p) for _p in (_ROOT,_os.path.join(_ROOT,"code")) if _p not in _sys.path]
_os.chdir(_ROOT)
import json, time, numpy as np
from dsceos_validation import ClusterConfig, SimulationConfig, run_simulation
from dsceos_controller import DSCEOSConfig

LADDER=[("soft",(0.18,0.02)),("feasible",(0.30,0.02)),("borderline-low",(0.42,0.02)),
        ("borderline-mid",(0.54,0.02)),("borderline-high",(0.70,0.05)),
        ("hard-stress",(0.75,0.20)),("extreme-infeasible",(0.85,0.35))]

from dsceos_controller import DistributedSCEOSController
from graph_config import LADDER_UNIT_MIX, ladder_cluster_kwargs  # single ladder source of truth
_orig_qp=DistributedSCEOSController._solve_local_clf_hocbf_qp
class _Rec:
    def reset(self,n_agents):
        self.n_agents=int(n_agents); self.empty_agent_steps=0; self.min_margin=np.inf
        self.max_final_res=0.0; self.n_calls=0; self.empty_step_idx=set()
        self.max_box_res=0.0; self.max_ball_res=0.0
    @property
    def empty_time_steps(self): return len(self.empty_step_idx)
_rec=_Rec(); _rec.reset(30)
def _patched_qp(self,u_nom,lb,ub,radius,e,vel,mass,damping):
    u,slack,ok,residual,V=_orig_qp(self,u_nom,lb,ub,radius,e,vel,mass,damping)  # u = FINAL applied force
    lb=np.asarray(lb,dtype=float); ub=np.asarray(ub,dtype=float); F=float(radius); u=np.asarray(u,dtype=float)
    # EXACT box-ball intersection test on the box the controller actually uses
    box_ok=bool(np.all(lb<=ub)); nearest=float(np.linalg.norm(np.clip(0.0,lb,ub))); margin=F-nearest
    step_idx=_rec.n_calls//max(1,_rec.n_agents)
    if (not box_ok) or (margin<0.0):
        _rec.empty_agent_steps+=1; _rec.empty_step_idx.add(step_idx)
    if box_ok: _rec.min_margin=min(_rec.min_margin,margin)
    # EXACT box-ball residual of the FINAL applied force u (NOT the QCQP initialization u0)
    # Task 3: box (operating-envelope) and ball (actuator) residuals are recorded SEPARATELY, so the
    # published JSON can report each independently rather than only their maximum.
    box_res=float(np.max(np.maximum(np.maximum(lb-u,u-ub),0.0)))
    ball_res=float(max(np.linalg.norm(u)-F,0.0))
    _rec.max_box_res=max(_rec.max_box_res,box_res)
    _rec.max_ball_res=max(_rec.max_ball_res,ball_res)
    _rec.max_final_res=max(_rec.max_final_res,box_res,ball_res); _rec.n_calls+=1
    return u,slack,ok,residual,V

def run(controller,target):
    cc=ClusterConfig(**LADDER_UNIT_MIX, **ladder_cluster_kwargs(),
        initial_spread=0.0,initial_speed_scale=0.0)
    ccfg=DSCEOSConfig(aggregate_tracking_weight=10.0,loss_weight_scale=0.03,sharing_weight=0.15,
        internal_weight=0.03,aggregate_consensus_gain=0.10,adaptive_consensus_gain=False)
    cfg=SimulationConfig(cluster=cc,controller=controller,scenario="static_request",
        dsceos_config=ccfg,horizon=12.0,dt=0.04,gateway_fraction=0.15,target_consensus_gain=0.18,
        target_gateway_gain=0.85,compute_reference_optimum=False,target_override=np.array(target),safety_filter=True,
        dpg_step_size=0.10,pd_kp=0.75,pd_kd=2.5)  # best-tested comparator tuning
    return run_simulation(cfg)

rows=[]; t0=time.time()
DistributedSCEOSController._solve_local_clf_hocbf_qp=_patched_qp  # install exact-test + FINAL-force recorder
print(f"{'regime':<20}{'ctrl':<6}{'emp_ag':>7}{'emp_t':>7}{'min_mg':>9}{'fin_res':>11}{'max_slk':>10}{'N_fb':>6}{'max_cv':>11}")
for regime,tgt in LADDER:
    for ctrl in ["dsceos","projected_gradient_hocbf","independent_tracking"]:
        _rec.reset(30); r=run(ctrl,tgt)
        marr=getattr(r,"metric_arrays",None) or getattr(r,"metrics",None)
        def arr(k):
            a=marr.get(k) if isinstance(marr,dict) else None
            return np.asarray(a,dtype=float) if a is not None else np.array([np.nan])
        slack=arr("clf_slack"); qp=arr("local_qp_success_rate"); cv=arr("capacity_violation")
        max_slack=float(np.nanmax(slack)) if not np.all(np.isnan(slack)) else float("nan")
        n_fallback=int(np.sum(qp<0.999)) if not np.all(np.isnan(qp)) else -1
        max_cv=float(np.nanmax(cv)); mm=_rec.min_margin
        lab={'dsceos':'DS','projected_gradient_hocbf':'DPG','independent_tracking':'PD'}[ctrl]
        mmp=(mm if np.isfinite(mm) else float("nan"))
        print(f"{regime:<20}{lab:<6}{_rec.empty_agent_steps:>7}{_rec.empty_time_steps:>7}{mmp:>9.3f}{_rec.max_final_res:>11.2e}{max_slack:>10.3f}{n_fallback:>6}{max_cv:>11.2e}")
        measured=(ctrl=="dsceos")  # the exact-test recorder only instruments the D-SCEOS local QP
        rows.append(dict(regime=regime,controller=ctrl,
            safety_diagnostics_measured=measured,
            empty_intersection_steps=(_rec.empty_time_steps if measured else None),
            empty_agent_steps=(_rec.empty_agent_steps if measured else None),
            min_intersection_margin=(round(float(mm),6) if (measured and np.isfinite(mm)) else None),
            max_final_force_residual=(round(_rec.max_final_res,12) if measured else None),
            max_hard_residual=(round(_rec.max_final_res,12) if measured else None),
            max_box_residual=(round(_rec.max_box_res,12) if measured else None),
            max_ball_residual=(round(_rec.max_ball_res,12) if measured else None),
            max_clf_slack=(round(max_slack,4) if (measured and max_slack==max_slack) else None),
            fallback_time_steps=(n_fallback if measured else None),
            max_capacity_violation=round(max_cv,10)))
DistributedSCEOSController._solve_local_clf_hocbf_qp=_orig_qp
json.dump(rows,open("ladder_diagnostics.json","w"),indent=2)
print(f"\n(ladder diagnostics in {time.time()-t0:.0f}s) -> ladder_diagnostics.json")
