"""M-04 (source-integrity C-01 fix): package the REAL realistic N15 Scenario-A control-modification
(||u - u_nom||) diagnostic quoted in the paper. This reproduces the released realistic runner path
EXACTLY -- build_realistic_units + compute_fleet_scaling + build_signal_for, with dsceos_validation's
make_units / select_target monkey-patched to inject the realistic units and the Scenario-A signal
(this is what run_realistic_scenario.py does) -- NOT the generic static_request harness. It
instruments the D-SCEOS local QP to record ||u - u_nom|| at every agent-and-time step and writes
control_displacement_N15A.json (scenario, fleet, J_T, n_agent_steps, max, mean, p95, fraction>1e-3)."""
import os as _os, sys as _sys
_ROOT=_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
[_sys.path.insert(0,_p) for _p in (_ROOT,_os.path.join(_ROOT,"code")) if _p not in _sys.path]
_os.chdir(_ROOT)
import json, numpy as np
import dsceos_validation as dv
from dsceos_validation import ClusterConfig, SimulationConfig, run_simulation
from dsceos_controller import DSCEOSConfig, DistributedSCEOSController
from realistic_cpes_catalog import build_realistic_units, compute_fleet_scaling
from realistic_scenarios import REALISTIC_SCENARIOS, build_signal_for
from graph_config import (AUTHORITATIVE_COMMUNICATION_RADIUS as _R045,
                          AUTHORITATIVE_NEIGHBOUR_COUNT as _K4,
                          AUTHORITATIVE_LAYOUT_SPREAD as _LS,
                          AUTHORITATIVE_SEED as _SEED)  # single authoritative graph config

_orig_qp=DistributedSCEOSController._solve_local_clf_hocbf_qp
_disp=[]
def _patched_qp(self,u_nom,lb,ub,radius,e,vel,mass,damping):
    u,slack,ok,residual,V=_orig_qp(self,u_nom,lb,ub,radius,e,vel,mass,damping)
    _disp.append(float(np.linalg.norm(np.asarray(u,dtype=float)-np.asarray(u_nom,dtype=float))))
    return u,slack,ok,residual,V

def main():
    FLEET="realistic_15"; SCEN="scenario_a_winter_morning_step"
    scenario=REALISTIC_SCENARIOS[SCEN]
    units,physical=build_realistic_units(FLEET)          # real catalogue units
    scaling=compute_fleet_scaling(physical)              # DSO <-> internal scaling
    signal=build_signal_for(scenario,scaling)            # Scenario-A signal
    # inject exactly as run_realistic_scenario.py does
    orig_mu, orig_st = dv.make_units, dv.select_target
    dv.make_units=lambda cfg: units
    dv.select_target=lambda cfg: signal
    cluster_cfg=ClusterConfig(n_thermal=3,n_storage=3,n_hydrogen=3,n_emobility=3,n_industrial=3,seed=_SEED,
        initial_spread=0.0,initial_speed_scale=0.0,communication_radius=_R045,neighbour_count=_K4,layout_spread=_LS)
    ccfg=DSCEOSConfig(aggregate_tracking_weight=2.0/3.0,loss_weight_scale=0.03,sharing_weight=0.15,
        internal_weight=0.03,aggregate_consensus_gain=0.42,adaptive_consensus_gain=True,gershgorin_safety_factor=0.95)
    cfg=SimulationConfig(cluster=cluster_cfg,controller="dsceos",scenario="static_request",dsceos_config=ccfg,
        dt=0.05,horizon=float(scenario.horizon_sim),gateway_fraction=0.15,target_consensus_gain=0.18,
        target_gateway_gain=0.85,compute_reference_optimum=False,target_override=None,safety_filter=True)
    DistributedSCEOSController._solve_local_clf_hocbf_qp=_patched_qp
    try:
        result=run_simulation(cfg)
    finally:
        DistributedSCEOSController._solve_local_clf_hocbf_qp=_orig_qp
        dv.make_units, dv.select_target = orig_mu, orig_st
    JT=float(result.summary.get("integrated_objective_value", float("nan")))
    d=np.asarray(_disp,dtype=float)
    out=dict(scenario="A", fleet=FLEET, J_T=(round(JT,8) if JT==JT else None),
        n_agent_steps=int(d.size), max=round(float(d.max()),6), mean=round(float(d.mean()),6),
        p95=round(float(np.percentile(d,95)),6), frac_gt_1em3=round(float(np.mean(d>1e-3)),6))
    json.dump(out,open("control_displacement_N15A.json","w"),indent=2)
    print("control displacement N15-A:", out)
    return out

if __name__=="__main__":
    main()
