"""Filter on/off diagnostic, reproducible from the parameterized main code
(runtime options dpg_filter_always + containment_margin). Run from package root."""
import os as _os, sys as _sys
_ROOT=_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
[_sys.path.insert(0,_p) for _p in (_ROOT,_os.path.join(_ROOT,"code")) if _p not in _sys.path]
_os.chdir(_ROOT)
import json, numpy as np
import dsceos_validation as dv
from dsceos_validation import ClusterConfig, SimulationConfig, run_simulation
from dsceos_controller import DSCEOSConfig
from realistic_cpes_catalog import build_realistic_units, compute_fleet_scaling
from realistic_scenarios import REALISTIC_SCENARIOS, build_signal_for
from graph_config import (AUTHORITATIVE_COMMUNICATION_RADIUS as _R045,
                          AUTHORITATIVE_NEIGHBOUR_COUNT as _K4,
                          AUTHORITATIVE_LAYOUT_SPREAD as _LS,
                          AUTHORITATIVE_SEED as _SEED)  # single authoritative graph config
def run(controller, filt):
    scen=REALISTIC_SCENARIOS["scenario_a_winter_morning_step"]
    units,phys=build_realistic_units("realistic_15"); sc=compute_fleet_scaling(phys); sig=build_signal_for(scen,sc)
    om,osel=dv.make_units,dv.select_target; dv.make_units=lambda cfg:units; dv.select_target=lambda cfg:sig
    cc=ClusterConfig(n_thermal=3,n_storage=3,n_hydrogen=3,n_emobility=3,n_industrial=3,seed=_SEED,
        initial_spread=0.0,initial_speed_scale=0.0,communication_radius=_R045,neighbour_count=_K4,layout_spread=_LS)
    ccfg=DSCEOSConfig(aggregate_tracking_weight=2/3,loss_weight_scale=0.03,sharing_weight=0.15,internal_weight=0.03,
        adaptive_consensus_gain=True,gershgorin_safety_factor=0.95)
    cfg=SimulationConfig(cluster=cc,controller=controller,scenario="static_request",dsceos_config=ccfg,dt=0.05,
        horizon=float(scen.horizon_sim),gateway_fraction=0.15,target_consensus_gain=0.18,target_gateway_gain=0.85,
        compute_reference_optimum=False,target_override=None,safety_filter=filt,
        dpg_filter_always=filt, containment_margin=0.05,
        dpg_step_size=0.10, pd_kp=0.75, pd_kd=2.5)
    r=run_simulation(cfg); dv.make_units,dv.select_target=om,osel; return r.summary
rows=[]; exp=json.load(open("reruns/filter_on_off_reference.json"))
expmap={(r["controller"],r["filter"]):r for r in exp}
print(f"{'controller':<24}{'filter':>8}{'J_T':>10}{'cap_viol':>12}{'check':>8}")
ok=True
for ctrl in ["dsceos","projected_gradient_hocbf","independent_tracking"]:
    for filt in [True,False]:
        s=run(ctrl,filt); jt=round(s['integrated_objective_value'],4); cv=round(s['max_capacity_violation'],6)
        e=expmap[(ctrl,filt)]; good=abs(jt-e['J_T'])<1e-3 and abs(cv-e['cap_viol'])<1e-3
        ok=ok and good
        print(f"{ctrl:<24}{str(filt):>8}{jt:>10.4f}{cv:>12.4e}{'OK' if good else 'FAIL':>8}")
        rows.append(dict(controller=ctrl,filter=filt,J_T=jt,cap_viol=cv))
print("ASSERT", "PASS" if ok else "FAIL", "(reproduces bundled r16_reproducible.json)")
