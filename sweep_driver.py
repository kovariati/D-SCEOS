"""
Reusable single-run driver that mirrors run_realistic_scenario.main() exactly
but exposes ALL controller/objective parameters (including CLF gains, HOCBF
alpha_0/alpha_1, DPG step size, gateway fraction) for the sensitivity sweeps.

Returns the summary.json dict without writing state history (fast).
Reproduces the released deterministic D-SCEOS value: run_config("A","realistic_15","dsceos") -> J_T 0.414216.
"""
from __future__ import annotations
from dataclasses import replace
import os as _os, sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__))
for _p in (_HERE, _os.path.join(_HERE, "code")):      # runnable directly from the package root
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
import numpy as np

from realistic_cpes_catalog import build_realistic_units, compute_fleet_scaling
from realistic_scenarios import REALISTIC_SCENARIOS, build_signal_for
import dsceos_validation as dv
from dsceos_validation import (ClusterConfig, SimulationConfig, run_simulation)
from dsceos_controller import DSCEOSConfig
from graph_config import (AUTHORITATIVE_COMMUNICATION_RADIUS as _R045,
                          AUTHORITATIVE_NEIGHBOUR_COUNT as _K4,
                          AUTHORITATIVE_LAYOUT_SPREAD as _LS,
                          AUTHORITATIVE_SEED as _SEED)  # single authoritative graph config

SCEN = {"A":"scenario_a_winter_morning_step",
        "B":"scenario_b_wind_ramp_down_event",
        "C":"scenario_c_winter_balancing_mfrr"}

def run_config(scen_key, cluster, controller, *,
               w_bar=2.0/3.0, w_loss=0.03, lam_s=0.15, lam_int=0.03,
               alpha0=1.0, alpha1=2.0, clf_rate=0.80,
               dpg_step=0.06, dpg_substeps=1,
               gateway_fraction=0.15, dt=0.05, adaptive_gain=True,
               fixed_consensus_gain=0.42, gershgorin_factor=0.95,
               return_result=False):
    scenario = REALISTIC_SCENARIOS[SCEN[scen_key]]
    tm = 4.0 if cluster=="realistic_60" else 1.0
    if tm != 1.0:
        scenario = replace(scenario,
            target_P_max_GW=scenario.target_P_max_GW*tm,
            target_Q_max_GVAR=scenario.target_Q_max_GVAR*tm)
    units, physical = build_realistic_units(cluster)
    n=len(units)
    scaling = compute_fleet_scaling(physical)
    signal = build_signal_for(scenario, scaling)

    orig_make, orig_sel = dv.make_units, dv.select_target
    dv.make_units = lambda cfg: units
    dv.select_target = lambda cfg: signal

    # DPG step-size / substeps injection (same mechanism as the runner)
    from distributed_projected_gradient_controller import DistributedProjectedGradientController as _DPG
    _s,_sub=dpg_step,dpg_substeps
    class _TunedDPG(_DPG):
        def __init__(self, problem, cfg, **kw):
            kw.setdefault("step_size",_s); kw.setdefault("optimizer_substeps",_sub)
            super().__init__(problem, cfg, **kw)
    orig_dpg = dv.DistributedProjectedGradientController
    dv.DistributedProjectedGradientController = _TunedDPG

    cluster_cfg = ClusterConfig(n_thermal=3,n_storage=3,n_hydrogen=3,n_emobility=3,
        n_industrial=3, seed=_SEED, initial_spread=0.0, initial_speed_scale=0.0,
        communication_radius=_R045, neighbour_count=_K4, layout_spread=_LS)
    ccfg = DSCEOSConfig(aggregate_tracking_weight=w_bar, loss_weight_scale=w_loss,
        sharing_weight=lam_s, internal_weight=lam_int,
        hocbf_alpha0=alpha0, hocbf_alpha1=alpha1, clf_rate=clf_rate,
        aggregate_consensus_gain=fixed_consensus_gain,
        adaptive_consensus_gain=adaptive_gain, gershgorin_safety_factor=gershgorin_factor)
    sim_cfg = SimulationConfig(cluster=cluster_cfg, controller=controller,
        scenario="static_request", dsceos_config=ccfg, dt=dt,
        horizon=float(scenario.horizon_sim), gateway_fraction=gateway_fraction,
        target_consensus_gain=0.18, target_gateway_gain=0.85,
        dpg_step_size=dpg_step,
        compute_reference_optimum=False, target_override=None, safety_filter=True)
    try:
        result = run_simulation(sim_cfg)
    finally:
        # Always restore the module-level overrides, even if run_simulation raises, so a failed
        # sweep point cannot leave the next point running against a poisoned global state.
        dv.make_units, dv.select_target = orig_make, orig_sel
        dv.DistributedProjectedGradientController = orig_dpg
    if return_result:
        return result
    s=result.summary
    return dict(J_T=s["integrated_objective_value"],
                energy=s["total_control_energy"],
                agg_err=s["final_aggregate_error"],
                cap_viol=s["max_capacity_violation"],
                Ky=s["active_aggregate_consensus_gain"])

if __name__=="__main__":
    # self-test against the released deterministic D-SCEOS J_T (symmetric combinatorial-Laplacian
    # internal-disagreement operator; these are the authoritative values used in the paper).
    _EXPECT = {"A": 0.414216, "B": 0.165303, "C": 1.539969}
    _ok = True
    for sk in ["A","B","C"]:
        r=run_config(sk,"realistic_15","dsceos")
        _match = abs(r["J_T"] - _EXPECT[sk]) < 1e-4
        _ok = _ok and _match
        print(f"N15/{sk} dsceos J_T={r['J_T']:.6f}  (expected {_EXPECT[sk]:.6f})  {'ok' if _match else 'MISMATCH'}")
    print("self-test:", "PASS" if _ok else "FAIL")
    import sys as _s; _s.exit(0 if _ok else 1)
