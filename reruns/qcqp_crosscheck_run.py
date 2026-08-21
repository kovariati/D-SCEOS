#!/usr/bin/env python3
"""Run Block A (QCQP/fallback cross-check) over the 6 realistic configs + 7 stress-ladder points."""
from __future__ import annotations
import sys, os, json, math, time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "code"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dsceos_controller as dc
import dsceos_validation as dv
from dsceos_validation import ClusterConfig, SimulationConfig, run_simulation, make_units as _mk, select_target as _st
from dsceos_controller import DSCEOSConfig
from graph_config import (AUTHORITATIVE_COMMUNICATION_RADIUS as R, AUTHORITATIVE_NEIGHBOUR_COUNT as NB,
                          AUTHORITATIVE_LAYOUT_SPREAD as LS, AUTHORITATIVE_SEED as SD)
import realistic_cpes_catalog as cat
import realistic_scenarios as scen

from qcqp_crosscheck import QCQPCapture, crosscheck_records, make_clarabel_solver

SCEN = {"a": "scenario_a_winter_morning_step",
        "b": "scenario_b_wind_ramp_down_event",
        "c": "scenario_c_winter_balancing_mfrr"}

from graph_config import LADDER_UNIT_MIX, ladder_cluster_kwargs
LADDER = [("soft", (0.18, 0.02)), ("feasible", (0.30, 0.02)), ("borderline-low", (0.42, 0.02)),
          ("borderline-mid", (0.54, 0.02)), ("borderline-high", (0.70, 0.05)),
          ("hard-stress", (0.75, 0.20)), ("extreme-infeasible", (0.85, 0.35))]


def build_ladder_cfg(target):
    cc = ClusterConfig(**LADDER_UNIT_MIX, **ladder_cluster_kwargs(),
                       initial_spread=0.0, initial_speed_scale=0.0)
    ccfg = DSCEOSConfig(aggregate_tracking_weight=10.0, loss_weight_scale=0.03, sharing_weight=0.15,
                        internal_weight=0.03, aggregate_consensus_gain=0.10, adaptive_consensus_gain=False)
    scfg = SimulationConfig(cluster=cc, controller="dsceos", scenario="static_request",
                            dsceos_config=ccfg, horizon=12.0, dt=0.04, gateway_fraction=0.15,
                            target_consensus_gain=0.18, target_gateway_gain=0.85,
                            compute_reference_optimum=False, target_override=target, safety_filter=True,
                            dpg_step_size=0.10, pd_kp=0.75, pd_kd=2.5)
    return scfg


def build_realistic_cfg(cluster, scen_key):
    scenario = scen.REALISTIC_SCENARIOS[SCEN[scen_key]]
    tm = 4.0 if cluster == "realistic_60" else 1.0
    if tm != 1.0:
        from dataclasses import replace
        scenario = replace(scenario, target_P_max_GW=scenario.target_P_max_GW * tm,
                           target_Q_max_GVAR=scenario.target_Q_max_GVAR * tm)
    units, physical = cat.build_realistic_units(cluster)
    scaling = cat.compute_fleet_scaling(physical)
    signal = scen.build_signal_for(scenario, scaling)
    dv.make_units = lambda cfg: units
    dv.select_target = lambda cfg: signal
    cc = ClusterConfig(n_thermal=3, n_storage=3, n_hydrogen=3, n_emobility=3, n_industrial=3,
                       seed=SD, initial_spread=0.0, initial_speed_scale=0.0,
                       communication_radius=R, neighbour_count=NB, layout_spread=LS)
    ccfg = DSCEOSConfig(aggregate_tracking_weight=2/3, loss_weight_scale=0.03, sharing_weight=0.15,
                        internal_weight=0.03, aggregate_consensus_gain=0.42,
                        adaptive_consensus_gain=True, gershgorin_safety_factor=0.95)
    scfg = SimulationConfig(cluster=cc, controller="dsceos", scenario="static_request", dsceos_config=ccfg,
                            dt=0.05, horizon=float(scenario.horizon_sim), gateway_fraction=0.15,
                            target_consensus_gain=0.18, target_gateway_gain=0.85,
                            compute_reference_optimum=False, target_override=None, safety_filter=True)
    return scfg


def _make_controller_for(scfg):
    """Replicate the controller construction that run_simulation performs, so we can
    attach the capture wrapper to the exact object the simulation will use."""
    # run_simulation builds the controller internally; we instead capture by patching the
    # class method for the duration of the run (all agents share one controller instance).
    return scfg


def run_openloop_scfg(scfg):
    records = []
    orig = dc.DistributedSCEOSController._solve_local_clf_hocbf_qp

    def wrapped(self, u_nom, lb, ub, radius, e, vel, mass, damping):
        u, slack, success, residual, V = orig(self, u_nom, lb, ub, radius, e, vel, mass, damping)
        Vv, coeff_u, drift = self._local_clf_terms(e, vel, mass, damping)
        rhs = drift + float(self.config.clf_rate) * Vv
        records.append(dict(u_nom=np.asarray(u_nom, float).copy(),
                            lb=np.asarray(lb, float).copy(), ub=np.asarray(ub, float).copy(),
                            radius=float(radius), coeff_u=np.asarray(coeff_u, float).copy(),
                            rhs=float(rhs), u_applied=np.asarray(u, float).copy(),
                            slack=float(slack), success=bool(success), residual=float(residual)))
        return u, slack, success, residual, V

    dc.DistributedSCEOSController._solve_local_clf_hocbf_qp = wrapped
    try:
        res = run_simulation(scfg)
    finally:
        dc.DistributedSCEOSController._solve_local_clf_hocbf_qp = orig
    w_s = scfg.dsceos_config.clf_slack_weight
    summary = crosscheck_records(records, w_s)
    summary["J_T_production"] = float(res.summary["integrated_objective_value"])
    summary["max_capacity_violation_production"] = float(res.summary["max_capacity_violation"])
    return summary


def run_closedloop_scfg(scfg):
    """Re-run with CLARABEL as the in-loop QCQP solver; return J_T and trajectory distance."""
    orig = dc.DistributedSCEOSController._solve_local_clf_hocbf_qp

    def clarabel_method(self, u_nom, lb, ub, radius, e, vel, mass, damping):
        return make_clarabel_solver(self)(u_nom, lb, ub, radius, e, vel, mass, damping)

    dc.DistributedSCEOSController._solve_local_clf_hocbf_qp = clarabel_method
    try:
        res = run_simulation(scfg)
    finally:
        dc.DistributedSCEOSController._solve_local_clf_hocbf_qp = orig
    return dict(J_T_clarabel=float(res.summary["integrated_objective_value"]),
                max_capacity_violation_clarabel=float(res.summary["max_capacity_violation"]))


def build_all_configs():
    cfgs = []
    for name, tgt in LADDER:
        cfgs.append((f"ladder_{name}", lambda t=tgt: build_ladder_cfg(t)))
    for cl, tag in [("realistic_15", "N15"), ("realistic_60", "N60")]:
        for sk in ["a", "b", "c"]:
            cfgs.append((f"{tag}_{sk}", lambda c=cl, s=sk: build_realistic_cfg(c, s)))
    return cfgs


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None, help="substring filter, e.g. ladder or N15_a")
    ap.add_argument("--no-closedloop", action="store_true")
    args = ap.parse_args()

    cfgs = build_all_configs()
    if args.only:
        cfgs = [c for c in cfgs if args.only in c[0]]

    _respath = os.path.join(os.path.dirname(__file__), "..", "results", "qcqp_crosscheck.json")
    out = {}
    if os.path.exists(_respath):
        try:
            out = json.load(open(_respath))
        except Exception:
            out = {}
    for name, builder in cfgs:
        t0 = time.time()
        scfg = builder()
        s = run_openloop_scfg(scfg)
        if not args.no_closedloop:
            scfg2 = builder()
            s.update(run_closedloop_scfg(scfg2))
            s["DeltaJ_T"] = s["J_T_clarabel"] - s["J_T_production"]
        s["wall_s"] = round(time.time() - t0, 1)
        out[name] = s
        print(f"[{name:<24}] steps={s['n_steps']:>5} fb_rate={s['fallback_rate']:.4f} "
              f"du_max={s['du_max']:.2e} du_mean={s['du_mean']:.2e} "
              f"indep_feas_max={s['indep_feas_resid_max']:.1e} "
              + (f"DJ_T={s.get('DeltaJ_T', float('nan')):+.2e} J_T={s['J_T_production']:.4f} " if not args.no_closedloop else "")
              + f"({s['wall_s']}s)")
        # incremental save
        os.makedirs(os.path.join(os.path.dirname(__file__), "..", "results"), exist_ok=True)
        with open(os.path.join(os.path.dirname(__file__), "..", "results", "qcqp_crosscheck.json"), "w") as f:
            json.dump(out, f, indent=2)
    print("saved results/qcqp_crosscheck.json")
