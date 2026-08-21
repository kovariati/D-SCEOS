#!/usr/bin/env python3
"""
Technical point 5 -- diagnostics of the centralized static box-constrained reference solve.

Reports, per realistic configuration:
  * projected-gradient (KKT) residual of the reference point,
      r = || P_X(p* - grad J(p*)) - p* ||_inf,     (zero at a KKT point of the box-constrained problem)
  * objective gap  J(p_final^DSCEOS) - J(p*),
  * FULL allocation error ||p_final - p*||_2 and per-unit RMS,
  * aggregate error ||y(p_final) - y(p*)||_2  (for contrast: the aggregate alone understates the gap).

Gradients are computed by central finite differences on the released objective, so the diagnostic is
independent of the controller's analytic gradient implementation.
"""
from __future__ import annotations
import os, sys, json, math
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "code"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import dsceos_validation as dv
from dsceos_validation import (ClusterConfig, SimulationConfig, run_simulation, arrays_from_units,
                               make_aggregate_blocks, make_fixed_local_graph, make_physical_layout,
                               objective_terms, solve_reference_optimum, aggregate_output)
from dsceos_controller import DSCEOSConfig
from graph_config import (AUTHORITATIVE_COMMUNICATION_RADIUS as R, AUTHORITATIVE_NEIGHBOUR_COUNT as NB,
                          AUTHORITATIVE_LAYOUT_SPREAD as LS, AUTHORITATIVE_SEED as SD)
import realistic_cpes_catalog as cat
import realistic_scenarios as scen

SCEN = {"a": "scenario_a_winter_morning_step", "b": "scenario_b_wind_ramp_down_event",
        "c": "scenario_c_winter_balancing_mfrr"}
FULL_W = dict(aggregate_tracking_weight=2.0 / 3.0, loss_weight_scale=0.03,
              sharing_weight=0.15, internal_weight=0.03)


def build(cluster, sk):
    scenario = scen.REALISTIC_SCENARIOS[SCEN[sk]]
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
    ccfg = DSCEOSConfig(**FULL_W, aggregate_consensus_gain=0.42, adaptive_consensus_gain=True,
                        gershgorin_safety_factor=0.95)
    scfg = SimulationConfig(cluster=cc, controller="dsceos", scenario="static_request", dsceos_config=ccfg,
                            dt=0.05, horizon=float(scenario.horizon_sim), gateway_fraction=0.15,
                            target_consensus_gain=0.18, target_gateway_gain=0.85,
                            compute_reference_optimum=False, target_override=None, safety_filter=True)
    return scfg, units, signal


def fd_grad(fun, x, h=1e-7):
    g = np.zeros_like(x)
    for k in range(x.size):
        e = np.zeros_like(x); e[k] = h
        g[k] = (fun(x + e) - fun(x - e)) / (2.0 * h)
    return g


def analyse(cluster, sk):
    scfg, units, signal = build(cluster, sk)
    res = run_simulation(scfg)
    p_final = np.asarray(res.positions[-1], dtype=float)

    arrays = arrays_from_units(units)
    n = p_final.shape[0]
    layout = make_physical_layout(n, scfg.cluster)
    W = make_fixed_local_graph(layout, scfg.cluster.communication_radius, scfg.cluster.neighbour_count)
    Ablk = make_aggregate_blocks(arrays["aggregate_weight"])
    target = np.asarray(signal.position(float(scfg.horizon)), dtype=float)
    cfg = scfg.dsceos_config

    Jstar, pstar, ok = solve_reference_optimum(p_final, arrays, W, Ablk, target, cfg)
    if pstar is None:
        return dict(status="reference solve failed")

    shape = p_final.shape
    lower = arrays["lower"]; upper = arrays["upper"]
    f = lambda x: objective_terms(x.reshape(shape), arrays, W, Ablk, target, cfg)["objective_value"]

    g = fd_grad(f, pstar.reshape(-1)).reshape(shape)
    proj = np.minimum(np.maximum(pstar - g, lower), upper)
    kkt = float(np.max(np.abs(proj - pstar)))

    J_final = f(p_final.reshape(-1))
    dp = p_final - pstar
    y_final = aggregate_output(Ablk, p_final); y_star = aggregate_output(Ablk, pstar)

    return dict(
        n_agents=int(n),
        reference_solver="L-BFGS-B (box-constrained, ftol 1e-9, maxiter 150)",
        reference_converged=bool(ok),
        kkt_projected_gradient_residual_inf=kkt,
        J_reference=float(Jstar),
        J_dsceos_final=float(J_final),
        objective_gap=float(J_final - Jstar),
        objective_gap_rel=float((J_final - Jstar) / max(abs(Jstar), 1e-12)),
        allocation_error_l2=float(np.linalg.norm(dp)),
        allocation_error_per_unit_rms=float(np.sqrt(np.mean(np.sum(dp * dp, axis=1)))),
        allocation_error_max_unit=float(np.max(np.linalg.norm(dp, axis=1))),
        aggregate_error_l2=float(np.linalg.norm(y_final - y_star)),
    )


if __name__ == "__main__":
    out = {}
    for cl, tag in [("realistic_15", "N15"), ("realistic_60", "N60")]:
        for sk in ["a", "b", "c"]:
            name = f"{tag}_{sk}"
            r = analyse(cl, sk)
            out[name] = r
            print(f"[{name}] KKT={r['kkt_projected_gradient_residual_inf']:.2e} "
                  f"gap={r['objective_gap']:+.3e} ({r['objective_gap_rel']*100:+.2f}%) "
                  f"||p-p*||={r['allocation_error_l2']:.4f} rms={r['allocation_error_per_unit_rms']:.4f} "
                  f"||y-y*||={r['aggregate_error_l2']:.4f}")
    path = os.path.join(os.path.dirname(__file__), "..", "results", "reference_kkt_diagnostics.json")
    json.dump(out, open(path, "w"), indent=2)
    print("saved results/reference_kkt_diagnostics.json")
