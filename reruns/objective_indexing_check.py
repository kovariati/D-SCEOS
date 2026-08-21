#!/usr/bin/env python3
"""
Quantify the effect of the discrete objective time-indexing convention.

The simulation loop evaluates the running cost with the state AFTER the integration step but with
the target sampled BEFORE it:

    J_k = J( p_{k+1} ; y^T(t_k) )                     (implemented, left-target convention)

A right-endpoint quadrature of the continuous integral would instead pair the same state with the
target at the same instant:

    J_k = J( p_{k+1} ; y^T(t_{k+1}) )                 (aligned convention)

This script recomputes J_T from the stored trajectories under BOTH conventions, so the sensitivity
of the headline metric to the convention is measured rather than assumed. It runs no simulation.

Output: objective_indexing_check.json
"""
from __future__ import annotations
import os, sys, json, math
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "code"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import dsceos_validation as dv
from dsceos_validation import (arrays_from_units, make_aggregate_blocks, make_fixed_local_graph,
                               make_physical_layout, objective_terms, ClusterConfig)
from dsceos_controller import DSCEOSConfig
from graph_config import (AUTHORITATIVE_COMMUNICATION_RADIUS as R, AUTHORITATIVE_NEIGHBOUR_COUNT as NB,
                          AUTHORITATIVE_LAYOUT_SPREAD as LS, AUTHORITATIVE_SEED as SD)
import realistic_cpes_catalog as cat
import realistic_scenarios as scen

SCEN = {"a": "scenario_a_winter_morning_step", "b": "scenario_b_wind_ramp_down_event",
        "c": "scenario_c_winter_balancing_mfrr"}
CTRL = {"dsceos": "D-SCEOS", "projected_gradient_hocbf": "DPG-HOCBF",
        "independent_tracking": "PD baseline"}
FULL_W = dict(aggregate_tracking_weight=2.0 / 3.0, loss_weight_scale=0.03,
              sharing_weight=0.15, internal_weight=0.03)


def setup(cluster, sk):
    scenario = scen.REALISTIC_SCENARIOS[SCEN[sk]]
    tm = 4.0 if cluster == "realistic_60" else 1.0
    if tm != 1.0:
        from dataclasses import replace
        scenario = replace(scenario, target_P_max_GW=scenario.target_P_max_GW * tm,
                           target_Q_max_GVAR=scenario.target_Q_max_GVAR * tm)
    units, physical = cat.build_realistic_units(cluster)
    scaling = cat.compute_fleet_scaling(physical)
    signal = scen.build_signal_for(scenario, scaling)
    arrays = arrays_from_units(units)
    n = len(units)
    cc = ClusterConfig(n_thermal=3, n_storage=3, n_hydrogen=3, n_emobility=3, n_industrial=3,
                       seed=SD, initial_spread=0.0, initial_speed_scale=0.0,
                       communication_radius=R, neighbour_count=NB, layout_spread=LS)
    layout = make_physical_layout(n, cc)
    W = make_fixed_local_graph(layout, R, NB)
    Ablk = make_aggregate_blocks(arrays["aggregate_weight"])
    cfg = DSCEOSConfig(**FULL_W, aggregate_consensus_gain=0.42, adaptive_consensus_gain=True,
                       gershgorin_safety_factor=0.95)
    return arrays, W, Ablk, cfg, signal


def integrate(positions, time, signal, arrays, W, Ablk, cfg, shift):
    """shift=0 -> left-target (implemented); shift=1 -> aligned right-endpoint."""
    dt = float(time[1] - time[0])
    total = 0.0
    steps = positions.shape[0] - 1
    for k in range(steps):
        p_next = positions[k + 1]
        t_used = float(time[k + shift])
        y = np.asarray(signal.position(t_used), dtype=float)
        total += objective_terms(p_next, arrays, W, Ablk, y, cfg)["objective_value"]
    return dt * total


def main():
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    out = {}
    print(f"{'config':<10} {'controller':<12} {'implemented':>12} {'aligned':>12} {'rel diff %':>11}")
    for cluster, tag in [("realistic_15", "N15"), ("realistic_60", "N60")]:
        for sk in ["a", "b", "c"]:
            arrays, W, Ablk, cfg, signal = setup(cluster, sk)
            for ctrl, label in CTRL.items():
                npz = os.path.join(root, "results", f"{tag}_{sk}_{ctrl}", "state_history.npz")
                if not os.path.exists(npz):
                    continue
                d = np.load(npz)
                pos, time = d["positions"], d["time"]
                a = integrate(pos, time, signal, arrays, W, Ablk, cfg, 0)
                b = integrate(pos, time, signal, arrays, W, Ablk, cfg, 1)
                rel = 100.0 * (b - a) / max(abs(a), 1e-12)
                key = f"{tag}_{sk}/{label}"
                out[key] = dict(implemented_left_target=a, aligned_right_endpoint=b,
                                abs_diff=b - a, rel_diff_pct=rel)
                print(f"{tag+'_'+sk:<10} {label:<12} {a:12.5f} {b:12.5f} {rel:+11.3f}")
    worst = max((abs(v["rel_diff_pct"]) for v in out.values()), default=0.0)
    summary = dict(convention_implemented="J_k = J(p_{k+1}; y^T(t_k))  (left-target)",
                   convention_aligned="J_k = J(p_{k+1}; y^T(t_{k+1}))  (right-endpoint)",
                   worst_abs_rel_diff_pct=worst,
                   note=("Scenario A has a constant request, so the two conventions coincide there. "
                         "The comparison is a re-integration of stored trajectories; no simulation was re-run, "
                         "and the controller trajectories themselves are identical under both readings."),
                   per_run=out)
    path = os.path.join(root, "objective_indexing_check.json")
    json.dump(summary, open(path, "w"), indent=2)
    print(f"\nworst |relative difference| = {worst:.3f}%")
    print("saved objective_indexing_check.json")


if __name__ == "__main__":
    main()
