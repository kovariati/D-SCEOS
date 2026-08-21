"""stronger distributed comparator under an equal budget.

Runs the distributed primal-dual (Arrow-Hurwicz) comparator DPD-HOCBF on the six realistic
configurations and reports its integrated objective against D-SCEOS, DPG-HOCBF and the PD baseline.
DPD-HOCBF uses the SAME fixed communication graph, the SAME gateway-injected target information, one
local update per sampling instant, the SAME nine-scalar per-neighbour payload and the SAME posthoc
HOCBF safety filter as the other comparators; its dual variables are private per-agent state and are
never transmitted, so the communication budget is identical (the review's parity requirement).

PORTABILITY (source-integrity): this runs the simulation IN-PROCESS (it imports the harness and calls
run_simulation directly, exactly as run_realistic_scenario.py does), rather than shelling out to a
subprocess. That removes the interpreter-name and path fragility that made the earlier subprocess
version fail on Windows, and surfaces any error directly instead of hiding it behind DEVNULL.

Run from the package root:  python3 reruns/primal_dual_comparator.py   (python on Windows)
"""
import json
import os
import sys
from dataclasses import replace

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_ROOT, os.path.join(_ROOT, "code")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.chdir(_ROOT)
sys.dont_write_bytecode = True

import dsceos_validation as dv                                     # noqa: E402
from dsceos_controller import DSCEOSConfig                         # noqa: E402
from graph_config import (AUTHORITATIVE_COMMUNICATION_RADIUS,      # noqa: E402
                          AUTHORITATIVE_NEIGHBOUR_COUNT, AUTHORITATIVE_LAYOUT_SPREAD,
                          AUTHORITATIVE_SEED)
from realistic_cpes_catalog import build_realistic_units, compute_fleet_scaling  # noqa: E402
from realistic_scenarios import REALISTIC_SCENARIOS, build_signal_for            # noqa: E402

SCEN = {"a": "scenario_a_winter_morning_step",
        "b": "scenario_b_wind_ramp_down_event",
        "c": "scenario_c_winter_balancing_mfrr"}
CTRL = ["dsceos", "distributed_primal_dual_hocbf", "projected_gradient_hocbf", "independent_tracking"]
LABEL = {"dsceos": "D-SCEOS", "distributed_primal_dual_hocbf": "DPD-HOCBF",
         "projected_gradient_hocbf": "DPG-HOCBF", "independent_tracking": "PD baseline"}


def simulate(tag, sc, controller):
    """In-process equivalent of run_realistic_scenario.py for one (fleet, scenario, controller)."""
    cluster = "realistic_15" if tag == "N15" else "realistic_60"
    units, physical = build_realistic_units(cluster)
    scenario = REALISTIC_SCENARIOS[SCEN[sc]]
    if tag == "N60":                                   # the released N60 study uses a 4x request
        scenario = replace(scenario,
                           target_P_max_GW=scenario.target_P_max_GW * 4.0,
                           target_Q_max_GVAR=scenario.target_Q_max_GVAR * 4.0)
    scaling = compute_fleet_scaling(physical)
    signal = build_signal_for(scenario, scaling)
    cluster_cfg = dv.ClusterConfig(
        n_thermal=3, n_storage=3, n_hydrogen=3, n_emobility=3, n_industrial=3,
        seed=AUTHORITATIVE_SEED, initial_spread=0.0, initial_speed_scale=0.0,
        communication_radius=AUTHORITATIVE_COMMUNICATION_RADIUS,
        neighbour_count=AUTHORITATIVE_NEIGHBOUR_COUNT, layout_spread=AUTHORITATIVE_LAYOUT_SPREAD)
    sim = dv.SimulationConfig(
        cluster=cluster_cfg, controller=controller, scenario="static_request",
        dsceos_config=DSCEOSConfig(adaptive_consensus_gain=True), dt=0.05,
        horizon=float(scenario.horizon_sim), gateway_fraction=0.15,
        target_consensus_gain=0.18, target_gateway_gain=0.85,
        compute_reference_optimum=False, target_override=None, safety_filter=True,
        dpg_step_size=0.10, pd_kp=0.75, pd_kd=2.5)
    om, osel = dv.make_units, dv.select_target
    dv.make_units = lambda _c: units
    dv.select_target = lambda _c: signal
    try:
        res = dv.run_simulation(sim)
    finally:
        dv.make_units, dv.select_target = om, osel
    return res.summary


rows = []
print(f"{'cfg':8}{'D-SCEOS':>10}{'DPD-HOCBF':>11}{'DPG-HOCBF':>11}{'PD baseline':>12}"
      f"{'DPD/DS':>8}{'maxviol':>10}")
for tag in ("N15", "N60"):
    for sc in "abc":
        vals = {}
        for c in CTRL:
            rel = os.path.join("results", f"{tag}_{sc}_{c}", "summary.json")
            if c == "distributed_primal_dual_hocbf" or not os.path.exists(rel):
                vals[c] = simulate(tag, sc, c)          # only DPD needs running; others are released
            else:
                vals[c] = json.load(open(rel))
        ds = vals["dsceos"]["integrated_objective_value"]
        dpd = vals["distributed_primal_dual_hocbf"]
        row = dict(cluster=tag, scenario=sc.upper(),
                   J_T={LABEL[c]: round(vals[c]["integrated_objective_value"], 4) for c in CTRL},
                   dpd_over_dsceos=round(dpd["integrated_objective_value"] / ds, 4),
                   dpd_max_cap_violation=round(dpd["max_capacity_violation"], 8),
                   dsceos_lowest=all(ds <= vals[c]["integrated_objective_value"] for c in CTRL))
        rows.append(row)
        print(f"{tag}-{sc.upper():6}{row['J_T']['D-SCEOS']:10.4f}{row['J_T']['DPD-HOCBF']:11.4f}"
              f"{row['J_T']['DPG-HOCBF']:11.4f}{row['J_T']['PD baseline']:12.4f}"
              f"{row['dpd_over_dsceos']:8.2f}{row['dpd_max_cap_violation']:10.1e}")

json.dump(rows, open("primal_dual_comparator.json", "w"), indent=1)
print(f"\n-> primal_dual_comparator.json  ({sum(r['dsceos_lowest'] for r in rows)}/{len(rows)} "
      f"configs: D-SCEOS lowest; DPD capacity violation max "
      f"{max(r['dpd_max_cap_violation'] for r in rows):.1e})")
