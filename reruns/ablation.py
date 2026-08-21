#!/usr/bin/env python3
"""
Block B -- component ablation.

Turns each D-SCEOS mechanism off in isolation and scores the resulting trajectory on the
FIXED full size-consistent objective J_T, so the number measures each component's contribution.

Variants:
  full            : all mechanisms on (baseline).
  no_N_scaling    : controller drops the size-consistency N (tracking weight -> w_bar/N).
  no_sharing      : lambda_s = 0 (no capacity-normalised utilisation sharing).
  no_internal     : lambda_int = 0 (no internal counter-action Laplacian).
  perfect_estimate: controller uses the TRUE aggregate instead of the consensus estimate.
  direct_broadcast: every agent receives the target directly (gateway_fraction = 1.0).
  hocbf_only      : CLF row removed; only the HOCBF-box/actuator-ball projection (no CLF/QCQP).

Scoring: the per-step objective is always recomputed with FULL weights (fixed reference cfg),
independent of the controller's ablated weights, so J_T is comparable across variants.

Outputs per (config, variant): full J_T, final aggregate error, max sampled capacity violation,
mean local-QP success (1 - fallback rate), mean CLF slack, runtime.  Minimum configs N15-A, N60-A,
N60-C; the most informative variants also over 10 paired seeds.
"""
from __future__ import annotations
import sys, os, json, math, time, copy
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "code"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dsceos_controller as dc
import dsceos_validation as dv
from dsceos_validation import (ClusterConfig, SimulationConfig, run_simulation, aggregate_output)
from dsceos_controller import DSCEOSConfig
from graph_config import (AUTHORITATIVE_COMMUNICATION_RADIUS as R, AUTHORITATIVE_NEIGHBOUR_COUNT as NB,
                          AUTHORITATIVE_LAYOUT_SPREAD as LS, AUTHORITATIVE_SEED as SD)
import realistic_cpes_catalog as cat
import realistic_scenarios as scen

SCEN = {"a": "scenario_a_winter_morning_step", "b": "scenario_b_wind_ramp_down_event",
        "c": "scenario_c_winter_balancing_mfrr"}

# Full (baseline) weights -- the fixed objective used for scoring every variant.
FULL_W = dict(aggregate_tracking_weight=2.0 / 3.0, loss_weight_scale=0.03,
              sharing_weight=0.15, internal_weight=0.03)

VARIANTS = ["full", "no_N_scaling", "no_sharing", "no_internal",
            "perfect_estimate", "direct_broadcast", "hocbf_only"]


def build_base(cluster, scen_key, seed=None, jitter=0.0):
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
                       seed=(SD if seed is None else seed),
                       initial_spread=(0.0 if seed is None else jitter),
                       initial_speed_scale=0.0, communication_radius=R, neighbour_count=NB,
                       layout_spread=LS)
    n = len(units)
    return cc, scenario, n


def variant_cfg(variant, n):
    w = dict(FULL_W)
    kwargs = dict(aggregate_consensus_gain=0.42, adaptive_consensus_gain=True,
                  gershgorin_safety_factor=0.95, use_local_qp=True, actuator_constraint="norm")
    if variant == "no_N_scaling":
        w["aggregate_tracking_weight"] = FULL_W["aggregate_tracking_weight"] / float(n)
    elif variant == "no_sharing":
        w["sharing_weight"] = 0.0
    elif variant == "no_internal":
        w["internal_weight"] = 0.0
    elif variant == "hocbf_only":
        kwargs["use_local_qp"] = False
    return DSCEOSConfig(**w, **kwargs)


def run_variant(cluster, scen_key, variant, seed=None, jitter=0.0):
    cc, scenario, n = build_base(cluster, scen_key, seed=seed, jitter=jitter)
    ccfg = variant_cfg(variant, n)
    gw_frac = 1.0 if variant == "direct_broadcast" else 0.15
    scfg = SimulationConfig(cluster=cc, controller="dsceos", scenario="static_request", dsceos_config=ccfg,
                            dt=0.05, horizon=float(scenario.horizon_sim), gateway_fraction=gw_frac,
                            target_consensus_gain=0.18, target_gateway_gain=0.85,
                            compute_reference_optimum=False, target_override=None, safety_filter=True)

    # ---- FULL-objective capture (score any variant on the fixed full objective) ----
    full_cfg = DSCEOSConfig(**FULL_W, aggregate_consensus_gain=0.42, adaptive_consensus_gain=True,
                            gershgorin_safety_factor=0.95)
    full_obj = []
    orig_ot = dv.objective_terms

    def patched_ot(p, arrays, W, Ablk, target, cfg):
        full_obj.append(orig_ot(p, arrays, W, Ablk, target, full_cfg)["objective_value"])
        return orig_ot(p, arrays, W, Ablk, target, cfg)

    # ---- perfect_estimate monkeypatch (only this variant) ----
    orig_update = dc.DistributedSCEOSController.update_estimators
    def perfect_update(self, p):
        orig_update(self, p)
        y_true = aggregate_output(self.problem.aggregate_blocks, p)
        self.state.y_hat[:] = y_true[None, :]

    dv.objective_terms = patched_ot
    if variant == "perfect_estimate":
        dc.DistributedSCEOSController.update_estimators = perfect_update
    try:
        t0 = time.time()
        res = run_simulation(scfg)
        wall = time.time() - t0
    finally:
        dv.objective_terms = orig_ot
        dc.DistributedSCEOSController.update_estimators = orig_update

    s = res.summary
    full_JT = float(scfg.dt * np.sum(full_obj))
    qp_rate = s.get("mean_local_qp_success_rate", float("nan"))
    return dict(
        full_J_T=full_JT,
        reported_J_T=float(s["integrated_objective_value"]),
        final_aggregate_error=float(s["final_aggregate_error"]),
        max_capacity_violation=float(s["max_capacity_violation"]),
        mean_qp_success=(float(qp_rate) if qp_rate == qp_rate else float("nan")),
        fallback_rate=(float(1.0 - qp_rate) if qp_rate == qp_rate else float("nan")),
        mean_clf_slack=float(s.get("mean_clf_slack", float("nan"))),
        total_control_energy=float(s.get("total_control_energy", float("nan"))),
        wall_s=round(wall, 1),
    )


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", default="N15_a,N60_a,N60_c")
    ap.add_argument("--variant", default=None, help="run a single variant")
    ap.add_argument("--seeds", type=int, default=1, help="paired seeds (1 = deterministic only)")
    args = ap.parse_args()

    cfg_map = {"N15_a": ("realistic_15", "a"), "N15_b": ("realistic_15", "b"), "N15_c": ("realistic_15", "c"),
               "N60_a": ("realistic_60", "a"), "N60_b": ("realistic_60", "b"), "N60_c": ("realistic_60", "c")}
    configs = [c.strip() for c in args.configs.split(",")]
    variants = [args.variant] if args.variant else VARIANTS

    respath = os.path.join(os.path.dirname(__file__), "..", "results", "ablation.json")
    out = {}
    if os.path.exists(respath):
        try: out = json.load(open(respath))
        except Exception: out = {}

    for cname in configs:
        cl, sk = cfg_map[cname]
        out.setdefault(cname, {})
        for v in variants:
            r = run_variant(cl, sk, v)
            out[cname][v] = r
            print(f"[{cname}/{v:<16}] full_J_T={r['full_J_T']:.5f} "
                  f"agg_err={r['final_aggregate_error']:.4f} cap={r['max_capacity_violation']:.1e} "
                  f"fb={r['fallback_rate']:.4f} slack={r['mean_clf_slack']:.3g} ({r['wall_s']}s)")
            os.makedirs(os.path.dirname(respath), exist_ok=True)
            json.dump(out, open(respath, "w"), indent=2)
    print("saved results/ablation.json")
