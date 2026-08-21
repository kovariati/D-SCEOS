"""independent fleet compositions and a real solver profile.

The submitted scalability evidence used only two fleet sizes, and the larger one (N=60) was a
fourfold *replication* of the N=15 catalog (``cluster_60() == cluster_15() * 4``), which does not
establish transferability to an arbitrary fleet composition or topology. The N*n_fg "unit-equivalent
solver-evaluation proxy" is also not a runtime or complexity measure. This harness supplies both
things the review asked for:

EXPERIMENT 1 -- INDEPENDENT COMPOSITIONS AT SEVERAL SIZES
  For each fleet size N in {15, 30, 60, 120} and several seeds, a fleet is drawn by INDEPENDENT
  sampling from the ten physical archetypes of the catalog (not by replication), on a fresh random
  connected geometric graph, and the three controllers are run on Scenario A. Because sampling is a
  pure function of the seed and identical across controllers, the per-seed comparison is paired.
  We report, per size: the D-SCEOS win rate over the seeds, and the paired mean ratio to each
  comparator, so that "does it still win on fleets it was not built from?" is answered directly.

EXPERIMENT 2 -- REAL SOLVER PROFILE
  With the opt-in per-agent QCQP trace enabled, one representative independent fleet per size is run
  and the following are measured directly (not proxied):
    - per-agent local solve time: mean / p95 / p99 / max (ms);
    - worst-case per-instant latency = the slowest single agent-solve in any sampling instant (ms);
    - total solver CPU time over the run (s) and the mean number of agent-solves per instant;
    - communication load: bytes/s per agent = 9 scalars * 8 bytes * degree / dt, min/mean/max over
      agents, which is exact for this protocol and depends on the LOCAL DEGREE, not on N;
    - per-degree load: mean solve time grouped by weighted degree, to expose any degree dependence.
  All wall-clock figures are environment-specific and are reported and validated as such (presence,
  type, plausibility), never asserted numerically.

Run from the package root:
    python3 reruns/fleet_scaling_profile.py --experiment compositions --sizes 15,30,60,120 --seeds 20
    python3 reruns/fleet_scaling_profile.py --experiment profile --sizes 15,30,60,120
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from operator import itemgetter

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_ROOT, os.path.join(_ROOT, "code")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.chdir(_ROOT)
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
sys.dont_write_bytecode = True

import scipy                                                       # noqa: E402
from scipy import stats                                            # noqa: E402

import dsceos_validation as dv                                     # noqa: E402
from dsceos_controller import DSCEOSConfig                         # noqa: E402
from graph_config import (authoritative_cluster_kwargs,            # noqa: E402
                          AUTHORITATIVE_COMMUNICATION_RADIUS,
                          AUTHORITATIVE_NEIGHBOUR_COUNT)
import realistic_cpes_catalog as rc                                # noqa: E402
from realistic_scenarios import REALISTIC_SCENARIOS                # noqa: E402
from run_realistic_scenario import build_signal_for, compute_fleet_scaling  # noqa: E402

# the ten physical archetypes, sampled independently
ARCHETYPES = [rc.NUCLEAR_BLOCK, rc.LIGNITE_BLOCK, rc.CCGT_BLOCK, rc.GAS_ENGINE_PEAKER,
              rc.BATTERY_STORAGE, rc.PUMPED_HYDRO, rc.HYDROGEN_P2X, rc.HEAT_PUMP_AGGREGATOR,
              rc.EV_V2G_HUB, rc.INDUSTRIAL_DEMAND_RESPONSE]
CONTROLLERS = [("dsceos", "D-SCEOS"),
               ("projected_gradient_hocbf", "DPG-HOCBF"),
               ("independent_tracking", "PD baseline")]
SCEN = REALISTIC_SCENARIOS["scenario_a_winter_morning_step"]
SCALAR_BYTES = 8
PAYLOAD_SCALARS = 9      # nine-scalar per-neighbour payload (B3, main text)
# Physical sampling period. One simulation time unit is one wall-clock minute (Section 7), so the
# dt=0.05 simulation-unit sampling step is 3 physical seconds. The communication rate is per PHYSICAL
# second and MUST divide by DT_PHYS_S, not by the raw 0.05 (that would overstate it by 60x).
DT_SIM_MIN = 0.05                 # sampling step in simulation units (= minutes)
DT_PHYS_S = 60.0 * DT_SIM_MIN     # = 3.0 physical seconds per sampling instant

# DESIGN-MATCHED RELATIVE LOADING (C-01 fix). Earlier versions used the fixed absolute
# Scenario-A request (2 GW / 0.3 GVAR) for every fleet size, so larger fleets solved an easier
# RELATIVE task (target/capacity fell from 30.8% at N=15 to 3.2% at N=120) and the size-scaling
# conclusion was invalid. We instead scale each sampled fleet's target to a FIXED FRACTION of that
# fleet's own capacity, equal to the N=15 reference loading (2 GW / 6.5 GW = 0.3077 active,
# 0.3 GVAR / 3.2 GVAR = 0.09375 reactive). This is the same ratio the main N=60 study realises with
# its target_multiplier=4 (8 GW / 26 GW = 0.3077), so all sizes now solve the same relative problem.
REF_LOAD_RATIO_P = 2.0 / 6.5      # 0.30769...  (N=15 Scenario-A active-power loading)
REF_LOAD_RATIO_Q = 0.3 / 3.2      # 0.09375     (N=15 Scenario-A reactive-power loading)


def independent_fleet(seed, n):
    """A fleet of n units sampled INDEPENDENTLY from the archetypes (no replication)."""
    rng = np.random.default_rng(7_000_000 + seed * 131 + n)
    idx = rng.integers(0, len(ARCHETYPES), size=n)
    physical = [ARCHETYPES[i] for i in idx]
    units = [rc.to_energy_flexibility_unit(p) for p in physical]
    return units, physical


def matched_scenario(physical):
    """Scenario A rescaled so target = REF_LOAD_RATIO * this fleet's own capacity (C-01)."""
    from dataclasses import replace
    cap_P = float(sum(p.P_max_GW for p in physical))
    cap_Q = float(sum(p.Q_max_GVAR for p in physical))
    tgt_P = REF_LOAD_RATIO_P * cap_P
    tgt_Q = REF_LOAD_RATIO_Q * cap_Q
    return replace(SCEN, target_P_max_GW=tgt_P, target_Q_max_GVAR=tgt_Q), cap_P, cap_Q, tgt_P, tgt_Q


def connected_graph(seed, n):
    rng = np.random.default_rng(8_000_000 + seed * 977 + n)
    for _ in range(500):
        layout = rng.uniform(0.0, 1.0, size=(n, 2))
        W = dv.make_fixed_local_graph(layout, AUTHORITATIVE_COMMUNICATION_RADIUS,
                                      AUTHORITATIVE_NEIGHBOUR_COUNT)
        deg = W.sum(axis=1)
        if np.any(deg <= 1e-12):
            continue
        if np.linalg.eigvalsh(np.diag(deg) - W)[1] > 1e-9:
            return W, layout
    raise RuntimeError("no connected graph drawn")


def run_one(seed, n, controller, trace=None):
    units, physical = independent_fleet(seed, n)
    W, layout = connected_graph(seed, n)
    ccfg = dv.ClusterConfig(n_thermal=0, n_storage=0, n_hydrogen=0, n_emobility=0, n_industrial=0,
                            initial_spread=0.10, initial_speed_scale=0.025,
                            **authoritative_cluster_kwargs())
    scen_matched, cap_P, cap_Q, tgt_P, tgt_Q = matched_scenario(physical)
    signal = build_signal_for(scen_matched, compute_fleet_scaling(physical))
    run_one.last_loading = dict(cap_P=round(cap_P, 4), cap_Q=round(cap_Q, 4),
                                target_P=round(tgt_P, 4), target_Q=round(tgt_Q, 4),
                                ratio_P=round(tgt_P / cap_P, 4), ratio_Q=round(tgt_Q / cap_Q, 4),
                                edges=int((W > 0).sum() // 2),
                                mean_neighbours=round(float((W > 0).sum(axis=1).mean()), 3))
    sim = dv.SimulationConfig(cluster=ccfg, controller=controller, scenario="static_request",
                              dsceos_config=DSCEOSConfig(), dt=0.05, horizon=float(SCEN.horizon_sim),
                              gateway_fraction=0.15, target_consensus_gain=0.18,
                              target_gateway_gain=0.85, compute_reference_optimum=False,
                              target_override=None, safety_filter=True, dpg_filter_always=True,
                              dpg_step_size=0.10, pd_kp=0.75, pd_kd=2.5)
    om, og, ol, ost = dv.make_units, dv.make_fixed_local_graph, dv.make_physical_layout, dv.select_target
    obase = dv.DistributedSCEOSController
    dv.make_units = lambda _c: units
    dv.make_physical_layout = lambda _n, _c: layout
    dv.make_fixed_local_graph = lambda _l, _r, _k: W
    dv.select_target = lambda _c: signal
    if trace is not None:
        class _Traced(obase):
            def __init__(self, problem, config=None):
                super().__init__(problem, config)
                self.qcqp_trace = trace
        dv.DistributedSCEOSController = _Traced
    try:
        res = dv.run_simulation(sim)
    finally:
        dv.make_units, dv.make_fixed_local_graph, dv.make_physical_layout, dv.select_target = om, og, ol, ost
        dv.DistributedSCEOSController = obase
    return res.summary, W, len(units)


def env():
    return dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
                platform=platform.platform(), machine=platform.machine())


def experiment_compositions(sizes, seeds):
    # Per-size checkpointing: each completed size block is persisted so a restart skips sizes already
    # done (the largest, N=120, is the slowest and most likely to be interrupted). Each size is an
    # independent, seed-deterministic computation, so this is byte-identical to an uninterrupted run.
    ckpt = "fleet_compositions.partial.json"
    out = []
    done_sizes = set()
    if os.path.exists(ckpt):
        try:
            _cp = json.load(open(ckpt))
            if _cp.get("seeds") == seeds:
                out = _cp["results"]
                done_sizes = set(b["N"] for b in out)
                print(f"  [resume] loaded sizes {sorted(done_sizes)} from {ckpt}", flush=True)
        except Exception as _e:
            print(f"  [resume] ignoring unreadable checkpoint: {_e}", flush=True)
    for n in sizes:
        if n in done_sizes:
            continue
        recs = {c: [] for c, _ in CONTROLLERS}
        loading = []
        for sd in range(seeds):
            for c, _ in CONTROLLERS:
                s, W, nn = run_one(sd, n, c)
                recs[c].append(dict(seed=sd, J_T=float(s["integrated_objective_value"]),
                                    cap_viol=float(s["max_capacity_violation"])))
            loading.append(run_one.last_loading)     # per-seed capacity/target/ratio + neighbours
            print(f"  N={n} seed {sd + 1}/{seeds}", flush=True)
        ds = np.array([r["J_T"] for r in sorted(recs["dsceos"], key=itemgetter("seed"))])
        # C-01 evidence: the realised relative loading is fixed across sizes (design-matched)
        ratios_P = [d["ratio_P"] for d in loading]
        # Per-controller max capacity violation. D-SCEOS is HOCBF-filtered so its applied force is
        # always admissible (structurally zero); the comparators can violate under a hard absolute
        # target on the largest fleet, which we report honestly rather than hide in a single max.
        per_ctrl_viol = {lbl: round(max(r["cap_viol"] for r in recs[c]), 8) for c, lbl in CONTROLLERS}
        blk = dict(N=n, seeds=seeds,
                   max_cap_violation=max(per_ctrl_viol.values()),
                   max_cap_violation_by_controller=per_ctrl_viol,
                   dsceos_max_cap_violation=per_ctrl_viol["D-SCEOS"],
                   mean_target_over_capacity_P=round(float(np.mean(ratios_P)), 4),
                   mean_target_over_capacity_Q=round(float(np.mean([d["ratio_Q"] for d in loading])), 4),
                   mean_capacity_P_GW=round(float(np.mean([d["cap_P"] for d in loading])), 3),
                   mean_neighbours=round(float(np.mean([d["mean_neighbours"] for d in loading])), 3))
        for c, lbl in CONTROLLERS[1:]:
            other = np.array([r["J_T"] for r in sorted(recs[c], key=itemgetter("seed"))])
            d = other - ds
            k = len(d)
            h = stats.t.ppf(0.975, k - 1) * d.std(ddof=1) / np.sqrt(k) if k > 1 else 0.0
            blk[lbl] = dict(win_rate_dsceos=round(float(np.mean(d > 0)), 4),
                            mean_ratio=round(float(np.mean(other / ds)), 4),
                            paired_diff_ci_low=round(float(d.mean() - h), 4),
                            paired_diff_ci_high=round(float(d.mean() + h), 4))
        out.append(blk)
        json.dump(dict(seeds=seeds, results=out), open(ckpt, "w"))
        print(f"N={n}: DPG win {blk['DPG-HOCBF']['win_rate_dsceos']:.2f} "
              f"ratio {blk['DPG-HOCBF']['mean_ratio']:.2f}x | "
              f"PD win {blk['PD baseline']['win_rate_dsceos']:.2f} "
              f"ratio {blk['PD baseline']['mean_ratio']:.2f}x | maxviol {blk['max_cap_violation']:.1e}")
    out = sorted(out, key=lambda b: b["N"])
    return dict(experiment="compositions", sizes=sizes, seeds=seeds, results=out, environment=env())


def experiment_profile(sizes):
    out = []
    n_seeds = 5      # M-07: several fleets per size, not a single representative draw
    for n in sizes:
        per_seed = []
        for sd in range(n_seeds):
            trace = []
            wall0 = time.perf_counter(); cpu0 = time.process_time()
            s, W, nn = run_one(sd, n, "dsceos", trace=trace)
            wall = time.perf_counter() - wall0; cpu = time.process_time() - cpu0
            tr = np.asarray(trace, dtype=float)  # (agent, fallback, solve_s, resid, slack, box, ball)
            agents = tr[:, 0].astype(int)
            steps = len(tr) // n
            t_ms = tr[:, 2] * 1e3
            # ideal-parallel critical path: slowest single agent-solve in any instant (NOT measured
            # parallel latency, which would require a real parallel runtime; labelled as such).
            worst_instant = max(t_ms[k * n:(k + 1) * n].max() for k in range(steps))
            # C-04 fix: communication goes to each ACTUAL neighbour, so use the NEIGHBOUR COUNT
            # (W>0).sum, not the distance-weighted degree W.sum. The rate is per PHYSICAL second:
            # one payload is exchanged per sampling instant, and one instant is DT_SIM_MIN=0.05
            # simulation units = DT_PHYS_S=3 s of wall-clock time (one sim unit = one minute, as used
            # throughout Section 7). Dividing by the raw 0.05 would treat the step as 0.05 s and
            # overstate the physical rate by a factor of 60.
            neighbours = (W > 0).sum(axis=1)
            bytes_per_s = PAYLOAD_SCALARS * SCALAR_BYTES * neighbours / DT_PHYS_S
            by_deg = {}
            for i in range(n):
                by_deg.setdefault(int(neighbours[i]), []).append(t_ms[agents == i].mean())
            per_seed.append(dict(
                summed_solve_time_s=float(tr[:, 2].sum()), process_cpu_s=float(cpu), wall_clock_s=float(wall),
                solve_ms=t_ms, worst_instant_ms=float(worst_instant), neighbours=neighbours,
                bytes_per_s=bytes_per_s, by_deg=by_deg, steps=steps, n_solves=len(tr)))
        # aggregate across the seeds
        all_t = np.concatenate([p["solve_ms"] for p in per_seed])
        all_bytes = np.concatenate([p["bytes_per_s"] for p in per_seed])
        merged_deg = {}
        for p in per_seed:
            for k, v in p["by_deg"].items():
                merged_deg.setdefault(k, []).extend(v)
        deg_profile = {str(k): round(float(np.mean(v)), 4) for k, v in sorted(merged_deg.items())}
        out.append(dict(N=n, seeds=n_seeds, steps=per_seed[0]["steps"],
                        summed_solve_time_s=round(float(np.mean([p["summed_solve_time_s"] for p in per_seed])), 4),
                        process_cpu_s=round(float(np.mean([p["process_cpu_s"] for p in per_seed])), 4),
                        wall_clock_s=round(float(np.mean([p["wall_clock_s"] for p in per_seed])), 3),
                        mean_solves_per_instant=round(per_seed[0]["n_solves"] / per_seed[0]["steps"], 3),
                        solve_ms_mean=round(float(all_t.mean()), 4),
                        solve_ms_p95=round(float(np.percentile(all_t, 95)), 4),
                        solve_ms_p99=round(float(np.percentile(all_t, 99)), 4),
                        solve_ms_max=round(float(all_t.max()), 4),
                        ideal_parallel_critical_path_ms_mean=round(float(np.mean([p["worst_instant_ms"] for p in per_seed])), 4),
                        ideal_parallel_critical_path_ms_max=round(float(max(p["worst_instant_ms"] for p in per_seed)), 4),
                        comm_bytes_per_s_per_agent=dict(
                            min=round(float(all_bytes.min()), 1),
                            mean=round(float(all_bytes.mean()), 1),
                            max=round(float(all_bytes.max()), 1)),
                        mean_neighbours=round(float(np.mean([p["neighbours"].mean() for p in per_seed])), 3),
                        max_neighbours=int(max(int(p["neighbours"].max()) for p in per_seed)),
                        solve_ms_by_neighbour_count=deg_profile))
        print(f"N={n}: mean {out[-1]['solve_ms_mean']:.3f} ms, p99 {out[-1]['solve_ms_p99']:.3f} ms, "
              f"crit-path(ideal) {out[-1]['ideal_parallel_critical_path_ms_mean']:.3f} ms, "
              f"CPU {out[-1]['process_cpu_s']:.2f} s, "
              f"comm {out[-1]['comm_bytes_per_s_per_agent']['mean']:.0f} B/s/agent "
              f"(nbr {out[-1]['mean_neighbours']:.1f})")
    return dict(experiment="profile", sizes=sizes, seeds_per_size=n_seeds, results=out, environment=env(),
                dt_sim_min=DT_SIM_MIN, dt_phys_s=DT_PHYS_S,
                note=("Solve times and process CPU are environment-specific and are NOT "
                      "asserted numerically. summed_solve_time_s is the sum of per-agent solve times; "
                      "process_cpu_s is measured process CPU; the ideal-parallel critical path is the "
                      "slowest single agent-solve per instant, an idealised lower bound on parallel "
                      "latency, not a measured parallel runtime. The communication load uses the "
                      "neighbour count (W>0) and is per PHYSICAL second: 9 scalars x 8 bytes per "
                      "neighbour per sampling instant, one instant = 3 physical seconds (dt=0.05 "
                      "simulation units = 0.05 minutes). It is exact for this protocol and depends "
                      "only on the local neighbour count, not on N."))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--experiment", choices=["compositions", "profile"], default="compositions")
    ap.add_argument("--sizes", default="15,30,60,120")
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    sizes = [int(x) for x in args.sizes.split(",")]
    if args.experiment == "compositions":
        payload = experiment_compositions(sizes, args.seeds)
        out = args.out or "fleet_compositions.json"
    else:
        payload = experiment_profile(sizes)
        out = args.out or "fleet_solver_profile.json"
    json.dump(payload, open(out, "w"), indent=1)
    if args.experiment == "compositions" and os.path.exists("fleet_compositions.partial.json"):
        os.remove("fleet_compositions.partial.json")
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
