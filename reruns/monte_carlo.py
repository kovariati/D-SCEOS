"""Monte Carlo statistical validation.

The submitted study reported deterministic point estimates from an exactly-at-rest initial state on
one fixed communication graph, with a single-parameter one-at-a-time sensitivity sweep. The review
asked for repeated samples, confidence intervals, several random connected graphs, measurement
noise / packet loss / variable delay, multi-parameter sensitivity, an equal tuning budget for the
comparators, and separate tuning and test scenarios. This harness supplies that evidence.

DESIGN
------
* PAIRED design. All randomisation is a pure function of the seed and is applied identically to the
  three controllers, so the per-seed differences are paired and the reported confidence intervals on
  the differences are valid. Changing the controller never changes the draw.
* Per-seed randomisation (experiment MC-A):
    - initial state          non-zero spread and non-zero initial rates;
    - cost coefficients      per-unit, per-component lognormal jitter on the operating-loss weights;
    - capacities             per-unit uniform jitter on the operating box and service capacity,
                             applied so that the rest point stays strictly inside the box;
    - communication graph    a fresh random geometric graph, RESAMPLED UNTIL CONNECTED, with the
                             released communication radius and neighbour cap. The realised
                             (edges, d_max, lambda_2, lambda_N) are recorded per seed.
* Channel impairments (experiment MC-B), applied to the neighbour messages only:
    - additive Gaussian measurement noise on the exchanged aggregate estimate;
    - Bernoulli packet loss  (a lost message is replaced by the neighbour's previous value);
    - variable delay         (a delayed message is served from a one-step buffer).
  VERIFIED SCOPE: these act on the D-SCEOS consensus channel ONLY. The comparators do not expose an
  equivalent aggregate-consensus entry point, so experiment B is a deliberately CONSERVATIVE,
  ONE-SIDED stress test of D-SCEOS against undisturbed comparators, not a budget-matched comparison.
  A matched-impairment protocol requires comparator-side channel models and is stated as future work.
* Multi-parameter sensitivity (experiment MC-C): a two-dimensional grid over the two weights the
  one-at-a-time sweep found most influential, evaluated for all three controllers, replacing the
  one-factor-at-a-time evidence with a joint grid.
* Tuning / test split. Scenario A is the scenario on which the comparators were tuned
  (reruns/baseline_tuning.py); Scenarios B and C are therefore reported SEPARATELY as
  held-out test scenarios, and the headline claim is the held-out one.

STATISTICS
----------
For each (scenario, controller): n, mean, sd, and a Student-t 95% confidence interval.
For each comparator: the PAIRED difference (comparator - D-SCEOS) with its 95% CI, the win rate,
and a Wilcoxon signed-rank test. A paired CI whose lower bound is above zero is the evidence that
the ordering is not an artefact of one particular draw.

USAGE
-----
    python3 reruns/monte_carlo.py --experiment A --cluster realistic_15 --seeds 30
    python3 reruns/monte_carlo.py --experiment B --cluster realistic_15 --seeds 30
    python3 reruns/monte_carlo.py --experiment C --cluster realistic_15 --seeds 10
    python3 reruns/monte_carlo.py --experiment A --cluster realistic_60 --seeds 30   # long

Results are written to monte_carlo_<experiment>_<cluster>.json together with full environment
metadata. Wall-clock timings are never asserted.
"""
from __future__ import annotations

import argparse
import copy
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
from realistic_cpes_catalog import build_realistic_units           # noqa: E402
from realistic_scenarios import REALISTIC_SCENARIOS                # noqa: E402
from run_realistic_scenario import build_signal_for, compute_fleet_scaling  # noqa: E402

CONTROLLERS = [("dsceos", "D-SCEOS"),
               ("projected_gradient_hocbf", "DPG-HOCBF"),
               ("independent_tracking", "PD baseline")]
SCEN = {"A": "scenario_a_winter_morning_step",
        "B": "scenario_b_wind_ramp_down_event",
        "C": "scenario_c_winter_balancing_mfrr"}
TUNING_SCENARIO = "A"          # comparators were tuned here -> B and C are held out

# perturbation magnitudes (documented, fixed, not tuned post hoc)
COST_SIGMA_LOG = 0.20          # lognormal sd on the operating-loss weights
CAP_JITTER = 0.10              # +/-10 % on the operating box and service capacity
INIT_SPREAD = 0.10             # initial-state spread
INIT_SPEED = 0.025             # initial-rate scale
NOISE_REL = 0.005              # measurement noise sd, relative to the aggregate scale
LOSS_PROB = 0.05               # Bernoulli packet loss
DELAY_PROB = 0.10              # probability that a message is served one step late


# --------------------------------------------------------------------------------------- graphs
def random_connected_graph(rng, n, radius, k):
    """Random geometric layout on the unit square, made connected by the deterministic
    nearest-component repair inside make_fixed_local_graph(). This is NOT rejection sampling:
    disconnected draws are repaired rather than discarded, so the resulting ensemble is a
    repaired-geometric ensemble. The connectivity check below is a post-condition assertion."""
    for _ in range(500):
        layout = rng.uniform(0.0, 1.0, size=(n, 2))
        W = dv.make_fixed_local_graph(layout, radius, k)
        deg = W.sum(axis=1)
        if np.any(deg <= 1e-12):
            continue
        L = np.diag(deg) - W
        ev = np.linalg.eigvalsh(L)
        if ev[1] > 1e-9:                                    # algebraic connectivity > 0
            return W, layout, float(ev[1]), float(ev[-1])
    raise RuntimeError("failed to draw a connected graph")


# ------------------------------------------------------------------------------- perturbations
def perturbed_arrays(seed, cluster):
    """Seed-determined perturbation of costs and capacities (identical for all controllers)."""
    units, physical = build_realistic_units(cluster)
    rng = np.random.default_rng(1_000_000 + seed)
    units = copy.deepcopy(units)
    out = []
    for u in units:
        lw = np.asarray(u.loss_weight, dtype=float) * rng.lognormal(0.0, COST_SIGMA_LOG,
                                                                    size=np.shape(u.loss_weight))
        g = 1.0 + rng.uniform(-CAP_JITTER, CAP_JITTER)
        lo = np.asarray(u.rest, dtype=float) + (np.asarray(u.lower, float) - np.asarray(u.rest, float)) * g
        hi = np.asarray(u.rest, dtype=float) + (np.asarray(u.upper, float) - np.asarray(u.rest, float)) * g
        out.append(type(u)(**{**{f: getattr(u, f) for f in u.__dataclass_fields__},
                              "loss_weight": lw, "lower": lo, "upper": hi,
                              "service_capacity": (None if u.service_capacity is None
                                                    else float(u.service_capacity) * g)}))
    return out, physical


class _Impaired:
    """Mixin factory: perturb the neighbour messages a controller consumes."""

    @staticmethod
    def wrap(base, seed, n, scale):
        rng = np.random.default_rng(5_000_000 + seed)
        buf = {"prev": None, "held": None}

        class _C(base):
            def update_estimators(self, *a, **kw):
                out = super().update_estimators(*a, **kw)
                yh = self.state.y_hat
                if buf["prev"] is None:
                    buf["prev"] = yh.copy()
                new = yh + rng.normal(0.0, NOISE_REL * scale, size=yh.shape)     # measurement noise
                lost = rng.random(n) < LOSS_PROB                                  # packet loss
                new[lost] = buf["prev"][lost]
                late = rng.random(n) < DELAY_PROB                                 # variable delay
                if buf["held"] is not None:
                    new[late] = buf["held"][late]
                buf["held"] = buf["prev"]
                buf["prev"] = new.copy()
                self.state.y_hat = new
                return out
        return _C


# ------------------------------------------------------------------------------------ one run
def one_run(seed, cluster, scen_key, controller, impaired, cfg_over=None):
    units, physical = perturbed_arrays(seed, cluster)
    n = len(units)
    kw = authoritative_cluster_kwargs()
    # C-03 fix: the initial-state draw in sample_initial_state() keys off cluster.seed. If we leave
    # cluster.seed at the single authoritative value, EVERY Monte-Carlo seed shares the same
    # standardized initial state (only rescaled by the seed-dependent capacity box), which contradicts
    # the declared per-seed initial-state randomisation. We therefore derive the cluster seed from the
    # Monte-Carlo seed with a dedicated offset, so positions and velocities are independently redrawn
    # per seed while the protocol parameters (radius, neighbour cap, spread scale) stay authoritative.
    kw = {**kw, "seed": 3_000_000 + seed}
    ccfg = dv.ClusterConfig(n_thermal=3, n_storage=3, n_hydrogen=3, n_emobility=3, n_industrial=3,
                            initial_spread=INIT_SPREAD, initial_speed_scale=INIT_SPEED, **kw)
    grng = np.random.default_rng(2_000_000 + seed)
    # Only the LAYOUT is randomised: the communication radius and the neighbour cap are taken by
    # name from the single authoritative source, so the drawn topologies differ from the released
    # one in geometry alone and not in the protocol parameters.
    W, layout, lam2, lamN = random_connected_graph(grng, n, AUTHORITATIVE_COMMUNICATION_RADIUS,
                                                   AUTHORITATIVE_NEIGHBOUR_COUNT)
    scen = REALISTIC_SCENARIOS[SCEN[scen_key]]
    # DESIGN-MATCHED LOADING (C-01 fix): the main realistic study scales the Scenario request by
    # target_multiplier=4 at N=60 so that target/capacity is the same as at N=15 (8 GW / 26 GW =
    # 2 GW / 6.5 GW = 30.8%). The Monte Carlo must use the SAME relative task, otherwise the larger
    # fleet solves an easier problem and the comparison is not size-matched. We therefore apply the
    # same 4x multiplier for the realistic_60 cluster, matching the released N=60 protocol exactly.
    if cluster == "realistic_60":
        from dataclasses import replace as _replace
        scen = _replace(scen, target_P_max_GW=scen.target_P_max_GW * 4.0,
                        target_Q_max_GVAR=scen.target_Q_max_GVAR * 4.0)
    dcfg = DSCEOSConfig(**(cfg_over or {}))
    signal = build_signal_for(scen, compute_fleet_scaling(physical))
    sim = dv.SimulationConfig(cluster=ccfg, controller=controller, scenario="static_request",
                              dsceos_config=dcfg, dt=0.05, horizon=float(scen.horizon_sim),
                              gateway_fraction=0.15, target_consensus_gain=0.18,
                              target_gateway_gain=0.85, compute_reference_optimum=False,
                              target_override=None, safety_filter=True, dpg_filter_always=True,
                              dpg_step_size=0.10, pd_kp=0.75, pd_kd=2.5)
    om, og, ol, ost = dv.make_units, dv.make_fixed_local_graph, dv.make_physical_layout, dv.select_target
    obase_ds = dv.DistributedSCEOSController
    dv.make_units = lambda _c: units
    dv.select_target = lambda _c: signal
    dv.make_physical_layout = lambda _n, _c: layout
    dv.make_fixed_local_graph = lambda _l, _r, _k: W
    if impaired:
        scale = float(np.mean(np.abs(dv.arrays_from_units(units)["upper"]))) or 1.0
        # HONEST SCOPE (verified): the wrapper hooks ``update_estimators``, which only the D-SCEOS
        # consensus loop owns. DPG-HOCBF does not run an aggregate-consensus estimator through this
        # entry point and the PD baseline is a plain function, so neither is affected. Experiment B
        # is therefore a ONE-SIDED WORST-CASE stress test of the D-SCEOS communication channel
        # against UNDISTURBED comparators -- deliberately conservative, and NOT budget-matched.
        # The guard below makes that explicit and fails loudly if the hook ever silently no-ops.
        if not hasattr(obase_ds, "update_estimators"):
            raise RuntimeError("impairment hook missing on the D-SCEOS controller")
        dv.DistributedSCEOSController = _Impaired.wrap(obase_ds, seed, n, scale)
    try:
        res = dv.run_simulation(sim)
    finally:
        dv.make_units, dv.make_fixed_local_graph, dv.make_physical_layout = om, og, ol
        dv.select_target = ost
        dv.DistributedSCEOSController = obase_ds
    s = res.summary
    # M-08: canonical edge-list fingerprint. Serialise the sorted upper-triangle edge set (i<j pairs
    # where W>0) and hash it, so the released artefact proves each seed's topology is distinct rather
    # than inferring it from edge counts alone.
    import hashlib as _hashlib
    _iu, _ju = np.where(np.triu(W > 0, k=1))
    _edge_str = ";".join(f"{int(a)}-{int(b)}" for a, b in zip(_iu, _ju))
    _edge_sha = _hashlib.sha256(_edge_str.encode()).hexdigest()[:16]
    return dict(J_T=float(s["integrated_objective_value"]),
                cap_viol=float(s["max_capacity_violation"]),
                agg_err=float(s["final_aggregate_error"]),
                energy=float(s["total_control_energy"]),
                edges=int((W > 0).sum() // 2), d_max=float(W.sum(axis=1).max()),
                edge_sha=_edge_sha,
                lambda2=lam2, lambdaN=lamN)


# ---------------------------------------------------------------------------------- statistics
def ci95(x):
    x = np.asarray(x, float); m = x.mean(); k = len(x)
    if k < 2:
        return float(m), float(m), float(m), 0.0
    h = stats.t.ppf(0.975, k - 1) * x.std(ddof=1) / np.sqrt(k)
    return float(m), float(m - h), float(m + h), float(x.std(ddof=1))


def summarise(records, seeds):
    out = {}
    for sc in sorted({_x["scenario"] for _x in records}):
        blk = {}
        by = {c: [_x for _x in records if _x["scenario"] == sc and _x["controller"] == c]
              for c, _ in CONTROLLERS}
        for c, lbl in CONTROLLERS:
            j = [_x["J_T"] for _x in sorted(by[c], key=itemgetter("seed"))]
            m, lo, hi, sd = ci95(j)
            blk[lbl] = dict(n=len(j), mean=round(m, 4), ci95_low=round(lo, 4), ci95_high=round(hi, 4),
                            sd=round(sd, 4), min=round(float(np.min(j)), 4), max=round(float(np.max(j)), 4),
                            max_cap_violation=round(max(_x["cap_viol"] for _x in by[c]), 8))
        ds = np.array([_x["J_T"] for _x in sorted(by["dsceos"], key=itemgetter("seed"))])
        for c, lbl in CONTROLLERS[1:]:
            other = np.array([_x["J_T"] for _x in sorted(by[c], key=itemgetter("seed"))])
            d = other - ds
            m, lo, hi, sd = ci95(d)
            try:
                w = float(stats.wilcoxon(other, ds, alternative="greater").pvalue)
            except Exception:
                w = float("nan")
            blk[f"paired {lbl} - D-SCEOS"] = dict(
                mean_diff=round(m, 4), ci95_low=round(lo, 4), ci95_high=round(hi, 4),
                win_rate_dsceos=round(float(np.mean(d > 0)), 4),
                wilcoxon_p_greater=(round(w, 6) if np.isfinite(w) else None),
                ratio_mean=round(float(np.mean(other / ds)), 4))
        out[sc] = blk
    return out


def env():
    return dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
                platform=platform.platform(), machine=platform.machine())


# ---------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--experiment", choices=["A", "B", "C"], default="A",
                    help="A = state/cost/capacity/graph; B = A + channel impairments; C = 2D grid")
    ap.add_argument("--cluster", choices=["realistic_15", "realistic_60"], default="realistic_15")
    ap.add_argument("--seeds", type=int, default=30)
    ap.add_argument("--scenarios", default="A,B,C")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    scens = [s.strip().upper() for s in args.scenarios.split(",") if s.strip()]
    tag = "N15" if args.cluster == "realistic_15" else "N60"
    t0 = time.time()

    if args.experiment in ("A", "B"):
        impaired = args.experiment == "B"
        # Per-seed checkpointing so a long N=60 run survives interruption: completed seeds are
        # persisted to a .partial file and reloaded on restart (byte-identical, since each seed is a
        # pure function of its index). Delete the .partial file to force a clean run.
        ckpt = f"monte_carlo_{args.experiment}_{tag}.partial.json"
        recs = []
        done_seeds = set()
        if os.path.exists(ckpt):
            try:
                _cp = json.load(open(ckpt))
                if (_cp.get("cluster") == args.cluster and _cp.get("experiment") == args.experiment):
                    recs = _cp["records"]
                    done_seeds = set(r["seed"] for r in recs)
                    print(f"  [resume] loaded {len(done_seeds)} completed seeds from {ckpt}", flush=True)
            except Exception as _e:
                print(f"  [resume] ignoring unreadable checkpoint: {_e}", flush=True)
        for sd in range(args.seeds):
            if sd in done_seeds:
                continue
            for sc in scens:
                for c, lbl in CONTROLLERS:
                    r = one_run(sd, args.cluster, sc, c, impaired)
                    r.update(seed=sd, scenario=sc, controller=c, label=lbl)
                    recs.append(r)
            json.dump(dict(cluster=args.cluster, experiment=args.experiment, records=recs),
                      open(ckpt, "w"))
            print(f"  seed {sd + 1}/{args.seeds} done ({time.time() - t0:.0f}s)", flush=True)
        recs = sorted(recs, key=lambda r: (r["seed"], r["scenario"], r["controller"]))
        stat = summarise(recs, args.seeds)
        payload = dict(experiment=args.experiment, cluster=args.cluster, seeds=args.seeds,
                       impairments=(dict(noise_rel=NOISE_REL, loss_prob=LOSS_PROB,
                                         delay_prob=DELAY_PROB,
                                         applied_to=["dsceos"],
                                         scope=("one-sided: comparators are NOT impaired, so this is "
                                                "a conservative stress test of D-SCEOS, not a "
                                                "budget-matched comparison"))
                                    if impaired else None),
                       perturbations=dict(cost_sigma_log=COST_SIGMA_LOG, capacity_jitter=CAP_JITTER,
                                          initial_spread=INIT_SPREAD, initial_speed_scale=INIT_SPEED,
                                          graph="random geometric layout, made connected by deterministic nearest-component repair (not rejection sampling)"),
                       tuning_scenario=TUNING_SCENARIO,
                       held_out_scenarios=[s for s in scens if s != TUNING_SCENARIO],
                       graph_realisations=dict(
                           edges=[r["edges"] for r in recs if r["controller"] == "dsceos"],
                           d_max=[r["d_max"] for r in recs if r["controller"] == "dsceos"],
                           edge_sha=[r["edge_sha"] for r in recs if r["controller"] == "dsceos"],
                           n_distinct_topologies=len(set(r["edge_sha"] for r in recs
                                                         if r["controller"] == "dsceos")),
                           lambda2=[round(r["lambda2"], 5) for r in recs if r["controller"] == "dsceos"]),
                       statistics=stat, records=recs, environment=env(),
                       wall_clock_s=round(time.time() - t0, 1))
    else:
        grid_w = [0.33, 0.67, 1.00, 1.50]
        grid_s = [0.05, 0.15, 0.30, 0.50]
        cells = []
        for wy in grid_w:
            for ls in grid_s:
                row = {"w_bar_y": wy, "lambda_s": ls}
                for c, lbl in CONTROLLERS:
                    js = [one_run(sd, args.cluster, "A", c, False,
                                  cfg_over=dict(aggregate_tracking_weight=wy, sharing_weight=ls))["J_T"]
                          for sd in range(args.seeds)]
                    m, lo, hi, _ = ci95(js)
                    row[lbl] = dict(mean=round(m, 4), ci95_low=round(lo, 4), ci95_high=round(hi, 4))
                row["dsceos_lowest"] = all(row["D-SCEOS"]["mean"] <= row[l]["mean"]
                                           for _, l in CONTROLLERS)
                cells.append(row)
                print(f"  w_bar={wy} lambda_s={ls} done ({time.time() - t0:.0f}s)", flush=True)
        payload = dict(experiment="C", cluster=args.cluster, seeds=args.seeds,
                       grid=dict(w_bar_y=grid_w, lambda_s=grid_s), cells=cells,
                       dsceos_lowest_cells=int(sum(c["dsceos_lowest"] for c in cells)),
                       n_cells=len(cells), environment=env(),
                       wall_clock_s=round(time.time() - t0, 1))

    out = args.out or f"monte_carlo_{args.experiment}_{tag}.json"
    json.dump(payload, open(out, "w"), indent=1)
    # success -> remove the resume checkpoint so the next run starts clean
    _ckpt = f"monte_carlo_{args.experiment}_{tag}.partial.json"
    if os.path.exists(_ckpt):
        os.remove(_ckpt)
    print(f"\n-> {out}   ({time.time() - t0:.0f}s)")
    if args.experiment in ("A", "B"):
        for sc, blk in payload["statistics"].items():
            held = " (held-out)" if sc != TUNING_SCENARIO else " (tuning scenario)"
            print(f"\n=== Scenario {sc}{held} ===")
            for _, lbl in CONTROLLERS:
                b = blk[lbl]
                print(f"  {lbl:<12} J_T = {b['mean']:8.4f}  95% CI [{b['ci95_low']:.4f}, "
                      f"{b['ci95_high']:.4f}]  sd={b['sd']:.4f}  maxviol={b['max_cap_violation']:.1e}")
            for _, lbl in CONTROLLERS[1:]:
                d = blk[f"paired {lbl} - D-SCEOS"]
                print(f"  paired {lbl:<12} diff = {d['mean_diff']:+8.4f} 95% CI "
                      f"[{d['ci95_low']:+.4f}, {d['ci95_high']:+.4f}]  win={d['win_rate_dsceos']:.2f}"
                      f"  p={d['wilcoxon_p_greater']}")
    else:
        print(f"D-SCEOS lowest mean in {payload['dsceos_lowest_cells']}/{payload['n_cells']} grid cells")


if __name__ == "__main__":
    main()
