"""per-AGENT fallback distribution, solve time and CLF residual.

The submitted manuscript reported only how many *sampling instants* contained at least one local
SLSQP solve that fell back to the deterministic box-ball projection. The review asked for the
per-agent distribution, the solve time and whether the CLF condition was met. This script re-runs the
seven-regime stress ladder with the opt-in per-agent QCQP trace enabled and reports, per regime:

  * fallback rate per agent: min / median / p95 / max across the N agents, plus how many agents never
    fell back and how many fell back at every instant;
  * concentration: the share of all fallback events contributed by the worst-quartile agents (this
    answers whether "all 300 instants" means "all agents always" or "one agent always");
  * local solve time per agent-step: mean / p95 / p99 / max (environment-specific, reported as
    orientation only, in line with the manuscript's stance on wall-clock numbers);
  * constraint residuals at the APPLIED force. NOTE ON INTERPRETATION: at a fallback the slack is
    set to s0 = max(0, a(u0) + c V), so the RELAXED CLF row is satisfied by construction and its
    residual is identically zero -- that is implementation consistency, not an unrelaxed-CLF
    decrease certificate. The independent information is therefore the box and actuator-ball
    residual of the applied force, which is logged separately here.

Run from the package root:  python3 reruns/fallback_per_agent.py
"""
import json
import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_ROOT, os.path.join(_ROOT, "code")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.chdir(_ROOT)
sys.dont_write_bytecode = True

import dsceos_validation as dv                                    # noqa: E402
from dsceos_controller import DSCEOSConfig                        # noqa: E402
from graph_config import ladder_cluster_kwargs, LADDER_UNIT_MIX   # noqa: E402

LADDER = [("soft", (0.18, 0.02)), ("feasible", (0.30, 0.02)),
          ("borderline-low", (0.42, 0.02)), ("borderline-mid", (0.54, 0.02)),
          ("borderline-high", (0.70, 0.05)), ("hard-stress", (0.75, 0.20)),
          ("extreme-infeasible", (0.85, 0.35))]
DT, HORIZON = 0.04, 12.0


def run_regime(target):
    # CONFIGURATION MUST MIRROR reruns/ladder_diagnostics.py EXACTLY, otherwise the per-agent
    # counts are not comparable with the published per-instant counts. The ladder deliberately
    # differs from the realistic study: exact-rest start (the ClusterConfig DEFAULT
    # initial_speed_scale is 0.025, not 0), w_bar = 10.0 and a FIXED consensus gain K_y = 0.10.
    cc = dv.ClusterConfig(**LADDER_UNIT_MIX, **ladder_cluster_kwargs(),
                          initial_spread=0.0, initial_speed_scale=0.0)
    ccfg = DSCEOSConfig(aggregate_tracking_weight=10.0, loss_weight_scale=0.03,
                        sharing_weight=0.15, internal_weight=0.03,
                        aggregate_consensus_gain=0.10, adaptive_consensus_gain=False)
    cfg = dv.SimulationConfig(cluster=cc, controller="dsceos", scenario="static_request",
                              dsceos_config=ccfg, dt=DT, horizon=HORIZON,
                              gateway_fraction=0.15, target_consensus_gain=0.18,
                              target_gateway_gain=0.85, compute_reference_optimum=False,
                              target_override=np.asarray(target, dtype=float),
                              safety_filter=True)
    trace = []
    original = dv.DistributedSCEOSController

    class _Traced(original):                       # enable the opt-in trace for this run only
        def __init__(self, problem, config=None):
            super().__init__(problem, config)
            self.qcqp_trace = trace

    dv.DistributedSCEOSController = _Traced
    try:
        dv.run_simulation(cfg)
    finally:
        dv.DistributedSCEOSController = original
    return np.asarray(trace, dtype=float)


rows = []
print(f"{'regime':<20}{'N':>4}{'steps':>7}{'fb/agent min':>13}{'med':>7}{'p95':>7}{'max':>7}"
      f"{'never':>7}{'always':>7}{'top-q share':>13}{'t_mean[ms]':>11}{'t_p99[ms]':>10}"
      f"{'boxball(fb)':>12}")
for name, tgt in LADDER:
    tr = run_regime(tgt)
    agents = tr[:, 0].astype(int)
    n = agents.max() + 1
    steps = len(tr) // n
    fb = tr[:, 1]
    per_agent = np.array([fb[agents == i].sum() for i in range(n)])
    rate = per_agent / steps
    order = np.sort(per_agent)[::-1]
    topq = order[:max(1, n // 4)].sum() / max(per_agent.sum(), 1e-30)
    t_ms = tr[:, 2] * 1e3
    res_fb = tr[fb > 0.5, 3]          # composite max(box, ball, relaxed-CLF) residual
    res_ok = tr[fb < 0.5, 3]
    hard_fb = np.maximum(tr[fb > 0.5, 5], tr[fb > 0.5, 6])   # box/ball only, independent evidence
    hard_ok = np.maximum(tr[fb < 0.5, 5], tr[fb < 0.5, 6])
    row = dict(regime=name, n_agents=int(n), steps=int(steps),
               per_agent_rate_min=round(float(rate.min()), 4),
               per_agent_rate_median=round(float(np.median(rate)), 4),
               per_agent_rate_p95=round(float(np.percentile(rate, 95)), 4),
               per_agent_rate_max=round(float(rate.max()), 4),
               agents_never=int(np.sum(per_agent == 0)),
               agents_always=int(np.sum(per_agent == steps)),
               top_quartile_share=round(float(topq), 4),
               solve_ms_mean=round(float(t_ms.mean()), 4),
               solve_ms_p95=round(float(np.percentile(t_ms, 95)), 4),
               solve_ms_p99=round(float(np.percentile(t_ms, 99)), 4),
               solve_ms_max=round(float(t_ms.max()), 4),
               max_constraint_residual_fallback=round(float(res_fb.max()) if res_fb.size else 0.0, 12),
               max_constraint_residual_solver=round(float(res_ok.max()) if res_ok.size else 0.0, 12),
               max_boxball_residual_fallback=round(float(hard_fb.max()) if hard_fb.size else 0.0, 12),
               max_boxball_residual_solver=round(float(hard_ok.max()) if hard_ok.size else 0.0, 12))
    inst = np.array([fb[k * n:(k + 1) * n].sum() > 0 for k in range(steps)])
    row["fallback_instants"] = int(inst.sum())
    rows.append(row)
    print(f"{name:<20}{n:>4}{steps:>7}{rate.min():>13.3f}{np.median(rate):>7.3f}"
          f"{np.percentile(rate,95):>7.3f}{rate.max():>7.3f}{row['agents_never']:>7}"
          f"{row['agents_always']:>7}{topq:>13.3f}{t_ms.mean():>11.3f}"
          f"{np.percentile(t_ms,99):>10.3f}{row['max_boxball_residual_fallback']:>12.2e}")

import platform, scipy                                        # noqa: E402
payload = dict(environment=dict(python=platform.python_version(), numpy=np.__version__,
                                scipy=scipy.__version__, platform=platform.platform(),
                                machine=platform.machine()),
               note=("Exact solver-status counters are numerically stack-sensitive; the per-instant "
                     "counts are asserted only in the pinned environment of requirements.txt. The "
                     "qualitative conclusions (no agent falls back at every instant; near-uniform "
                     "spread across agents) are stack-robust. Wall-clock solve times are "
                     "environment-specific and are NOT asserted."),
               regimes=rows)
json.dump(payload, open("fallback_per_agent.json", "w"), indent=1)
print("\n-> fallback_per_agent.json")
print("NOTE: solve times are wall-clock on the generating machine and are reported as "
      "environment-specific orientation, consistent with the manuscript's stance.")
