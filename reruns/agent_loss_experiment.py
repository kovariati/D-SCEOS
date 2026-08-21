"""agent-loss: agent-loss numerical example with a realistic scenario signal.
Removes one unit at mid-horizon, rebuilds the fixed graph, re-tunes the
Gershgorin K_y, and reports the transient recovery + safety during the event."""
import os as _os, sys as _sys
_ROOT=_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
[_sys.path.insert(0,_p) for _p in (_ROOT,_os.path.join(_ROOT,"code")) if _p not in _sys.path]
_os.chdir(_ROOT)  # results/ paths resolve against the package root
import json, numpy as np
import os, uuid, platform, tempfile, hashlib
from dataclasses import replace
from realistic_cpes_catalog import build_realistic_units, compute_fleet_scaling
from realistic_scenarios import REALISTIC_SCENARIOS, build_signal_for
import dsceos_validation as dv
from dsceos_validation import ClusterConfig, SimulationConfig, run_simulation
from dsceos_controller import DSCEOSConfig
from graph_config import (AUTHORITATIVE_COMMUNICATION_RADIUS as _R045,
                          AUTHORITATIVE_NEIGHBOUR_COUNT as _K4,
                          AUTHORITATIVE_LAYOUT_SPREAD as _LS,
                          AUTHORITATIVE_SEED as _SEED)  # single authoritative graph config

# Provenance token (Design note, fail-closed agent-loss evidence): a per-run id that a downstream
# checker requires to appear in the freshly written PROVENANCE SIDECAR. A caller (run_all.py) sets
# DSCEOS_RUN_ID; the checker then knows the producer -- not a `touch` on a stale packaged file -- actually
# (re)wrote the artefacts this run. Content-based, so it is not defeatable by a timestamp `touch`.
# S-03: the token and the environment metadata are written to a SEPARATE sidecar file, NOT into the
# numeric JSON. This keeps the numeric agent_loss_{A,B}.json byte-reproducible on a fixed stack (a random
# per-run id would otherwise make byte-identity impossible), while the volatile provenance lives beside it.
_RUN_ID = os.environ.get("DSCEOS_RUN_ID") or uuid.uuid4().hex
_ENV_META = {"numpy": np.__version__,
             "scipy": __import__("scipy").__version__,
             "python": platform.python_version(),
             "platform": platform.platform()}

def _atomic_write_json(path, obj):
    # atomic temp-file -> os.replace so a reader never sees a half-written file, and a fresh inode is
    # created each run (defeats mtime-only spoofing).
    d = os.path.dirname(os.path.abspath(path)) or "."
    fd, tmp = tempfile.mkstemp(dir=d, prefix=".tmp_", suffix=".json")
    try:
        with os.fdopen(fd, "w") as fh:
            json.dump(obj, fh, indent=2)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)

SCEN={"A":"scenario_a_winter_morning_step","B":"scenario_b_wind_ramp_down_event"}

def run_agent_loss(scen_key, cluster="realistic_15"):
    scenario=REALISTIC_SCENARIOS[SCEN[scen_key]]
    units,physical=build_realistic_units(cluster)
    scaling=compute_fleet_scaling(physical)
    signal=build_signal_for(scenario,scaling)
    om,os=dv.make_units,dv.select_target
    dv.make_units=lambda cfg: units
    dv.select_target=lambda cfg: signal
    ccfg=DSCEOSConfig(aggregate_tracking_weight=2/3,loss_weight_scale=0.03,
        sharing_weight=0.15,internal_weight=0.03,adaptive_consensus_gain=True,
        gershgorin_safety_factor=0.95,aggregate_consensus_gain=0.42)
    ccluster=ClusterConfig(n_thermal=3,n_storage=3,n_hydrogen=3,n_emobility=3,
        n_industrial=3,seed=_SEED,initial_spread=0.0,initial_speed_scale=0.0,
        communication_radius=_R045,neighbour_count=_K4,layout_spread=_LS)
    cfg=SimulationConfig(cluster=ccluster,controller="dsceos",
        scenario="fault_tolerance",   # triggers run_fault_tolerance_simulation
        dsceos_config=ccfg,dt=0.05,horizon=float(scenario.horizon_sim),
        gateway_fraction=0.15,target_consensus_gain=0.18,target_gateway_gain=0.85,
        compute_reference_optimum=False,target_override=None,safety_filter=True)
    res=run_simulation(cfg)
    dv.make_units,dv.select_target=om,os
    return res,scenario

for sk in ["A","B"]:
    # Fail-closed provenance: remove any stale packaged artefact AND its sidecar BEFORE producing, so a
    # downstream check cannot pass on a pre-existing file if this run does not actually write a new one.
    _outpath = f"agent_loss_{sk}.json"
    _provpath = f"agent_loss_{sk}.provenance.json"
    for _p in (_outpath, _provpath):
        if os.path.exists(_p):
            os.remove(_p)
    res,scenario=run_agent_loss(sk)
    s=res.summary; m=res.metrics
    dt=0.05
    drop_t=s["drop_time"]; k_drop=int(round(drop_t/dt))
    agg=m["aggregate_error"]; est=m["estimator_spread"]; cap=m["capacity_violation"]
    # recovery: FIRST-RETURN time -- the first sample after the drop at which the aggregate
    # error is back within 10% of the pre-drop level (agg <= 1.1*pre). This is a first-return
    # criterion, NOT a sustained dwell-window settling time.
    pre=float(np.mean(agg[max(0,k_drop-20):k_drop]))
    post_peak=float(np.max(agg[k_drop:k_drop+40]))
    # first sample with agg <= 1.1*pre after the drop (first return, not sustained settling)
    settle=None
    for k in range(k_drop, len(agg)):
        if agg[k]<=1.1*pre:
            settle=(k-k_drop)*dt; break
    print(f"\n=== Scenario {sk} agent-loss ===")
    print(f"  n_agents: {s['n_agents_initial']} -> {s['n_agents_final']} (dropped idx {s['dropped_agent']} at t={drop_t:.2f} min)")
    print(f"  Gershgorin after reconfig: d_max={s['gershgorin_max_weighted_degree']:.4f}, "
          f"lambda_N_bound={s['gershgorin_lambda_N_bound']:.4f}, K_y={s['active_aggregate_consensus_gain']:.5f}")
    print(f"  max capacity violation over WHOLE run (incl. transition): {s['max_capacity_violation']:.3e}")
    print(f"  aggregate err: pre-drop mean={pre:.5f}, post-drop peak={post_peak:.5f}, "
          f"first return within 10% in {settle:.2f} min" if settle else "  no first return within horizon")
    print(f"  final aggregate error={s['final_aggregate_error']:.5f}, integrated J_T={s['integrated_objective_value']:.4f}")
    # Numeric artefact: byte-reproducible on a fixed stack (NO volatile run_id/environment here).
    _atomic_write_json(_outpath,
              {"scenario":sk,
               "n_before":s["n_agents_initial"],"n_after":s["n_agents_final"],
               "dropped":s["dropped_agent"],"drop_time_min":drop_t,
               "Ky_after":s["active_aggregate_consensus_gain"],
               "dmax_after":s["gershgorin_max_weighted_degree"],
               "max_cap_violation":s["max_capacity_violation"],
               "agg_pre":pre,"agg_post_peak":post_peak,"first_return_min":settle,
               "final_agg_err":s["final_aggregate_error"],"J_T":s["integrated_objective_value"]})
    # Provenance sidecar: per-run id + environment (NOT byte-reproducible by design). It carries a
    # SHA-256 CONSISTENCY binding bound_sha256 = sha256(run_id || numeric_bytes) plus the plain numeric
    # hash and size. This is a consistency check computed from PUBLIC data (NOT a secret-keyed MAC) and it
    # does NOT by itself prove fresh generation: a no-op could recompute it over stale bytes. FRESHNESS is
    # instead enforced upstream by run_all's pre-clean, which deletes these files before the producer runs
    # so a no-op/crashed producer leaves them missing and the stage fails closed (audit source-integrity C-01). The
    # binding here only catches a mismatched/swapped numeric+sidecar pair.
    with open(_outpath, "rb") as _fh:
        _numbytes = _fh.read()
    _bound = hashlib.sha256(_RUN_ID.encode() + b"\x00" + _numbytes).hexdigest()
    _atomic_write_json(_provpath,
              {"scenario":sk,"run_id":_RUN_ID,"environment":_ENV_META,
               "of":_outpath,
               "numeric_sha256":hashlib.sha256(_numbytes).hexdigest(),
               "numeric_bytes":len(_numbytes),
               "bound_sha256":_bound})
