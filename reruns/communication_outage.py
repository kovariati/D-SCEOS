"""baseline tuning0: temporary communication disconnection for a few time steps.
Two failure modes during a mid-horizon window:
  (i)  FULL BLACKOUT  - all neighbour messages lost (adjacency -> 0)
  (ii) PARTITION      - graph split into two disconnected components
No topology *reconfiguration* is applied (the outage is transient, not a
commissioning change): K_y is held fixed and the estimator states are CONTINUED
(not reset), so we test native re-synchronization after links return.
Records estimator disagreement, aggregate error and capacity feasibility."""
import os as _os, sys as _sys
_ROOT=_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
[_sys.path.insert(0,_p) for _p in (_ROOT,_os.path.join(_ROOT,"code")) if _p not in _sys.path]
_os.chdir(_ROOT)  # results/ paths resolve against the package root
import json, numpy as np
from realistic_cpes_catalog import build_realistic_units, compute_fleet_scaling
from realistic_scenarios import REALISTIC_SCENARIOS, build_signal_for
import dsceos_validation as dv
from dsceos_validation import (ClusterConfig, arrays_from_units, make_aggregate_blocks,
    make_fixed_local_graph, make_physical_layout, aggregate_output, gateway_mask,
    update_peer_estimate, row_norm, capacity_violation)
from dsceos_controller import DSCEOSProblemData, DSCEOSConfig, DistributedSCEOSController
from hocbf_safety_filter import HOCBFSafetyFilter
from graph_config import (AUTHORITATIVE_COMMUNICATION_RADIUS as _R045,
                          AUTHORITATIVE_NEIGHBOUR_COUNT as _K4,
                          AUTHORITATIVE_LAYOUT_SPREAD as _LS,
                          AUTHORITATIVE_SEED as _SEED)  # single authoritative graph config

def build(cluster="realistic_15"):
    units,physical=build_realistic_units(cluster)
    arrays=arrays_from_units(units)
    scaling=compute_fleet_scaling(physical)
    ccluster=ClusterConfig(n_thermal=3,n_storage=3,n_hydrogen=3,n_emobility=3,
        n_industrial=3,seed=_SEED,initial_spread=0.0,initial_speed_scale=0.0,
        communication_radius=_R045,neighbour_count=_K4,layout_spread=_LS)
    layout=make_physical_layout(len(units),ccluster)
    W=make_fixed_local_graph(layout,ccluster.communication_radius,ccluster.neighbour_count)
    Ablk=make_aggregate_blocks(arrays["aggregate_weight"])
    return units,arrays,scaling,W,Ablk,layout

def partition_graph(W):
    """Zero out edges crossing the median index -> two components."""
    n=W.shape[0]; Wp=W.copy(); h=n//2
    Wp[:h,h:]=0.0; Wp[h:,:h]=0.0
    return Wp

def run_blackout(scen_key, mode, win_min, cluster="realistic_15"):
    scenario=REALISTIC_SCENARIOS[{"A":"scenario_a_winter_morning_step",
                                  "B":"scenario_b_wind_ramp_down_event"}[scen_key]]
    units,arrays,scaling,W,Ablk,layout=build(cluster)
    signal=build_signal_for(scenario,scaling)
    n=len(units); dt=0.05; horizon=float(scenario.horizon_sim)
    steps=int(round(horizon/dt))
    cfg=DSCEOSConfig(aggregate_tracking_weight=2/3,loss_weight_scale=0.03,
        sharing_weight=0.15,internal_weight=0.03,adaptive_consensus_gain=True,
        gershgorin_safety_factor=0.95)
    problem=DSCEOSProblemData(aggregate_blocks=Ablk,lower=arrays["lower"],upper=arrays["upper"],
        rest=arrays["rest"],loss_weight=arrays["loss_weight"],masses=arrays["masses"],
        dampings=arrays["dampings"],force_limits=arrays["force_limits"],
        service_selector=np.tile(np.array([[1.0,0.0]]),(n,1)),
        service_capacity=arrays["service_capacity"],adjacency=W.copy())
    ctl=DistributedSCEOSController(problem,cfg)
    safety=HOCBFSafetyFilter(problem,cfg)
    p=arrays["rest"].copy(); v=np.zeros_like(p)
    gateways=gateway_mask(n,0.15)
    target_est=np.tile(aggregate_output(Ablk,p)[None,:],(n,1)); target_est[gateways]=signal.position(0.0)
    W_good=W.copy()
    W_bad=(np.zeros_like(W) if mode=="blackout" else partition_graph(W))
    k1=steps//3; k2=k1+int(round(win_min/dt))   # outage window
    spread=[]; agg=[]; cap=[]; Wsum_target=W_good  # for target consensus too
    for k in range(steps):
        t=k*dt
        outage = (k1<=k<k2)
        Wnow = W_bad if outage else W_good
        problem.adjacency[:]=Wnow           # controller reads this in update_estimators
        Wt = Wnow                            # target-estimate consensus also degraded
        y=signal.position(t)
        target_est=update_peer_estimate(target_est,y,Wt,gateways,0.18,0.85)
        u,diag=ctl.control(p,v,target_est)
        acc=(u-arrays["dampings"][:,None]*v)/arrays["masses"][:,None]
        v=v+dt*acc; p=p+dt*v
        cv=capacity_violation(p,arrays["lower"],arrays["upper"])
        p=np.minimum(np.maximum(p,arrays["lower"]-0.05),arrays["upper"]+0.05)
        yhat=ctl.state.y_hat
        spread.append(float(np.max(row_norm(yhat-np.mean(yhat,axis=0,keepdims=True)))))
        agg.append(float(np.linalg.norm(aggregate_output(Ablk,p)-y)))
        cap.append(cv)
    spread=np.array(spread); agg=np.array(agg); cap=np.array(cap)
    pre=float(np.mean(spread[max(0,k1-20):k1]))
    peak=float(np.max(spread[k1:k2+1]))
    # recovery: steps after window end until spread returns within 1.2*pre
    rec=None
    for k in range(k2,len(spread)):
        if spread[k]<=max(1.2*pre,pre+1e-4): rec=(k-k2)*dt; break
    return dict(scenario=scen_key,mode=mode,win_min=win_min,
                pre_spread=pre,peak_spread=peak,
                spread_ratio=peak/max(pre,1e-9),
                recovery_min=rec,max_cap_violation=float(np.max(cap)),
                final_agg_err=float(agg[-1]))

print(f"{'scen':>5}{'mode':>10}{'win[min]':>9}{'pre_spread':>12}{'peak_spread':>12}{'x':>7}{'recov[min]':>11}{'cap_viol':>10}")
rows=[]
for sk in ["A","B"]:
    for mode in ["blackout","partition"]:
        r=run_blackout(sk,mode,win_min=3.0)
        rows.append(r)
        print(f"{sk:>5}{mode:>10}{r['win_min']:>9.1f}{r['pre_spread']:>12.5f}"
              f"{r['peak_spread']:>12.5f}{r['spread_ratio']:>7.1f}"
              f"{(r['recovery_min'] if r['recovery_min'] is not None else -1):>11.2f}{r['max_cap_violation']:>10.1e}")
json.dump(rows,open("communication_outage.json","w"),indent=2)
