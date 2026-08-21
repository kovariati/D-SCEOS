"""Computational timing and centralized benchmark: timing + centralized benchmark with ANALYTIC GRADIENT and
a proper reachability check (design note). Reports finite-difference vs analytic-gradient
solve, target reachability, per-node D-SCEOS timing (honestly labelled)."""
import os as _os, sys as _sys
_ROOT=_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
[_sys.path.insert(0,_p) for _p in (_ROOT,_os.path.join(_ROOT,"code")) if _p not in _sys.path]
_os.chdir(_ROOT)  # results/ paths resolve against the package root
import json, time, numpy as np
from realistic_cpes_catalog import build_realistic_units
from realistic_scenarios import REALISTIC_SCENARIOS, build_signal_for
from dsceos_validation import (ClusterConfig, arrays_from_units, make_aggregate_blocks,
    make_fixed_local_graph, make_physical_layout, aggregate_output, utilization, objective_terms)
from dsceos_controller import DSCEOSProblemData, DSCEOSConfig, DistributedSCEOSController
from scipy.optimize import minimize
from benchmark_objective import obj_and_grad, reachable_range
from graph_config import (AUTHORITATIVE_COMMUNICATION_RADIUS as _R045,
                          AUTHORITATIVE_NEIGHBOUR_COUNT as _K4,
                          AUTHORITATIVE_LAYOUT_SPREAD as _LS,
                          AUTHORITATIVE_SEED as _SEED)  # single authoritative graph config



if __name__ == "__main__":
    rows=[]
    for cluster,tag,tm in [("realistic_15","N15",1.0),("realistic_60","N60",4.0)]:
        units,physical=build_realistic_units(cluster); arr=arrays_from_units(units); n=len(units)
        cc=ClusterConfig(n_thermal=3,n_storage=3,n_hydrogen=3,n_emobility=3,n_industrial=3,seed=_SEED,
            initial_spread=0.0,initial_speed_scale=0.0,communication_radius=_R045,neighbour_count=_K4,layout_spread=_LS)
        W=make_fixed_local_graph(make_physical_layout(n,cc),cc.communication_radius,cc.neighbour_count); Ablk=make_aggregate_blocks(arr["aggregate_weight"])
        cfg=DSCEOSConfig(adaptive_consensus_gain=True)
        problem=DSCEOSProblemData(aggregate_blocks=Ablk,lower=arr["lower"],upper=arr["upper"],rest=arr["rest"],
            loss_weight=arr["loss_weight"],masses=arr["masses"],dampings=arr["dampings"],force_limits=arr["force_limits"],
            service_selector=np.tile(np.array([[1.0,0.0]]),(n,1)),service_capacity=arr["service_capacity"],adjacency=W)
        ctl=DistributedSCEOSController(problem,cfg); p0=arr["rest"].copy(); v0=np.zeros_like(p0); ctl.reset_estimators(p0)
        tgt=np.tile(np.array([0.3,0.02]),(n,1))
        for _ in range(3): ctl.control(p0,v0,tgt)
        M=60; t0=time.perf_counter()
        for _ in range(M): ctl.control(p0,v0,tgt)
        tstep=(time.perf_counter()-t0)/M
        ymin,ymax=reachable_range(Ablk,arr["lower"],arr["upper"])
        lo=arr["lower"].reshape(-1); up=arr["upper"].reshape(-1); shape=p0.shape
        for label,target in [("nominal",np.array([0.30,0.02])),("extreme",np.array([0.85,0.35]))]:
            reachable=bool(ymin[0]<=target[0]<=ymax[0] and ymin[1]<=target[1]<=ymax[1])
            x0=np.clip(p0.reshape(-1),lo,up)
            # analytic-gradient solve
            t0=time.perf_counter()
            rA=minimize(lambda x: obj_and_grad(x,arr,W,Ablk,target,cfg,shape),x0,jac=True,
                        method="L-BFGS-B",bounds=list(zip(lo,up)),options={"maxiter":300,"ftol":1e-9})
            tA=time.perf_counter()-t0
            # finite-difference solve (what the submitted driver did)
            t0=time.perf_counter()
            rF=minimize(lambda x: objective_terms(x.reshape(shape),arr,W,Ablk,target,cfg)["objective_value"],x0,
                        method="L-BFGS-B",bounds=list(zip(lo,up)),options={"maxiter":300,"ftol":1e-9})
            tF=time.perf_counter()-t0
            rows.append(dict(cluster=tag,n=n,target=label,reachable=reachable,
                dsceos_per_agent_ms=tstep/n*1e3,dsceos_step_ms=tstep*1e3,
                analytic_ms=tA*1e3,analytic_nit=int(rA.nit),analytic_nfev=int(rA.nfev),
                findiff_ms=tF*1e3,findiff_nfev=int(rF.nfev),
                obj_gap=float(abs(rA.fun-rF.fun))))
            print(f"{tag} {label:<8} reachable={str(reachable):<5} | D-SCEOS/agent={tstep/n*1e3:.3f}ms | "
                  f"analytic {tA*1e3:.1f}ms/{rA.nit}it/{rA.nfev}fev  vs  fin-diff {tF*1e3:.1f}ms/{rF.nfev}fev  (gap {abs(rA.fun-rF.fun):.1e})")
    # M-03: record the environment, and state explicitly that the millisecond fields are a
    # machine-specific artifact of THIS run, not the illustration quoted in the manuscript.
    import platform, sys as _sys
    try:
        import numpy as _np, scipy as _sp
        _libs = {"numpy": _np.__version__, "scipy": _sp.__version__}
    except Exception:
        _libs = {}
    env = dict(
        note=("Millisecond fields are HARDWARE-DEPENDENT and specific to the machine and run that "
              "produced this file; they are NOT the wall-clock illustration quoted in the "
              "manuscript and are deliberately excluded from the validated claims. Only the "
              "deterministic fields (cluster, n, target, reachable, analytic_nit, analytic_nfev, "
              "findiff_nfev, obj_gap) are comparable across machines and are asserted by "
              "validate_results.py."),
        platform=platform.platform(), machine=platform.machine(),
        processor=(platform.processor() or "unknown"),
        python=_sys.version.split()[0], libraries=_libs)
    json.dump({"environment": env, "rows": rows}, open("computational_cost_benchmark.json", "w"), indent=2)
