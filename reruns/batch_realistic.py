"""Re-run all 18 realistic configurations with the physically-exact aggregate fix.
Reports new J_T and the (now physically-exact) GW tracking error = physical fleet-sum error."""
import os as _os, sys as _sys
_ROOT=_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
[_sys.path.insert(0,_p) for _p in (_ROOT,_os.path.join(_ROOT,"code")) if _p not in _sys.path]
_os.chdir(_ROOT)  # results/ paths resolve against the package root
import json, subprocess, numpy as np, time
from pathlib import Path
SCEN={"a":"scenario_a_winter_morning_step","b":"scenario_b_wind_ramp_down_event","c":"scenario_c_winter_balancing_mfrr"}
CTRL={"dsceos":"D-SCEOS","projected_gradient_hocbf":"DPG-HOCBF","independent_tracking":"PD"}
rows=[]; t0=time.time()
for cluster,tag,tm in [("realistic_15","N15",1.0),("realistic_60","N60",4.0)]:
    for sk,sname in SCEN.items():
        for ctrl in CTRL:
            out=Path(f"results/{tag}_{sk}_{ctrl}")
            # source-integrity baseline retune: comparators use the BEST-TESTED settings from the N15/A sweep
            # (DPG step 0.10; PD kp=0.75, kd=2.5). D-SCEOS has no such knob and is unchanged.
            tune=["--dpg-step-size","0.10","--pd-kp","0.75","--pd-kd","2.5"]
            subprocess.run([_sys.executable,"code/run_realistic_scenario.py","--controller",ctrl,
                "--scenario",sname,"--cluster",cluster,"--target-multiplier",str(tm),
                "--safety-filter","--adaptive-consensus-gain","--outdir",str(out)]+tune,
                capture_output=True,check=True)
            s=json.load(open(out/"summary.json"))
            d=np.load(out/"state_history.npz"); p=d["positions"][-1]
            # physical fleet-sum error: un-scale the internal aggregate to GW/GVAR
            from realistic_cpes_catalog import build_realistic_units, compute_fleet_scaling
            from realistic_scenarios import REALISTIC_SCENARIOS, build_signal_for
            from dsceos_validation import arrays_from_units, make_aggregate_blocks, aggregate_output
            from dataclasses import replace as _replace
            _units,_phys=build_realistic_units(cluster); _sc=compute_fleet_scaling(_phys)
            _Ablk=make_aggregate_blocks(arrays_from_units(_units)["aggregate_weight"])
            _scen=REALISTIC_SCENARIOS[sname]
            if tm!=1.0: _scen=_replace(_scen,target_P_max_GW=_scen.target_P_max_GW*tm,target_Q_max_GVAR=_scen.target_Q_max_GVAR*tm)
            _sig=build_signal_for(_scen,_sc); _yTi=_sig.position(_scen.horizon_sim)
            _yP,_yQ=aggregate_output(_Ablk,p)
            dP=_yP/_sc.P_scale_DSO_GW_to_internal - _yTi[0]/_sc.P_scale_DSO_GW_to_internal
            dQ=_yQ/_sc.Q_scale_DSO_GVAR_to_internal - _yTi[1]/_sc.Q_scale_DSO_GVAR_to_internal
            yTnorm=float(np.hypot(_yTi[0]/_sc.P_scale_DSO_GW_to_internal,_yTi[1]/_sc.Q_scale_DSO_GVAR_to_internal))
            pqnorm=float(np.hypot(dP,dQ))
            rows.append(dict(cluster=tag,scenario=sk.upper(),controller=ctrl,
                J_T=s["integrated_objective_value"],
                agg_err_internal=s["final_aggregate_error"],
                final_error_P_GW=round(float(dP),6), final_error_Q_GVAR=round(float(dQ),6),
                final_error_PQ_GVA_equiv=round(pqnorm,6),
                R1_pass=bool(pqnorm<=0.05*yTnorm),
                max_cap_viol=s["max_capacity_violation"],
                energy=s["total_control_energy"]))
print(f"(re-run {len(rows)} configs in {time.time()-t0:.0f}s)\n")
print(f"{'cfg':<12}{'D-SCEOS':>10}{'DPG-HOCBF':>11}{'PD':>10}   D-SCEOS lowest?")
by={}
for r in rows: by[(r['cluster'],r['scenario'],r['controller'])]=r['J_T']
for cl in ["N15","N60"]:
    for sc in ["A","B","C"]:
        d=by[(cl,sc,'dsceos')]; g=by[(cl,sc,'projected_gradient_hocbf')]; p=by[(cl,sc,'independent_tracking')]
        low="YES" if d==min(d,g,p) else "NO"
        print(f"{cl+'/'+sc:<12}{d:>10.4f}{g:>11.4f}{p:>10.4f}   {low}")
json.dump(rows,open("technology_representative_summary.json","w"),indent=2)
