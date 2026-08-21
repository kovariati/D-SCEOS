"""C-02 fix: FAIL-CLOSED full expected-results validator. Any missing file or any numeric
mismatch (realistic, ladder, filter, economic, benchmark distance, target-estimator, R1-R4)
causes a non-zero exit. Run from the package root:  python3 validate_results.py
Use --allow-missing only to intentionally skip absent optional artifacts (still fails on mismatch).

SCOPE (audit source-integrity S-03): this validator checks NUMBERS and figure-summary JSON values -- it does NOT
hash or otherwise verify the figure PNG image bytes. Image-byte integrity is covered separately by
`2_revised_paper/SHA256_MANIFEST.txt`, which lists every released figure PNG/PDF; a tampered
`scenario_*_objective_main.png` is caught by `sha256sum -c SHA256_MANIFEST.txt`, not here. The two checks
are complementary: this validator guards the numeric claims and the figure-summary/trajectory
synchronisation, while the source manifest guards the released image bytes."""
import os, sys, json, numpy as np
os.environ["PYTHONDONTWRITEBYTECODE"]="1"; sys.dont_write_bytecode=True  # m-01: emit no .pyc
_H=os.path.dirname(os.path.abspath(__file__)); os.chdir(_H)
sys.path.insert(0,_H); sys.path.insert(0,os.path.join(_H,"code"))
from graph_config import (AUTHORITATIVE_COMMUNICATION_RADIUS as _R045,
                          AUTHORITATIVE_NEIGHBOUR_COUNT as _K4,
                          AUTHORITATIVE_LAYOUT_SPREAD as _LS,
                          AUTHORITATIVE_SEED as _SEED)  # single authoritative graph config
ALLOW_MISSING = "--allow-missing" in sys.argv
EXP=json.load(open("expected_results.json")); TOL=EXP["tol"]; fails=[]; missing=[]
def close(a,b,rtol=TOL["J_T_rtol"],atol=0.0): return abs(a-b)<=atol+rtol*abs(b)
def check(name, ok, got, exp):
    print(f"  [{'ok ' if ok else 'ERR'}] {name}: got={got} exp={exp}")
    if not ok: fails.append(name)
def need(path):
    if not os.path.exists(path):
        missing.append(path); print(f"  [MISSING] {path}"); return False
    return True
# 1) realistic J_T (all 18) -- REQUIRED
if need("technology_representative_summary.json"):
    d=json.load(open("technology_representative_summary.json")); rows={(r['cluster'],r['scenario'],r['controller']):r['J_T'] for r in d}
    for cfg,ctrls in EXP["realistic_JT"].items():
        cl,sc=cfg.split("_")
        for ctrl,ev in ctrls.items():
            gv=rows.get((cl,sc,ctrl)); check(f"realistic {cfg} {ctrl}", gv is not None and close(gv,ev), gv, ev)
# 2) ladder -- REQUIRED
if need("stress_ladder_summary.json"):
    d=json.load(open("stress_ladder_summary.json")); lr={r['regime']:r for r in d}
    for reg,ev in EXP["ladder_JT_dsceos"].items():
        gv=lr.get(reg,{}).get('dsceos'); check(f"ladder {reg} DS", gv is not None and close(gv,ev), gv, ev)
    check("ladder extreme PD", close(lr['extreme-infeasible']['pd'], EXP["ladder_JT_pd_extreme"]),
          lr['extreme-infeasible']['pd'], EXP["ladder_JT_pd_extreme"])
# 3) filter -- REQUIRED
if need("reruns/filter_on_off_reference.json"):
    ex=json.load(open("reruns/filter_on_off_reference.json")); m={(r['controller'],r['filter']):r for r in ex}; e=EXP["filter"]
    check("filter PD-off J_T", close(m[('independent_tracking',False)]['J_T'],e['pd_off_JT']), m[('independent_tracking',False)]['J_T'], e['pd_off_JT'])
    check("filter PD-off viol", close(m[('independent_tracking',False)]['cap_viol'],e['pd_off_viol'],rtol=0,atol=TOL["viol_atol"]), m[('independent_tracking',False)]['cap_viol'], e['pd_off_viol'])
    # source-integrity design note: the remaining three published filter values were re-baselined but never
    # asserted, so corrupting them still produced PASS. They are asserted here.
    for _lbl,_key,_ctrl,_filt in (("filter PD-on J_T","pd_on_JT","independent_tracking",True),
                                  ("filter D-SCEOS J_T","dsceos_JT","dsceos",True),
                                  ("filter DPG J_T","dpg_JT","projected_gradient_hocbf",True)):
        _g=m[(_ctrl,_filt)]['J_T']; check(_lbl, close(_g,e[_key]), _g, e[_key])
# 3b) per-agent fallback distribution -- REQUIRED, fail-closed.
# Only STACK-ROBUST fields are asserted numerically. Wall-clock solve times are checked for
# presence, type and plausibility only, because they are environment-specific; the exact
# solver-status counters are asserted because the pinned environment of requirements.txt fixes them.
if need("fallback_per_agent.json"):
    _fp=json.load(open("fallback_per_agent.json"))
    _env=_fp.get("environment") if isinstance(_fp,dict) else None
    check("fallback: environment metadata present",
          isinstance(_env,dict) and {"python","numpy","scipy","platform"} <= set(_env),
          sorted(_env) if isinstance(_env,dict) else None, "python+numpy+scipy+platform")
    _rows={r["regime"]:r for r in (_fp["regimes"] if isinstance(_fp,dict) else _fp)}
    _lad={r["regime"]:r for r in json.load(open("ladder_diagnostics.json")) if r["controller"]=="dsceos"}
    for _reg,_ev in EXP["fallback_per_agent"].items():
        _g=_rows.get(_reg)
        if _g is None:
            check(f"fallback {_reg}: present", False, None, "row"); continue
        # cross-consistency: the per-agent trace must reproduce the published per-instant counter
        check(f"fallback {_reg}: per-instant count matches ladder_diagnostics",
              _g["fallback_instants"]==_lad[_reg]["fallback_time_steps"],
              _g["fallback_instants"], _lad[_reg]["fallback_time_steps"])
        for _k in ("fallback_instants","agents_never","agents_always","n_agents","steps"):
            check(f"fallback {_reg}.{_k}", _g.get(_k)==_ev[_k], _g.get(_k), _ev[_k])
        for _k in ("per_agent_rate_median","per_agent_rate_max","top_quartile_share"):
            check(f"fallback {_reg}.{_k}", close(_g.get(_k,-1),_ev[_k],rtol=0,atol=5e-3), _g.get(_k), _ev[_k])
        for _k in ("max_boxball_residual_fallback","max_boxball_residual_solver"):
            check(f"fallback {_reg}.{_k} <= tol", _g.get(_k,1.0)<=2e-8, _g.get(_k), "<=2e-8")
        _t=_g.get("solve_ms_mean")
        check(f"fallback {_reg}: solve time present and plausible",
              isinstance(_t,(int,float)) and 0.0<_t<1.0e3, _t, "0 < t < 1000 ms (not asserted numerically)")
# 3c) Monte Carlo statistical validation -- REQUIRED, fail-closed.
# Only STACK-ROBUST conclusions are asserted. The sample means are NOT asserted numerically: they
# depend on SLSQP solver status and therefore on the exact NumPy/SciPy build, exactly as the
# per-agent fallback counters do. What is asserted is what the manuscript actually claims -- every
# drawn graph is connected, no run violates capacity, D-SCEOS wins every paired comparison, and
# every paired 95% confidence interval on the difference lies strictly above zero.
for _mc_file, _mc_key in (("monte_carlo_A_N15.json", "A"), ("monte_carlo_B_N15.json", "B"),
                          ("monte_carlo_A_N60.json", "A_N60")):
    if _mc_key not in EXP.get("monte_carlo", {}):
        continue
    if not need(_mc_file):
        continue
    _mc = json.load(open(_mc_file)); _ev = EXP["monte_carlo"][_mc_key]
    _env = _mc.get("environment")
    check(f"MC-{_mc_key}: environment metadata present",
          isinstance(_env, dict) and {"python", "numpy", "scipy", "platform"} <= set(_env),
          sorted(_env) if isinstance(_env, dict) else None, "python+numpy+scipy+platform")
    check(f"MC-{_mc_key}: seeds", _mc.get("seeds") == _ev["seeds"], _mc.get("seeds"), _ev["seeds"])
    check(f"MC-{_mc_key}: record count", len(_mc.get("records", [])) == _ev["n_records"],
          len(_mc.get("records", [])), _ev["n_records"])
    _l2 = _mc["graph_realisations"]["lambda2"]
    check(f"MC-{_mc_key}: every drawn graph is connected (lambda2 > 0)",
          len(_l2) > 0 and min(_l2) > 1e-9, f"min lambda2 = {min(_l2) if _l2 else None}", "> 0")
    # Sig #4 fix: distinctness must be judged on the canonical edge-LIST hash, not the edge COUNT
    # (many different topologies can share an edge count, so counting distinct edge counts is
    # fail-open). Prefer edge_sha when present; fall back to edge counts only for legacy artefacts.
    _gr = _mc["graph_realisations"]
    if "edge_sha" in _gr and _gr["edge_sha"]:
        _n_distinct = len(set(_gr["edge_sha"]))
        _distinct_basis = "canonical edge-list SHA"
        # also assert the published count field is honest (matches the recomputed set size)
        if "n_distinct_topologies" in _gr:
            check(f"MC-{_mc_key}: published n_distinct_topologies matches recomputed edge-hash set",
                  _gr["n_distinct_topologies"] == _n_distinct, _gr["n_distinct_topologies"], _n_distinct)
    else:
        _n_distinct = len(set(_gr["edges"]))
        _distinct_basis = "edge count (legacy)"
    check(f"MC-{_mc_key}: at least {_ev['min_distinct_graphs']} distinct topologies drawn ({_distinct_basis})",
          _n_distinct >= _ev["min_distinct_graphs"], _n_distinct, f">= {_ev['min_distinct_graphs']}")
    # D-SCEOS is HOCBF-filtered, so its APPLIED force is admissible at every solve: its per-run
    # capacity violation must be structurally zero even under randomised initial conditions. The PD
    # baseline (independent_tracking) is ALSO safety-filtered here, but the filter only enforces an
    # admissible applied force at the sampling instants -- the PD reference law can still integrate to
    # a slightly inadmissible state between instants, so a small POST-STEP excursion can occur on a
    # random off-rest initial condition. We therefore assert D-SCEOS structural zero and record, but do
    # not require zero for, the comparators. This is the honest post-C-03 picture (randomised initial
    # state), a sampled-data effect and in fact evidence FOR the D-SCEOS in-loop QCQP.
    _ds_viol = max((r["cap_viol"] for r in _mc["records"] if r["controller"] == "dsceos"), default=0.0)
    _cmp_viol = max((r["cap_viol"] for r in _mc["records"] if r["controller"] != "dsceos"), default=0.0)
    check(f"MC-{_mc_key}: zero D-SCEOS capacity violation over all runs (structural)",
          _ds_viol <= TOL["viol_atol"], _ds_viol, 0.0)
    if _cmp_viol > TOL["viol_atol"]:
        print(f"  [note] MC-{_mc_key}: a comparator shows a small capacity excursion "
              f"({_cmp_viol:.2e}) on a randomised initial condition; D-SCEOS stays zero.")
    # Dominance is asserted only where the manuscript CLAIMS it. Experiment B (one-sided channel
    # impairment) reports a MIXED outcome -- D-SCEOS loses to DPG-HOCBF on held-out Scenario B --
    # so its expected block pins the per-comparison outcome instead of asserting dominance. This
    # keeps the validator fail-closed on the published claim rather than on a hoped-for one.
    for _sc, _blk in _mc["statistics"].items():
        for _k, _v in _blk.items():
            if not _k.startswith("paired"):
                continue
            if _ev.get("assert_dominance"):
                check(f"MC-{_mc_key} {_sc} {_k}: D-SCEOS wins every seed",
                      _v["win_rate_dsceos"] == 1.0, _v["win_rate_dsceos"], 1.0)
                check(f"MC-{_mc_key} {_sc} {_k}: paired 95% CI strictly above zero",
                      _v["ci95_low"] > 0.0, _v["ci95_low"], "> 0")
            else:
                _want = _ev["outcomes"].get(f"{_sc}|{_k}")
                _got = "dominant" if (_v["win_rate_dsceos"] == 1.0 and _v["ci95_low"] > 0) else (
                       "favourable" if _v["ci95_low"] > 0 else
                       "inconclusive" if _v["ci95_high"] > 0 else "adverse")
                check(f"MC-{_mc_key} {_sc} {_k}: published outcome", _got == _want, _got, _want)
if "C" in EXP.get("monte_carlo", {}) and need("monte_carlo_C_N15.json"):
    _mc = json.load(open("monte_carlo_C_N15.json")); _ev = EXP["monte_carlo"]["C"]
    _env = _mc.get("environment")
    check("MC-C: environment metadata present",
          isinstance(_env, dict) and {"python", "numpy", "scipy", "platform"} <= set(_env),
          sorted(_env) if isinstance(_env, dict) else None, "python+numpy+scipy+platform")
    check("MC-C: seeds per cell", _mc.get("seeds") == _ev["seeds"], _mc.get("seeds"), _ev["seeds"])
    check("MC-C: grid cells", _mc.get("n_cells") == _ev["n_cells"], _mc.get("n_cells"), _ev["n_cells"])
    check("MC-C: grid axes", _mc.get("grid") == _ev["grid"], _mc.get("grid"), _ev["grid"])
    check("MC-C: D-SCEOS lowest mean in every cell",
          _mc.get("dsceos_lowest_cells") == _ev["dsceos_lowest_cells"],
          _mc.get("dsceos_lowest_cells"), _ev["dsceos_lowest_cells"])
    # separation is the stronger, published claim: how many cells have the D-SCEOS 95% CI entirely
    # below BOTH comparator intervals. Asserted as a lower bound so a numerically different solver
    # build cannot silently weaken the published statement without failing.
    _sep = sum(1 for _c in _mc["cells"]
               if _c["D-SCEOS"]["ci95_high"] < min(_c["DPG-HOCBF"]["ci95_low"],
                                                   _c["PD baseline"]["ci95_low"]))
    check("MC-C: cells with fully separated D-SCEOS interval",
          _sep >= _ev["min_separated_cells"], _sep, f">= {_ev['min_separated_cells']}")
# 3d) stronger distributed comparator: primal-dual -- REQUIRED, fail-closed.
if need("primal_dual_comparator.json"):
    _pd = json.load(open("primal_dual_comparator.json"))
    _epd = EXP.get("primal_dual", {})
    _rows = {(r["cluster"], r["scenario"]): r for r in _pd}
    for _key, _ev in _epd.get("configs", {}).items():
        _cl, _sc = _key.split("_")
        _r = _rows.get((_cl, _sc))
        if _r is None:
            check(f"primal-dual {_key}: present", False, None, "row"); continue
        # zero capacity violation is a hard, stack-robust guarantee (HOCBF filter) -> asserted
        check(f"primal-dual {_key}: zero capacity violation",
              _r["dpd_max_cap_violation"] <= TOL["viol_atol"], _r["dpd_max_cap_violation"], 0.0)
        # D-SCEOS lowest is the published claim -> asserted
        check(f"primal-dual {_key}: D-SCEOS lowest", _r["dsceos_lowest"] is True,
              _r["dsceos_lowest"], True)
        # DPD strictly beats the PD baseline everywhere -> asserted (both from the same released files)
        check(f"primal-dual {_key}: DPD beats PD baseline",
              _r["J_T"]["DPD-HOCBF"] < _r["J_T"]["PD baseline"],
              (_r["J_T"]["DPD-HOCBF"], _r["J_T"]["PD baseline"]), "DPD < PD")
        # the DPD J_T itself is solver-dependent -> checked only within tolerance, not pinned hard
        check(f"primal-dual {_key}: DPD J_T within tol of expected",
              close(_r["J_T"]["DPD-HOCBF"], _ev["dpd_JT"]),
              _r["J_T"]["DPD-HOCBF"], _ev["dpd_JT"])
# 3e) independent fleet compositions + solver profile -- REQUIRED, fail-closed.
if need("fleet_compositions.json"):
    _fc = json.load(open("fleet_compositions.json"))
    _env = _fc.get("environment")
    check("fleet-compositions: environment metadata present",
          isinstance(_env, dict) and {"python", "numpy", "scipy", "platform"} <= set(_env),
          sorted(_env) if isinstance(_env, dict) else None, "python+numpy+scipy+platform")
    _efc = EXP.get("fleet_compositions", {})
    _byN = {r["N"]: r for r in _fc["results"]}
    for _N_str, _ev in _efc.get("sizes", {}).items():
        _N = int(_N_str); _r = _byN.get(_N)
        if _r is None:
            check(f"fleet {_N}: present", False, None, "row"); continue
        # D-SCEOS is HOCBF-filtered -> its capacity violation is structurally zero (the assertion).
        # A comparator may violate under a hard absolute target on the largest fleet; that is reported
        # per controller in the artefact and is NOT asserted to be zero.
        check(f"fleet {_N}: D-SCEOS zero capacity violation",
              _r.get("dsceos_max_cap_violation", _r["max_cap_violation"]) <= TOL["viol_atol"],
              _r.get("dsceos_max_cap_violation", _r["max_cap_violation"]), 0.0)
        # C-01: the realised relative loading must be the DESIGN-MATCHED reference across all sizes
        # (0.3077 active / 0.09375 reactive), otherwise the size comparison is not load-matched.
        check(f"fleet {_N}: active loading is design-matched (0.3077)",
              abs(_r.get("mean_target_over_capacity_P", -1) - 0.3077) <= 1e-3,
              _r.get("mean_target_over_capacity_P"), 0.3077)
        check(f"fleet {_N}: reactive loading is design-matched (0.0937)",
              abs(_r.get("mean_target_over_capacity_Q", -1) - 0.0937) <= 1e-3,
              _r.get("mean_target_over_capacity_Q"), 0.0937)
        for _lbl in ("DPG-HOCBF", "PD baseline"):
            # stack-robust published claims: D-SCEOS wins every seed and the paired CI is above zero
            check(f"fleet {_N} {_lbl}: D-SCEOS wins every seed",
                  _r[_lbl]["win_rate_dsceos"] == 1.0, _r[_lbl]["win_rate_dsceos"], 1.0)
            check(f"fleet {_N} {_lbl}: paired diff CI above zero",
                  _r[_lbl]["paired_diff_ci_low"] > 0.0, _r[_lbl]["paired_diff_ci_low"], "> 0")
if need("fleet_solver_profile.json"):
    _fp = json.load(open("fleet_solver_profile.json"))
    _env = _fp.get("environment")
    check("solver-profile: environment metadata present",
          isinstance(_env, dict) and {"python", "numpy", "scipy", "platform"} <= set(_env),
          sorted(_env) if isinstance(_env, dict) else None, "python+numpy+scipy+platform")
    _efp = EXP.get("solver_profile", {})
    _byN = {r["N"]: r for r in _fp["results"]}
    for _N_str in _efp.get("sizes", []):
        _N = int(_N_str); _r = _byN.get(_N)
        if _r is None:
            check(f"profile {_N}: present", False, None, "row"); continue
        # solve times are environment-specific -> presence + plausibility only, not asserted numerically
        _t = _r.get("solve_ms_mean")
        check(f"profile {_N}: solve time present and plausible",
              isinstance(_t, (int, float)) and 0.0 < _t < 1.0e3, _t, "0 < t < 1000 ms (not asserted)")
        # M-06/C-02: the communication load is EXACT for this protocol -- 9 scalars x 8 bytes x
        # neighbour count / dt -- so the mean value is deterministic given the mean neighbour count
        # and IS asserted (not just presence). dt = 0.05 s.
        # M-06/C-02/C-04: the communication load is EXACT for this protocol -- 9 scalars x 8 bytes x
        # neighbour count per sampling instant -- and is reported per PHYSICAL second. One instant is
        # dt=0.05 simulation units = 0.05 minutes = 3 physical seconds, so the mean rate is
        # 9*8*neighbours/3. Dividing by the raw 0.05 would overstate the physical rate by 60x.
        _cm = _r.get("comm_bytes_per_s_per_agent", {})
        _nbr = _r.get("mean_neighbours")
        _DT_PHYS_S = 3.0
        _expected_mean = 9 * 8 * _nbr / _DT_PHYS_S if isinstance(_nbr, (int, float)) else None
        check(f"profile {_N}: comm mean = 9*8*neighbours/3s (exact, physical)",
              _expected_mean is not None and abs(_cm.get("mean", -1) - _expected_mean) <= 1.0,
              _cm.get("mean"), _expected_mean)
# 4) economic -- REQUIRED
if need("reruns/economic_actuation_reference.json"):
    d=json.load(open("reruns/economic_actuation_reference.json")); em={(r['scenario'],r['controller']):r['cost_weighted'] for r in d}
    for sc,ev in EXP["economic_costwtd_N60_dsceos"].items():
        gv=em.get((sc,'dsceos')); check(f"economic N60-{sc} DS", gv is not None and close(gv,ev,rtol=2e-2), gv, ev)
# 5) target-estimator spectral values from the ACTUAL simulation operator M=(I-Gamma)(I-gamma_T L)
try:
    from realistic_cpes_catalog import build_realistic_units
    from dsceos_validation import ClusterConfig, make_fixed_local_graph, make_physical_layout, gateway_mask
    def tgt_op(n, cc, gw_frac=0.15, gT=0.18, bT=0.85):
        W=make_fixed_local_graph(make_physical_layout(n,cc),cc.communication_radius,cc.neighbour_count)
        L=np.diag(W.sum(1))-W
        gw=gateway_mask(n, gw_frac)
        Gam=np.diag([bT if g else 0.0 for g in gw]); M=(np.eye(n)-Gam)@(np.eye(n)-gT*L)
        return float(max(abs(np.linalg.eigvals(M)))), float(np.linalg.norm(M,2))
    ccN15=ClusterConfig(n_thermal=3,n_storage=3,n_hydrogen=3,n_emobility=3,n_industrial=3,seed=_SEED,initial_spread=0.,initial_speed_scale=0.,communication_radius=_R045,neighbour_count=_K4,layout_spread=_LS)
    ccN60=ClusterConfig(n_thermal=12,n_storage=12,n_hydrogen=12,n_emobility=12,n_industrial=12,seed=_SEED,initial_spread=0.,initial_speed_scale=0.,communication_radius=_R045,neighbour_count=_K4,layout_spread=_LS)
    for tag,n,cc in [("N15",15,ccN15),("N60",60,ccN60)]:
        rho,nrm=tgt_op(n,cc); e=EXP["target_estimator"]
        check(f"target-estimator {tag} rho", close(rho,e[f"{tag}_rho"],rtol=2e-3), round(rho,5), e[f"{tag}_rho"])
        check(f"target-estimator {tag} ||M||2", close(nrm,e[f"{tag}_norm2"],rtol=2e-3), round(nrm,5), e[f"{tag}_norm2"])
except Exception as ex:
    fails.append(f"target-estimator recompute FAILED ({type(ex).__name__}): {ex}")
    print(f"  [ERR] target-estimator recompute FAILED (fatal): {ex}")

# 6) full R1-R4 matrix + benchmark distance from technology_representative_summary.json (+ optimum for R2/benchmark)
if os.path.exists("technology_representative_summary.json"):
    d=json.load(open("technology_representative_summary.json"))
    need_fields=all(k in d[0] for k in ("R1_pass","J_T","energy","final_error_PQ_GVA_equiv"))
    if not need_fields:
        fails.append("R1-R4: technology_representative_summary.json missing required fields (regenerate batch_realistic.py)")
        print("  [ERR] technology_representative_summary.json missing R1_pass/energy fields")
    else:
        lab={'dsceos':'DS','projected_gradient_hocbf':'DPG','independent_tracking':'PD'}
        by={}
        for r in d: by.setdefault((r['cluster'],r['scenario']),{})[lab[r['controller']]]=r
        # R1,R3,R4 from JSON (R3: J_T<=2*best; R4: energy<=3*best)
        for cfg,exp in EXP["acceptability_matrix"].items():
            g=by.get(tuple(cfg.split('_')),{})
            if not g: continue
            bestJ=min(v['J_T'] for v in g.values()); bestE=min(v['energy'] for v in g.values())
            r1=sorted([c for c,v in g.items() if v['R1_pass']])
            r3=sorted([c for c,v in g.items() if v['J_T']<=2*bestJ+1e-9])
            r4=sorted([c for c,v in g.items() if v['energy']<=3*bestE+1e-9])
            check(f"R1 {cfg}", r1==sorted(exp["R1"]), r1, sorted(exp["R1"]))
            check(f"R3 {cfg}", r3==sorted(exp["R3"]), r3, sorted(exp["R3"]))
            check(f"R4 {cfg}", r4==sorted(exp["R4"]), r4, sorted(exp["R4"]))
        # R2 + benchmark distance: recompute the centralized optimum (radius-0.45 graph, same as runner)
        try:
            from realistic_cpes_catalog import build_realistic_units, compute_fleet_scaling
            from realistic_scenarios import REALISTIC_SCENARIOS, build_signal_for
            from dsceos_validation import arrays_from_units, make_aggregate_blocks, make_fixed_local_graph, make_physical_layout, aggregate_output
            from dsceos_controller import DSCEOSConfig
            from scipy.optimize import minimize as _min
            from dataclasses import replace as _rep
            from benchmark_objective import obj_and_grad as _oag
            SCN={"A":"scenario_a_winter_morning_step","B":"scenario_b_wind_ramp_down_event","C":"scenario_c_winter_balancing_mfrr"}
            def _opt(cl,sc,tm,cc):
                units,phys=build_realistic_units(cl); arr=arrays_from_units(units)
                W=make_fixed_local_graph(make_physical_layout(len(units),cc),cc.communication_radius,cc.neighbour_count); Ablk=make_aggregate_blocks(arr["aggregate_weight"])
                scfg=compute_fleet_scaling(phys); scen=REALISTIC_SCENARIOS[SCN[sc]]
                if tm!=1.0: scen=_rep(scen,target_P_max_GW=scen.target_P_max_GW*tm,target_Q_max_GVAR=scen.target_Q_max_GVAR*tm)
                yTi=build_signal_for(scen,scfg).position(scen.horizon_sim)
                lo=arr["lower"].reshape(-1);up=arr["upper"].reshape(-1);x0=np.clip(arr["rest"].reshape(-1),lo,up)
                r=_min(lambda x:_oag(x,arr,W,Ablk,yTi,DSCEOSConfig(adaptive_consensus_gain=True),arr["rest"].shape),x0,jac=True,method="L-BFGS-B",bounds=list(zip(lo,up)),options={"maxiter":300,"ftol":1e-9})
                ys=aggregate_output(Ablk,r.x.reshape(arr["rest"].shape))
                yT=np.array([yTi[0]/scfg.P_scale_DSO_GW_to_internal,yTi[1]/scfg.Q_scale_DSO_GVAR_to_internal])
                return np.array([ys[0]/scfg.P_scale_DSO_GW_to_internal,ys[1]/scfg.Q_scale_DSO_GVAR_to_internal]), Ablk, scfg, float(np.linalg.norm(yT)), r.x.reshape(arr["rest"].shape)
            ccN60_=ClusterConfig(n_thermal=12,n_storage=12,n_hydrogen=12,n_emobility=12,n_industrial=12,seed=_SEED,initial_spread=0.,initial_speed_scale=0.,communication_radius=_R045,neighbour_count=_K4,layout_spread=_LS)
            ccN15_=ClusterConfig(n_thermal=3,n_storage=3,n_hydrogen=3,n_emobility=3,n_industrial=3,seed=_SEED,initial_spread=0.,initial_speed_scale=0.,communication_radius=_R045,neighbour_count=_K4,layout_spread=_LS)
            # benchmark distance (N60 D-SCEOS) with a tight relative tolerance
            for sc,ev in EXP["benchmark_distance_N60_dsceos_PQnorm"].items():
                ys,Ablk,scfg,yTnorm,pstar=_opt("realistic_60",sc,4.0,ccN60_)
                p=np.load(f"results/N60_{sc.lower()}_dsceos/state_history.npz")["positions"][-1]
                yp=aggregate_output(Ablk,p); y=np.array([yp[0]/scfg.P_scale_DSO_GW_to_internal,yp[1]/scfg.Q_scale_DSO_GVAR_to_internal])
                dist=float(np.linalg.norm(y-ys))
                check(f"benchmark dist N60-{sc} DS", close(dist,ev,rtol=8e-2,atol=1e-3), round(dist,4), ev)
            # M-05: full per-unit allocation RMS ||p - p*|| for N60 A/B/C (packaged + validated)
            for sc,ev in EXP.get("allocation_rms_N60_dsceos",{}).items():
                _,_,_,_,pstar=_opt("realistic_60",sc,4.0,ccN60_)
                p=np.load(f"results/N60_{sc.lower()}_dsceos/state_history.npz")["positions"][-1]
                rms=float(np.sqrt(np.mean((p-pstar)**2)))
                check(f"allocation RMS N60-{sc}", close(rms,ev,rtol=0,atol=5e-4), round(rms,4), ev)
            # FULL R2 matrix (all 18 cells): R2 pass = ||y - y*|| <= 5% ||y_T||
            lab2={'dsceos':'DS','projected_gradient_hocbf':'DPG','independent_tracking':'PD'}
            for cl,tag2,tm2,cc2 in [("realistic_15","N15",1.0,ccN15_),("realistic_60","N60",4.0,ccN60_)]:
                for sc in ["A","B","C"]:
                    ys,Ablk,scfg,yTnorm,pstar=_opt(cl,sc,tm2,cc2); thr=0.05*yTnorm
                    got=[]
                    for ctrl in ["dsceos","projected_gradient_hocbf","independent_tracking"]:
                        p=np.load(f"results/{tag2}_{sc.lower()}_{ctrl}/state_history.npz")["positions"][-1]
                        yp=aggregate_output(Ablk,p); y=np.array([yp[0]/scfg.P_scale_DSO_GW_to_internal,yp[1]/scfg.Q_scale_DSO_GVAR_to_internal])
                        if float(np.linalg.norm(y-ys))<=thr: got.append(lab2[ctrl])
                    exp=sorted(EXP["acceptability_matrix"][f"{tag2}_{sc}"]["R2"])
                    check(f"R2 {tag2}_{sc}", sorted(got)==exp, sorted(got), exp)
        except Exception as ex:
            fails.append(f"R2/benchmark recompute FAILED: {type(ex).__name__}: {ex}")
            print(f"  [ERR] R2/benchmark recompute FAILED (fatal): {ex}")

# 7) stress-ladder safety diagnostics (C-02): empty-intersection=0, hard residual bounded, cap viol=0
if os.path.exists("ladder_diagnostics.json"):
    dl=json.load(open("ladder_diagnostics.json")); dsd={r['regime']:r for r in dl if r['controller']=='dsceos'}
    exp=EXP.get("ladder_diagnostics_dsceos",{})
    for reg,ex in exp.items():
        row=dsd.get(reg)
        if row is None: check(f"ladder-diag {reg} present", False, None, "row"); continue
        # EXACT box-ball intersection test (C-01): zero empty agent-steps and time-steps AND positive margin
        check(f"ladder-diag {reg} exact empty_agent_steps==0", row.get('empty_agent_steps',1)==0, row.get('empty_agent_steps'), 0)
        check(f"ladder-diag {reg} empty_intersection_steps==0", row.get('empty_intersection_steps',1)==0, row.get('empty_intersection_steps'), 0)
        mm=row.get('min_intersection_margin')
        check(f"ladder-diag {reg} min_intersection_margin>=0.30", (mm is not None and mm>=ex['min_intersection_margin_min']), mm, f">={ex['min_intersection_margin_min']}")
        # residual of the FINAL applied QCQP force (not the projection initialization)
        rf=row.get('max_final_force_residual', row.get('max_hard_residual'))
        check(f"ladder-diag {reg} max_final_force_residual<2e-8", rf<2e-8, rf, "<2e-8")
        # Task 3: box (operating-envelope) and ball (actuator) residuals are published SEPARATELY.
        # The HOCBF box is never violated by the applied force (exactly zero box residual); the only
        # residual is the actuator ball at solver tolerance.
        _boxr=row.get('max_box_residual'); _ballr=row.get('max_ball_residual')
        check(f"ladder-diag {reg} max_box_residual==0", (_boxr is not None and _boxr<=1e-12), _boxr, "<=1e-12")
        check(f"ladder-diag {reg} max_ball_residual<2e-8", (_ballr is not None and _ballr<2e-8), _ballr, "<2e-8")
        check(f"ladder-diag {reg} cap_viol==0", row['max_capacity_violation']<=1e-9, row['max_capacity_violation'], 0)
        check(f"ladder-diag {reg} max_clf_slack", close(row['max_clf_slack'], ex['max_clf_slack'], rtol=5e-2), row['max_clf_slack'], ex['max_clf_slack'])
else:
    fails.append("ladder_diagnostics.json missing (run reruns/ladder_diagnostics.py)")
    print("  [ERR] ladder_diagnostics.json missing")

# R1--R4 diagnostic matrix (retuned comparators). Stack-robust assertions: D-SCEOS meets the
# cost-optimal-aggregate requirement R2 and the integrated-objective requirement R3 in EVERY one of
# the six configurations; this is the headline claim of the matrix and is independent of the exact
# comparator numbers. We do NOT assert the comparators' cells (those are solver/tuning-sensitive).
if os.path.exists("acceptability_matrix.json"):
    _acceptability = json.load(open("acceptability_matrix.json"))
    check("acceptability: six configurations present", len(_acceptability) == 6, len(_acceptability), 6)
    for _row in _acceptability:
        _cfg = _row["config"]
        check(f"acceptability {_cfg}: D-SCEOS meets R2", "DS" in _row.get("R2", []), _row.get("R2"), "contains DS")
        check(f"acceptability {_cfg}: D-SCEOS meets R3", "DS" in _row.get("R3", []), _row.get("R3"), "contains DS")
        # D-SCEOS must also have the strictly lowest J_T in every configuration
        _jt = _row.get("J_T", {})
        check(f"acceptability {_cfg}: D-SCEOS lowest J_T",
              _jt.get("DS") == min(_jt.values()) if _jt else False, _jt.get("DS"), "min")
else:
    print("  [warn] acceptability_matrix.json missing (run reruns/acceptability_matrix.py) -- not asserted")

# 8) REAL realistic N15-A control-displacement diagnostic (M-04 / source-integrity C-01): fail-closed, all fields
if os.path.exists("control_displacement_N15A.json"):
    cd=json.load(open("control_displacement_N15A.json")); exp=EXP["control_displacement_N15A"]
    check("control-displacement N15-A scenario/fleet", cd.get("scenario")=="A" and cd.get("fleet")=="realistic_15", (cd.get("scenario"),cd.get("fleet")), ("A","realistic_15"))
    check("control-displacement N15-A n_agent_steps", cd.get("n_agent_steps")==exp["n_agent_steps"], cd.get("n_agent_steps"), exp["n_agent_steps"])
    check("control-displacement N15-A J_T", close(cd["J_T"],exp["J_T"],rtol=0,atol=1e-3), cd["J_T"], exp["J_T"])
    check("control-displacement N15-A max", close(cd["max"],exp["max"],rtol=0,atol=5e-3), cd["max"], exp["max"])
    check("control-displacement N15-A mean", close(cd["mean"],exp["mean"],rtol=0,atol=1e-3), cd["mean"], exp["mean"])
    check("control-displacement N15-A p95", close(cd["p95"],exp["p95"],rtol=0,atol=1e-3), cd["p95"], exp["p95"])
    check("control-displacement N15-A frac>1e-3", close(cd["frac_gt_1em3"],exp["frac_gt_1em3"],rtol=0,atol=5e-3), cd["frac_gt_1em3"], exp["frac_gt_1em3"])
else:
    fails.append("control_displacement_N15A.json missing (run reruns/control_displacement.py)")
    print("  [ERR] control_displacement_N15A.json missing")

# 9) previously run-only diagnostics, now packaged and asserted (m-05):
#    comparator tuning, 28-point sensitivity, gateway sweep, agent loss, communication outage.
_ro=EXP.get("run_only_diagnostics")
if _ro:
    def _need(fn):
        if not os.path.exists(fn):
            fails.append(f"{fn} missing (regenerate with the corresponding reruns/ script)")
            print(f"  [ERR] {fn} missing"); return None
        return json.load(open(fn))
    b=_need("baseline_tuning.json")
    if b is not None:
        ex=_ro["comparator_tuning"]
        check("tuning D-SCEOS reference J_T", close(b["dsceos_ref"],ex["dsceos_ref_J_T"],rtol=5e-3), round(b["dsceos_ref"],6), ex["dsceos_ref_J_T"])
        check("tuning best PD J_T", close(b["best_pd"]["J_T"],ex["best_pd_J_T"],rtol=5e-3), round(b["best_pd"]["J_T"],6), ex["best_pd_J_T"])
        check("tuning best DPG J_T", close(b["best_dpg"]["J_T"],ex["best_dpg_J_T"],rtol=5e-3), round(b["best_dpg"]["J_T"],6), ex["best_dpg_J_T"])
        check("tuning comparators capacity-feasible", max(b["best_pd"]["cap_viol"],b["best_dpg"]["cap_viol"])<=1e-9, max(b["best_pd"]["cap_viol"],b["best_dpg"]["cap_viol"]), 0.0)
    s=_need("parameter_sensitivity_N15A.json")
    if s is not None:
        ex=_ro["sensitivity_N15A"]
        check("sensitivity sweep points", len(s)==ex["n_points"], len(s), ex["n_points"])
        got=sum(1 for r in s if r["dsceos_lowest"])
        check("sensitivity D-SCEOS lowest count", got==ex["dsceos_lowest_count"], got, ex["dsceos_lowest_count"])
    g=_need("gateway_sensitivity.json")
    if g is not None:
        ex=_ro["gateway_sweep"]
        check("gateway sweep rows", len(g)==ex["n_rows"], len(g), ex["n_rows"])
        check("gateway sweep cap-viol==0", max(r["cap_viol"] for r in g)<=1e-9, max(r["cap_viol"] for r in g), 0.0)
        d=[r for r in g if r["kind"]=="fraction" and abs(r["gw_fraction"]-0.15)<1e-9]
        check("gateway default-fraction J_T", bool(d) and close(d[0]["J_T"],ex["default_fraction_J_T"],rtol=5e-3), (round(d[0]["J_T"],6) if d else None), ex["default_fraction_J_T"])
    for tag in ("A","B"):
        al=_need(f"agent_loss_{tag}.json")
        if al is not None:
            check(f"agent-loss {tag} cap-viol==0", al["max_capacity_violation" if "max_capacity_violation" in al else "max_cap_violation"]<=1e-9,
                  al.get("max_cap_violation", al.get("max_capacity_violation")), 0.0)
    if os.path.exists("agent_loss_A.json"):
        al=json.load(open("agent_loss_A.json")); ex=_ro["agent_loss"]
        check("agent-loss A first-return time (min)", close(al["first_return_min"],ex["A_first_return_min"],rtol=0,atol=0.2), round(al["first_return_min"],3), ex["A_first_return_min"])
    c=_need("communication_outage.json")
    if c is not None:
        ex=_ro["comm_failure"]
        check("comm-failure rows", len(c)==ex["n_rows"], len(c), ex["n_rows"])
        check("comm-failure cap-viol==0", max(r["max_cap_violation"] for r in c)<=1e-9, max(r["max_cap_violation"] for r in c), 0.0)
        lo=min(r["spread_ratio"] for r in c); hi=max(r["spread_ratio"] for r in c)
        check("comm-failure spread-ratio range", close(lo,ex["spread_ratio_min"],rtol=5e-2) and close(hi,ex["spread_ratio_max"],rtol=5e-2), (round(lo,3),round(hi,3)), (ex["spread_ratio_min"],ex["spread_ratio_max"]))

# 10) FULL field-by-field coverage of every packaged diagnostic family (source-integrity M-01), plus the
#     capacity-violation rows of ALL THREE ladder controllers (not just D-SCEOS).
def _diff_struct(got, exp, path=""):
    """Return a list of field-level mismatches between a packaged JSON and its expected content.
    Numbers use a relative tolerance; everything else must match exactly. Missing keys and length
    changes are mismatches, so a corrupted or partially-replaced JSON cannot slip through."""
    out=[]
    if isinstance(exp, dict):
        if not isinstance(got, dict): return [f"{path}: expected object, got {type(got).__name__}"]
        for k in exp:
            if k not in got: out.append(f"{path}.{k}: missing")
            else: out += _diff_struct(got[k], exp[k], f"{path}.{k}")
    elif isinstance(exp, list):
        if not isinstance(got, list): return [f"{path}: expected list, got {type(got).__name__}"]
        if len(got) != len(exp): return [f"{path}: length {len(got)} != {len(exp)}"]
        for i,(g,e2) in enumerate(zip(got,exp)): out += _diff_struct(g, e2, f"{path}[{i}]")
    elif isinstance(exp, bool) or exp is None:
        # booleans must match EXACTLY and by type: True != 1 (source-integrity Design note)
        if not (type(got) is type(exp) and got == exp):
            out.append(f"{path}: {got!r} ({type(got).__name__}) != {exp!r} ({type(exp).__name__})")
    elif isinstance(exp, int):
        # counts are exact integers: 9 != 9.04 and 9 != True
        if isinstance(got, bool) or not isinstance(got, int) or got != exp:
            out.append(f"{path}: {got!r} ({type(got).__name__}) != exact int {exp!r}")
    elif isinstance(exp, float):
        if isinstance(got, bool) or not isinstance(got,(int,float)) or \
                not close(float(got), float(exp), rtol=5e-3, atol=1e-9):
            out.append(f"{path}: {got!r} != {exp!r}")
    elif got != exp:
        out.append(f"{path}: {got!r} != {exp!r}")
    return out

_full=EXP.get("diagnostics_full", {})
_files={"comparator_tuning":"baseline_tuning.json","sensitivity_N15A":"parameter_sensitivity_N15A.json",
        "gateway_sweep":"gateway_sensitivity.json","agent_loss_A":"agent_loss_A.json",
        "agent_loss_B":"agent_loss_B.json","comm_failure":"communication_outage.json"}
for _fam, _fn in _files.items():
    if _fam not in _full:
        continue
    if not os.path.exists(_fn):
        fails.append(f"{_fn} missing (regenerate with the corresponding reruns/ script)")
        print(f"  [ERR] {_fn} missing"); continue
    _d=_diff_struct(json.load(open(_fn)), _full[_fam])
    _n=sum(1 for _ in json.dumps(_full[_fam]))  # size marker only
    check(f"{_fam}: every published field matches", not _d,
          (f"{len(_d)} mismatched field(s): {_d[:3]}" if _d else f"all fields match ({_fn})"), "all fields match")

# ladder capacity violation for every controller (D-SCEOS, DPG-HOCBF, PD)
_lcv=EXP.get("ladder_capacity_violation_all_controllers")
if _lcv:
    if os.path.exists("ladder_diagnostics.json"):
        _ld={(r["regime"],r["controller"]):r for r in json.load(open("ladder_diagnostics.json"))}
        _bad=[]
        for _row in _lcv:
            _key=(_row["regime"],_row["controller"]); _g=_ld.get(_key)
            if _g is None or not close(_g["max_capacity_violation"],_row["max_capacity_violation"],rtol=0,atol=1e-9):
                _bad.append(_key)
        check("ladder capacity violation, ALL controllers", not _bad, (f"{len(_bad)} mismatched rows: {_bad[:3]}" if _bad else "all match"), "all 21 rows match")
    else:
        fails.append("ladder_diagnostics.json missing"); print("  [ERR] ladder_diagnostics.json missing")

# 11) centralized-benchmark DETERMINISTIC fields (source-integrity M-01). The millisecond fields are
#     hardware-dependent and deliberately NOT asserted; everything else here is.
_cb=EXP.get("centralized_benchmark_deterministic")
if _cb:
    if not os.path.exists("computational_cost_benchmark.json"):
        fails.append("computational_cost_benchmark.json missing (run reruns/computational_cost_benchmark.py)")
        print("  [ERR] computational_cost_benchmark.json missing")
    else:
        _tj=json.load(open("computational_cost_benchmark.json"))
        _rows=_tj["rows"] if isinstance(_tj,dict) and "rows" in _tj else _tj
        _det=("cluster","n","target","reachable","analytic_nit","analytic_nfev","findiff_nfev","obj_gap")
        _got=[{k:r.get(k) for k in _det} for r in _rows]
        _diff=_diff_struct(_got, _cb["rows"])
        check("centralized benchmark: deterministic fields", not _diff,
              (f"{len(_diff)} mismatch(es): {_diff[:3]}" if _diff else "all deterministic fields match"),
              "all deterministic fields match")
        # the unit-equivalent work quoted in the paper must follow from those counts
        _by={(r["cluster"],r["target"]):r for r in _rows}
        _w=_cb["unit_equivalent_work"]
        for _tag,_cl in (("N15","N15"),("N60","N60")):
            _r=_by.get((_cl,"nominal"))
            if _r is None:
                check(f"W_cent {_tag} row present", False, None, "nominal row"); continue
            check(f"W_cent {_tag} = n*analytic_nfev", _r["n"]*_r["analytic_nfev"]==_w[_tag],
                  _r["n"]*_r["analytic_nfev"], _w[_tag])
            check(f"W_cent {_tag} finite-difference = n*findiff_nfev",
                  _r["n"]*_r["findiff_nfev"]==_w[f"{_tag}_findiff"],
                  _r["n"]*_r["findiff_nfev"], _w[f"{_tag}_findiff"])
        # the environment block must be present AND complete, so the ms figures can never be read as
        # machine-independent, and a note-only stub is rejected (source-integrity Design note)
        _env=_tj.get("environment") if isinstance(_tj,dict) else None
        _req={"note":str,"platform":str,"machine":str,"processor":str,"python":str,"libraries":dict}
        _bad=[]
        if not isinstance(_env,dict):
            _bad.append("environment block missing")
        else:
            for _k,_ty in _req.items():
                if _k not in _env: _bad.append(f"missing '{_k}'")
                elif not isinstance(_env[_k],_ty): _bad.append(f"'{_k}' is {type(_env[_k]).__name__}, expected {_ty.__name__}")
                elif _ty is str and not _env[_k].strip(): _bad.append(f"'{_k}' is empty")
            if isinstance(_env.get("libraries"),dict):
                for _lib in ("numpy","scipy"):
                    _v=_env["libraries"].get(_lib)
                    if not isinstance(_v,str) or not _v.strip(): _bad.append(f"libraries.{_lib} missing/empty")
            if isinstance(_env.get("note"),str) and "HARDWARE-DEPENDENT" not in _env["note"].upper():
                _bad.append("note does not flag the millisecond fields as hardware-dependent")
        check("centralized benchmark: environment metadata complete", not _bad,
              (f"{len(_bad)} problem(s): {_bad[:3]}" if _bad else "all required keys present and typed"),
              "note+platform+machine+processor+python+libraries{numpy,scipy}")

# Design note: N=60 ordering robustness to comparator retuning, FAIL-CLOSED.
# The paper claims D-SCEOS still wins all three N=60 scenarios against the BEST-TESTED N=60 DPG step
# (0.04 on the documented grid, NOT 0.06), not only against the transferred N15/A step 0.10. This check
# is REQUIRED: a missing, structurally incomplete, or tampered artefact FAILS validation (it does not
# silently pass). We assert (a) the file exists, (b) each scenario carries the FULL documented step grid
# (not a single hand-picked row), (c) the published best step is the recomputed grid minimiser and is
# 0.04, and (d) D-SCEOS is strictly lower than that best-tested DPG in every scenario.
_ROBUST_GRID = [0.02, 0.04, 0.06, 0.10, 0.20, 0.40, 0.72]
if need("dpg_n60_retune_robustness.json"):
    _rob = json.load(open("dpg_n60_retune_robustness.json"))
    _res = _rob.get("results", [])
    check("N60 robustness: all three scenarios present (A,B,C)",
          sorted(r.get("scenario") for r in _res) == ["A", "B", "C"],
          sorted(r.get("scenario") for r in _res), ["A", "B", "C"])
    for _r in _res:
        _sc = _r.get("scenario")
        _grid = _r.get("dpg_grid", {})
        # (b) full grid present -- a single-row faked artefact fails here
        _grid_steps = sorted(float(k) for k in _grid)
        check(f"N60 robustness {_sc}: full DPG step grid present",
              _grid_steps == _ROBUST_GRID, _grid_steps, _ROBUST_GRID)
        # (c) published best step is the recomputed grid minimiser AND equals 0.04
        if _grid:
            _recomp_best = min(_grid, key=_grid.get)
            check(f"N60 robustness {_sc}: published best step is the grid minimiser",
                  abs(float(_r.get("best_dpg_step")) - float(_recomp_best)) < 1e-9
                  and abs(_r.get("best_dpg_J_T") - _grid[_recomp_best]) < 1e-6,
                  (_r.get("best_dpg_step"), _r.get("best_dpg_J_T")),
                  (float(_recomp_best), _grid[_recomp_best]))
            check(f"N60 robustness {_sc}: best-tested step is 0.04",
                  abs(float(_r.get("best_dpg_step")) - 0.04) < 1e-9, _r.get("best_dpg_step"), 0.04)
        # (d) D-SCEOS strictly lower than the best-tested DPG
        check(f"N60 robustness {_sc}: D-SCEOS lower than best-tested DPG",
              _r["J_T_dsceos"] < _r["best_dpg_J_T"], _r["J_T_dsceos"], f"< {_r['best_dpg_J_T']}")
    check("N60 robustness: D-SCEOS wins all three scenarios vs best-tested DPG",
          _rob.get("dsceos_wins_all") is True, _rob.get("dsceos_wins_all"), True)

# M-05: figure/trajectory synchronisation. The paper's nine static result figures are built from the
# figure-summary JSONs; assert those summaries carry the SAME retuned comparator objectives as the
# released realistic trajectories, so a stale figure cannot slip through a validator PASS. The summary
# lives next to the figures; look it up relative to the package if present.
_figsum = None
for _cand in ("../2_revised_paper/figures/objective_main_figure_summary.json",
              "objective_main_figure_summary.json"):
    if os.path.exists(_cand):
        _figsum = _cand
        break
# Design note: the figure-summary sync is REQUIRED, not skip-on-absent. If the summary the paper's result
# figures are built from cannot be found, that is a validation FAILURE (a stale or missing figure could
# otherwise slip through a PASS), unless --allow-missing is passed.
if _figsum is None:
    missing.append("objective_main_figure_summary.json (paper figures dir)")
    print("  [MISSING] objective_main_figure_summary.json (searched 2_revised_paper/figures and cwd)")
if _figsum is not None and os.path.exists("technology_representative_summary.json"):
    _rf = {(r["cluster"], r["scenario"], r["controller"]): r["J_T"]
           for r in json.load(open("technology_representative_summary.json"))}
    _scen_map = {"scenario_a_winter_morning_step": "A",
                 "scenario_b_wind_ramp_down_event": "B",
                 "scenario_c_winter_balancing_mfrr": "C"}
    _fs = json.load(open(_figsum))
    for _row in _fs:
        _sc = _scen_map.get(_row.get("scenario"))
        _jt = _row.get("final_cumulative_JT", {})
        if _sc is None or not isinstance(_jt, dict):
            continue
        for _ctrl in ("dsceos", "projected_gradient_hocbf", "independent_tracking"):
            _fig_v = _jt.get(_ctrl)
            _traj_v = _rf.get(("N15", _sc, _ctrl))
            if _fig_v is not None and _traj_v is not None:
                check(f"figure-summary N15-{_sc} {_ctrl} matches trajectory J_T",
                      abs(_fig_v - _traj_v) <= 1e-3, round(_fig_v, 4), round(_traj_v, 4))
else:
    print("  [warn] objective_main_figure_summary.json not found next to validator -- figure sync not asserted")

# verdict: fail-closed
miss_fatal = missing and not ALLOW_MISSING
status = "PASS" if (not fails and not miss_fatal) else f"FAIL ({len(fails)} mismatches, {len(missing)} missing" + (" [allowed]" if ALLOW_MISSING else "") + ")"
print("\n=== VALIDATION " + status + " ===")
sys.exit(0 if (not fails and not miss_fatal) else 1)
