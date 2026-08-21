"""Reproduction driver + manifest for the published D-SCEOS v1.0.0 release. Run from package root.
  python3 run_all.py            # list all entries (no execution)
  python3 run_all.py --list     # same
  python3 run_all.py --all      # execute every entry IN ORDER, fail-fast, check tolerances
  python3 run_all.py <substr>   # execute the entries whose command matches <substr>
Exit code is non-zero if any executed entry fails (subprocess error or tolerance check).
"""
import sys as _sys
_sys.dont_write_bytecode = True  # never leave __pycache__ inside the released package (source-integrity Design note); also avoids stale-bytecode confusion when a module is
# edited and restored within the same second during fault injection
import subprocess, sys, os, json, re
os.chdir(os.path.dirname(os.path.abspath(__file__)))
# the inline sanity checkers read their reference values from expected_results.json, the SINGLE
# source of published expectations, instead of repeating literals here. Hard-coded literals in this
# file silently went stale when the objective was re-defined  even though
# expected_results.json had been re-baselined, so the drift class is removed structurally.
_EXP = json.load(open("expected_results.json"))

# Design note (fail-closed agent-loss evidence): a per-run provenance token. run_all mints one id
# per invocation, exports it as DSCEOS_RUN_ID before running any stage, and the agent-loss producer
# writes it INTO each artefact. The checker then requires the on-disk JSON to carry exactly this run's
# id -- so a stale packaged file (even one `touch`-ed to look fresh) fails, because it cannot contain a
# token that is minted fresh each run. This is content-based provenance, not a timestamp heuristic.
import uuid as _uuid
_RUN_ID = _uuid.uuid4().hex
os.environ["DSCEOS_RUN_ID"] = _RUN_ID

def chk_agent_loss(_):
    # Fail-closed agent-loss evidence. FRESHNESS is enforced upstream by run_one's pre-clean (audit source-integrity
    # C-01): the numeric files and their sidecars are DELETED before the producer runs, so a no-op/crashed
    # producer leaves them missing and this checker fails at the existence test. This checker then verifies,
    # for BOTH scenarios:
    #  (a) the numeric JSON and its sidecar exist (they can only exist if the producer actually ran, given
    #      the pre-clean);
    #  (b) the sidecar carries THIS run's id (defence in depth; the pre-clean already guarantees freshness);
    #  (c) the sidecar is CONSISTENT with the numeric artefact -- bound_sha256 == sha256(run_id||numeric)
    #      and numeric_sha256/size match -- so a mismatched/swapped pair is caught (this is a consistency
    #      binding computed from public data, NOT a secret-keyed MAC and NOT by itself a freshness proof);
    #  (d) an environment stamp is present; and
    #  (e) the published capacity/first-return numbers match.
    import hashlib as _hl
    ex = _EXP.get("run_only_diagnostics", {}).get("agent_loss", {})
    msgs = []
    for tag in ("A", "B"):
        path = f"agent_loss_{tag}.json"
        prov = f"agent_loss_{tag}.provenance.json"
        if not os.path.exists(path):
            return False, f"{path} missing (producer did not write it)"
        if not os.path.exists(prov):
            return False, f"{prov} missing (no provenance sidecar; producer did not write it this run)"
        pj = json.load(open(prov))
        if pj.get("run_id") != _RUN_ID:
            return False, (f"{prov} is stale / not regenerated this run "
                           f"(run_id {pj.get('run_id')!r} != {_RUN_ID!r}); a touched packaged file cannot pass")
        # (c) sidecar<->numeric CONSISTENCY (not a freshness proof; freshness is the pre-clean's job).
        # bound_sha256 == sha256(run_id || numeric_bytes) catches a mismatched/swapped numeric+sidecar pair.
        with open(path, "rb") as _fh:
            _num = _fh.read()
        _recomp_plain = _hl.sha256(_num).hexdigest()
        _recomp_bound = _hl.sha256(pj.get("run_id", "").encode() + b"\x00" + _num).hexdigest()
        if pj.get("bound_sha256") != _recomp_bound:
            return False, (f"{tag} numeric/sidecar inconsistent "
                           f"(bound hash mismatch): the numeric JSON and its sidecar do not correspond")
        if pj.get("numeric_sha256") != _recomp_plain:
            return False, (f"{tag} numeric artefact hash != sidecar numeric_sha256 "
                           f"({_recomp_plain[:12]}... vs {str(pj.get('numeric_sha256'))[:12]}...)")
        if pj.get("numeric_bytes") != len(_num):
            return False, f"{tag} numeric byte size {len(_num)} != sidecar {pj.get('numeric_bytes')}"
        if not pj.get("environment", {}).get("python"):
            return False, f"{prov} missing environment stamp"
        d = json.loads(_num.decode())
        cv = d.get("max_cap_violation", d.get("max_capacity_violation", 1.0))
        if cv > 1e-9:
            return False, f"{tag} capacity violation {cv} > 0"
        _env = pj["environment"]
        msgs.append(f"{tag} provenance+hash-ok (np {_env.get('numpy')}, py {_env.get('python')}), cap-viol {cv:.1e}")
    if "A_first_return_min" in ex:
        a = json.load(open("agent_loss_A.json"))
        if abs(a["first_return_min"] - ex["A_first_return_min"]) > 0.2:
            return False, f"A first-return {a['first_return_min']} != expected {ex['A_first_return_min']}"
    return True, "; ".join(msgs)

def chk_r16(_):
    got=json.load(open("reruns/filter_on_off_reference.json"))
    pd_off=[r for r in got if r["controller"]=="independent_tracking" and not r["filter"]][0]
    ref=_EXP["filter"]
    ok=(abs(pd_off["J_T"]-ref["pd_off_JT"])<1e-2
        and abs(pd_off["cap_viol"]-ref["pd_off_viol"])<1e-2)
    return ok, f"PD-off J_T={pd_off['J_T']} viol={pd_off['cap_viol']}"
def chk_energy(_):
    d=json.load(open("reruns/economic_actuation_reference.json"))
    a=[r for r in d if r["scenario"]=="A" and r["controller"]=="dsceos"][0]
    ref=_EXP["economic_costwtd_N60_dsceos"]["A"]
    return abs(a["cost_weighted"]-ref)<1e-2, f"D-SCEOS A cost-wtd={a['cost_weighted']} (exp {ref})"
def chk_realistic(_):
    d=json.load(open("technology_representative_summary.json")); rows={(r['cluster'],r['scenario'],r['controller']):r for r in d}
    a=rows[('N15','A','dsceos')]; ref=_EXP["realistic_JT"]["N15_A"]["dsceos"]
    return abs(a['J_T']-ref)<2e-3, f"N15-A DS J_T={a['J_T']:.4f} (exp {ref})"
def chk_ladder(_):
    d=json.load(open("stress_ladder_summary.json"))
    soft=[r for r in d if r['regime']=='soft'][0]; ref=_EXP["ladder_JT_dsceos"]["soft"]
    return abs(soft['dsceos']-ref)<1e-2, f"soft DS J_T={soft['dsceos']:.3f} (exp {ref})"

# Status categories (m-07):
#   [inline-check]    this stage has an inline sanity checker in run_all itself
#   [validator-asserted] this stage regenerates a JSON whose published fields are asserted
#                        field-by-field by validate_results.py (fail-closed)
#   [run-only]        reproduced by the script, deliberately NOT asserted (hardware-dependent)
# Freshness-checked producer outputs deleted BEFORE the stage runs (audit source-integrity C-01). A no-op/crashed
# producer then leaves these MISSING and the stage checker fails closed; only an actually-executing
# producer recreates them. Keyed by stage label.
_PRE_RUN_CLEAN = {
    "Agent-loss (agent-loss/agent-loss)": [
        "agent_loss_A.json", "agent_loss_B.json",
        "agent_loss_A.provenance.json", "agent_loss_B.provenance.json",
    ],
}
_VALIDATOR_ASSERTED = {
    "Realistic J_T (Tab. scalability)", "Stress ladder J_T (Tab. ladder)",
    "Parameter sensitivity (parameter sensitivity)", "Baseline tuning (baseline tuning)", "Gateway sweep (gateway sensitivity)",
    "Economic actuation (economic actuation)", "Agent-loss (agent-loss/agent-loss)", "Comm-outage (baseline tuning0)",
    "Filter on/off (filter on/off, parameterized)", "Stress-ladder safety diagnostics",
    "N15-A control-displacement diagnostic", "Gershgorin vs optimal gain (consensus-gain comparison)",
    "Timing + benchmark (computational timing/3.8)",
    # these stages ARE checked (stack-robust conclusions) by validate_results.py
    "Per-agent fallback distribution",
    "Monte Carlo A: state/cost/capacity/graph",
    "Monte Carlo B: + noise/loss/delay on the consensus channel",
    "Monte Carlo C: two-dimensional weight grid",
    "Stronger distributed comparator: primal-dual",
    "Independent fleet compositions at several sizes",
    "Real solver + communication profile",
    "R1-R4 diagnostic matrix (retuned comparators)",
    # audit: these N=60 stages ARE asserted (stack-robust conclusions / strict ordering)
    "Monte Carlo A at N=60 (matched loading, the design rationale)",
    "N=60 ordering robustness to DPG retuning",
}
# standalone regression tests: they assert internally, they do NOT produce a JSON that the final
# expected-results validator re-checks.
_STANDALONE_TESTS = {
    "Physical-aggregate regression", "Graph-config regression (effective + source config)",
    "Source-hygiene regression (single-source + undefined names)",
    "Full expected-results validator",
    # self-asserting deterministic regressions (assert internally to fixed tolerances, not hardware)
    "Exact-gradient regression",
    "Exact-CLF/Vdot derivation",
    "Communication physical-units regression (C-04)",
    "N15/N60 graph guard",
    # deterministic: regenerates each MC topology from its seed and asserts the published edge_sha,
    # fail-closed on any missing required campaign (audit Sig#7); not hardware-dependent.
    "Topology reproduction from seed",
}

_PARTIALLY_ASSERTED = {"Timing + benchmark (computational timing/3.8)":
    "[deterministic fields asserted by validate_results.py; wall-clock ms NOT asserted]"}


def _status(label, checker):
    if label in _PARTIALLY_ASSERTED:
        return _PARTIALLY_ASSERTED[label]
    if label in _STANDALONE_TESTS:
        return "[standalone self-asserting test]"
    if label in _VALIDATOR_ASSERTED:
        return ("[inline-check + output asserted by validate_results.py]" if checker
                else "[output asserted by validate_results.py]")
    return "[inline-check]" if checker else "[run-only: hardware-dependent, not asserted]"

MANIFEST = [
 ("Realistic J_T (Tab. scalability)", "python3 reruns/batch_realistic.py", chk_realistic),
 ("Stress ladder J_T (Tab. ladder)", "python3 reruns/ladder_rerun.py", chk_ladder),
 ("Gershgorin vs optimal gain (consensus-gain comparison)", "python3 reruns/consensus_gain_comparison.py", None),
 ("Parameter sensitivity (parameter sensitivity)", "python3 reruns/parameter_sensitivity.py", None),
 ("Baseline tuning (baseline tuning)", "python3 reruns/baseline_tuning.py", None),
 ("Gateway sweep (gateway sensitivity)", "python3 reruns/gateway_sensitivity.py", None),
 ("Timing + benchmark (computational timing/3.8)", "python3 reruns/computational_cost_benchmark.py", None),
 ("Economic actuation (economic actuation)", "python3 reruns/economic_actuation_metric.py", chk_energy),
 ("Agent-loss (agent-loss/agent-loss)", "python3 reruns/agent_loss_experiment.py", chk_agent_loss),
 ("Comm-outage (baseline tuning0)", "python3 reruns/communication_outage.py", None),
 ("Filter on/off (filter on/off, parameterized)", "python3 reruns/filter_on_off.py", chk_r16),
 ("Physical-aggregate regression", "python3 test_physical_aggregate.py", None),
 ("Communication physical-units regression (C-04)", "python3 test_comm_physical_units.py", None),
 ("Topology reproduction from seed", "python3 test_topology_reproduction.py", None),
 ("Source-hygiene regression (single-source + undefined names)", "python3 test_source_hygiene.py", None),
 ("Graph-config regression (effective + source config)", "python3 test_graph_config.py", None),
 ("Stress-ladder safety diagnostics", "python3 reruns/ladder_diagnostics.py", None),
 ("N15-A control-displacement diagnostic", "python3 reruns/control_displacement.py", None),
 ("Per-agent fallback distribution", "python3 reruns/fallback_per_agent.py", None),
 ("N=60 scalability table (component-wise P/Q)", "python3 reruns/scalability_table.py", None),
 ("R1-R4 diagnostic matrix (retuned comparators)", "python3 reruns/acceptability_matrix.py", None),
 ("Exact-gradient regression", "python3 test_exact_gradient.py", None),
 ("Exact-CLF/Vdot derivation", "python3 test_clf_derivation.py", None),
 ("Monte Carlo A: state/cost/capacity/graph",
  "python3 reruns/monte_carlo.py --experiment A --cluster realistic_15 --seeds 30", None),
 ("Monte Carlo B: + noise/loss/delay on the consensus channel",
  "python3 reruns/monte_carlo.py --experiment B --cluster realistic_15 --seeds 30", None),
 ("Monte Carlo C: two-dimensional weight grid",
  "python3 reruns/monte_carlo.py --experiment C --cluster realistic_15 --seeds 10", None),
 ("Monte Carlo A at N=60 (matched loading, the design rationale)",
  "python3 reruns/monte_carlo.py --experiment A --cluster realistic_60 --seeds 30", None),
 ("N=60 ordering robustness to DPG retuning",
  "python3 reruns/dpg_n60_retune_robustness.py", None),
 ("Result figures: main objective",
  "python3 code/generate_objective_main_figures.py --results-dir results --output-dir ../2_revised_paper/figures", None),
 ("Result figures: appendix certification",
  "python3 code/generate_appendix_certification_figures.py --results-dir results --output-dir ../2_revised_paper/figures", None),
 ("Stronger distributed comparator: primal-dual",
  "python3 reruns/primal_dual_comparator.py", None),
 ("Independent fleet compositions at several sizes",
  "python3 reruns/fleet_scaling_profile.py --experiment compositions --sizes 15,30,60,120 --seeds 20", None),
 ("Real solver + communication profile",
  "python3 reruns/fleet_scaling_profile.py --experiment profile --sizes 15,30,60,120", None),
 ("Full expected-results validator", "python3 validate_results.py", None),
]
def run_one(label, cmd, checker):
    # Portability (source-integrity): the manifest commands are written with a leading "python3 " for readability,
    # but "python3" is not the interpreter name on every platform (e.g. Windows, where it triggers the
    # Microsoft Store alias). Rewrite the leading token to the interpreter that is actually running
    # this script, so `python`, `python3` or an absolute path all work identically.
    exec_cmd = re.sub(r"^python3\b", subprocess.list2cmdline([sys.executable]), cmd)
    # FRESH-GENERATION ENFORCEMENT (audit source-integrity C-01). A checker that only inspects the produced artefacts
    # cannot prove they were generated THIS run: every field (including a public bound hash) is
    # recomputable by a no-op that reads the run id and the stale numeric bytes. Freshness must instead be
    # enforced by the ENVIRONMENT: before running a producer whose outputs are freshness-checked, delete
    # those outputs, so a no-op/crashed producer leaves them MISSING and the checker fails closed. Only an
    # actually-executing producer can recreate them (it atomically rewrites both the numeric file and its
    # sidecar). This defeats the "no-op recomputes a valid sidecar over stale numeric JSON" attack.
    for _p in _PRE_RUN_CLEAN.get(label, ()):
        try:
            if os.path.exists(_p):
                os.remove(_p)
                print(f"  [pre-clean] removed stale {_p} (must be regenerated this run)")
        except OSError as _e:
            print(f"  [pre-clean] WARNING could not remove {_p}: {_e}")
    print(f"\n### {label}\n$ {exec_cmd}")
    rc = subprocess.run(exec_cmd, shell=True).returncode
    if rc != 0:
        print(f"  [RUN] FAILED (exit {rc})"); return False
    if checker:
        try: ok, msg = checker(rc)
        except Exception as e: print(f"  [CHECK] FAIL — {e}"); return False
        print(f"  [CHECK] {'PASS' if ok else 'FAIL'} — {msg}"); return ok
    print("  [RUN] ok"); return True
def main():
    args = sys.argv[1:]
    if args and args[0] == "--cold":
        # Cold regeneration: remove every artefact a previous run could be resumed from, so that
        # --all really starts from nothing. Without this the Monte Carlo scripts silently reuse their
        # *.partial.json checkpoints and the run is not a cold reproduction.
        import glob
        removed = []
        for pat in ("*.partial.json", "results/*.partial.json", "results/*/state_history.npz",
                    "agent_loss_*.json", "agent_loss_*.provenance.json"):
            for f in glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)), pat)):
                try:
                    os.remove(f); removed.append(os.path.basename(f))
                except OSError:
                    pass
        print(f"[cold] removed {len(removed)} resumable artefact(s): "
              f"{', '.join(sorted(set(removed))[:8])}{' ...' if len(removed) > 8 else ''}")
        args = ["--all"] + list(args[1:])

    if not args or args[0] in ("--list", "-l", "--help", "-h"):
        print("Entries (pass --all to execute fail-fast, or a substring to run a subset):")
        print(f"(the leading 'python3' shown below is auto-rewritten to the running interpreter "
              f"'{sys.executable}' at execution time, so 'python'/'python3' both work.)")
        for label, cmd, ch in MANIFEST:
            print(f"  - {label}\n      {cmd}   {_status(label, ch)}")
        return 0
    fail_fast = (args[0] == "--all")
    sel = MANIFEST if fail_fast else [m for m in MANIFEST
           if any(a in m[1] or a.lower() in m[0].lower() for a in args)]  # match command OR label
    if not sel:
        print(f"No manifest entry matches {args!r}. Use --list to see entries."); return 2
    results=[]
    for m in sel:
        ok = run_one(*m); results.append((m[0], ok))
        if fail_fast and not ok:
            print(f"\n=== FAIL-FAST: stopping at '{m[0]}' ==="); break
    # If any executed stage regenerated an output that validate_results.py asserts, run the final
    # validator automatically -- otherwise a subset run could leave a regenerated JSON unchecked
    # while still printing "[RUN] ok" (source-integrity Design note). BUT if any stage already FAILED, skip the
    # auto-validator: running it would print a misleading "VALIDATION PASS" on the packaged JSONs even
    # though a producer failed this run (audit source-integrity M-01). The aggregate exit stays 1.
    ran_labels = {m[0] for m in sel}
    any_failed = any(not r for _, r in results)
    if any_failed:
        print("\n### Skipping the auto-validator: a stage FAILED this run, so a numeric PASS on the "
              "packaged artefacts would be misleading. Fix the failing stage and re-run.")
    elif ran_labels & _VALIDATOR_ASSERTED and "Full expected-results validator" not in ran_labels:
        print("\n### Auto-running the final validator (a validator-asserted output was regenerated)")
        ok = run_one("Full expected-results validator", "python3 validate_results.py", None)
        results.append(("Full expected-results validator (auto)", ok))
    print("\n=== SUMMARY ===")
    for nme,r in results: print(f"  {'ok ' if r else 'ERR'}  {nme}")
    return 0 if all(r for _,r in results) else 1
if __name__ == "__main__":
    sys.exit(main())
