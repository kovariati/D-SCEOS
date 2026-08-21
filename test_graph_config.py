"""Regression guard for the released communication-graph configuration.

Three complementary layers, with HONEST scope:

1. SOURCE-LITERAL checks (AST): the runner/validator ClusterConfig and the validator's
   make_fixed_local_graph calls must use the literal authoritative radius, and any NON-literal
   radius is rejected outright (so a decoy literal cannot mask a variable). These catch direct
   source edits; they cannot catch a runtime mutation.

2. EFFECTIVE DRY-RUN check: the realistic runner is actually executed up to the point where it
   would call run_simulation, and the SimulationConfig it really built is inspected. The effective
   ClusterConfig must carry the authoritative radius/neighbour count, and the graph rebuilt from
   that effective config must match the released spectral fingerprint. This DOES catch a runtime
   mutation such as ``cluster_cfg.communication_radius = 0.30`` after construction, which layer 1
   alone would miss.

3. FINGERPRINT check: the released N=60 graph rebuilt at the authoritative radius must match the
   released d_max / lambda_N used in the Gershgorin table.

The authoritative constants live in ``graph_config.py`` (single source of truth). Note that the
full validator recomputes ITS OWN graphs; it does not execute the realistic runner, so layer 2 --
not the validator -- is what protects against effective runner-configuration drift.
"""
import sys as _sys
_sys.dont_write_bytecode = True  # never leave __pycache__ inside the released package (source-integrity Design note); also avoids stale-bytecode confusion when a module is
# edited and restored within the same second during fault injection
import os, ast
import numpy as np

_H = os.path.dirname(os.path.abspath(__file__))

import ast

def _cluster_radius_findings(path):
    """Return (constants, has_non_constant) for every communication_radius passed to a
    ClusterConfig(...) call, parsed via AST. `constants` are the literal float values; a
    True `has_non_constant` means at least one ClusterConfig used a NON-literal (variable /
    expression) radius, which a decoy literal ClusterConfig(0.45) could otherwise mask."""
    tree = ast.parse(open(path, encoding="utf-8").read())
    constants, has_non_constant, approved = [], False, 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and (
                (isinstance(node.func, ast.Name) and node.func.id == "ClusterConfig") or
                (isinstance(node.func, ast.Attribute) and node.func.attr == "ClusterConfig")):
            for kw in node.keywords:
                if kw.arg == "communication_radius":
                    if isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, (int, float)):
                        constants.append(float(kw.value.value))
                    elif _is_approved_ref(kw.value):
                        approved += 1
                    else:
                        has_non_constant = True
    return constants, has_non_constant, approved

# names that are the APPROVED single source of the radius (graph_config.py). Any other
# non-literal radius is rejected, because a decoy literal could otherwise mask it.
_APPROVED_RADIUS_NAMES = {"AUTHORITATIVE_COMMUNICATION_RADIUS", "_R045"}
_APPROVED_COUNT_NAMES = {"AUTHORITATIVE_NEIGHBOUR_COUNT", "_K4"}


def _is_approved_ref(node):
    if isinstance(node, ast.Name):
        return node.id in _APPROVED_RADIUS_NAMES
    if isinstance(node, ast.Attribute):
        return node.attr in _APPROVED_RADIUS_NAMES
    return False


def _assert_all_radius_045(path, who):
    _authoritative()                      # ensure code/ is on sys.path
    import graph_config as gc             # direct import: controlled names are never aliased
    assert abs(gc.AUTHORITATIVE_COMMUNICATION_RADIUS - 0.45) < 1e-9, \
        "graph_config.AUTHORITATIVE_COMMUNICATION_RADIUS is no longer 0.45"
    consts, has_var, approved = _cluster_radius_findings(path)
    assert consts or approved, f"no ClusterConfig(communication_radius=...) found in {who}"
    # every radius must be either the literal 0.45 or the approved graph_config constant
    assert not has_var, (
        f"{who} builds a ClusterConfig with a communication_radius that is neither the literal 0.45 "
        f"nor the approved graph_config constant (cannot verify it)")
    assert all(abs(v - 0.45) < 1e-9 for v in consts), f"{who} ClusterConfig radii != 0.45: {consts}"

def _fixed_graph_radius_literals(path):
    """Return literal radius args (2nd positional or radius=... kw) of make_fixed_local_graph(...)
    calls, and a flag for any non-literal radius, parsed via AST."""
    tree = ast.parse(open(path, encoding="utf-8").read())
    consts, has_non_constant = [], False
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and (
                (isinstance(node.func, ast.Name) and node.func.id == "make_fixed_local_graph") or
                (isinstance(node.func, ast.Attribute) and node.func.attr == "make_fixed_local_graph")):
            rad = None
            if len(node.args) >= 2:
                rad = node.args[1]
            for kw in node.keywords:
                if kw.arg in ("radius", "communication_radius"):
                    rad = kw.value
            if rad is None:
                continue
            if isinstance(rad, ast.Constant) and isinstance(rad.value, (int, float)):
                consts.append(float(rad.value))
            elif isinstance(rad, ast.Attribute) and rad.attr == "communication_radius":
                pass  # cc.communication_radius: already covered by the ClusterConfig check
            elif _is_approved_ref(rad):
                pass  # the approved graph_config constant
            else:
                has_non_constant = True
    return consts, has_non_constant

def test_validator_benchmark_graph_radius_045():
    """The validator's benchmark/R2 graph builder must use radius 0.45 (literal make_fixed_local_graph
    calls), so a change to that literal is caught here, not only via the spectral expected values."""
    consts, has_var = _fixed_graph_radius_literals(os.path.join(_H, "validate_results.py"))
    assert not has_var, "validate_results.py builds make_fixed_local_graph with an unrecognized non-literal radius"
    assert all(abs(v - 0.45) < 1e-9 for v in consts), f"benchmark-graph radii != 0.45: {consts}"

def test_runner_builds_cluster_at_radius_045():
    """The realistic runner must construct its ClusterConfig at communication_radius == 0.45,
    with no non-literal radius that a decoy literal could mask."""
    _assert_all_radius_045(os.path.join(_H, "code", "run_realistic_scenario.py"), "run_realistic_scenario.py")

def test_validator_builds_cluster_at_radius_045():
    """The validator must recompute its optimum/target-estimator graphs at communication_radius == 0.45."""
    _assert_all_radius_045(os.path.join(_H, "validate_results.py"), "validate_results.py")

def test_ladder_graph_matches_released_fingerprint():
    """The 30-unit stress-ladder graph, rebuilt from graph_config's LADDER_* constants, must match
    the released fingerprint exactly (edges, d_max, lambda_N and the adjacency hash). The ladder now
    passes ALL FOUR graph parameters explicitly, so it no longer depends on any dataclass default
    (source-integrity audit major 1)."""
    import sys
    sys.path.insert(0, os.path.join(_H, "code")); sys.path.insert(0, _H)
    _authoritative()                      # ensure code/ is on sys.path
    import graph_config as gc             # direct import: controlled names are never aliased
    from dsceos_validation import ClusterConfig, make_fixed_local_graph, make_physical_layout
    cc = ClusterConfig(initial_spread=0., initial_speed_scale=0.,
                       **gc.LADDER_UNIT_MIX, **gc.ladder_cluster_kwargs())
    n = gc.LADDER_FINGERPRINT["n"]
    # the DECLARED fleet composition must add up to the fingerprinted fleet size, so the source of
    # truth cannot drift away from the graph it is supposed to describe (source-integrity Design note)
    assert sum(gc.LADDER_UNIT_MIX.values()) == n, (
        f"LADDER_UNIT_MIX sums to {sum(gc.LADDER_UNIT_MIX.values())}, fingerprint says {n}")
    W = make_fixed_local_graph(make_physical_layout(n, cc),
                               cc.communication_radius, cc.neighbour_count)
    L = np.diag(W.sum(1)) - W
    fp = gc.LADDER_FINGERPRINT
    assert int((W > 0).sum() // 2) == fp["edges"], f"ladder edge count drift: {int((W > 0).sum() // 2)}"
    assert abs(float(W.sum(1).max()) - fp["d_max"]) < gc.FINGERPRINT_TOL, "ladder d_max drift"
    assert abs(float(max(np.linalg.eigvalsh(L))) - fp["lambda_N"]) < gc.FINGERPRINT_TOL, \
        "ladder lambda_N drift"
    assert gc.adjacency_hash(W) == fp["adjacency_sha256"], "ladder adjacency hash drift"


def test_graph_properties_match_gershgorin_table():
    """Rebuild the N=60 radius-0.45 graph and check d_max and lambda_N against the released values."""
    import sys
    sys.path.insert(0, os.path.join(_H, "code")); sys.path.insert(0, _H)
    from dsceos_validation import (ClusterConfig, make_fixed_local_graph, make_physical_layout)
    _authoritative()                      # ensure code/ is on sys.path
    import graph_config as gc             # direct import: controlled names are never aliased
    cc = ClusterConfig(n_thermal=12, n_storage=12, n_hydrogen=12, n_emobility=12, n_industrial=12,
                       initial_spread=0., initial_speed_scale=0.,
                       **gc.authoritative_cluster_kwargs())
    W = make_fixed_local_graph(make_physical_layout(60, cc), cc.communication_radius, cc.neighbour_count)
    L = np.diag(W.sum(1)) - W
    d_max = float(W.sum(1).max()); lam_N = float(max(np.linalg.eigvalsh(L)))
    assert abs(d_max - 6.983) < 5e-2, f"d_max {d_max} != 6.983"
    assert abs(lam_N - 8.016) < 5e-2, f"lambda_N {lam_N} != 8.016"


def _authoritative():
    """The single authoritative graph configuration (code/graph_config.py), imported by the
    realistic runner, the validator and these tests alike."""
    import sys
    sys.path.insert(0, os.path.join(_H, "code")); sys.path.insert(0, _H)
    import graph_config as _graph_config_module
    return _graph_config_module


def _dry_run_capture(cluster_arg, extra_args=()):
    """Execute the realistic runner up to its run_simulation call and return the SimulationConfig it
    actually built (never runs the simulation itself)."""
    import sys, tempfile
    sys.path.insert(0, os.path.join(_H, "code")); sys.path.insert(0, _H)
    import dsceos_validation as dv
    import run_realistic_scenario as rrs

    captured = {}

    class _Stop(Exception):
        pass

    def _capture(cfg):
        captured["cfg"] = cfg
        raise _Stop()

    orig_run, orig_argv = dv.run_simulation, sys.argv
    dv.run_simulation = _capture
    tmp = tempfile.mkdtemp()
    sys.argv = ["run_realistic_scenario.py", "--scenario", "scenario_a_winter_morning_step",
                "--controller", "dsceos", "--cluster", cluster_arg,
                "--outdir", tmp] + list(extra_args)
    try:
        try:
            rrs.main()
        except _Stop:
            pass
    finally:
        dv.run_simulation, sys.argv = orig_run, orig_argv
    assert "cfg" in captured, f"runner ({cluster_arg}) never reached run_simulation; dry-run capture failed"
    return captured["cfg"]


def _check_effective_graph(cluster_arg, n_units, extra_args=()):
    import sys
    sys.path.insert(0, os.path.join(_H, "code")); sys.path.insert(0, _H)
    import dsceos_validation as dv
    _authoritative()                      # ensure code/ is on sys.path
    import graph_config as gc             # direct import: controlled names are never aliased
    cfg = _dry_run_capture(cluster_arg, extra_args)
    cc = cfg.cluster  # provenance-ok: effective SimulationConfig capture
    gc.assert_authoritative_cluster(cc, who=f"realistic runner effective ClusterConfig ({cluster_arg})")
    W = dv.make_fixed_local_graph(dv.make_physical_layout(n_units, cc),
                                  cc.communication_radius, cc.neighbour_count)
    L = np.diag(W.sum(1)) - W
    fp = gc.GRAPH_FINGERPRINTS[n_units]
    assert int((W > 0).sum() // 2) == fp["edges"], (
        f"{cluster_arg}: effective edge count {int((W > 0).sum() // 2)} != released {fp['edges']}")
    assert abs(float(W.sum(1).max()) - fp["d_max"]) < gc.FINGERPRINT_TOL, f"{cluster_arg}: d_max drift"
    assert abs(float(max(np.linalg.eigvalsh(L))) - fp["lambda_N"]) < gc.FINGERPRINT_TOL, \
        f"{cluster_arg}: lambda_N drift"
    # exact edge-set identity, not just spectral summaries
    got = gc.adjacency_hash(W)
    assert got == fp["adjacency_sha256"], (
        f"{cluster_arg}: effective adjacency hash {got[:16]}... != released {fp['adjacency_sha256'][:16]}...")


def test_runner_effective_graph_N15():
    """EFFECTIVE-CONFIG dry run for the N=15 realistic runner: the SimulationConfig the runner really
    builds must carry the authoritative graph parameters, and the graph rebuilt from it must match
    the released fingerprint AND adjacency hash. Catches runtime configuration drift that the
    source-literal AST checks cannot see."""
    _check_effective_graph("realistic_15", 15)


def test_runner_effective_graph_N60():
    """EFFECTIVE-CONFIG dry run for the N=60 realistic runner branch (with the --target-multiplier 4.0
    path actually used to produce the released N=60 results), so a drift introduced only for the
    larger fleet is caught as well."""
    _check_effective_graph("realistic_60", 60, extra_args=("--target-multiplier", "4.0"))


if __name__ == "__main__":
    # direct-run mode: run the checks and print a summary (no sys.exit at import time)
    import sys, traceback
    ok = True
    for fn in (test_runner_builds_cluster_at_radius_045, test_validator_builds_cluster_at_radius_045, test_validator_benchmark_graph_radius_045, test_runner_effective_graph_N15, test_runner_effective_graph_N60,
               test_ladder_graph_matches_released_fingerprint,
               test_graph_properties_match_gershgorin_table):
        try:
            fn(); print(f"PASS: {fn.__name__}")
        except AssertionError as e:
            ok = False; print(f"FAIL: {fn.__name__}: {e}")
        except Exception:
            ok = False; print(f"ERROR: {fn.__name__}"); traceback.print_exc()
    sys.exit(0 if ok else 1)
