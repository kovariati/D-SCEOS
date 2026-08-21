"""Source-hygiene regression tests for every released reproduction path.

SCOPE, STATED HONESTLY. This is a REGRESSION guard, not a general static proof about Python. What
the package actually guarantees is:
  * the three released BASE graphs are pinned by exact fingerprints (N=15 and N=60 realistic and
    N=30 stress ladder: edge count, d_max, lambda_N and a SHA-256 adjacency hash), and the realistic
    runner is additionally verified by an effective-configuration dry run. Event-modified graphs
    (the communication-outage blackout/partition topologies and the agent-loss reconfiguration) are
    NOT separately fingerprinted: they are defined by their event scripts and protected through the
    asserted output artefacts of those experiments;
  * the checks below catch every construction that a past audit demonstrated could break that
    single-sourcing or introduce an immediate NameError -- all of them are kept as a permanent
    corpus and re-verified on every run;
  * `pyflakes` is a MANDATORY, pinned, independent cross-check of the F821/F823 classes: if it is
    not installed the test FAILS rather than skipping.
  * The four entry points set `sys.dont_write_bytecode`, so running them directly leaves no
    `__pycache__` inside the package. NOTE: `pytest` rewrites assertions and manages its own caches
    BEFORE that flag can take effect, so a bare `python -m pytest` still creates `.pytest_cache/`
    and `__pycache__/`. Run `python3 -B -m pytest -q -p no:cacheprovider --assert=plain` for a fully
    cache-free pytest run (verified to leave zero artefacts); the packaging step strips them anyway.
It is NOT a claim that no conceivable Python construction can evade the static analysis. Dynamic
reassignment via `globals()`/`setattr`, `exec`, or an object created in another module remains out
of scope; the runtime dry run and the exact fingerprints are what protect the released results.

Two guards:

1. NAME RESOLUTION. Real Python scoping is modelled: module, function, lambda, class and
   COMPREHENSION scopes; definition-time expressions (decorators, defaults, annotations, class
   bases) attributed to the enclosing scope; `except ... as e` alias lifetime; `del` ending a
   binding; walrus targets binding in the enclosing scope; `match` capture patterns; and PEP 563
   (`from __future__ import annotations`) making annotations non-evaluated. Conditional bindings
   (`if cond: x = 1`) are deliberately NOT reported, since that is valid Python.

2. GRAPH-CONFIG PROVENANCE. Every released path must take the graph parameters from
   `code/graph_config.py`: all four parameters must be supplied explicitly (never left to a
   dataclass default), literals and untraceable `**kwargs` are rejected, constructor and
   graph-builder aliases (import-as, assignment, `functools.partial`) are resolved,
   `dataclasses.replace` overrides are checked, `ClusterConfig` / `SimulationConfig` /
   `authoritative_cluster_kwargs` must come from the right module and may not be shadowed, only the
   exact modules `graph_config` / `code.graph_config` count, aliases may not be re-bound in any
   scope, helpers reachable through an indirect alias stay unproven, and the capture marker is
   honoured only in the dry-run test for a `.cluster` assignment.

Run directly (``python3 test_source_hygiene.py``) or under pytest.
"""

import sys as _sys
_sys.dont_write_bytecode = True  # never leave __pycache__ inside the released package (source-integrity Design note); also avoids stale-bytecode confusion when a module is
# edited and restored within the same second during fault injection
import ast
import builtins
import os

_H = os.path.dirname(os.path.abspath(__file__))

_SKIP_FILES = {"graph_config.py"}   # the single place the authoritative numbers may live

# The simulator library exposes a generic CLI whose graph parameters are user-supplied arguments
# (with dataclass defaults); it is not a fixed released reproduction path, so the provenance rule
# does not apply to it. Its NAME analysis is still performed.
# The Monte Carlo harness exists precisely to REPLACE the authoritative fixed
# graph by freshly drawn random connected topologies, so it must substitute the graph builder;
# that is the experiment, not a deviation. It still takes the communication radius and the
# neighbour cap BY NAME from graph_config, so only the layout is randomised. Its NAME analysis
# remains active, as for dsceos_validation.py.
_PROVENANCE_EXEMPT = {os.path.join("code", "dsceos_validation.py"),
                      os.path.join("reruns", "monte_carlo.py"),
                      os.path.join("reruns", "ablation.py"),
                      # 3.8 harness: draws independent fleets on fresh random connected graphs, so it
                      # substitutes the graph builder by design; it takes the communication radius and
                      # neighbour cap BY NAME from graph_config. NAME analysis stays active.
                      os.path.join("reruns", "fleet_scaling_profile.py")}


_SKIP_DIRS = {"__pycache__", ".pytest_cache", ".ipynb_checkpoints", ".git", "audit_history"}


def released_python_paths():
    """Every released Python file, discovered RECURSIVELY from the package root (source-integrity design note: a
    file in a sub-directory such as reruns/nested/ previously escaped the guard entirely)."""
    out = []
    for root, dirs, files in os.walk(_H):
        dirs[:] = sorted(d for d in dirs if d not in _SKIP_DIRS)
        for name in sorted(files):
            if not name.endswith(".py") or name in _SKIP_FILES:
                continue
            out.append(os.path.relpath(os.path.join(root, name), _H))
    return sorted(out)


def _parse(rel):
    path = os.path.join(_H, rel)
    with open(path, encoding="utf-8") as fh:
        src = fh.read()
    tree = ast.parse(src, filename=rel)
    tree._hygiene_src = src
    tree._hygiene_file = os.path.basename(rel)
    return path, tree


# ------------------------------------------------------------- name analysis

_BUILTINS = set(dir(builtins)) | {"__file__", "__name__", "__doc__", "__spec__", "__package__"}

# set while analysing a module that uses `from __future__ import annotations` (PEP 563): annotations
# are then strings and are never evaluated, so they cannot raise NameError (source-integrity policy check 2)
_FUTURE_ANNOTATIONS = False


def _has_future_annotations(tree):
    return any(isinstance(n, ast.ImportFrom) and n.module == "__future__"
               and any(a.name == "annotations" for a in n.names) for n in ast.walk(tree))


def _params_of(fn):
    a = fn.args
    names = {x.arg for x in list(a.posonlyargs) + list(a.args) + list(a.kwonlyargs)}
    if a.vararg:
        names.add(a.vararg.arg)
    if a.kwarg:
        names.add(a.kwarg.arg)
    return names


def _bindings(node):
    """Earliest SOURCE line at which each name is bound anywhere inside ``node``.
    (ast.walk is breadth-first, so the minimum line must be taken explicitly.)"""
    bound = {}

    def _put(name, lineno):
        if name not in bound or lineno < bound[name]:
            bound[name] = lineno

    for n in ast.walk(node):
        if isinstance(n, ast.Name) and isinstance(n.ctx, (ast.Store, ast.Del)):
            _put(n.id, n.lineno)
        elif isinstance(n, (ast.Import, ast.ImportFrom)):
            for a in n.names:
                _put((a.asname or a.name).split(".")[0], n.lineno)
        elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            _put(n.name, n.lineno)
            for p in _params_of(n):
                _put(p, n.lineno)
        elif isinstance(n, ast.ClassDef):
            _put(n.name, n.lineno)
        elif isinstance(n, ast.Lambda):
            for p in _params_of(n):
                _put(p, n.lineno)
        elif isinstance(n, ast.ExceptHandler) and n.name:
            _put(n.name, n.lineno)
        elif isinstance(n, ast.Global):
            for nm in n.names:
                _put(nm, n.lineno)
    return bound


def _loads_in(node):
    """Every Name load anywhere inside ``node`` (used for definition-time expressions)."""
    return [(n.id, n.lineno) for n in ast.walk(node)
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)]


def _definition_time_loads(node):
    """Loads that a def/class/lambda evaluates IN ITS ENCLOSING SCOPE at definition time:
    decorators, default arguments, annotations and class bases/keywords. These run when the
    statement executes, so a name bound later in the enclosing scope is a genuine NameError
    (source-integrity design note cases 3-6)."""
    out = []
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        for d in node.decorator_list:
            out += _loads_in(d)
        a = node.args
        for d in list(a.defaults) + [k for k in a.kw_defaults if k is not None]:
            out += _loads_in(d)
        if not _FUTURE_ANNOTATIONS:
            for arg in list(a.posonlyargs) + list(a.args) + list(a.kwonlyargs) + \
                    [x for x in (a.vararg, a.kwarg) if x is not None]:
                if arg.annotation is not None:
                    out += _loads_in(arg.annotation)
            if node.returns is not None:
                out += _loads_in(node.returns)
    elif isinstance(node, ast.ClassDef):
        for d in node.decorator_list:
            out += _loads_in(d)
        for b in node.bases:
            out += _loads_in(b)
        for kw in node.keywords:
            out += _loads_in(kw.value)
    elif isinstance(node, ast.Lambda):
        a = node.args
        for d in list(a.defaults) + [k for k in a.kw_defaults if k is not None]:
            out += _loads_in(d)
    return out


def _direct_loads(node):
    """Loaded names evaluated in THIS scope. Nested function/lambda/class BODIES are skipped (they
    are their own scopes), but their definition-time expressions are attributed here."""
    out = []

    def _walk(n):
        for c in ast.iter_child_nodes(n):
            if isinstance(c, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
                out.extend(_definition_time_loads(c))
                continue
            if isinstance(c, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
                # only the first iterable is evaluated in THIS scope; everything else belongs to the
                # comprehension's own scope (source-integrity design note, 3.2.1, 3.2.5)
                if c.generators:
                    out.extend(_loads_in(c.generators[0].iter))
                continue
            if isinstance(c, ast.Name) and isinstance(c.ctx, ast.Load):
                out.append((c.id, c.lineno))
            _walk(c)

    # a lambda's scope is its single body expression: wrap it so the same walker applies
    _walk(ast.Expr(value=node.body) if isinstance(node, ast.Lambda) else node)
    return out


def _comprehension_targets(root):
    """Comprehension/generator targets have their own scope, and the element expression is written
    before the target, so they must not be reported as use-before-assignment."""
    out = set()
    for n in ast.walk(root):
        if isinstance(n, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
            for gen in n.generators:
                for sub in ast.walk(gen.target):
                    if isinstance(sub, ast.Name):
                        out.add(sub.id)
    return out


def _scope_bindings(node):
    """Names bound in THIS scope only: nested function/lambda/class bodies are their own scopes, so
    their locals must not be treated as available here (they would hide a real undefined name)."""
    bound = {}

    def _put(name, lineno):
        if name not in bound or lineno < bound[name]:
            bound[name] = lineno

    def _walk(n, top):
        for c in ast.iter_child_nodes(n):
            if isinstance(c, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                _put(c.name, c.lineno)        # the def binds its own name here
                continue
            if isinstance(c, ast.Lambda):
                continue
            if isinstance(c, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
                # comprehensions are their own scope, BUT a walrus inside one binds in THIS scope
                for sub in ast.walk(c):
                    if isinstance(sub, ast.NamedExpr) and isinstance(sub.target, ast.Name):
                        _put(sub.target.id, sub.target.lineno)
                continue
            if isinstance(c, ast.MatchAs) and c.name:
                _put(c.name, c.lineno)
            elif isinstance(c, ast.MatchStar) and c.name:
                _put(c.name, c.lineno)
            elif isinstance(c, ast.MatchMapping) and c.rest:
                _put(c.rest, c.lineno)
            if isinstance(c, ast.Name) and isinstance(c.ctx, ast.Store):
                _put(c.id, c.lineno)
            elif isinstance(c, (ast.Import, ast.ImportFrom)):
                for a in c.names:
                    _put((a.asname or a.name).split(".")[0], c.lineno)
            # NOTE: an `except ... as e` alias is deleted at the end of its handler, so it is NOT
            # bound at scope level (source-integrity design note.2); it is made available only inside the handler.
            elif isinstance(c, ast.Global):
                for nm in c.names:
                    _put(nm, c.lineno)
            _walk(c, False)

    _walk(node, True)
    return bound


def _comprehensions_of(scope_node):
    """Comprehension nodes belonging directly to this scope (not to a nested function/class)."""
    out = []

    def _walk(n):
        for c in ast.iter_child_nodes(n):
            if isinstance(c, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
                continue
            if isinstance(c, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
                out.append(c)
                continue
            _walk(c)

    _walk(ast.Expr(value=scope_node.body) if isinstance(scope_node, ast.Lambda) else scope_node)
    return out


def _handler_windows(scope_node):
    """`except ... as e` aliases with the line window in which they exist."""
    out = []
    for n in ast.walk(scope_node):
        if isinstance(n, ast.ExceptHandler) and n.name:
            last = max([n.lineno] + [ln for s in ast.walk(n)
                                     for ln in [getattr(s, "end_lineno", None) or
                                                getattr(s, "lineno", None)] if ln])
            out.append((n.name, n.lineno, last))
    return out


def _del_events(scope_node):
    """Lines at which names are deleted in this scope (a deleted name is gone until re-bound)."""
    out = {}
    for n in ast.walk(scope_node):
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Del):
            out.setdefault(n.id, []).append(n.lineno)
    return out


def _binding_lines(scope_node):
    """All lines at which each name is (re)bound in this scope, for del/rebind ordering."""
    out = {}
    for n in ast.walk(scope_node):
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store):
            out.setdefault(n.id, []).append(n.lineno)
        elif isinstance(n, (ast.Import, ast.ImportFrom)):
            for a in n.names:
                out.setdefault((a.asname or a.name).split(".")[0], []).append(n.lineno)
    return out


def _analyze_comprehension(comp, outer_available, scope_name, problems):
    """A comprehension is its own scope: its targets are bound only inside it, and only the FIRST
    iterable is evaluated outside (already checked by the enclosing scope)."""
    targets = set()
    for gen in comp.generators:
        for sub in ast.walk(gen.target):
            if isinstance(sub, ast.Name):
                targets.add(sub.id)
    available = set(outer_available) | targets | _BUILTINS
    inner_nodes = []
    for i, gen in enumerate(comp.generators):
        if i > 0:
            inner_nodes.append(gen.iter)      # later iterables run inside the comprehension scope
        inner_nodes.extend(gen.ifs)
    if isinstance(comp, ast.DictComp):
        inner_nodes.extend([comp.key, comp.value])
    else:
        inner_nodes.append(comp.elt)
    for nd in inner_nodes:
        # only the loads evaluated in THIS comprehension scope: a nested comprehension or lambda
        # has its own scope and is analysed separately below
        for nm, ln in _direct_loads(ast.Expr(value=nd)):
            if nm not in available:
                problems.append(("undefined", nm, ln, f"{scope_name}/comprehension"))
    for nd in inner_nodes:
        for sub in _comprehensions_of(ast.Expr(value=nd)):
            _analyze_comprehension(sub, available, f"{scope_name}/comprehension", problems)
        for sub in ast.walk(nd):
            if isinstance(sub, ast.Lambda):
                _analyze_scope(sub, available, False, f"{scope_name}/<lambda:{sub.lineno}>",
                               problems)


def _analyze_scope(node, outer_available, is_module, scope_name, problems):
    """Check one scope, then recurse into its nested scopes with the proper scope chain."""
    own = _scope_bindings(node)
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
        own.update({p: node.lineno for p in _params_of(node)})
    declared_global = {nm for n in ast.walk(node) if isinstance(n, ast.Global) for nm in n.names}
    handlers = _handler_windows(node)
    handler_names = {h[0] for h in handlers}
    dels = _del_events(node)
    binds = _binding_lines(node)
    available = set(outer_available) | set(own) | _BUILTINS

    for nm, ln in _direct_loads(node):
        if nm in _BUILTINS or nm in declared_global:
            continue
        if nm in handler_names and nm not in own and nm not in outer_available:
            # an except-alias exists only inside its handler (source-integrity design note.2)
            if not any(h == nm and lo <= ln <= hi for h, lo, hi in handlers):
                problems.append(("undefined", nm, ln, scope_name))
            continue
        if nm not in available:
            problems.append(("undefined", nm, ln, scope_name))
            continue
        if nm in dels:
            last_del = max([d for d in dels[nm] if d < ln], default=None)
            if last_del is not None:
                last_bind = max([b for b in binds.get(nm, []) if last_del < b < ln], default=None)
                if last_bind is None:
                    # deleted and not re-bound before this use (source-integrity design note.3)
                    problems.append(("use-after-del", nm, ln, scope_name))
                    continue
        if nm in own and own[nm] > ln:
            problems.append(("use-before-assignment", nm, ln, scope_name))

    inner_available = set(outer_available) | set(own)
    if isinstance(node, ast.ClassDef):
        # methods and comprehensions do NOT see class-body names (source-integrity design note.5)
        inner_available = set(outer_available)
    for comp in _comprehensions_of(node):
        _analyze_comprehension(comp, inner_available, scope_name, problems)
    children = [ast.Expr(value=node.body)] if isinstance(node, ast.Lambda) else \
        list(ast.iter_child_nodes(node))
    for n in children:
        _recurse_nested(n, inner_available, problems)


def _recurse_nested(n, inner_available, problems):
    if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
        _analyze_scope(n, inner_available, False, n.name, problems)
        return
    if isinstance(n, ast.Lambda):
        # a lambda body is its own scope (source-integrity design note cases 1 and 7)
        _analyze_scope(n, inner_available, False, f"<lambda:{n.lineno}>", problems)
        return
    if isinstance(n, ast.ClassDef):
        # a class body executes immediately in its own scope; its methods see the ENCLOSING scope,
        # not the class scope, so they are recursed with the same outer availability
        _analyze_scope(n, inner_available, True, f"class {n.name}", problems)
        return
    for c in ast.iter_child_nodes(n):
        _recurse_nested(c, inner_available, problems)


def analyze_names_source(src):
    """Same analysis as :func:`analyze_names`, on a source string (used by the self-tests)."""
    return _analyze_tree(ast.parse(src))


def analyze_names(rel):
    """Name problems in one file, using a proper module -> function -> nested-function scope chain."""
    return _analyze_tree(_parse(rel)[1])


def _analyze_tree(tree):
    global _FUTURE_ANNOTATIONS
    _FUTURE_ANNOTATIONS = _has_future_annotations(tree)
    problems = []
    _analyze_scope(tree, set(), True, "module", problems)
    seen, out = set(), []
    for p in problems:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def test_released_paths_have_no_name_errors():
    """No undefined name and no use-before-assignment in any released path, in module OR function
    scope. Covers the source-integrity `cc` NameError, a module-level stale binding and UnboundLocalError."""
    offenders = {}
    for rel in released_python_paths():
        probs = analyze_names(rel)
        if probs:
            offenders[rel] = probs
    assert not offenders, f"name problems in released paths: {offenders}"


def test_pyflakes_cross_check():
    """MANDATORY independent cross-check (source-integrity policy check: one clear policy).

    ``pyflakes`` is part of the pinned runtime in ``requirements.txt`` precisely because it is a
    second, independent opinion on the F821/F823 classes that this module also checks itself. If it
    is missing, the environment does not match the pinned one, so this test FAILS rather than
    silently skipping -- the guard must never appear to pass with only half of its coverage active.
    """
    try:
        from pyflakes.api import check
        from pyflakes.reporter import Reporter
    except Exception as exc:                                     # pragma: no cover - env problem
        raise AssertionError(
            "pyflakes is required for the independent name-resolution cross-check and is part of "
            "the pinned environment: pip install -r requirements.txt "
            f"(import failed: {exc})")
    import io
    import re
    bad = {}
    for rel in released_python_paths():
        out, err = io.StringIO(), io.StringIO()
        with open(os.path.join(_H, rel), encoding="utf-8") as fh:
            check(fh.read(), rel, Reporter(out, err))
        hits = [ln for ln in out.getvalue().splitlines()
                if re.search(r"undefined name|referenced before assignment|"
                             r"defined in enclosing scope", ln)]
        if hits:
            bad[rel] = hits
    assert not bad, f"pyflakes reports name errors: {bad}"


def test_pyflakes_agrees_with_builtin_guard_on_the_bypass_corpus():
    """Where pyflakes covers a class of error, it must agree with the built-in analyser on the
    stored bypass corpus. Cases pyflakes does not model (e.g. a class-body comprehension reaching a
    class-level name) are reported for the record, not failed."""
    from pyflakes.api import check
    from pyflakes.reporter import Reporter
    import io
    import re
    disagree = []
    for label, src in _NAME_BYPASSES.items():
        mine = bool(analyze_names_source(src))
        out, err = io.StringIO(), io.StringIO()
        check(src, "<bypass>", Reporter(out, err))
        pf = bool([ln for ln in out.getvalue().splitlines()
                   if re.search(r"undefined name|referenced before assignment", ln)])
        if not mine:
            disagree.append(f"{label}: built-in guard missed it")
        elif not pf:
            print(f"    (note: pyflakes does not model '{label}'; built-in guard covers it)")
    assert not disagree, disagree


# --------------------------------------------------------------- provenance

_GRAPH_KWARGS = {"communication_radius", "neighbour_count", "layout_spread", "seed"}
# only THESE module names are the authoritative graph configuration (a name that merely ends with
# "graph_config" is a different module and must not be trusted -- source-integrity design note case 6)
_GRAPH_CONFIG_MODULES = {"graph_config", "code.graph_config"}
# names whose ORIGIN must be the simulator/graph-config, never a local definition
_ORIGIN_CONTROLLED = {"ClusterConfig": "dsceos_validation",
                      "SimulationConfig": "dsceos_validation",
                      "make_fixed_local_graph": "dsceos_validation",
                      "authoritative_cluster_kwargs": "graph_config",
                      "ladder_cluster_kwargs": "graph_config"}


def _node_index(tree):
    """Walk the AST ONCE and index the node kinds the provenance analysis needs. Re-walking the tree
    inside the fixed-point loops made the guard quadratic-to-cubic and effectively hung it on larger
    files (source-integrity Design note); everything below now iterates these pre-built lists."""
    idx = getattr(tree, "_hygiene_index", None)
    if idx is not None:
        return idx
    idx = {"Assign": [], "AnnAssign": [], "Call": [], "For": [], "ImportFrom": [], "Import": [],
           "FunctionDef": [], "Name": []}
    for n in ast.walk(tree):
        k = type(n).__name__
        if k == "AsyncFunctionDef":
            k = "FunctionDef"
        if k in idx:
            idx[k].append(n)
    idx["assign_like"] = idx["Assign"] + idx["AnnAssign"]
    # --- scope map: every node -> its enclosing scope, and every scope -> its parent scope.
    # Provenance must be scope-aware: a proven `cc` at module level must NOT make a same-named
    # PARAMETER of some function proven (source-integrity Design note).
    scope_of, parent_scope = {}, {id(tree): None}
    scopes = {id(tree): tree}

    _SCOPE_TYPES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef,
                    ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)

    def _descend(node, scope):
        for child in ast.iter_child_nodes(node):
            scope_of[id(child)] = scope
            if isinstance(child, _SCOPE_TYPES):
                parent_scope[id(child)] = scope
                scopes[id(child)] = child
                _descend(child, child)
            else:
                _descend(child, scope)

    scope_of[id(tree)] = tree
    _descend(tree, tree)
    idx["scope_of"] = scope_of
    idx["parent_scope"] = parent_scope
    idx["scopes"] = scopes
    tree._hygiene_index = idx
    return idx


def _declared_scope_redirect(tree):
    """`global x` / `nonlocal x` inside a function make a Store of `x` bind an OUTER scope, not the
    local one (source-integrity Design note). Returns (scope_id, name) -> target scope node. Cached per tree."""
    cached = getattr(tree, "_hygiene_redirect", None)
    if cached is not None:
        return cached
    idx = _node_index(tree)
    scope_of, parent = idx["scope_of"], idx["parent_scope"]

    redirect = {}
    # preliminary binding set (without redirection) so `nonlocal` can find the nearest binder
    _prelim_binds = set()
    for node, name in _all_binders(tree):
        _prelim_binds.add((id(scope_of.get(id(node), tree)), name))
    for fn in idx["FunctionDef"] + [n for n in ast.walk(tree) if isinstance(n, ast.Lambda)]:
        for p in _params_of(fn):
            _prelim_binds.add((id(fn), p))
    for n in ast.walk(tree):
        if isinstance(n, (ast.Global, ast.Nonlocal)):
            sc = scope_of.get(id(n), tree)
            for nm in n.names:
                if isinstance(n, ast.Global):
                    redirect[(id(sc), nm)] = tree
                else:
                    # `nonlocal x` binds the NEAREST ENCLOSING FUNCTION scope that binds x, which is
                    # not necessarily the immediate parent (source-integrity design note, nested nonlocal)
                    outer = parent.get(id(sc))
                    while outer is not None and outer is not tree:
                        if isinstance(outer, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)) \
                                and any(nm == b for (s, b) in _prelim_binds if s == id(outer)):
                            redirect[(id(sc), nm)] = outer
                            break
                        outer = parent.get(id(outer))
    tree._hygiene_redirect = redirect
    return redirect


def _scope_bindings_map(tree):
    """(scope_id, name) -> True for every name BOUND in that scope, including function/lambda
    parameters (`ast.arg`) and comprehension targets, and honouring global/nonlocal redirection.
    Cached per tree: rebuilding it per name resolution made the analysis quadratic."""
    cached = getattr(tree, "_hygiene_scope_binds", None)
    if cached is not None:
        return cached
    idx = _node_index(tree)
    scope_of = idx["scope_of"]
    redirect = _declared_scope_redirect(tree)
    out = {}
    for node, name in _all_binders(tree):
        sc = scope_of.get(id(node), tree)
        sc = redirect.get((id(sc), name), sc)
        out[(id(sc), name)] = True
    for fn in idx["FunctionDef"] + [n for n in ast.walk(tree) if isinstance(n, ast.Lambda)]:
        for p in _params_of(fn):
            out[(id(fn), p)] = True          # parameters bind in the function's OWN scope
    for comp in [n for n in ast.walk(tree)
                 if isinstance(n, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp))]:
        for gen in comp.generators:
            for sub in ast.walk(gen.target):
                if isinstance(sub, ast.Name):
                    out[(id(comp), sub.id)] = True   # comprehension targets bind in their own scope
    tree._hygiene_scope_binds = out
    return out


def _resolve_scope(tree, name, use_node):
    """Innermost scope that binds ``name`` as seen from ``use_node``, following Python's rules:
    a CLASS body is visible only to code executing directly in that body, never to a method defined
    inside it (source-integrity design note), so class scopes are skipped once the search leaves them."""
    idx = _node_index(tree)
    binds = _scope_bindings_map(tree)
    sc = idx["scope_of"].get(id(use_node), tree)
    first = True
    while sc is not None:
        skip_class = isinstance(sc, ast.ClassDef) and not first
        if not skip_class and (id(sc), name) in binds:
            return sc
        sc = idx["parent_scope"].get(id(sc))
        first = False
    return None


def _alias_names_of(tree, canonical):
    """Local handles of a helper (`replace`, `partial`): import-as and plain assignment.
    Cached per tree and per canonical name."""
    cache = getattr(tree, "_hygiene_alias_cache", None)
    if cache is None:
        cache = tree._hygiene_alias_cache = {}
    if canonical in cache:
        return cache[canonical]
    idx = _node_index(tree)
    names = {canonical}
    for _ in range(2):
        for n in idx["ImportFrom"]:
            for a in n.names:
                if a.name == canonical and a.asname:
                    names.add(a.asname)
        for n in idx["assign_like"]:
            v = getattr(n, "value", None)
            if isinstance(v, ast.Name) and v.id in names:
                targets = n.targets if isinstance(n, ast.Assign) else [n.target]
                for tgt in targets:
                    for sub in ast.walk(tgt):
                        if isinstance(sub, ast.Name):
                            names.add(sub.id)
    cache[canonical] = names
    return names


def _ctor_aliases(tree, canonical):
    """Every local handle that refers to ``canonical`` (import-as, plain or annotated assignment, or
    functools.partial), so an aliased constructor cannot escape the provenance guard."""
    idx = _node_index(tree)
    partial_names = _alias_names_of(tree, "partial")      # computed ONCE, not per assignment
    names = {canonical}
    for _ in range(3):
        for n in idx["ImportFrom"]:
            for a in n.names:
                if a.name == canonical and a.asname:
                    names.add(a.asname)
        for n in idx["assign_like"]:
            v = getattr(n, "value", None)
            if v is None:
                continue
            hit = (isinstance(v, ast.Name) and v.id in names) or (
                isinstance(v, ast.Call)
                and (getattr(v.func, "id", None) or getattr(v.func, "attr", None)) in partial_names
                and v.args and isinstance(v.args[0], ast.Name) and v.args[0].id in names)
            if hit:
                targets = n.targets if isinstance(n, ast.Assign) else [n.target]
                for tgt in targets:
                    for sub in ast.walk(tgt):
                        if isinstance(sub, ast.Name):
                            names.add(sub.id)
    return names


def _graph_config_module_handles(tree):
    """Names that provably hold the graph_config MODULE: a direct `import graph_config as X`, or a
    variable assigned from a local helper whose body imports graph_config and returns it."""
    handles = set()
    returning = set()
    for fn in ast.walk(tree):
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        local = set()
        for n in ast.walk(fn):
            if isinstance(n, ast.Import):
                for a in n.names:
                    if a.name in _GRAPH_CONFIG_MODULES:
                        local.add(a.asname or a.name.split(".")[0])
        for n in ast.walk(fn):
            if isinstance(n, ast.Return) and isinstance(n.value, ast.Name) and n.value.id in local:
                returning.add(fn.name)
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            for a in n.names:
                if a.name in _GRAPH_CONFIG_MODULES:
                    handles.add(a.asname or a.name.split(".")[0])
        elif isinstance(n, ast.Assign) and isinstance(n.value, ast.Call) \
                and getattr(n.value.func, "id", None) in returning:
            for tgt in n.targets:
                for sub in ast.walk(tgt):
                    if isinstance(sub, ast.Name):
                        handles.add(sub.id)
    return handles


def _graph_config_imports(tree, module_level_only=False):
    names = {}
    nodes = tree.body if module_level_only else ast.walk(tree)
    for n in nodes:
        if isinstance(n, ast.ImportFrom) and (n.module or "") in _GRAPH_CONFIG_MODULES:
            for a in n.names:
                names[a.asname or a.name] = a.name
        elif isinstance(n, ast.Import):
            for a in n.names:
                if a.name in _GRAPH_CONFIG_MODULES:
                    names[a.asname or a.name.split(".")[0]] = a.name
    return names


_CAPTURE_MARKER = "provenance-ok: effective SimulationConfig capture"


_CAPTURE_MARKER_FILE = "test_graph_config.py"


def _marked_capture_names(tree):
    """Names a source line explicitly marks as an effective SimulationConfig captured at runtime.
    This is a deliberate, auditable opt-in with a deliberately narrow scope: it is honoured ONLY in
    the effective-configuration dry-run test, where the object is captured from the runner itself and
    cannot be proven statically, and only for an assignment whose value is a `.cluster` attribute
    (source-integrity design note case 7). Everywhere else the marker has no effect."""
    if getattr(tree, "_hygiene_file", "") != _CAPTURE_MARKER_FILE:
        return set()
    src = getattr(tree, "_hygiene_src", "") or ""
    lines = src.splitlines()
    out = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Assign) and 0 < n.lineno <= len(lines):
            if not (isinstance(n.value, ast.Attribute) and n.value.attr == "cluster"):
                continue
            if _CAPTURE_MARKER in lines[n.lineno - 1]:
                for tgt in n.targets:
                    for sub in ast.walk(tgt):
                        if isinstance(sub, ast.Name):
                            out.add(sub.id)
    return out


def _all_binders(tree):
    """(cached per tree) Every (node, name) pair where Python binds a name, for ALL binder forms. Used to revoke a
    cluster-typed name whose value can be replaced by a non-authoritative object. Comprehension
    targets are deliberately excluded: they live in the comprehension's own scope and cannot rebind
    an outer name (source-integrity Design note asked for full binder coverage; this is the canonical list)."""
    cached = getattr(tree, "_hygiene_binders", None)
    if cached is not None:
        return cached
    comp_targets = set()
    for n in ast.walk(tree):
        if isinstance(n, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
            for gen in n.generators:
                for sub in ast.walk(gen.target):
                    if isinstance(sub, ast.Name):
                        comp_targets.add(id(sub))
    out = []

    def _names_of(target):
        return [sub for sub in ast.walk(target)
                if isinstance(sub, ast.Name) and isinstance(sub.ctx, (ast.Store, ast.Del))]

    for n in ast.walk(tree):
        if isinstance(n, ast.Assign):
            for tgt in n.targets:
                out += [(n, s.id) for s in _names_of(tgt) if id(s) not in comp_targets]
        elif isinstance(n, (ast.AnnAssign, ast.AugAssign)):
            out += [(n, s.id) for s in _names_of(n.target) if id(s) not in comp_targets]
        elif isinstance(n, (ast.For, ast.AsyncFor)):
            out += [(n, s.id) for s in _names_of(n.target) if id(s) not in comp_targets]
        elif isinstance(n, (ast.With, ast.AsyncWith)):
            for item in n.items:
                if item.optional_vars is not None:
                    out += [(n, s.id) for s in _names_of(item.optional_vars)]
        elif isinstance(n, ast.NamedExpr):
            out += [(n, s.id) for s in _names_of(n.target) if id(s) not in comp_targets]
        elif isinstance(n, ast.ExceptHandler) and n.name:
            out.append((n, n.name))
        elif isinstance(n, (ast.Import, ast.ImportFrom)):
            for a in n.names:
                out.append((n, (a.asname or a.name).split(".")[0]))
        elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            out.append((n, n.name))
        elif isinstance(n, ast.MatchAs) and n.name:
            out.append((n, n.name))
        elif isinstance(n, ast.MatchStar) and n.name:
            out.append((n, n.name))
        elif isinstance(n, ast.MatchMapping) and n.rest:
            out.append((n, n.rest))
        elif isinstance(n, ast.Delete):
            for tgt in n.targets:
                out += [(n, s.id) for s in _names_of(tgt)]
    tree._hygiene_binders = out
    return out


def _cluster_typed_names(tree):
    """Names that provably hold a cluster configuration: bound from a ``ClusterConfig(...)`` call,
    from ``dataclasses.replace(<cluster>, ...)``, or from a ``....cluster`` attribute of a captured
    SimulationConfig. Only attributes of THESE names count as an effective cluster object, so a
    locally defined look-alike class cannot fake graph-parameter provenance (source-integrity design note)."""
    idx = _node_index(tree)                  # ONE walk, reused by every step below
    names = set()
    _param_calls = {}
    funcs = {n.name: n for n in idx["FunctionDef"]}

    # a function used as a VALUE may be invoked through an alias we cannot see: fail closed
    _func_names = set(funcs)
    # a function NAME appearing as the callee of a call is not an alias; only a reference used as a
    # VALUE (assigned, passed, stored) means the function can be reached indirectly
    _callee_nodes = {id(c.func) for c in idx["Call"] if isinstance(c.func, ast.Name)}
    _aliased_funcs = {n.id for n in idx["Name"]
                      if isinstance(n.ctx, ast.Load) and n.id in _func_names
                      and id(n) not in _callee_nodes}
    _simcfg = {tgt.id for n in idx["Assign"]
               if isinstance(n.value, ast.Call)
               and _expr_origin(n.value.func, n, tree) == "SimulationConfig"
               for tgt in n.targets if isinstance(tgt, ast.Name)}
    _marked_capture = _marked_capture_names(tree)
    scope_of = idx["scope_of"]
    _node_by_id = {id(n): n for n in ast.walk(tree)}
    _calls = idx["Call"]
    _assigns = idx["Assign"]
    _fors = idx["For"]
    _proving = set()                         # (id(node), name) pairs that legitimately prove a name
    _param_proven = set()                    # (id(function scope), parameter name) proven by calls

    def _live_pairs():
        """(scope_id, name) proven so far in this fixed point."""
        pairs = set(_param_proven)
        for nid, nm in _proving:
            nd = _node_by_id.get(nid)
            if nd is not None:
                pairs.add((id(scope_of.get(id(nd), tree)), nm))
        return pairs

    def _arg_proven(arg, call_node=None):
        """An argument is proven only if the name resolves LEXICALLY at the call site to a binding
        that was itself proven (source-integrity Design note, call-argument scope contamination)."""
        if isinstance(arg, ast.Call):
            return _expr_origin(arg.func, arg, tree) == "ClusterConfig"
        if not isinstance(arg, ast.Name):
            return False
        if call_node is None:
            return arg.id in names
        sc = _resolve_scope(tree, arg.id, call_node)
        return sc is not None and (id(sc), arg.id) in _live_pairs()
    for _ in range(4):                       # small fixed point over assignments and call sites
        # (a) parameters that provably receive a cluster object at every local call site
        for call in _calls:
            fname = getattr(call.func, "id", None)
            fn = funcs.get(fname)
            if fn is None or fname in _aliased_funcs:
                continue                       # indirectly callable -> parameters stay unproven
            params = [a.arg for a in list(fn.args.posonlyargs) + list(fn.args.args)]
            kwonly = [a.arg for a in fn.args.kwonlyargs]
            _param_calls.setdefault(fname, (params, [], kwonly, fn))[1].append(call)
        # (a2) a parameter counts as cluster-typed only if EVERY local call site supplies a proven
        # cluster object there (source-integrity design note: one good call site must not bless a mixed helper)
        for fname, (params, calls, kwonly, fn) in _param_calls.items():
            if not calls:
                continue
            # keyword-only parameters: proven when EVERY call site supplies a proven cluster
            for kwname in kwonly:
                supplied = []
                for call in calls:
                    match = [k for k in call.keywords if k.arg == kwname]
                    if match:
                        supplied.append(_arg_proven(match[0].value, call))
                    elif any(k.arg is None for k in call.keywords):
                        supplied.append(False)
                if supplied and all(supplied):
                    names.add(kwname)
                    _param_proven.add((id(fn), kwname))
            # a parameter whose DEFAULT is a proven cluster and that is never overridden
            defaults = list(zip([a.arg for a in fn.args.args][-len(fn.args.defaults):],
                                fn.args.defaults)) if fn.args.defaults else []
            defaults += [(a.arg, d) for a, d in zip(fn.args.kwonlyargs, fn.args.kw_defaults)
                         if d is not None]
            for pname, dflt in defaults:
                if not _arg_proven(dflt, fn):
                    continue
                overridden = False
                for call in calls:
                    idxp = params.index(pname) if pname in params else -1
                    if idxp >= 0 and idxp < len(call.args):
                        overridden = not _arg_proven(call.args[idxp], call)
                    kwm = [k for k in call.keywords if k.arg == pname]
                    if kwm:
                        overridden = overridden or not _arg_proven(kwm[0].value, call)
                    if any(k.arg is None for k in call.keywords):
                        overridden = True
                if not overridden:
                    names.add(pname)
                    _param_proven.add((id(fn), pname))
            for i, pname in enumerate(params):
                supplied = []
                for call in calls:
                    if i < len(call.args):
                        supplied.append(_arg_proven(call.args[i], call))
                    else:
                        kwmatch = [k for k in call.keywords if k.arg == pname]
                        if kwmatch:
                            supplied.append(_arg_proven(kwmatch[0].value, call))
                        elif any(k.arg is None for k in call.keywords):
                            supplied.append(False)      # **kwargs: unverifiable -> not proven
                if supplied and all(supplied):
                    names.add(pname)
                    _param_proven.add((id(funcs[fname]), pname))
        # (a3) loop variables unpacked from a literal list/tuple of tuples: proven when EVERY
        # element supplies a proven cluster object at that position
        for node in _fors:
            if not isinstance(node.target, ast.Tuple):
                continue
            if not isinstance(node.iter, (ast.List, ast.Tuple)) or not node.iter.elts:
                continue
            tnames = [e.id if isinstance(e, ast.Name) else None for e in node.target.elts]
            for i, tn in enumerate(tnames):
                if tn is None:
                    continue
                comps = []
                for elt in node.iter.elts:
                    if isinstance(elt, (ast.Tuple, ast.List)) and i < len(elt.elts):
                        comps.append(_arg_proven(elt.elts[i], node))
                    else:
                        comps.append(False)
                if comps and all(comps):
                    names.add(tn)
                    _proving.add((id(node), tn))
        # (b) assignments
        for n in ast.walk(tree):
            if not isinstance(n, ast.Assign):
                continue
            v = n.value
            ok = False
            if isinstance(v, ast.Call):
                fn = getattr(v.func, "id", None) or getattr(v.func, "attr", None)
                if _expr_origin(v.func, n, tree) == "ClusterConfig":
                    ok = True
                elif _expr_origin(v.func, n, tree) == "replace" and v.args \
                        and isinstance(v.args[0], ast.Name) and v.args[0].id in names:
                    ok = True
            elif isinstance(v, ast.Attribute) and v.attr == "cluster":
                # only a PROVEN SimulationConfig may yield a cluster (source-integrity design note), or an
                # assignment explicitly marked as a runtime capture in the dry-run test
                base = v.value
                marked = any(isinstance(sub, ast.Name) and sub.id in _marked_capture
                             for tgt in n.targets for sub in ast.walk(tgt))
                ok = marked or (isinstance(base, ast.Name) and base.id in _simcfg)
            elif isinstance(v, ast.Name) and v.id in names:
                ok = True
            if ok:
                for tgt in n.targets:
                    for sub in ast.walk(tgt):
                        if isinstance(sub, ast.Name):
                            names.add(sub.id)
                            _proving.add((id(n), sub.id))
    # A name is cluster-typed only if EVERY binding of it is proven. Re-binding a proven name to a
    # look-alike object (unconditionally, in a branch, or before building a SimulationConfig) must
    # revoke its status (source-integrity Design note).
    # --- scope-qualify: a proven name belongs to the scope in which it was proven, and a binding in
    # ANY scope (including a same-named parameter of another function) can only revoke within that
    # scope (source-integrity Design note).
    scope_of = idx["scope_of"]
    proven_pairs = {(id(scope_of.get(nid_node, tree)), nm)
                    for nid_node, nm in []}                      # placeholder, filled below
    proven_pairs = set()
    node_by_id = {id(n): n for n in ast.walk(tree)}
    for nid, nm in _proving:
        node = node_by_id.get(nid)
        if node is not None:
            proven_pairs.add((id(scope_of.get(id(node), tree)), nm))
    binders = _all_binders(tree)
    _redirect = _declared_scope_redirect(tree)

    def _bind_scope(node, nm):
        sc = scope_of.get(id(node), tree)
        return _redirect.get((id(sc), nm), sc)

    binder_pairs = [(id(_bind_scope(node, nm)), nm, node) for node, nm in binders]
    # parameters bind in their own function scope and are never proving bindings
    for fn in idx["FunctionDef"] + [n for n in ast.walk(tree) if isinstance(n, ast.Lambda)]:
        for p in _params_of(fn):
            binder_pairs.append((id(fn), p, fn))
    for _ in range(2):
        revoked = {(sc, nm) for sc, nm, node in binder_pairs
                   if (sc, nm) in proven_pairs and (id(node), nm) not in _proving}
        if not revoked:
            break
        proven_pairs -= revoked
    proven_pairs |= _param_proven            # parameters proven at every call site
    names = proven_pairs
    return names


def _cluster_proven(name, use_node, tree, cluster_names):
    """Is ``name``, as seen from ``use_node``, bound to a PROVEN cluster configuration?
    Resolves lexically, so a same-named parameter in an inner scope does not inherit the module's
    proven status (source-integrity Design note)."""
    sc = _resolve_scope(tree, name, use_node)
    return sc is not None and (id(sc), name) in cluster_names


# --------------------------------------------------------------- binding origins
# Controlled callables/modules are tracked as (scope, name) -> ORIGIN binding records rather than as
# file-level name sets. An alias therefore CARRIES its origin (so `CC = ClusterConfig; CC(...)` is
# recognised), and any other binding of that name in that scope REVOKES it (so a re-bound or
# parameter-shadowed alias is not trusted) -- source-integrity Design note and O-01 are the two sides of this.
_ORIGIN_OF_IMPORT = {
    ("dsceos_validation", "ClusterConfig"): "ClusterConfig",
    ("dsceos_validation", "SimulationConfig"): "SimulationConfig",
    ("dsceos_validation", "make_fixed_local_graph"): "make_fixed_local_graph",
    ("dataclasses", "replace"): "replace",
    ("functools", "partial"): "partial",
}
_GC_HELPERS = {"authoritative_cluster_kwargs", "ladder_cluster_kwargs"}
_MODULE_ORIGIN = {"dsceos_validation": "dsceos_module",
                  "graph_config": "graph_config_module",
                  "code.graph_config": "graph_config_module",
                  "dataclasses": "dataclasses_module",
                  "functools": "functools_module"}
# which attribute of which authoritative module yields which controlled origin
_MODULE_ATTR_ORIGIN = {
    "dsceos_module": {"ClusterConfig": "ClusterConfig", "SimulationConfig": "SimulationConfig",
                      "make_fixed_local_graph": "make_fixed_local_graph"},
    "graph_config_module": {"authoritative_cluster_kwargs": "gc_kwargs",
                            "ladder_cluster_kwargs": "gc_kwargs"},
    "dataclasses_module": {"replace": "replace"},
    "functools_module": {"partial": "partial"},
}


def _origin_bindings(tree):
    """(scope_id, name) -> origin tag for every controlled callable, constant or module handle."""
    cached = getattr(tree, "_hygiene_origins", None)
    if cached is not None:
        return cached
    idx = _node_index(tree)
    scope_of = idx["scope_of"]
    origins, conferring, aliased = {}, set(), set()

    def _put(node, name, origin, by_import=False):
        sc = scope_of.get(id(node), tree)
        origins[(id(sc), name)] = origin
        conferring.add((id(sc), name, id(node)))
        if not by_import:
            aliased.add((id(sc), name, node.lineno))

    for n in ast.walk(tree):
        if isinstance(n, ast.ImportFrom):
            mod = n.module or ""
            for a in n.names:
                local = a.asname or a.name
                if (mod, a.name) in _ORIGIN_OF_IMPORT:
                    _put(n, local, _ORIGIN_OF_IMPORT[(mod, a.name)], by_import=True)
                elif mod in _GRAPH_CONFIG_MODULES:
                    _put(n, local, "gc_kwargs" if a.name in _GC_HELPERS else "gc_const",
                         by_import=True)
        elif isinstance(n, ast.Import):
            for a in n.names:
                if a.name in _MODULE_ORIGIN:
                    _put(n, a.asname or a.name.split(".")[0], _MODULE_ORIGIN[a.name],
                         by_import=True)

    # propagate through plain aliases, module attributes and functools.partial wrappers, to a fixed
    # point. NOTE: this uses a LOCAL resolver over the partially-built map -- calling the cached
    # _expr_origin/_origin_of here would recurse into this very function (source-integrity fix).
    def _local_origin(expr, at_node):
        if isinstance(expr, ast.Name):
            sc = _resolve_scope(tree, expr.id, at_node)
            return origins.get((id(sc if sc is not None else tree), expr.id))
        if isinstance(expr, ast.Attribute) and isinstance(expr.value, ast.Name):
            base = _local_origin(expr.value, at_node)
            return _MODULE_ATTR_ORIGIN.get(base, {}).get(expr.attr)
        return None

    for _ in range(3):
        for n in idx["assign_like"]:
            v = getattr(n, "value", None)
            if v is None:
                continue
            src = None
            if isinstance(v, (ast.Name, ast.Attribute)):
                src = _local_origin(v, n)
            elif isinstance(v, ast.Call):
                forig = _local_origin(v.func, n)
                fname = getattr(v.func, "id", None)
                if forig == "partial" and v.args:
                    src = _local_origin(v.args[0], n)
                elif fname is not None and fname in _graph_config_module_handles_helpers(tree):
                    _hsc = _resolve_scope(tree, fname, n)
                    _def = next((d for d in idx["FunctionDef"] if d.name == fname), None)
                    if _def is not None and _hsc is not None and \
                            id(_hsc) == id(scope_of.get(id(_def), tree)):
                        src = "graph_config_module"
            if src is None:
                continue
            targets = n.targets if isinstance(n, ast.Assign) else [n.target]
            for tgt in targets:
                for sub in ast.walk(tgt):
                    if isinstance(sub, ast.Name):
                        _put(n, sub.id, src)

    # revoke on ANY other binding of the same (scope, name)
    binders = list(_all_binders(tree))
    for fn in idx["FunctionDef"] + [n for n in ast.walk(tree) if isinstance(n, ast.Lambda)]:
        binders += [(fn, p) for p in _params_of(fn)]
    for node, name in binders:
        sc = scope_of.get(id(node), tree)
        key = (id(sc), name)
        if key in origins and (id(sc), name, id(node)) not in conferring:
            del origins[key]
    tree._hygiene_origins = origins
    tree._hygiene_aliased = aliased
    return origins


_CONTROLLED_CALLABLES = {"ClusterConfig", "SimulationConfig", "make_fixed_local_graph",
                         "authoritative_cluster_kwargs", "ladder_cluster_kwargs"}


def _parent_map(tree):
    cached = getattr(tree, "_hygiene_parents", None)
    if cached is not None:
        return cached
    parents = {}
    for n in ast.walk(tree):
        for c in ast.iter_child_nodes(n):
            parents[id(c)] = n
    tree._hygiene_parents = parents
    return parents


def _controlled_value_uses(tree):
    """A controlled callable may appear ONLY as the callee of a call (directly, or as
    `<module handle>.<name>`), or inside an import statement. Any other VALUE position -- tuple
    unpacking, a default argument, a `functools.partial` argument, a conditional expression, a
    container element or subscript, a class attribute, a return value, a `for`/`with` target source
    -- creates an alias, and tracking every such form has proven unbounded (source-integrity Design note).
    Returns (name, lineno) for each violating use."""
    parents = _parent_map(tree)
    out = []
    for n in ast.walk(tree):
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load):
            name = n.id
            if name not in _CONTROLLED_CALLABLES and \
                    _origin_of(name, n, tree) not in ("ClusterConfig", "SimulationConfig",
                                                      "make_fixed_local_graph", "gc_kwargs"):
                continue
            p = parents.get(id(n))
            if isinstance(p, ast.Call) and p.func is n:
                continue                                   # direct call: allowed
            if isinstance(p, ast.Attribute):
                gp = parents.get(id(p))
                if isinstance(gp, ast.Call) and gp.func is p:
                    continue                               # module-handle call: allowed
            if isinstance(p, (ast.Import, ast.ImportFrom, ast.alias)):
                continue
            out.append((name, n.lineno))
        elif isinstance(n, ast.Attribute) and n.attr in _CONTROLLED_CALLABLES:
            p = parents.get(id(n))
            if isinstance(p, ast.Call) and p.func is n:
                continue
            out.append((n.attr, n.lineno))
    return out


def _aliased_controlled_bindings(tree):
    """(scope, name, line) triples where a controlled callable/module was introduced by something
    OTHER than an import. POLICY (source-integrity Design note): controlled callables may only be introduced by a
    direct import. Tracking every alias-creating construction (tuple unpacking, defaults, partial
    with preset arguments, conditional expressions, subscripts, class namespaces, helper returns,
    for/with targets, ...) proved unbounded, so re-exporting them through any other binding form is
    rejected outright. The released paths contain no such alias, so the rule costs nothing."""
    _origin_bindings(tree)
    return getattr(tree, "_hygiene_aliased", set())


def _graph_config_module_handles_helpers(tree):
    """Names of local helpers that import graph_config and return it."""
    cached = getattr(tree, "_hygiene_gc_helpers", None)
    if cached is not None:
        return cached
    out = set()
    for fn in ast.walk(tree):
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        local = {a.asname or a.name.split(".")[0] for n in ast.walk(fn)
                 if isinstance(n, ast.Import) for a in n.names if a.name in _GRAPH_CONFIG_MODULES}
        if any(isinstance(n, ast.Return) and isinstance(n.value, ast.Name) and n.value.id in local
               for n in ast.walk(fn)):
            out.add(fn.name)
    tree._hygiene_gc_helpers = out
    return out


def _controlled_name_roles(tree):
    """name -> the origin it is SUPPOSED to have, for every name that carries a controlled origin
    anywhere in the file. A call through such a name that resolves elsewhere is a violation, not an
    unknown call (source-integrity Design note: revoking an origin must FLAG, not silently ignore)."""
    cached = getattr(tree, "_hygiene_roles", None)
    if cached is not None:
        return cached
    # the canonical names always carry an expected role, so calling `ClusterConfig(...)` without an
    # authoritative import, from a foreign module, or through a local shadow is a violation
    roles = {"ClusterConfig": "ClusterConfig", "SimulationConfig": "SimulationConfig",
             "make_fixed_local_graph": "make_fixed_local_graph",
             "authoritative_cluster_kwargs": "gc_kwargs", "ladder_cluster_kwargs": "gc_kwargs"}
    idx = _node_index(tree)
    scope_of = idx["scope_of"]
    for n in ast.walk(tree):
        if isinstance(n, ast.ImportFrom):
            mod = n.module or ""
            for a in n.names:
                local = a.asname or a.name
                if (mod, a.name) in _ORIGIN_OF_IMPORT:
                    roles[local] = _ORIGIN_OF_IMPORT[(mod, a.name)]
                elif mod in _GRAPH_CONFIG_MODULES and a.name in _GC_HELPERS:
                    roles[local] = "gc_kwargs"
                elif a.name in roles and mod not in _GRAPH_CONFIG_MODULES \
                        and mod != "dsceos_validation":
                    roles[local] = roles[a.name]      # foreign import of a canonical name
        elif isinstance(n, ast.Import):
            for a in n.names:
                if a.name in _MODULE_ORIGIN:
                    roles[a.asname or a.name.split(".")[0]] = _MODULE_ORIGIN[a.name]
    # plain aliases of a controlled name inherit its role
    for _ in range(2):
        for n in idx["assign_like"]:
            v = getattr(n, "value", None)
            if isinstance(v, ast.Name) and v.id in roles:
                targets = n.targets if isinstance(n, ast.Assign) else [n.target]
                for tgt in targets:
                    for sub in ast.walk(tgt):
                        if isinstance(sub, ast.Name):
                            roles.setdefault(sub.id, roles[v.id])
    tree._hygiene_roles = roles
    return roles


def _expr_origin(expr, use_node, tree):
    """Origin of a callable/value EXPRESSION: a bare name resolves lexically, and
    `<module handle>.<attr>` resolves through the base binding, so `dv.ClusterConfig(...)` is
    classified exactly like `ClusterConfig(...)` (source-integrity Design note)."""
    if tree is None or expr is None:
        return None
    if isinstance(expr, ast.Name):
        return _origin_of(expr.id, use_node, tree)
    if isinstance(expr, ast.Attribute) and isinstance(expr.value, ast.Name):
        base_origin = _origin_of(expr.value.id, use_node, tree)
        return _MODULE_ATTR_ORIGIN.get(base_origin, {}).get(expr.attr)
    return None


def _origin_of(name, use_node, tree):
    """Origin tag of ``name`` as resolved lexically from ``use_node`` (None if not controlled)."""
    if tree is None or name is None:
        return None
    sc = _resolve_scope(tree, name, use_node)
    return _origin_bindings(tree).get((id(sc if sc is not None else tree), name))


def _gc_import_pairs(tree):
    """(scope_id, name) for every name imported FROM graph_config, in whatever scope the import
    occurs (a function-local `from graph_config import X` is perfectly authoritative)."""
    cached = getattr(tree, "_hygiene_gc_pairs", None)
    if cached is not None:
        return cached
    idx = _node_index(tree)
    scope_of = idx["scope_of"]
    pairs = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.ImportFrom) and (n.module or "") in _GRAPH_CONFIG_MODULES:
            for a in n.names:
                pairs.add((id(scope_of.get(id(n), tree)), a.asname or a.name))
        elif isinstance(n, ast.Import):
            for a in n.names:
                if a.name in _GRAPH_CONFIG_MODULES:
                    pairs.add((id(scope_of.get(id(n), tree)), a.asname or a.name.split(".")[0]))
    tree._hygiene_gc_pairs = pairs
    return pairs


def _gc_handle_pairs(tree):
    """(scope_id, name) for names that provably hold the graph_config MODULE: a direct import alias,
    or an assignment from a local helper that imports and returns it."""
    cached = getattr(tree, "_hygiene_gc_handle_pairs", None)
    if cached is not None:
        return cached
    idx = _node_index(tree)
    scope_of = idx["scope_of"]
    handles = _graph_config_module_handles(tree)
    pairs = set(_gc_import_pairs(tree))
    for n in idx["Assign"]:
        for tgt in n.targets:
            for sub in ast.walk(tgt):
                if isinstance(sub, ast.Name) and sub.id in handles:
                    pairs.add((id(scope_of.get(id(n), tree)), sub.id))
    tree._hygiene_gc_handle_pairs = pairs
    return pairs


def _gc_visible(name, use_node, tree, gc_names):
    """A graph_config name counts only where the binding it lexically RESOLVES to is the
    graph_config import; a re-bound or parameter-shadowed alias is not trusted."""
    if tree is None or use_node is None:
        return name in gc_names
    return _origin_of(name, use_node, tree) in ("gc_const", "gc_kwargs")


def _is_from_graph_config(node, gc_names, cluster_names, simcfg_names=frozenset(), tree=None,
                          use_node=None):
    """True if the value comes from graph_config, or is read off a PROVEN cluster object."""
    if isinstance(node, ast.Name):
        return _gc_visible(node.id, use_node if use_node is not None else node, tree, gc_names)
    if isinstance(node, ast.Attribute):
        if node.attr in _GRAPH_KWARGS:
            base = node.value
            if isinstance(base, ast.Name):
                if _gc_visible(base.id, use_node if use_node is not None else node, tree, gc_names):
                    return True
                if tree is not None:
                    return _cluster_proven(base.id, use_node if use_node is not None else node,
                                           tree, cluster_names)
                return (id(tree), base.id) in cluster_names
            if isinstance(base, ast.Attribute) and base.attr == "cluster":
                # `<x>.cluster.<param>` counts only when <x> is a PROVEN SimulationConfig
                inner = base.value
                return isinstance(inner, ast.Name) and inner.id in simcfg_names
            return False
        return _is_from_graph_config(node.value, gc_names, cluster_names, simcfg_names, tree,
                                     use_node)
    if isinstance(node, ast.Call):
        nm = getattr(node.func, "id", None) or getattr(node.func, "attr", None)
        return nm in gc_names or nm == "authoritative_cluster_kwargs"
    return False


def provenance_problems_source(src):
    """Same analysis as :func:`provenance_problems`, on a source string (used by the self-tests)."""
    tree = ast.parse(src)
    tree._hygiene_src = src
    return _provenance_tree(tree)


def provenance_problems(rel):
    return _provenance_tree(_parse(rel)[1])


def _starred_problems(value, lineno, gc_names, cluster_names, where, simcfg_names=frozenset(),
                      gc_handles=frozenset(), tree=None):
    """Check a ``**kwargs`` argument. A literal dict is inspected key by key; anything else cannot be
    verified statically and is rejected (fail-closed, source-integrity design note)."""
    out = []
    if isinstance(value, ast.Dict):
        for k, v in zip(value.keys, value.values):
            if isinstance(k, ast.Constant) and k.value in _GRAPH_KWARGS | {"radius", "k"}:
                if isinstance(v, ast.Constant):
                    out.append((f"literal {k.value}={v.value} via **kwargs in {where}", lineno))
                elif not _is_from_graph_config(v, gc_names, cluster_names, simcfg_names, tree, value):
                    out.append((f"{k.value} via **kwargs in {where} not sourced from graph_config",
                                lineno))
    elif isinstance(value, ast.Call) and (
            getattr(value.func, "id", None) or getattr(value.func, "attr", None)
    ) in ("authoritative_cluster_kwargs", "ladder_cluster_kwargs"):
        # the helper counts only if the NAME it is reached through resolves lexically to the
        # graph_config import / module handle -- a same-named parameter shadows it (source-integrity design note)
        if isinstance(value.func, ast.Attribute):
            base = value.func.value
            _sc = _resolve_scope(tree, base.id, value) if (tree is not None
                                                           and isinstance(base, ast.Name)) else None
            ok = isinstance(base, ast.Name) and base.id in gc_handles and (
                tree is None or _sc is None or (id(_sc), base.id) in _gc_handle_pairs(tree))
            if not ok:
                out.append((f"{value.func.attr} called through an unverified object in {where}",
                            lineno))
        elif tree is not None:
            nm = value.func.id
            sc = _resolve_scope(tree, nm, value)
            if sc is not None and (id(sc), nm) not in _gc_import_pairs(tree):
                out.append((f"{nm} is shadowed by a local binding in {where}", lineno))
    elif isinstance(value, ast.Attribute) and isinstance(value.value, ast.Name) \
            and value.value.id in gc_handles:
        pass                                   # a constant taken from the authoritative module
    elif isinstance(value, ast.Name) and value.id in gc_names:
        pass                                   # a constant imported from graph_config
    else:
        out.append((f"unverifiable **kwargs passed to {where} (graph parameters cannot be traced)",
                    lineno))
    return out


def _origin_problems(tree, used_names):
    """Names whose ORIGIN is controlled must be imported from the right module and never redefined
    or re-bound locally (source-integrity design note cases 3, 4 and the earlier ClusterConfig shadowing)."""
    out = []
    for name, module in _ORIGIN_CONTROLLED.items():
        if name not in used_names:
            continue
        redefined = [n.lineno for n in ast.walk(tree)
                     if (isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                         and n.name == name)
                     or (isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store) and n.id == name)]
        if redefined:
            out.append((f"{name} is shadowed/redefined locally (lines {redefined})", 0))
            continue
        ok_modules = _GRAPH_CONFIG_MODULES if module == "graph_config" else {module}
        imported = any(isinstance(n, ast.ImportFrom) and (n.module or "") in ok_modules
                       and any(a.name == name and a.asname is None for a in n.names)
                       for n in ast.walk(tree))
        if not imported:
            out.append((f"{name} is used without importing it from {module}", 0))
    return out


def _used_call_names(tree):
    """Bare-name calls only: `helper()` must come from the right import, whereas
    `graph_config_module.helper()` is attribute access on an already-checked module alias."""
    return {n.func.id for n in ast.walk(tree)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}


def _simconfig_typed_names(tree):
    """Names bound from a genuine SimulationConfig(...) call."""
    return {tgt.id for n in ast.walk(tree) if isinstance(n, ast.Assign)
            and isinstance(n.value, ast.Call)
            and (getattr(n.value.func, "id", None) or getattr(n.value.func, "attr", None))
            == "SimulationConfig"
            for tgt in n.targets if isinstance(tgt, ast.Name)}


def _foreign_alias_problems(tree):
    """An `import ... as X` that brings a controlled name in from the WRONG module (source-integrity design note)."""
    out = []
    for n in ast.walk(tree):
        if not isinstance(n, ast.ImportFrom):
            continue
        for a in n.names:
            if a.name in _ORIGIN_CONTROLLED:
                want = _ORIGIN_CONTROLLED[a.name]
                ok = _GRAPH_CONFIG_MODULES if want == "graph_config" else {want}
                if (n.module or "") not in ok:
                    out.append((f"{a.name} imported from '{n.module}' instead of {want}"
                                + (f" (as {a.asname})" if a.asname else ""), n.lineno))
    return out
def _provenance_tree(tree):
    gc_names = _graph_config_imports(tree)
    gc_handles = _graph_config_module_handles(tree)
    _replace_names = _alias_names_of(tree, "replace")
    cluster_ctors = _ctor_aliases(tree, "ClusterConfig")
    _simcfg_ctors = _ctor_aliases(tree, "SimulationConfig")
    graph_builders = _ctor_aliases(tree, "make_fixed_local_graph")
    simcfg_names = _simconfig_typed_names(tree)
    cluster_names = _cluster_typed_names(tree)
    problems, builds_graph = [], False
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fname = getattr(node.func, "id", None) or getattr(node.func, "attr", None)
        _bare = getattr(node.func, "id", None)
        if _bare is not None:
            _role = _controlled_name_roles(tree).get(_bare)
            if _role is not None and _origin_of(_bare, node, tree) != _role:
                problems.append((f"'{_bare}' is a controlled {_role} handle but resolves to a "
                                 f"different binding at this call site", node.lineno))
                builds_graph = True
        # an attribute call naming a controlled function must come from the right module handle
        if isinstance(node.func, ast.Attribute) and node.func.attr in _ORIGIN_CONTROLLED:
            _want = ("graph_config_module"
                     if _ORIGIN_CONTROLLED[node.func.attr] == "graph_config" else "dsceos_module")
            _got = _expr_origin(node.func.value, node, tree) \
                if isinstance(node.func.value, ast.Name) else None
            if _got != _want:
                problems.append((f"'{node.func.attr}' is called on an object that is not the "
                                 f"authoritative {_want.replace('_', ' ')}", node.lineno))
                builds_graph = True
        _abase = node.func.value if isinstance(node.func, ast.Attribute) else None
        if isinstance(_abase, ast.Name):
            _mrole = _controlled_name_roles(tree).get(_abase.id)
            if _mrole in ("dsceos_module", "graph_config_module") \
                    and _origin_of(_abase.id, node, tree) != _mrole:
                problems.append((f"'{_abase.id}' is a controlled module handle but resolves to a "
                                 f"different binding at this call site", node.lineno))
                builds_graph = True
        _corig = _expr_origin(node.func, node, tree)
        if _corig == "ClusterConfig":
            builds_graph = True
            supplied = {kw.arg for kw in node.keywords if kw.arg}
            splatted = [kw for kw in node.keywords if kw.arg is None]
            if not splatted and not _GRAPH_KWARGS <= supplied:
                missing = sorted(_GRAPH_KWARGS - supplied)
                problems.append((f"ClusterConfig leaves {missing} to a dataclass default instead of "
                                 f"sourcing them from graph_config", node.lineno))
            for kw in node.keywords:
                if kw.arg is None:
                    # **kwargs: inspect a literal dict, otherwise it is unverifiable -> FAIL CLOSED
                    problems.extend(_starred_problems(kw.value, node.lineno, gc_names,
                                                     cluster_names, "ClusterConfig", simcfg_names,
                                                     gc_handles, tree))
                elif kw.arg in _GRAPH_KWARGS:
                    if isinstance(kw.value, ast.Constant):
                        problems.append((f"literal {kw.arg}={kw.value.value}", node.lineno))
                    elif not _is_from_graph_config(kw.value, gc_names, cluster_names, simcfg_names, tree, node):
                        problems.append((f"{kw.arg} not sourced from graph_config", node.lineno))
        elif _corig == "SimulationConfig":
            builds_graph = True
            kwnames = {kw.arg for kw in node.keywords if kw.arg}
            if "cluster" not in kwnames and not any(kw.arg is None for kw in node.keywords):
                problems.append(("SimulationConfig is built without an explicit cluster= argument, "
                                 "so the graph would come from a dataclass default", node.lineno))
            else:
                for kw in node.keywords:
                    if kw.arg == "cluster" and not (
                            isinstance(kw.value, ast.Name)
                            and _cluster_proven(kw.value.id, node, tree, cluster_names)):
                        problems.append(("SimulationConfig cluster= is not a proven cluster "
                                         "configuration", node.lineno))
        elif _corig == "make_fixed_local_graph":
            builds_graph = True
            rad = node.args[1] if len(node.args) >= 2 else None
            cnt = node.args[2] if len(node.args) >= 3 else None
            for kw in node.keywords:
                if kw.arg is None:
                    problems.extend(_starred_problems(kw.value, node.lineno, gc_names,
                                                      cluster_names, "make_fixed_local_graph",
                                                      simcfg_names, gc_handles, tree))
                elif kw.arg in ("radius", "communication_radius"):
                    rad = kw.value
                elif kw.arg in ("neighbour_count", "k"):
                    cnt = kw.value
            for label, val in (("radius", rad), ("neighbour_count", cnt)):
                if val is None:
                    continue
                if isinstance(val, ast.Constant):
                    problems.append((f"literal make_fixed_local_graph {label}={val.value}",
                                     node.lineno))
                elif not _is_from_graph_config(val, gc_names, cluster_names, simcfg_names, tree, node):
                    problems.append((f"make_fixed_local_graph {label} not sourced from graph_config",
                                     node.lineno))
    # dataclasses.replace(<cluster>, communication_radius=...) rewrites graph parameters too
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and (
                getattr(node.func, "id", None) or getattr(node.func, "attr", None)
        ) and _expr_origin(node.func, node, tree) == "replace":
            if node.args and isinstance(node.args[0], ast.Name) and _cluster_proven(
                    node.args[0].id, node, tree, cluster_names):
                for kw in node.keywords:
                    if kw.arg in _GRAPH_KWARGS:
                        if isinstance(kw.value, ast.Constant):
                            problems.append((f"dataclasses.replace overrides {kw.arg} with the "
                                             f"literal {kw.value.value}", node.lineno))
                        elif not _is_from_graph_config(kw.value, gc_names, cluster_names,
                                                       simcfg_names, tree, node):
                            problems.append((f"dataclasses.replace sets {kw.arg} from a "
                                             f"non-authoritative value", node.lineno))
                    elif kw.arg is None:
                        problems.append(("dataclasses.replace with unverifiable **kwargs on a "
                                         "cluster configuration", node.lineno))
    for n in ast.walk(tree):
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store) and n.id in gc_names:
            # a graph_config alias may never be re-bound, in ANY scope (source-integrity design note case 2)
            problems.append((f"re-binds graph_config alias '{n.id}'", n.lineno))
    if os.path.basename(getattr(tree, "_hygiene_file", "")).startswith("ladder_"):
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and (
                    getattr(node.func, "id", None) in cluster_ctors):
                literal_units = sorted(kw.arg for kw in node.keywords
                                       if kw.arg and kw.arg.startswith("n_")
                                       and isinstance(kw.value, ast.Constant))
                if literal_units:
                    problems.append((f"ladder runner hard-codes the fleet composition "
                                     f"{literal_units} instead of splatting LADDER_UNIT_MIX",
                                     node.lineno))
    for _nm, _ln in _controlled_value_uses(tree):
        problems.append((f"'{_nm}' is used as a VALUE rather than called directly; controlled "
                         f"callables may only be imported and called, never aliased", _ln))
    for _sc, _nm, _ln in sorted(_aliased_controlled_bindings(tree), key=lambda x: x[2]):
        problems.append((f"controlled callable/module re-bound to '{_nm}' by a non-import binding; "
                         f"controlled names may only be introduced by a direct import", _ln))
    if builds_graph:
        if not gc_names:
            problems.append(("builds a cluster/graph but never imports graph_config", 0))
        problems.extend(_origin_problems(tree, _used_call_names(tree)))
        problems.extend(_foreign_alias_problems(tree))
        # `<something>.ClusterConfig(...)` / `<something>.authoritative_cluster_kwargs()` is only
        # acceptable when <something> is a module alias we have already verified
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) \
                    and node.func.attr in _ORIGIN_CONTROLLED:
                base = node.func.value
                want = ("graph_config_module" if _ORIGIN_CONTROLLED[node.func.attr] == "graph_config"
                        else "dsceos_module")
                base_origin = _origin_of(base.id, node, tree) if isinstance(base, ast.Name) else None
                if base_origin != want:
                    problems.append((f"{node.func.attr} called through an object whose binding is "
                                     f"not the {want.replace('_', ' ')}", node.lineno))
    return problems


def test_released_paths_take_graph_params_from_graph_config():
    """Graph parameters must be single-sourced from graph_config.py in every released path: no
    literal, no foreign name, no locally re-bound alias."""
    offenders = {}
    for rel in released_python_paths():
        if rel in _PROVENANCE_EXEMPT:
            continue
        probs = provenance_problems(rel)
        if probs:
            offenders[rel] = probs
    assert not offenders, f"graph-parameter provenance violations: {offenders}"



# ------------------------------------------------------- self-tests of the guards themselves
# Each snippet is a bypass that a past audit demonstrated against an earlier version of this file.
# Keeping them here proves, inside the package and WITHOUT pyflakes, that the guards still fire.
_NAME_BYPASSES = {
    "source-integrity C-01 stale reference after rename": """
def build(cluster):
    ccluster = ClusterConfig()
    return make_fixed_local_graph(layout, cc.communication_radius, cc.neighbour_count)
""",
    "source-integrity 4.2 module-level undefined name": """
BROKEN_MODULE_BINDING = missing_module_name
""",
    "source-integrity 3.2 function use-before-assignment": """
def main():
    _audit = audit_uba
    audit_uba = 1
    return _audit
""",
    "source-integrity 3.3 cross-function shadow of an undefined global": """
def broken():
    return audit_shadow_name

def unrelated():
    audit_shadow_name = 1
    return audit_shadow_name

AUDIT_TRIGGER = broken()
""",
    "source-integrity 4.1.1 undefined name in a lambda body": "f = lambda: audit_missing_lambda\n",
    "source-integrity 4.1.2 undefined name in a class body": "class B:\n    value = audit_missing_class_body\n",
    "source-integrity 4.1.3 default argument bound later": (
        "def f(a=audit_later_default):\n    return a\naudit_later_default = 1\n"),
    "source-integrity 4.1.4 decorator bound later": (
        "@audit_later_decorator\ndef f():\n    return 1\n"
        "def audit_later_decorator(x):\n    return x\n"),
    "source-integrity 4.1.5 class base defined later": (
        "class C(AuditLaterBase):\n    pass\nclass AuditLaterBase:\n    pass\n"),
    "source-integrity 4.1.6 annotation defined later": (
        "def f(x: AuditLaterAnn) -> int:\n    return 1\nclass AuditLaterAnn:\n    pass\n"),
    "source-integrity 4.1.7 undefined name in a nested lambda": (
        "def g():\n    return (lambda: audit_missing_nested)()\n"),
    "source-integrity 3.1 comprehension target treated as scope-wide": (
        "print(audit_comp_name)\n_v = [audit_comp_name for audit_comp_name in range(1)]\n"),
    "source-integrity 3.2.1 comprehension iterable self-reference": "xs = [i for i in i]\n",
    "source-integrity 3.2.2 except alias used after its handler": (
        "try:\n    pass\nexcept ValueError as e:\n    pass\nprint(e)\n"),
    "source-integrity 3.2.3 name used after del": "x = 1\ndel x\nprint(x)\n",
    "source-integrity 3.2.5 class-body comprehension reaching a class-level name": (
        "class C:\n    vals = [1, 2]\n    d = [v * k for v in vals for k in vals]\n"),
}

_PROVENANCE_BYPASSES = {
    "source-integrity m-01 alias replaced by identical literals": """
from graph_config import AUTHORITATIVE_COMMUNICATION_RADIUS as _R045
cfg = ClusterConfig(seed=7, communication_radius=0.45, neighbour_count=4, layout_spread=1.0)
""",
    "source-integrity 4.4 alias re-bound after import": """
from graph_config import AUTHORITATIVE_COMMUNICATION_RADIUS as _R045
_R045 = 0.45
cfg = ClusterConfig(communication_radius=_R045)
""",
    "source-integrity 3.4 look-alike object faking cluster provenance": """
from dsceos_validation import ClusterConfig
from graph_config import AUTHORITATIVE_COMMUNICATION_RADIUS as _R045
class Fake:
    communication_radius = 0.45
    neighbour_count = 4
fake = Fake()
cfg = ClusterConfig(communication_radius=fake.communication_radius,
                    neighbour_count=fake.neighbour_count)
""",
    "source-integrity 5.1 graph parameters smuggled through a **dict literal": """
from dsceos_validation import ClusterConfig
from graph_config import AUTHORITATIVE_COMMUNICATION_RADIUS as _R045
cfg = ClusterConfig(**{"communication_radius": 0.2, "neighbour_count": 1,
                       "layout_spread": 2.0, "seed": 9})
""",
    "source-integrity 5.1b opaque **kwargs name": """
from dsceos_validation import ClusterConfig
from graph_config import AUTHORITATIVE_COMMUNICATION_RADIUS as _R045
d = {"communication_radius": 0.2}
cfg = ClusterConfig(**d)
""",
    "source-integrity 5.1c **kwargs into the graph builder": """
from dsceos_validation import ClusterConfig, make_fixed_local_graph
from graph_config import AUTHORITATIVE_COMMUNICATION_RADIUS as _R045
W = make_fixed_local_graph(layout, **{"radius": 0.2, "neighbour_count": 1})
""",
    "source-integrity 5.2 fake .cluster attribute": """
from dsceos_validation import ClusterConfig, make_fixed_local_graph
from graph_config import AUTHORITATIVE_COMMUNICATION_RADIUS as _R045
holder = something_else
cc = holder.cluster
W = make_fixed_local_graph(layout, cc.communication_radius, cc.neighbour_count)
""",
    "source-integrity 5.3 locally shadowed ClusterConfig": """
from dsceos_validation import make_fixed_local_graph
from graph_config import AUTHORITATIVE_COMMUNICATION_RADIUS as _R045
def ClusterConfig(**kwargs):
    return 1
cfg = ClusterConfig(communication_radius=0.2)
""",
    "source-integrity 5.4 helper blessed by one good call site only": """
from dsceos_validation import ClusterConfig
from graph_config import AUTHORITATIVE_COMMUNICATION_RADIUS as _R045
def helper(cc):
    return ClusterConfig(communication_radius=cc.communication_radius)
good = ClusterConfig(communication_radius=_R045)
helper(good)
helper(fake_obj)
""",
    "source-integrity 4.2.1 direct holder.cluster attribute chain": """
from dsceos_validation import ClusterConfig
from graph_config import AUTHORITATIVE_COMMUNICATION_RADIUS as _R045
cfg = ClusterConfig(communication_radius=holder.cluster.communication_radius,
                    neighbour_count=holder.cluster.neighbour_count)
""",
    "source-integrity 4.2.2 graph_config alias re-bound inside a function": """
from dsceos_validation import ClusterConfig
from graph_config import AUTHORITATIVE_COMMUNICATION_RADIUS as _R045
def f():
    from graph_config import AUTHORITATIVE_COMMUNICATION_RADIUS as _R2
    _R2 = 0.2
    return ClusterConfig(communication_radius=_R2)
""",
    "source-integrity 4.2.3 locally shadowed authoritative_cluster_kwargs": """
from dsceos_validation import ClusterConfig
from graph_config import AUTHORITATIVE_COMMUNICATION_RADIUS as _R045
def authoritative_cluster_kwargs():
    return {"communication_radius": 0.2}
cfg = ClusterConfig(**authoritative_cluster_kwargs())
""",
    "source-integrity 4.2.4 shadowed SimulationConfig yielding a fake cluster": """
from dsceos_validation import ClusterConfig, make_fixed_local_graph
from graph_config import AUTHORITATIVE_COMMUNICATION_RADIUS as _R045
def SimulationConfig(**k):
    return 1
s = SimulationConfig()
cc = s.cluster
W = make_fixed_local_graph(layout, cc.communication_radius, cc.neighbour_count)
""",
    "source-integrity 4.2.5 helper reachable through an indirect alias": """
from dsceos_validation import ClusterConfig
from graph_config import AUTHORITATIVE_COMMUNICATION_RADIUS as _R045
def helper(cc):
    return ClusterConfig(communication_radius=cc.communication_radius)
good = ClusterConfig(communication_radius=_R045)
helper(good)
h = helper
h(fake_obj)
""",
    "source-integrity 4.2.6 module whose name merely ends with graph_config": """
from fake_graph_config import AUTHORITATIVE_COMMUNICATION_RADIUS as _R045
from dsceos_validation import ClusterConfig
cfg = ClusterConfig(communication_radius=_R045)
""",
    "source-integrity 2.1 ClusterConfig() left to dataclass defaults": """
from dsceos_validation import ClusterConfig
from graph_config import AUTHORITATIVE_COMMUNICATION_RADIUS as _R045
cc = ClusterConfig()
""",
    "source-integrity 2.2 constructor aliased at import": """
from dsceos_validation import ClusterConfig as CC
from graph_config import AUTHORITATIVE_COMMUNICATION_RADIUS as _R045
cc = CC(communication_radius=0.2, neighbour_count=1, layout_spread=2.0, seed=9)
""",
    "source-integrity 2.3 dataclasses.replace overriding graph parameters": """
from dsceos_validation import ClusterConfig
from graph_config import AUTHORITATIVE_COMMUNICATION_RADIUS as _R045, authoritative_cluster_kwargs
from dataclasses import replace
cc = ClusterConfig(**authoritative_cluster_kwargs())
cc2 = replace(cc, communication_radius=0.2, neighbour_count=1)
""",
    "source-integrity 2.4 constructor aliased by assignment": """
from dsceos_validation import ClusterConfig
from graph_config import AUTHORITATIVE_COMMUNICATION_RADIUS as _R045
CC = ClusterConfig
cc = CC(communication_radius=0.2, neighbour_count=1, layout_spread=2.0, seed=9)
""",
    "source-integrity 2.5 functools.partial over the constructor": """
from dsceos_validation import ClusterConfig
from graph_config import AUTHORITATIVE_COMMUNICATION_RADIUS as _R045
from functools import partial
P = partial(ClusterConfig, communication_radius=0.2)
cc = P(neighbour_count=1, layout_spread=2.0, seed=9)
""",
    "source-integrity 2.6 aliased graph builder": """
from dsceos_validation import ClusterConfig, make_fixed_local_graph
from graph_config import AUTHORITATIVE_COMMUNICATION_RADIUS as _R045
mk = make_fixed_local_graph
W = mk(layout, 0.2, 1)
""",
    "source-integrity 2.7 fake ClusterConfig through an attribute": """
from dsceos_validation import ClusterConfig
from graph_config import AUTHORITATIVE_COMMUNICATION_RADIUS as _R045
cc = shady.ClusterConfig(communication_radius=0.2, neighbour_count=1, layout_spread=2.0, seed=9)
""",
    "source-integrity policy: constructor aliased from a module handle (controlled callable used as a value)": (
        "import dsceos_validation as dv\nfrom dsceos_validation import make_fixed_local_graph\nfrom graph_config import authoritative_cluster_kwargs\nlayout = None\nCC = dv.ClusterConfig\ncc = CC(**authoritative_cluster_kwargs())\nW = make_fixed_local_graph(layout, cc.communication_radius, cc.neighbour_count)\n"),
    "source-integrity policy: module-qualified functools.partial (controlled callable used as a value)": (
        "from dsceos_validation import ClusterConfig, make_fixed_local_graph\nimport functools\nfrom graph_config import authoritative_cluster_kwargs\nlayout = None\nP = functools.partial(ClusterConfig)\ncc = P(**authoritative_cluster_kwargs())\nW = make_fixed_local_graph(layout, cc.communication_radius, cc.neighbour_count)\n"),
    "source-integrity policy: constructor aliased by assignment (controlled callable used as a value)": (
        "from dsceos_validation import ClusterConfig, make_fixed_local_graph\nfrom graph_config import authoritative_cluster_kwargs\nlayout = None\nCC = ClusterConfig\ncc = CC(**authoritative_cluster_kwargs())\nW = make_fixed_local_graph(layout, cc.communication_radius, cc.neighbour_count)\n"),
    "source-integrity M-01a constructor and builder via tuple unpacking": (
        "from dsceos_validation import ClusterConfig, make_fixed_local_graph\nfrom graph_config import authoritative_cluster_kwargs\nlayout = None\nCC, mk = ClusterConfig, make_fixed_local_graph\ncc = CC(**authoritative_cluster_kwargs())\nW = mk(layout, 0.2, 1)\n"),
    "source-integrity M-01b controlled callables as default parameters": (
        "from dsceos_validation import ClusterConfig, make_fixed_local_graph\nfrom graph_config import authoritative_cluster_kwargs\nlayout = None\ndef use(_cc=ClusterConfig, _mk=make_fixed_local_graph):\n    c = _cc(**authoritative_cluster_kwargs())\n    return _mk(layout, 0.2, 1)\nuse()\n"),
    "source-integrity M-01c partial with preset positional literals": (
        "from dsceos_validation import ClusterConfig, make_fixed_local_graph\nfrom graph_config import authoritative_cluster_kwargs\nlayout = None\nfrom functools import partial\nmk = partial(make_fixed_local_graph, layout, 0.2)\nW = mk(1)\n"),
    "source-integrity M-01d partial with preset keyword literals": (
        "from dsceos_validation import ClusterConfig, make_fixed_local_graph\nfrom graph_config import authoritative_cluster_kwargs\nlayout = None\nfrom functools import partial\nmk = partial(make_fixed_local_graph, radius=0.2)\nW = mk(layout, neighbour_count=1)\n"),
    "source-integrity M-01e module-attribute alias later re-bound to a fake": (
        "import dsceos_validation as dv\nlayout = None\nmk = dv.make_fixed_local_graph\nmk = lambda l, r, k: 0\nW = mk(layout, 0.2, 1)\n"),
    "source-integrity M-01f alias from a conditional expression": (
        "from dsceos_validation import ClusterConfig, make_fixed_local_graph\nfrom graph_config import authoritative_cluster_kwargs\nlayout = None\nmk = make_fixed_local_graph if layout is None else None\nW = mk(layout, 0.2, 1)\n"),
    "source-integrity M-01g alias from a tuple subscript": (
        "from dsceos_validation import ClusterConfig, make_fixed_local_graph\nfrom graph_config import authoritative_cluster_kwargs\nlayout = None\n_t = (make_fixed_local_graph,)\nmk = _t[0]\nW = mk(layout, 0.2, 1)\n"),
    "source-integrity M-01h alias through a class namespace": (
        "from dsceos_validation import ClusterConfig, make_fixed_local_graph\nfrom graph_config import authoritative_cluster_kwargs\nlayout = None\nclass NS:\n    mk = make_fixed_local_graph\nW = NS.mk(layout, 0.2, 1)\n"),
    "source-integrity M-01i local helper returns the builder": (
        "from dsceos_validation import ClusterConfig, make_fixed_local_graph\nfrom graph_config import authoritative_cluster_kwargs\nlayout = None\ndef get():\n    return make_fixed_local_graph\nmk = get()\nW = mk(layout, 0.2, 1)\n"),
    "source-integrity M-01j for-target receives the builder": (
        "from dsceos_validation import ClusterConfig, make_fixed_local_graph\nfrom graph_config import authoritative_cluster_kwargs\nlayout = None\nfor mk in [make_fixed_local_graph]:\n    W = mk(layout, 0.2, 1)\n"),
    "source-integrity M-01k with-target receives the builder": (
        "from dsceos_validation import ClusterConfig, make_fixed_local_graph\nfrom graph_config import authoritative_cluster_kwargs\nlayout = None\nimport contextlib\nwith contextlib.nullcontext(make_fixed_local_graph) as mk:\n    W = mk(layout, 0.2, 1)\n"),
    "source-integrity M-01a dv.ClusterConfig and dv.builder with literals": (
        "import dsceos_validation as dv\nlayout=None\ncc = dv.ClusterConfig(communication_radius=0.2, neighbour_count=1, layout_spread=1.0, seed=7)\nW = dv.make_fixed_local_graph(layout, cc.communication_radius, cc.neighbour_count)\n"),
    "source-integrity M-01b authoritative cluster but literal dv.make_fixed_local_graph": (
        "import dsceos_validation as dv\nfrom graph_config import authoritative_cluster_kwargs\nlayout=None\ncc = dv.ClusterConfig(**authoritative_cluster_kwargs())\nW = dv.make_fixed_local_graph(layout, 0.2, 1)\n"),
    "source-integrity M-01c dv.SimulationConfig with a non-authoritative cluster": (
        "import dsceos_validation as dv\nlayout=None\ncc = dv.ClusterConfig(communication_radius=0.2, neighbour_count=1, layout_spread=1.0, seed=7)\ncfg = dv.SimulationConfig(cluster=cc)\n"),
    "source-integrity M-01d module-attribute aliases then literal calls": (
        "import dsceos_validation as dv\nlayout=None\nCC = dv.ClusterConfig\nmk = dv.make_fixed_local_graph\ncc = CC(communication_radius=0.2, neighbour_count=1, layout_spread=1.0, seed=7)\nW = mk(layout, 0.2, 1)\n"),
    "source-integrity M-01a dsceos module alias re-bound": (
        "import dsceos_validation as dv\nfrom graph_config import authoritative_cluster_kwargs, AUTHORITATIVE_COMMUNICATION_RADIUS as R, AUTHORITATIVE_NEIGHBOUR_COUNT as K\ndv = FakeModule\ncc = dv.ClusterConfig(**authoritative_cluster_kwargs())\nW = dv.make_fixed_local_graph(layout, cc.communication_radius, cc.neighbour_count)\n"),
    "source-integrity M-01b dsceos module alias shadowed by a parameter": (
        "import dsceos_validation as dv\nfrom graph_config import authoritative_cluster_kwargs, AUTHORITATIVE_COMMUNICATION_RADIUS as R, AUTHORITATIVE_NEIGHBOUR_COUNT as K\ndef use(dv):\n    cc = dv.ClusterConfig(**authoritative_cluster_kwargs())\n    return dv.make_fixed_local_graph(layout, cc.communication_radius, cc.neighbour_count)\nuse(FakeModule)\n"),
    "source-integrity M-01c graph-builder alias re-bound": (
        "from dsceos_validation import make_fixed_local_graph as build_graph\nfrom graph_config import authoritative_cluster_kwargs, AUTHORITATIVE_COMMUNICATION_RADIUS as R, AUTHORITATIVE_NEIGHBOUR_COUNT as K\ndef fake_builder(layout, radius, k):\n    return 0.2, 1\nbuild_graph = fake_builder\nW = build_graph(layout, R, K)\n"),
    "source-integrity M-01d graph-builder alias shadowed by a parameter": (
        "from dsceos_validation import make_fixed_local_graph as build_graph\nfrom graph_config import authoritative_cluster_kwargs, AUTHORITATIVE_COMMUNICATION_RADIUS as R, AUTHORITATIVE_NEIGHBOUR_COUNT as K\ndef use(build_graph):\n    return build_graph(layout, R, K)\nuse(fake_builder)\n"),
    "source-integrity M-01e module-returning helper shadowed by a parameter": (
        "from dsceos_validation import ClusterConfig\ndef _auth():\n    import graph_config as g\n    return g\ndef use(_auth):\n    gc = _auth()\n    return ClusterConfig(**gc.authoritative_cluster_kwargs())\nuse(FakeAuth)\n"),
    "source-integrity M-01f dataclasses.replace shadowed by a parameter": (
        "from dsceos_validation import ClusterConfig, make_fixed_local_graph\nfrom dataclasses import replace\nfrom graph_config import authoritative_cluster_kwargs\ncc = ClusterConfig(**authoritative_cluster_kwargs())\ndef use(replace):\n    cc2 = replace(cc, communication_radius=0.2, neighbour_count=1)\n    return make_fixed_local_graph(layout, cc2.communication_radius, cc2.neighbour_count)\nuse(fake_replace)\n"),
    "source-integrity M-01g SimulationConfig through a shadowed module alias": (
        "import dsceos_validation as dv\nfrom graph_config import authoritative_cluster_kwargs, AUTHORITATIVE_COMMUNICATION_RADIUS as R, AUTHORITATIVE_NEIGHBOUR_COUNT as K\ndef use(dv):\n    cfg = dv.SimulationConfig(cluster=FakeCluster())\n    cc = cfg.cluster\n    return dv.make_fixed_local_graph(layout, cc.communication_radius, cc.neighbour_count)\nuse(FakeModule)\n"),
    "source-integrity M-01a comprehension walrus re-binding": (
        "from dsceos_validation import ClusterConfig, SimulationConfig, make_fixed_local_graph\nfrom graph_config import authoritative_cluster_kwargs\ncc = ClusterConfig(**authoritative_cluster_kwargs())\nW = [ make_fixed_local_graph(layout, cc.communication_radius, cc.neighbour_count) for _ in [0] if (cc := FakeCluster()) ]\n"),
    "source-integrity M-01b nested nonlocal skipping an intermediate scope": (
        "from dsceos_validation import ClusterConfig, SimulationConfig, make_fixed_local_graph\nfrom graph_config import authoritative_cluster_kwargs\ncc = ClusterConfig(**authoritative_cluster_kwargs())\ndef a():\n    cc = ClusterConfig(**authoritative_cluster_kwargs())\n    def b():\n        def c():\n            nonlocal cc\n            cc = FakeCluster()\n        c()\n    b()\n    return make_fixed_local_graph(layout, cc.communication_radius, cc.neighbour_count)\n"),
    "source-integrity M-01c class body using a class-level look-alike": (
        "from dsceos_validation import ClusterConfig, SimulationConfig, make_fixed_local_graph\nfrom graph_config import authoritative_cluster_kwargs\ncc = ClusterConfig(**authoritative_cluster_kwargs())\nclass K:\n    cc = FakeCluster()\n    W = make_fixed_local_graph(layout, cc.communication_radius, cc.neighbour_count)\n"),
    "source-integrity M-01d ClusterConfig shadowed by a parameter": (
        "from dsceos_validation import ClusterConfig, make_fixed_local_graph\nfrom graph_config import authoritative_cluster_kwargs\ndef use(ClusterConfig):\n    return ClusterConfig(**authoritative_cluster_kwargs())\ncc = use(FakeCtor)\nW = make_fixed_local_graph(layout, cc.communication_radius, cc.neighbour_count)\n"),
    "source-integrity M-01e authoritative_cluster_kwargs shadowed by a parameter": (
        "from dsceos_validation import ClusterConfig\nfrom graph_config import authoritative_cluster_kwargs\ndef use(authoritative_cluster_kwargs):\n    return ClusterConfig(**authoritative_cluster_kwargs())\ncc = use(fake_kwargs)\n"),
    "source-integrity M-01f graph_config module alias shadowed by a parameter": (
        "from dsceos_validation import ClusterConfig\nimport graph_config as gc\ndef use(gc):\n    return ClusterConfig(**gc.authoritative_cluster_kwargs())\ncc = use(FakeMod())\n"),
    "source-integrity M-01g SimulationConfig shadowed by a parameter": (
        "from dsceos_validation import ClusterConfig, SimulationConfig, make_fixed_local_graph\nfrom graph_config import authoritative_cluster_kwargs\ncc = ClusterConfig(**authoritative_cluster_kwargs())\ndef use(SimulationConfig):\n    cfg = SimulationConfig(cluster=FakeCluster())\n    cc2 = cfg.cluster\n    return make_fixed_local_graph(layout, cc2.communication_radius, cc2.neighbour_count)\n"),
    "source-integrity M-01h make_fixed_local_graph shadowed by a parameter": (
        "from dsceos_validation import ClusterConfig, SimulationConfig, make_fixed_local_graph\nfrom graph_config import authoritative_cluster_kwargs\ncc = ClusterConfig(**authoritative_cluster_kwargs())\ndef use(make_fixed_local_graph):\n    return make_fixed_local_graph(layout, 0.2, 1)\nuse(fake_builder)\n"),
    "source-integrity M-01a comprehension target shadows a proven name": (
        "from dsceos_validation import ClusterConfig, make_fixed_local_graph\nfrom graph_config import authoritative_cluster_kwargs\ncc = ClusterConfig(**authoritative_cluster_kwargs())\nW = [ make_fixed_local_graph(layout, cc.communication_radius, cc.neighbour_count) for cc in [FakeCluster()] ]\n"),
    "source-integrity M-01b global re-binding from another scope": (
        "from dsceos_validation import ClusterConfig, make_fixed_local_graph\nfrom graph_config import authoritative_cluster_kwargs\ncc = ClusterConfig(**authoritative_cluster_kwargs())\ndef spoil():\n    global cc\n    cc = FakeCluster()\nspoil()\nW = make_fixed_local_graph(layout, cc.communication_radius, cc.neighbour_count)\n"),
    "source-integrity M-01c nonlocal re-binding in an inner function": (
        "from dsceos_validation import ClusterConfig, make_fixed_local_graph\nfrom graph_config import authoritative_cluster_kwargs\ncc = ClusterConfig(**authoritative_cluster_kwargs())\ndef outer():\n    cc = ClusterConfig(**authoritative_cluster_kwargs())\n    def inner():\n        nonlocal cc\n        cc = FakeCluster()\n    inner()\n    return make_fixed_local_graph(layout, cc.communication_radius, cc.neighbour_count)\n"),
    "source-integrity M-01d graph_config alias shadowed by a parameter": (
        "from dsceos_validation import ClusterConfig\nimport graph_config as gc\ndef use(gc):\n    return ClusterConfig(communication_radius=gc.AUTHORITATIVE_COMMUNICATION_RADIUS, neighbour_count=gc.AUTHORITATIVE_NEIGHBOUR_COUNT, layout_spread=gc.AUTHORITATIVE_LAYOUT_SPREAD, seed=gc.AUTHORITATIVE_SEED)\nuse(FakeModule())\n"),
    "source-integrity M-01e call-argument scope contamination": (
        "from dsceos_validation import ClusterConfig, make_fixed_local_graph\nfrom graph_config import authoritative_cluster_kwargs\ncc = ClusterConfig(**authoritative_cluster_kwargs())\ndef helper(cc):\n    return make_fixed_local_graph(layout, cc.communication_radius, cc.neighbour_count)\ndef caller():\n    cc = FakeCluster()\n    return helper(cc)\n"),
    "source-integrity M-01f unproven keyword-only argument": (
        "from dsceos_validation import ClusterConfig, make_fixed_local_graph\nfrom graph_config import authoritative_cluster_kwargs\ncc = ClusterConfig(**authoritative_cluster_kwargs())\ndef use(*, cc):\n    return make_fixed_local_graph(layout, cc.communication_radius, cc.neighbour_count)\nuse(cc=FakeCluster())\n"),
    "source-integrity M-01a same-named ordinary parameter in an inner scope": (
        "from dsceos_validation import ClusterConfig, make_fixed_local_graph\nfrom graph_config import authoritative_cluster_kwargs\ncc = ClusterConfig(**authoritative_cluster_kwargs())\n"
        "def use(cc):\n    return make_fixed_local_graph(layout, cc.communication_radius, cc.neighbour_count)\nW = use(FakeCluster())\n"),
    "source-integrity M-01b same-named positional-only parameter": (
        "from dsceos_validation import ClusterConfig, make_fixed_local_graph\nfrom graph_config import authoritative_cluster_kwargs\ncc = ClusterConfig(**authoritative_cluster_kwargs())\n"
        "def use(cc, /):\n    return make_fixed_local_graph(layout, cc.communication_radius, cc.neighbour_count)\nW = use(FakeCluster())\n"),
    "source-integrity M-01c same-named keyword-only parameter": (
        "from dsceos_validation import ClusterConfig, make_fixed_local_graph\nfrom graph_config import authoritative_cluster_kwargs\ncc = ClusterConfig(**authoritative_cluster_kwargs())\n"
        "def use(*, cc):\n    return make_fixed_local_graph(layout, cc.communication_radius, cc.neighbour_count)\nW = use(cc=FakeCluster())\n"),
    "source-integrity M-01d same-named lambda parameter": (
        "from dsceos_validation import ClusterConfig, make_fixed_local_graph\nfrom graph_config import authoritative_cluster_kwargs\ncc = ClusterConfig(**authoritative_cluster_kwargs())\n"
        "use = lambda cc: make_fixed_local_graph(layout, cc.communication_radius, cc.neighbour_count)\nW = use(FakeCluster())\n"),
    "source-integrity M-01a re-bound through `with ... as`": (
        "from dsceos_validation import ClusterConfig, make_fixed_local_graph\nfrom graph_config import authoritative_cluster_kwargs\ncc = ClusterConfig(**authoritative_cluster_kwargs())\n"
        "with fake_ctx() as cc:\n    pass\n"
        "W = make_fixed_local_graph(layout, cc.communication_radius, cc.neighbour_count)\n"),
    "source-integrity M-01b re-bound through `except ... as`": (
        "from dsceos_validation import ClusterConfig, make_fixed_local_graph\nfrom graph_config import authoritative_cluster_kwargs\ncc = ClusterConfig(**authoritative_cluster_kwargs())\n"
        "try:\n    pass\nexcept ValueError as cc:\n    pass\n"
        "W = make_fixed_local_graph(layout, cc.communication_radius, cc.neighbour_count)\n"),
    "source-integrity M-01c re-bound through a walrus": (
        "from dsceos_validation import ClusterConfig, make_fixed_local_graph\nfrom graph_config import authoritative_cluster_kwargs\ncc = ClusterConfig(**authoritative_cluster_kwargs())\n"
        "if (cc := FakeCluster()):\n    pass\n"
        "W = make_fixed_local_graph(layout, cc.communication_radius, cc.neighbour_count)\n"),
    "source-integrity M-01d re-bound through a match capture": (
        "from dsceos_validation import ClusterConfig, make_fixed_local_graph\nfrom graph_config import authoritative_cluster_kwargs\ncc = ClusterConfig(**authoritative_cluster_kwargs())\n"
        "match obj:\n    case cc:\n        pass\n"
        "W = make_fixed_local_graph(layout, cc.communication_radius, cc.neighbour_count)\n"),
    "source-integrity M-01e shadowed by a class definition": (
        "from dsceos_validation import ClusterConfig, make_fixed_local_graph\nfrom graph_config import authoritative_cluster_kwargs\ncc = ClusterConfig(**authoritative_cluster_kwargs())\n"
        "class cc:\n    communication_radius = 0.2\n    neighbour_count = 1\n"
        "W = make_fixed_local_graph(layout, cc.communication_radius, cc.neighbour_count)\n"),
    "source-integrity M-01f shadowed by an import alias": (
        "from dsceos_validation import ClusterConfig, make_fixed_local_graph\nfrom graph_config import authoritative_cluster_kwargs\ncc = ClusterConfig(**authoritative_cluster_kwargs())\n"
        "import fake_module as cc\n"
        "W = make_fixed_local_graph(layout, cc.communication_radius, cc.neighbour_count)\n"),
    "source-integrity M-02a proven cluster re-bound to a look-alike": """
from dsceos_validation import ClusterConfig, make_fixed_local_graph
from graph_config import authoritative_cluster_kwargs
cc = ClusterConfig(**authoritative_cluster_kwargs())
cc = FakeCluster()
W = make_fixed_local_graph(layout, cc.communication_radius, cc.neighbour_count)
""",
    "source-integrity M-02b conditional re-binding to a look-alike": """
from dsceos_validation import ClusterConfig, make_fixed_local_graph
from graph_config import authoritative_cluster_kwargs
cc = ClusterConfig(**authoritative_cluster_kwargs())
if flag:
    cc = FakeCluster()
W = make_fixed_local_graph(layout, cc.communication_radius, cc.neighbour_count)
""",
    "source-integrity M-02c re-bound cluster handed to SimulationConfig": """
from dsceos_validation import ClusterConfig, SimulationConfig
from graph_config import authoritative_cluster_kwargs
cc = ClusterConfig(**authoritative_cluster_kwargs())
cc = FakeCluster()
cfg = SimulationConfig(cluster=cc)
""",
    "source-integrity 4.1a locally shadowed ladder_cluster_kwargs": """
from dsceos_validation import ClusterConfig
from graph_config import LADDER_UNIT_MIX, ladder_cluster_kwargs
def ladder_cluster_kwargs():
    return {"communication_radius": 0.2, "neighbour_count": 1, "layout_spread": 2.0, "seed": 9}
cc = ClusterConfig(**LADDER_UNIT_MIX, **ladder_cluster_kwargs())
""",
    "source-integrity 4.1b SimulationConfig built without an explicit cluster": """
from dsceos_validation import SimulationConfig
from graph_config import AUTHORITATIVE_COMMUNICATION_RADIUS as _R045
cfg = SimulationConfig()
""",
    "source-integrity 4.1c dataclasses.replace reached through an alias": """
from dsceos_validation import ClusterConfig
from graph_config import LADDER_UNIT_MIX, ladder_cluster_kwargs
from dataclasses import replace as rp
cc = ClusterConfig(**LADDER_UNIT_MIX, **ladder_cluster_kwargs())
cc2 = rp(cc, communication_radius=0.2, neighbour_count=1)
""",
    "source-integrity 4.1d constructor imported from a foreign module": """
from audit_fake_defs import ClusterConfig as CC
from graph_config import AUTHORITATIVE_COMMUNICATION_RADIUS as _R045
cc = CC(communication_radius=_R045, neighbour_count=4, layout_spread=1.0, seed=7)
""",
    "source-integrity 4.1e graph builder imported from a foreign module": """
from audit_fake_builder_defs import make_fixed_local_graph as mk
from graph_config import AUTHORITATIVE_COMMUNICATION_RADIUS as _R045
W = mk(layout, _R045, 4)
""",
    "source-integrity 4.1f functools.partial reached through an alias": """
from dsceos_validation import ClusterConfig
from graph_config import AUTHORITATIVE_COMMUNICATION_RADIUS as _R045
from functools import partial
p = partial
CC = p(ClusterConfig, communication_radius=0.2)
cc = CC(neighbour_count=1, layout_spread=2.0, seed=9)
""",
    "source-integrity 4.1g constructor aliased through an annotated assignment": """
from dsceos_validation import ClusterConfig
from graph_config import AUTHORITATIVE_COMMUNICATION_RADIUS as _R045
CC: object = ClusterConfig
cc = CC(communication_radius=0.2, neighbour_count=1, layout_spread=2.0, seed=9)
""",
    "source-integrity 2.8 fake kwargs helper through an attribute": """
from dsceos_validation import ClusterConfig
from graph_config import AUTHORITATIVE_COMMUNICATION_RADIUS as _R045
cc = ClusterConfig(**shady.authoritative_cluster_kwargs())
""",
    "source-integrity 4.2.7 capture marker applied to a fake object": """
from dsceos_validation import ClusterConfig, make_fixed_local_graph
from graph_config import AUTHORITATIVE_COMMUNICATION_RADIUS as _R045
cc = fake_thing.cluster  # provenance-ok: effective SimulationConfig capture
W = make_fixed_local_graph(layout, cc.communication_radius, cc.neighbour_count)
""",
}


def test_name_guard_catches_known_bypasses():
    """Every historical name-resolution bypass must still be detected."""
    missed = [label for label, src in _NAME_BYPASSES.items() if not analyze_names_source(src)]
    assert not missed, f"name guard no longer detects: {missed}"


def test_provenance_guard_catches_known_bypasses():
    """Every historical graph-provenance bypass must still be detected."""
    missed = [label for label, src in _PROVENANCE_BYPASSES.items()
              if not provenance_problems_source(src)]
    assert not missed, f"provenance guard no longer detects: {missed}"


_VALID_CODE = {
    "PEP 563 forward-referenced annotations": (
        "from __future__ import annotations\n\ndef identity(x: Later) -> Later:\n"
        "    return x\n\nclass Later:\n    pass\n"),
    "nonlocal rebinding": (
        "def outer():\n    v = 1\n    def inner():\n        nonlocal v\n        v = 2\n"
        "        return v\n    return inner()\n"),
    "walrus target binding in the enclosing scope": (
        "xs = [1, 2, 3]\nres = [y for x in xs if (y := x * 2) > 2]\nprint(y, res)\n"),
    "module-qualified dataclasses.replace": (
        "from dsceos_validation import ClusterConfig, make_fixed_local_graph\nimport dataclasses\nfrom graph_config import authoritative_cluster_kwargs, AUTHORITATIVE_COMMUNICATION_RADIUS as R\nlayout = None\ncc = ClusterConfig(**authoritative_cluster_kwargs())\ncc2 = dataclasses.replace(cc, communication_radius=R)\nW = make_fixed_local_graph(layout, cc2.communication_radius, cc2.neighbour_count)\n"),
    "ClusterConfig imported under an alias": (
        "from dsceos_validation import ClusterConfig as CC, make_fixed_local_graph\nfrom graph_config import authoritative_cluster_kwargs\nlayout = None\ncc = CC(**authoritative_cluster_kwargs())\nW = make_fixed_local_graph(layout, cc.communication_radius, cc.neighbour_count)\n"),
    "dataclasses.replace imported under an alias": (
        "from dsceos_validation import ClusterConfig, make_fixed_local_graph\nfrom dataclasses import replace as repl\nfrom graph_config import authoritative_cluster_kwargs, AUTHORITATIVE_COMMUNICATION_RADIUS as R\nlayout = None\ncc = ClusterConfig(**authoritative_cluster_kwargs())\ncc2 = repl(cc, communication_radius=R)\nW = make_fixed_local_graph(layout, cc2.communication_radius, cc2.neighbour_count)\n"),
    "SimulationConfig imported under an alias": (
        "from dsceos_validation import ClusterConfig, SimulationConfig as SC\nfrom graph_config import authoritative_cluster_kwargs\ncc = ClusterConfig(**authoritative_cluster_kwargs())\ncfg = SC(cluster=cc)\n"),
    "method resolves to the module binding, not a class attribute": (
        "from dsceos_validation import ClusterConfig, make_fixed_local_graph\n"
        "from graph_config import authoritative_cluster_kwargs\n"
        "cc = ClusterConfig(**authoritative_cluster_kwargs())\n"
        "layout = None\n"
        "class FakeCluster:\n    communication_radius = 0.2\n    neighbour_count = 1\n"
        "class K:\n    cc = FakeCluster()\n"
        "    def m(self):\n"
        "        return make_fixed_local_graph(layout, cc.communication_radius, cc.neighbour_count)\n"),
    "keyword-only parameter proven at every call site": (
        "from dsceos_validation import ClusterConfig, make_fixed_local_graph\n"
        "from graph_config import authoritative_cluster_kwargs\n"
        "cc = ClusterConfig(**authoritative_cluster_kwargs())\n"
        "layout = None\n"
        "def use(*, cc):\n    return make_fixed_local_graph(layout, cc.communication_radius, cc.neighbour_count)\n"
        "use(cc=cc)\n"),
    "default parameter capturing a proven cluster": (
        "from dsceos_validation import ClusterConfig, make_fixed_local_graph\n"
        "from graph_config import authoritative_cluster_kwargs\n"
        "cc = ClusterConfig(**authoritative_cluster_kwargs())\n"
        "layout = None\n"
        "def use(cc=cc):\n    return make_fixed_local_graph(layout, cc.communication_radius, cc.neighbour_count)\n"
        "use()\n"),
    "match capture patterns": (
        "def f(v):\n    match v:\n        case [a, b]:\n            return a + b\n"
        "        case {'k': c}:\n            return c\n        case other:\n            return other\n"),
}


def test_name_guard_accepts_valid_python():
    """Negative controls: valid Python that a past version of the analyser rejected must pass, so
    the guard cannot block a legitimate future refactor (source-integrity policy check 2)."""
    wrong = {label: analyze_names_source(src) for label, src in _VALID_CODE.items()
             if analyze_names_source(src)}
    assert not wrong, f"false positives on valid code: {wrong}"


def test_provenance_guard_runtime_budget():
    """Runtime regression limit. A source-integrity defect made the provenance analysis quadratic-to-cubic (a
    single file took minutes and the stage effectively hung). The analysis is now index-based; this
    test fails if it ever regresses past a generous budget."""
    import time
    budget_s = 60.0
    t0 = time.time()
    for rel in released_python_paths():
        provenance_problems(rel)
    elapsed = time.time() - t0
    assert elapsed < budget_s, (
        f"provenance analysis took {elapsed:.1f}s over {len(released_python_paths())} files "
        f"(budget {budget_s:.0f}s) - likely an algorithmic regression")
    print(f"    (provenance analysis: {elapsed:.1f}s for {len(released_python_paths())} files)")


# Snippets whose EFFECT can be observed at runtime: each re-binds a proven cluster name to a
# look-alike and then builds a graph from it. The runtime test below executes them against stubs and
# records what the graph builder actually received, so the guard's finding is tied to real
# behaviour rather than to an AST pattern alone (source-integrity audit item 3).
#
# `except ... as cc` is deliberately NOT in this list: Python deletes the alias at the end of the
# handler, so that form raises NameError instead of silently supplying wrong parameters. It remains
# in the static corpus (it is still a binder that must revoke cluster provenance) and the NAME guard
# is what catches its runtime consequence.
_RUNTIME_REBIND_CASES = {
    "assignment": "cc = _Fake()\n",
    "conditional assignment": "if True:\n    cc = _Fake()\n",
    "with ... as": "with _fake_ctx() as cc:\n    pass\n",
    "walrus": "if (cc := _Fake()):\n    pass\n",
    "match capture": "match _Fake():\n    case cc:\n        pass\n",
    "class shadowing": "class cc:\n    communication_radius = 0.2\n    neighbour_count = 1\n",
}


def test_rebinding_bypasses_really_change_the_effective_graph_parameters():
    """Execute each re-binding snippet against stubs and confirm that (a) the graph builder really
    receives the non-authoritative (0.2, 1) pair instead of the authoritative one, and (b) the static
    guard flags exactly those snippets. This ties the guard to observable runtime behaviour."""
    authoritative = (0.45, 4)
    recorded = {}

    class _Fake:
        communication_radius = 0.2
        neighbour_count = 1

    class _FakeCtx:
        def __enter__(self):
            return _Fake()

        def __exit__(self, *a):
            return False

    class _RealCluster:
        communication_radius, neighbour_count = authoritative

    for label, rebind in _RUNTIME_REBIND_CASES.items():
        seen = {}

        def _builder(_layout, radius, count, _seen=seen):
            _seen["args"] = (radius, count)
            return "graph"

        src = ("cc = _RealCluster()\n" + rebind
               + "W = _builder(None, cc.communication_radius, cc.neighbour_count)\n")
        env = {"_Fake": _Fake, "_fake_ctx": _FakeCtx, "_RealCluster": _RealCluster,
               "_builder": _builder}
        exec(compile(src, f"<runtime:{label}>", "exec"), env)          # noqa: S102 - test sandbox
        recorded[label] = seen.get("args")

    wrong = {k: v for k, v in recorded.items() if v == authoritative}
    assert not wrong, f"these snippets did NOT actually change the parameters: {wrong}"

    # and the static guard must flag the same constructions
    header = ("from dsceos_validation import ClusterConfig, make_fixed_local_graph\n"
              "from graph_config import authoritative_cluster_kwargs\n"
              "cc = ClusterConfig(**authoritative_cluster_kwargs())\n")
    tail = "W = make_fixed_local_graph(layout, cc.communication_radius, cc.neighbour_count)\n"
    missed = [label for label, rebind in _RUNTIME_REBIND_CASES.items()
              if not provenance_problems_source(header + rebind + tail)]
    assert not missed, f"static guard misses runtime-confirmed bypasses: {missed}"
    print(f"    (runtime check: {len(recorded)} re-binding forms all reached the builder with "
          f"non-authoritative parameters, and all are flagged statically)")


def test_guards_do_not_fire_on_clean_reference_code():
    """Negative control: correct code must produce no findings (guards must not be trivially true)."""
    clean = """
from dsceos_validation import ClusterConfig, make_fixed_local_graph
from graph_config import (AUTHORITATIVE_COMMUNICATION_RADIUS as _R045,
                          AUTHORITATIVE_NEIGHBOUR_COUNT as _K4,
                          AUTHORITATIVE_LAYOUT_SPREAD as _LS,
                          AUTHORITATIVE_SEED as _SEED)

def build(layout):
    cc = ClusterConfig(seed=_SEED, communication_radius=_R045,
                       neighbour_count=_K4, layout_spread=_LS)
    return make_fixed_local_graph(layout, cc.communication_radius, cc.neighbour_count)
"""
    assert not analyze_names_source(clean), analyze_names_source(clean)
    assert not provenance_problems_source(clean), provenance_problems_source(clean)


if __name__ == "__main__":
    import sys
    import traceback

    ok = True
    for fn in (test_released_paths_have_no_name_errors,
               test_pyflakes_cross_check,
               test_pyflakes_agrees_with_builtin_guard_on_the_bypass_corpus,
               test_released_paths_take_graph_params_from_graph_config,
               test_name_guard_catches_known_bypasses,
               test_provenance_guard_catches_known_bypasses,
               test_name_guard_accepts_valid_python,
               test_provenance_guard_runtime_budget,
               test_rebinding_bypasses_really_change_the_effective_graph_parameters,
               test_guards_do_not_fire_on_clean_reference_code):
        try:
            fn()
            print(f"PASS: {fn.__name__}")
        except AssertionError as e:
            ok = False
            print(f"FAIL: {fn.__name__}: {e}")
        except Exception:
            ok = False
            print(f"ERROR: {fn.__name__}")
            traceback.print_exc()
    sys.exit(0 if ok else 1)
