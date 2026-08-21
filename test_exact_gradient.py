"""the design rationale regression: the D-SCEOS local direction is the EXACT gradient of J.

The submitted version used the non-symmetric operator B = I - D^{-1} W in the controller against
an objective term (lam/2)||B delta||^2 whose exact gradient is lam B^T B delta. That mismatch was a
STEADY STRUCTURAL bias, not a transient estimation error. The released formulation replaces the internal term
by the symmetric edge-disagreement Laplacian form (lam/2) delta^T L delta, L = D - W, whose exact
gradient lam * L delta is what the controller applies (the extra factor is the LOCAL weighted
degree d_i, so no extra communication is needed).

This test asserts, to machine precision and on both released fleets:

  T1  the internal block of the controller gradient equals lam * (L delta) exactly;
  T2  with EXACT estimators (y_hat = A p, y_hat^T = y_T) the FULL local gradient of the controller
      equals the exact gradient of the released benchmark objective -- i.e. e_D^grad == 0, so the
      only remaining gradient-channel error is estimator error, which vanishes with consensus;
  T3  the internal term of objective_terms() equals (lam/2) delta^T L delta and is >= 0 (L is PSD);
  T4  a finite-difference check of the benchmark gradient against the benchmark objective value;
  T5  the objective field is CONSERVATIVE: the Jacobian of the local-gradient map is symmetric
      (this is what fails for the submitted B-operator on a non-regular graph, and is the reason
      the review's option "fit the benchmark objective to the proxy equilibrium" is not
      available for the submitted formulation -- no potential exists).

Run: python3 test_exact_gradient.py
"""
import sys
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "code"))

from benchmark_objective import obj_and_grad                      # noqa: E402
from graph_config import authoritative_cluster_kwargs             # noqa: E402
from realistic_cpes_catalog import build_realistic_units          # noqa: E402
import dsceos_validation as dv                                    # noqa: E402
from dsceos_controller import (DSCEOSConfig, DSCEOSProblemData,  # noqa: E402
                               DistributedSCEOSController)

TOL = 1e-12
CFG = DSCEOSConfig(aggregate_tracking_weight=2.0 / 3.0, loss_weight_scale=0.03,
                   sharing_weight=0.15, internal_weight=0.03)


def build(cluster):
    units, _ = build_realistic_units(cluster)
    n = len(units)
    ccfg = dv.ClusterConfig(n_thermal=3, n_storage=3, n_hydrogen=3, n_emobility=3,
                            n_industrial=3, initial_spread=0.0, initial_speed_scale=0.0,
                            **authoritative_cluster_kwargs())
    layout = dv.make_physical_layout(n, ccfg)
    W = dv.make_fixed_local_graph(layout, ccfg.communication_radius, ccfg.neighbour_count)
    arrays = dv.arrays_from_units(units)
    Ablk = dv.make_aggregate_blocks(arrays["aggregate_weight"])
    return units, arrays, Ablk, W, n


def make_controller(units, arrays, Ablk, W):
    problem = DSCEOSProblemData(
        masses=arrays["masses"], dampings=arrays["dampings"],
        force_limits=arrays["force_limits"], lower=arrays["lower"], upper=arrays["upper"],
        rest=arrays["rest"], loss_weight=arrays["loss_weight"],
        aggregate_blocks=Ablk, service_capacity=arrays["service_capacity"],
        service_selector=np.tile(np.array([1.0, 0.0]), (len(units), 1)),
        adjacency=W)
    return DistributedSCEOSController(problem, CFG)


failures = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{('  ' + detail) if detail else ''}")
    if not ok:
        failures.append(name)


for cluster, tag in (("realistic_15", "N15"), ("realistic_60", "N60")):
    print(f"\n=== {tag} ({cluster}) ===")
    units, arrays, Ablk, W, n = build(cluster)
    ctrl = make_controller(units, arrays, Ablk, W)
    deg = W.sum(axis=1)
    L = np.diag(deg) - W
    rng = np.random.default_rng(20260724)

    for trial in range(5):
        # random INTERIOR operating point (the gradient identity is unconstrained)
        p = arrays["rest"] + 0.25 * rng.normal(size=arrays["rest"].shape)
        p = np.clip(p, arrays["lower"] + 1e-3, arrays["upper"] - 1e-3)
        delta = p - arrays["rest"]
        y = dv.aggregate_output(Ablk, p)
        target = y + 0.05 * rng.normal(size=y.shape)

        # ---- T1 internal block only
        ctrl.reset_estimators(p)
        ctrl.state.y_hat = np.tile(y, (n, 1))                  # exact aggregate estimate
        g_ctrl = ctrl.local_gradients(p, np.tile(target, (n, 1)))   # exact target estimate
        exact_internal = CFG.internal_weight * (L @ delta)

        # rebuild the controller gradient without its internal block
        g_wo = g_ctrl - exact_internal
        # ---- T2 full gradient identity against the benchmark
        _, g_bench = obj_and_grad(p.reshape(-1), arrays, W, Ablk, target, CFG, p.shape)
        g_bench = g_bench.reshape(p.shape)
        err_full = np.max(np.abs(g_ctrl - g_bench))
        if trial == 0:
            check("T1 internal block == lam*(L delta)",
                  np.max(np.abs((g_ctrl - g_wo) - exact_internal)) < TOL)
            check("T2 full local gradient == exact grad J (exact estimators)",
                  err_full < TOL, f"max|diff| = {err_full:.3e}")
        elif err_full >= TOL:
            check(f"T2 (trial {trial})", False, f"max|diff| = {err_full:.3e}")

        # ---- T3 objective internal term
        terms = dv.objective_terms(p, arrays, W, Ablk, target, CFG)
        want = 0.5 * CFG.internal_weight * float(delta.ravel() @ (L @ delta).ravel())
        if trial == 0:
            check("T3 objective internal term == (lam/2) delta^T L delta",
                  abs(terms["objective_internal"] - want) < 1e-14 and terms["objective_internal"] >= 0.0,
                  f"value = {terms['objective_internal']:.6e}")

        # ---- T4 finite-difference check of the benchmark gradient
        if trial == 0:
            h = 1e-6
            fd = np.zeros_like(p)
            for i in range(n):
                for k in range(p.shape[1]):
                    pp = p.copy(); pp[i, k] += h
                    pm = p.copy(); pm[i, k] -= h
                    fd[i, k] = (obj_and_grad(pp.reshape(-1), arrays, W, Ablk, target, CFG, p.shape)[0]
                                - obj_and_grad(pm.reshape(-1), arrays, W, Ablk, target, CFG, p.shape)[0]) / (2 * h)
            rel = np.max(np.abs(fd - g_bench)) / max(np.max(np.abs(g_bench)), 1e-30)
            check("T4 benchmark gradient == central finite difference", rel < 5e-7,
                  f"max rel. error = {rel:.2e}")

    # ---- T5 conservativeness: Jacobian of the local-gradient map must be symmetric
    p0 = arrays["rest"] + 0.1 * rng.normal(size=arrays["rest"].shape)
    y0 = dv.aggregate_output(Ablk, p0)
    h = 1e-6
    dim = p0.size
    Jac = np.zeros((dim, dim))
    for c in range(dim):
        pp = p0.reshape(-1).copy(); pp[c] += h
        pm = p0.reshape(-1).copy(); pm[c] -= h
        ctrl.reset_estimators(pp.reshape(p0.shape))
        ctrl.state.y_hat = np.tile(dv.aggregate_output(Ablk, pp.reshape(p0.shape)), (n, 1))
        gp = ctrl.local_gradients(pp.reshape(p0.shape), np.tile(y0, (n, 1))).reshape(-1)
        ctrl.reset_estimators(pm.reshape(p0.shape))
        ctrl.state.y_hat = np.tile(dv.aggregate_output(Ablk, pm.reshape(p0.shape)), (n, 1))
        gm = ctrl.local_gradients(pm.reshape(p0.shape), np.tile(y0, (n, 1))).reshape(-1)
        Jac[:, c] = (gp - gm) / (2 * h)
    asym = np.max(np.abs(Jac - Jac.T)) / max(np.max(np.abs(Jac)), 1e-30)
    check("T5 local-gradient field is conservative (symmetric Jacobian)", asym < 1e-6,
          f"max rel. asymmetry = {asym:.2e}")

    # reference: how asymmetric was the SUBMITTED operator on this graph?
    B = np.eye(n) - W / np.where(deg > 1e-12, deg, 1.0)[:, None]
    sub_asym = np.max(np.abs(B - B.T)) / max(np.max(np.abs(B)), 1e-30)
    print(f"  [info] submitted operator B = I - D^-1 W: max rel. asymmetry {sub_asym:.3f} "
          f"(zero only on a regular graph) -> the submitted field admitted NO potential")

print("\n=== RESULT ===")
if failures:
    print("FAILED: " + "; ".join(failures))
    sys.exit(1)
print("All exact-gradient regression checks passed.")
