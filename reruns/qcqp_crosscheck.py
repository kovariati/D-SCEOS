#!/usr/bin/env python3
"""
Block A -- Independent QCQP / fallback solver cross-check.

The D-SCEOS per-agent step solves the local CLF/HOCBF-QCQP

    min_{u,s}  0.5||u - u_nom||^2 + w_s s^2
    s.t.       lb <= u <= ub            (HOCBF box, hard)
               ||u|| <= radius          (actuator ball, hard)
               coeff_u^T u + rhs <= s   (CLF row, soft; s >= 0)

with SLSQP, falling back to the Dykstra box-ball projection on solver failure.
This module re-solves the SAME problem at every agent-step with an INDEPENDENT
interior-point conic solver (CLARABEL via cvxpy) and reports, per configuration:

  * solver status distribution and mean iteration count of the independent solve,
  * KKT/feasibility residual of the independent solution,
  * per-agent fallback rate of the production controller,
  * ||u_exact - u_applied|| (how far the applied force is from the exact QCQP optimum),
  * DeltaJ_T from a closed-loop re-run that uses CLARABEL as the in-loop QCQP solver.

No Monte Carlo. numpy/scipy + cvxpy(CLARABEL).
"""
from __future__ import annotations
import sys, os, math, json, time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "code"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# cvxpy/CLARABEL is only needed by the functions that actually solve the conic program.
# It is imported lazily so that other scripts which merely reuse the configuration builders
# from this package (e.g. graph_decoupling.py, headroom_normalisation.py) run without it.
cp = None


def _require_cvxpy():
    global cp
    if cp is None:
        import cvxpy as _cp
        cp = _cp
    return cp

import dsceos_controller as dc
import dsceos_validation as dv


# ---------------------------------------------------------------------------
# Independent QCQP solve (CLARABEL). Exactly the problem in _solve_local_clf_hocbf_qp.
# ---------------------------------------------------------------------------
_PROB_CACHE = {}


def _build_parametrized(d, w_s):
    _require_cvxpy()
    """DPP-parametrized CLARABEL problem, cached per (d, w_s) for fast repeated solves."""
    u = cp.Variable(d)
    s = cp.Variable(nonneg=True)
    p_unom = cp.Parameter(d)
    p_lb = cp.Parameter(d)
    p_ub = cp.Parameter(d)
    p_rad = cp.Parameter(nonneg=True)
    p_coeff = cp.Parameter(d)
    p_rhs = cp.Parameter()
    obj = cp.Minimize(0.5 * cp.sum_squares(u - p_unom) + w_s * cp.square(s))
    cons = [u >= p_lb, u <= p_ub, cp.norm(u, 2) <= p_rad, p_coeff @ u + p_rhs <= s]
    prob = cp.Problem(obj, cons)
    return dict(prob=prob, u=u, s=s, p_unom=p_unom, p_lb=p_lb, p_ub=p_ub,
                p_rad=p_rad, p_coeff=p_coeff, p_rhs=p_rhs)


def solve_qcqp_clarabel(u_nom, lb, ub, radius, coeff_u, rhs, w_s):
    _require_cvxpy()
    d = int(len(u_nom))
    key = (d, float(w_s))
    P = _PROB_CACHE.get(key)
    if P is None:
        P = _build_parametrized(d, float(w_s))
        _PROB_CACHE[key] = P
    P["p_unom"].value = np.asarray(u_nom, float)
    P["p_lb"].value = np.asarray(lb, float)
    P["p_ub"].value = np.asarray(ub, float)
    P["p_rad"].value = float(radius)
    P["p_coeff"].value = np.asarray(coeff_u, float)
    P["p_rhs"].value = float(rhs)
    try:
        P["prob"].solve(solver=cp.CLARABEL, verbose=False)
    except Exception:
        return None, None, "error", None
    stats = P["prob"].solver_stats
    iters = getattr(stats, "num_iters", None) if stats is not None else None
    uval = P["u"].value
    sval = P["s"].value
    return (np.asarray(uval, dtype=float) if uval is not None else None,
            (float(sval) if sval is not None else None),
            P["prob"].status, iters)


def kkt_feasibility_residual(u, lb, ub, radius):
    """Max hard-constraint feasibility residual of a force vector (0 = feasible)."""
    box = max(float(np.max(lb - u)), float(np.max(u - ub)), 0.0)
    ball = max(float(np.linalg.norm(u) - radius), 0.0)
    return max(box, ball)


# ---------------------------------------------------------------------------
# Capture wrapper: log every QCQP input + the applied output + fallback flag.
# ---------------------------------------------------------------------------
class QCQPCapture:
    def __init__(self, controller):
        self.controller = controller
        self.records = []
        self._orig = controller._solve_local_clf_hocbf_qp

    def __enter__(self):
        ctrl = self.controller
        cfg = ctrl.config
        orig = self._orig

        def wrapped(u_nom, lb, ub, radius, e, vel, mass, damping):
            u, slack, success, residual, V = orig(u_nom, lb, ub, radius, e, vel, mass, damping)
            # recompute the CLF terms exactly as the solver did
            Vv, coeff_u, drift = ctrl._local_clf_terms(e, vel, mass, damping)
            rhs = drift + float(cfg.clf_rate) * Vv
            self.records.append(dict(
                u_nom=np.asarray(u_nom, float).copy(),
                lb=np.asarray(lb, float).copy(), ub=np.asarray(ub, float).copy(),
                radius=float(radius), coeff_u=np.asarray(coeff_u, float).copy(),
                rhs=float(rhs), u_applied=np.asarray(u, float).copy(),
                slack=float(slack), success=bool(success), residual=float(residual)))
            return u, slack, success, residual, V

        ctrl._solve_local_clf_hocbf_qp = wrapped
        return self

    def __exit__(self, *a):
        self.controller._solve_local_clf_hocbf_qp = self._orig


# ---------------------------------------------------------------------------
# In-loop CLARABEL solver (for the closed-loop DeltaJ_T re-run).
# ---------------------------------------------------------------------------
def make_clarabel_solver(controller):
    cfg = controller.config
    def clarabel_solve(u_nom, lb, ub, radius, e, vel, mass, damping):
        V, coeff_u, drift = controller._local_clf_terms(e, vel, mass, damping)
        rhs = drift + float(cfg.clf_rate) * V
        u, s, status, iters = solve_qcqp_clarabel(u_nom, lb, ub, radius, coeff_u, rhs, cfg.clf_slack_weight)
        if u is None or not np.all(np.isfinite(u)):
            # extremely rare; fall back to the same deterministic projection
            u0, r0 = dc.project_to_box_ball(u_nom, np.asarray(lb, float), np.asarray(ub, float), radius)
            s0 = max(0.0, float(np.dot(coeff_u, u0) + rhs))
            return u0, s0, False, r0, V
        slack = float(max(0.0, s if s is not None else 0.0))
        box_res = max(float(np.max(lb - u)), float(np.max(u - ub)), 0.0)
        ball_res = max(float(np.linalg.norm(u) - radius), 0.0)
        clf_res = max(0.0, float(np.dot(coeff_u, u) + rhs - slack))
        return u, slack, True, max(box_res, ball_res, clf_res), V
    return clarabel_solve


# ---------------------------------------------------------------------------
# Per-step open-loop comparison over a captured run.
# ---------------------------------------------------------------------------
def crosscheck_records(records, w_s, max_solved_sample=4000, seed=0):
    """Cross-check every fallback step + a capped random sample of the solver-success steps.
    All fallback steps are always checked (they are the review-relevant ones); the success
    steps are sampled only to bound du_max_on_solved without O(n) conic solves at large N."""
    n = len(records)
    fallback = np.array([not r["success"] for r in records])
    fb_idx = np.where(fallback)[0]
    ok_idx = np.where(~fallback)[0]
    rng = np.random.default_rng(seed)
    if len(ok_idx) > max_solved_sample:
        ok_sample = rng.choice(ok_idx, size=max_solved_sample, replace=False)
    else:
        ok_sample = ok_idx
    check_idx = np.concatenate([fb_idx, ok_sample])
    du = {}
    feas_exact = []
    solved = []
    iters = []
    for k in check_idx:
        r = records[int(k)]
        u_ex, s_ex, status, it = solve_qcqp_clarabel(
            r["u_nom"], r["lb"], r["ub"], r["radius"], r["coeff_u"], r["rhs"], w_s)
        if u_ex is None:
            du[int(k)] = math.nan; continue
        solved.append(status in ("optimal", "optimal_inaccurate"))
        du[int(k)] = float(np.linalg.norm(u_ex - r["u_applied"]))
        feas_exact.append(kkt_feasibility_residual(u_ex, r["lb"], r["ub"], r["radius"]))
        if it is not None: iters.append(int(it))
    du_fb = np.array([du[int(k)] for k in fb_idx]) if len(fb_idx) else np.array([])
    du_ok = np.array([du[int(k)] for k in ok_sample]) if len(ok_sample) else np.array([])
    du_all = np.array([v for v in du.values()])
    return dict(
        n_steps=n,
        n_fallback=int(fallback.sum()),
        fallback_rate=float(fallback.mean()),
        n_checked=int(len(check_idx)),
        n_solved_sampled=int(len(ok_sample)),
        indep_solved_rate=(float(np.mean(solved)) if solved else math.nan),
        indep_mean_iters=(float(np.mean(iters)) if iters else math.nan),
        indep_feas_resid_max=(float(np.nanmax(feas_exact)) if feas_exact else math.nan),
        indep_feas_resid_mean=(float(np.nanmean(feas_exact)) if feas_exact else math.nan),
        du_max=float(np.nanmax(du_all)) if len(du_all) else 0.0,
        du_mean=float(np.nanmean(du_all)) if len(du_all) else 0.0,
        du_max_on_fallback=(float(np.nanmax(du_fb)) if len(du_fb) else 0.0),
        du_mean_on_fallback=(float(np.nanmean(du_fb)) if len(du_fb) else 0.0),
        du_max_on_solved=(float(np.nanmax(du_ok)) if len(du_ok) else 0.0),
        applied_feas_resid_max=float(np.max([r["residual"] for r in records])),
    )


if __name__ == "__main__":
    print("module: import and call run functions from qcqp_crosscheck_run.py")
