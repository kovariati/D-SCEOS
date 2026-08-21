"""Regression tests for the constraint-handling and graph-validation failure paths.

These cover the paths an external audit found unprotected: an inconsistent HOCBF box, an empty
box-ball intersection, runtime installation of a graph that violates the standing assumptions, and
the HOCBF gain admissibility condition. Each test states the property that must hold, not merely the
current numeric output.
"""
from __future__ import annotations

import numpy as np
import pytest

from dsceos_controller import (DSCEOSConfig, DSCEOSProblemData, DistributedSCEOSController,
                               project_to_box_ball, validate_communication_graph)


def _problem(n=3, d=2, m=2, W=None):
    if W is None:
        W = np.zeros((n, n))
        for i in range(n - 1):
            W[i, i + 1] = W[i + 1, i] = 1.0
    return DSCEOSProblemData(
        aggregate_blocks=np.tile(np.eye(m, d)[None], (n, 1, 1)),
        lower=-np.ones((n, d)), upper=np.ones((n, d)), rest=np.zeros((n, d)),
        loss_weight=np.ones((n, d)), masses=np.ones(n), dampings=np.ones(n),
        force_limits=np.ones(n), service_selector=np.tile([1.0, 0.0], (n, 1)),
        service_capacity=np.ones(n), adjacency=W)


# --------------------------------------------------------------------------------------
# Inconsistent HOCBF box must be reported, never silently repaired by swapping the bounds
# --------------------------------------------------------------------------------------
def test_inconsistent_box_reports_positive_residual_against_original_bounds():
    lb = np.array([1.0, -1.0])
    ub = np.array([-1.0, 1.0])          # lb > ub on coordinate 0 -> empty set there
    u, res, status = project_to_box_ball(np.zeros(2), lb, ub, 2.0, return_status=True)
    true_res = max(float(np.max(lb - u)), float(np.max(u - ub)), 0.0)
    assert res == pytest.approx(true_res), "residual must be measured against the ORIGINAL bounds"
    assert res > 0.0, "an empty box must not be reported as feasible"
    assert status["inconsistent_box"] is True
    assert status["feasible"] is False


def test_consistent_box_ball_projection_is_feasible():
    u, res, status = project_to_box_ball(np.array([5.0, 0.0]), -np.ones(2), np.ones(2), 3.0,
                                         return_status=True)
    assert res == pytest.approx(0.0, abs=1e-9)
    assert status["inconsistent_box"] is False and status["feasible"] is True


def test_empty_box_ball_intersection_reports_positive_residual():
    # Consistent box far outside a tiny ball: the intersection is empty.
    lb = np.array([2.0, 2.0])
    ub = np.array([3.0, 3.0])
    u, res, status = project_to_box_ball(np.array([2.5, 2.5]), lb, ub, 0.1, return_status=True)
    assert res > 0.0
    assert status["inconsistent_box"] is False


# --------------------------------------------------------------------------------------
# Graph preconditions must hold at construction AND at every runtime topology change
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("bad,reason", [
    (np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]]), "asymmetric"),
    (np.array([[0.0, -1.0, 0.0], [-1.0, 0.0, 1.0], [0.0, 1.0, 0.0]]), "negative weight"),
    (np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]), "disconnected"),
    (np.array([[1.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0]]), "self-loop"),
])
def test_validate_communication_graph_rejects(bad, reason):
    with pytest.raises(ValueError):
        validate_communication_graph(bad, context=reason)


@pytest.mark.parametrize("bad", [
    np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]]),
    np.array([[0.0, -1.0, 0.0], [-1.0, 0.0, 1.0], [0.0, 1.0, 0.0]]),
    np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
])
def test_set_adjacency_enforces_the_same_preconditions(bad):
    ctrl = DistributedSCEOSController(_problem(), DSCEOSConfig())
    with pytest.raises(ValueError):
        ctrl.set_adjacency(bad)


def test_set_adjacency_accepts_a_valid_graph():
    ctrl = DistributedSCEOSController(_problem(), DSCEOSConfig())
    good = np.array([[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]])
    ctrl.set_adjacency(good)
    assert np.allclose(ctrl.problem.adjacency, good)


# --------------------------------------------------------------------------------------
# HOCBF gain admissibility
# --------------------------------------------------------------------------------------
def test_hocbf_gain_double_root_condition_is_enforced():
    # alpha_1^2 >= 4 alpha_0 is the admissibility condition used by the formal statement.
    DSCEOSConfig(hocbf_alpha0=1.0, hocbf_alpha1=2.0)   # boundary case must be accepted
    with pytest.raises(ValueError):
        DSCEOSConfig(hocbf_alpha0=4.0, hocbf_alpha1=1.0)   # 1 < 16 -> not admissible


# --------------------------------------------------------------------------------------
# Comparator gradients must agree with the central objective's gradient (the design rationale)
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("n", [3, 5, 8])
def test_gradient_tracking_matches_projected_gradient_tracking_block(n):
    """The tracking block must not carry a spurious factor N.

    With y = mean_i A_i p_i the derivative w.r.t. p_i carries 1/N, which cancels the explicit N of
    the size-consistent objective. The gradient-tracking comparator must therefore produce the same
    tracking block as the projected-gradient comparator for the SAME state and the SAME aggregate
    estimate, at every N. The estimate is injected directly so that the test isolates the gradient
    and does not depend on the two estimators' initialisation conventions.
    """
    from distributed_gradient_tracking_controller import DistributedGradientTrackingController
    from distributed_projected_gradient_controller import DistributedProjectedGradientController

    W = np.zeros((n, n))
    for i in range(n - 1):
        W[i, i + 1] = W[i + 1, i] = 1.0
    pr = _problem(n=n, W=W)
    cfg = DSCEOSConfig(aggregate_tracking_weight=1.0, loss_weight_scale=0.0,
                       sharing_weight=0.0, internal_weight=0.0)     # isolate the tracking block
    rng = np.random.default_rng(0)
    z = rng.uniform(-0.5, 0.5, size=(n, 2))
    y_hat = rng.uniform(-0.3, 0.3, size=(n, 2))
    tgt = np.zeros(2)

    gt = DistributedGradientTrackingController(pr, cfg)
    gt.reset(z); gt.z = z.copy(); gt.y_hat = y_hat.copy()
    dpg = DistributedProjectedGradientController(pr, cfg)
    dpg.reset(z); dpg.z = z.copy(); dpg.y_hat = y_hat.copy()

    g_gt = gt._gradient(z, tgt)
    g_dpg = dpg._gradient(tgt)
    assert np.allclose(g_gt, g_dpg, rtol=1e-10, atol=1e-12), (
        f"tracking blocks differ at N={n}; a spurious factor N shows up as a ratio of {n}: "
        f"max ratio {np.max(np.abs(g_gt) / np.maximum(np.abs(g_dpg), 1e-15)):.3f}")
