"""
distributed_gradient_tracking_controller.py
===========================================

Convergent distributed gradient-tracking (DIGing-type) comparator.

Rationale
---------
The projected-gradient comparator takes a single projected step per sampling instant, so its
iterate carries a persistent optimisation bias: it does not converge to the minimiser of the
shared objective even in the limit of a static request. That makes it a weak stand-in for
"a properly convergent distributed optimiser". This module supplies that stronger reference.

It runs the standard gradient-tracking recursion on the same fixed communication graph,

    z_{k+1} = P_X[ W z_k - alpha * s_k ]
    s_{k+1} = W s_k + grad f(z_{k+1}) - grad f(z_k)

with W the Metropolis-weight doubly stochastic mixing matrix built from the same adjacency, P_X the
projection onto the local operating box, and grad f the SAME objective gradient the other
controllers use (aggregate tracking through the local estimator, local operating loss,
capacity-normalised utilisation sharing and the graph-local internal term). Under a connected graph
and a sufficiently small step, this recursion drives every agent to the exact minimiser, and the
gradient tracker s removes the steady-state bias of the plain projected step.

Design matching
---------------
Everything outside the inner optimiser is identical to the projected-gradient comparator: the same
reference-tracking gain and velocity damping map the iterate to a nominal force, the same actuator
truncation applies, and the same post-hoc HOCBF filter is used by the harness. The comparison
therefore isolates the inner optimiser.

Communication budget
--------------------
Gradient tracking exchanges BOTH the iterate z and the tracker s with each neighbour, i.e. 2d
scalars per neighbour per step, on top of the m-scalar aggregate estimate. That is strictly more
than the projected-gradient comparator uses. We deliberately grant it the larger budget: the point
of this comparator is to test whether the proposed controller's advantage survives against a
convergent optimiser, so any budget asymmetry is set in the comparator's favour and is reported.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dsceos_controller import DSCEOSConfig, DSCEOSProblemData, row_norm, truncate_rows_by_norm

Array = np.ndarray


@dataclass(frozen=True)
class GTDiagnostics:
    estimate_spread: float
    reference_spread: float
    mean_gradient_norm: float
    force_margin: float
    tracker_norm: float


def metropolis_weights(adjacency: Array) -> Array:
    """Doubly stochastic mixing matrix from a symmetric adjacency (Metropolis rule)."""
    A = (np.asarray(adjacency, dtype=float) > 0.0).astype(float)
    np.fill_diagonal(A, 0.0)
    deg = A.sum(axis=1)
    n = A.shape[0]
    W = np.zeros_like(A)
    for i in range(n):
        for j in range(n):
            if i != j and A[i, j] > 0.0:
                W[i, j] = 1.0 / (1.0 + max(deg[i], deg[j]))
    np.fill_diagonal(W, 1.0 - W.sum(axis=1))
    return W


class DistributedGradientTrackingController:
    """Gradient-tracking distributed optimiser driving the same second-order units."""

    def __init__(
        self,
        problem: DSCEOSProblemData,
        config: DSCEOSConfig | None = None,
        step_size: float = 0.06,
        reference_tracking_gain: float = 0.55,
        velocity_damping: float = 1.50,
        consensus_gain: float | None = None,
        optimizer_substeps: int = 1,
    ) -> None:
        self.problem = problem
        self.config = config or DSCEOSConfig()
        self.step_size = float(step_size)
        self.reference_tracking_gain = float(reference_tracking_gain)
        self.velocity_damping = float(velocity_damping)
        self.optimizer_substeps = max(1, int(optimizer_substeps))
        self.consensus_gain = (float(consensus_gain) if consensus_gain is not None
                               else float(self.config.aggregate_consensus_gain))
        self.W = metropolis_weights(problem.adjacency)
        self.z: Array | None = None
        self.s: Array | None = None
        self.y_hat: Array | None = None
        self._prev_local: Array | None = None

    # ---- aggregate estimate (identical mechanism to the other comparators) ----
    def reset(self, p0: Array) -> None:
        pr = self.problem
        self.z = np.minimum(np.maximum(np.asarray(p0, dtype=float).copy(), pr.lower), pr.upper)
        local = np.einsum("nmd,nd->nm", pr.aggregate_blocks, self.z)
        # Match the projected-gradient comparator's estimator initialisation exactly, so that the two
        # comparators differ only in their inner optimiser. Both start from the post-first-step form
        # y_hat = A_i p_i(0) with a matching stored previous output, i.e. the first update injects a
        # zero increment. D-SCEOS instead starts at zero with a zero stored output, which reaches the
        # same state after one update; the conventions coincide from step 1 onward and are recorded
        # here so the difference at step 0 is not mistaken for an algorithmic asymmetry.
        self.y_hat = local.copy()
        self._prev_local = local.copy()
        self.s = None

    def _aggregate_estimate_update(self) -> None:
        assert self.z is not None
        pr = self.problem
        local = np.einsum("nmd,nd->nm", pr.aggregate_blocks, self.z)
        A = pr.adjacency
        lap = A @ self.y_hat - A.sum(axis=1)[:, None] * self.y_hat
        self.y_hat = self.y_hat + (local - self._prev_local) + self.consensus_gain * lap
        self._prev_local = local

    # ---- objective gradient: the SAME four blocks the other controllers use ----
    def _gradient(self, z: Array, target_estimate: Array) -> Array:
        pr, cfg = self.problem, self.config
        n = z.shape[0]
        grad = cfg.loss_weight_scale * pr.loss_weight * (z - pr.rest)

        resid = self.y_hat - np.asarray(target_estimate, dtype=float)
        # The aggregate convention is y = mean_i A_i p_i, so d y / d p_i carries a factor 1/N which
        # exactly cancels the explicit N of the size-consistent tracking term. The per-agent tracking
        # gradient is therefore w_y * A_i^T (y - y_T) with NO further N, matching the D-SCEOS and
        # projected-gradient implementations. An earlier version of this file multiplied by n here,
        # which made this comparator optimise a different objective from the one it documents.
        grad = grad + cfg.aggregate_tracking_weight * np.einsum(
            "nmd,nm->nd", pr.aggregate_blocks, resid)

        rho = np.sum(pr.service_selector * z, axis=1) / pr.service_capacity
        A = pr.adjacency
        lap_rho = A.sum(axis=1) * rho - A @ rho
        grad = grad + cfg.sharing_weight * (lap_rho / pr.service_capacity)[:, None] * pr.service_selector

        delta = z - pr.rest
        lap_delta = A.sum(axis=1)[:, None] * delta - A @ delta
        grad = grad + cfg.internal_weight * lap_delta
        return grad

    def control(self, p: Array, v: Array, target_estimate: Array) -> tuple[Array, GTDiagnostics]:
        p = np.asarray(p, dtype=float)
        v = np.asarray(v, dtype=float)
        target_estimate = np.asarray(target_estimate, dtype=float)
        pr = self.problem
        if self.z is None or self.z.shape != p.shape:
            self.reset(p)
        assert self.z is not None

        self.z = np.minimum(np.maximum(self.z, pr.lower), pr.upper)
        grad_norms = []
        for _ in range(self.optimizer_substeps):
            self._aggregate_estimate_update()
            g_old = self._gradient(self.z, target_estimate)
            if self.s is None:
                self.s = g_old.copy()          # standard DIGing initialisation s_0 = grad f(z_0)
            grad_norms.append(row_norm(g_old))
            z_new = self.W @ self.z - self.step_size * self.s
            z_new = np.minimum(np.maximum(z_new, pr.lower), pr.upper)
            g_new = self._gradient(z_new, target_estimate)
            self.s = self.W @ self.s + (g_new - g_old)
            self.z = z_new

        a_nom = self.reference_tracking_gain * (self.z - p) - self.velocity_damping * v
        u_nom = pr.masses[:, None] * a_nom + pr.dampings[:, None] * v
        u_nom = truncate_rows_by_norm(u_nom, pr.force_limits)

        est_spread = 0.0
        if self.y_hat is not None and self.y_hat.size:
            est_spread = float(np.max(row_norm(self.y_hat - np.mean(self.y_hat, axis=0, keepdims=True))))
        return u_nom, GTDiagnostics(
            estimate_spread=est_spread,
            reference_spread=float(np.mean(row_norm(self.z - p))),
            mean_gradient_norm=float(np.mean(grad_norms[-1])) if grad_norms else 0.0,
            force_margin=float(np.min(pr.force_limits - row_norm(u_nom))),
            tracker_norm=float(np.mean(row_norm(self.s))) if self.s is not None else 0.0,
        )
