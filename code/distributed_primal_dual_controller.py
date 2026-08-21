"""distributed_primal_dual_controller.py
=======================================

the design rationale asked for at least one stronger distributed comparator --
specifically "a primal-dual or second-order distributed optimizer with a safety
filter" -- evaluated under the SAME communication and safety budget as D-SCEOS.

This module provides a distributed BOX-DUAL (Arrow-Hurwicz) reference generator,
then tracks that reference with the same local PD law and passes the result
through the SAME posthoc HOCBF safety filter used by the other baselines.

Name used by the validation harness:
    distributed_primal_dual_hocbf   (abbreviated DPD-HOCBF)

SCOPE AND HONEST LIMITATIONS
----------------------------
This is a genuine saddle-point (primal-descent / dual-ascent) method, but it is
a LOCAL BOX-DUAL comparator, NOT a coupled economic primal-dual optimizer:
  * it carries per-agent duals ONLY for the local operating-box constraints
    (``mu_lo``, ``mu_hi``), one pair per agent per coordinate;
  * there is NO global/coupled equality or resource-balance dual, and NO
    dual-consensus message between agents;
  * the step sizes are fixed (``primal_step``, ``dual_step``) with no dedicated
    tuning sweep;
  * the reference is lightly clamped to a widened box so it stays finite while
    the HOCBF filter enforces the hard limits.
It is therefore a CONTROLLED DIAGNOSTIC that probes whether a dual treatment of
the binding (box) constraints changes the ordering -- not a claim of a
strongest-available distributed optimizer. The released results show it is
numerically indistinguishable from DPG-HOCBF at N=15 (the box seldom binds) and
worse at N=60 (the un-projected reference overshoots), which is reported honestly
in the manuscript. A coupled economic primal-dual method with a resource-balance
dual and its own tuning remains stated as future comparator work.

DPG-HOCBF takes a single projected-gradient step per sampling instant and handles
the operating box only by Euclidean projection. This controller instead keeps the
box constraints in the Lagrangian with per-coordinate duals ``mu_lo``, ``mu_hi``
and does dual ascent on them:

    z_i      <- z_i - eta_p * grad_z L_i(z, mu)          (primal descent, no hard projection)
    mu_lo    <- [ mu_lo + eta_d * (lower - z) ]_+        (dual ascent, lower box)
    mu_hi    <- [ mu_hi + eta_d * (z - upper) ]_+        (dual ascent, upper box)

where ``L`` is the augmented Lagrangian of the SAME CPES objective used
everywhere in this study.

COMMUNICATION-BUDGET PARITY (the crux of the review's request)
----------------------------------------------------------------
The box duals are LOCAL to each agent and are never transmitted. The controller
runs one local update per sampling instant on the same fixed graph with the same
nine-scalar per-neighbour payload as D-SCEOS -- identical communication budget --
which is exactly the parity the review asked for.

Interpretation:
    DPD-HOCBF = distributed box-dual (Arrow-Hurwicz) reference update
                + PD tracking + posthoc HOCBF safety filter.
Safety is NOT native to this comparator, exactly as for DPG-HOCBF; the harness
applies the common HOCBF filter to the nominal output.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dsceos_controller import DSCEOSConfig, DSCEOSProblemData, row_norm, truncate_rows_by_norm

Array = np.ndarray


@dataclass
class DPDDiagnostics:
    estimate_spread: float
    mean_gradient_norm: float
    reference_spread: float
    force_margin: float
    mean_dual: float


class DistributedPrimalDualController:
    """Peer-to-peer primal-dual (Arrow-Hurwicz) reference generator + PD tracking.

    The controller mirrors ``DistributedProjectedGradientController`` in interface,
    communication pattern and objective, and differs only in the optimizer: it
    maintains, per agent and per coordinate, a pair of duals (``mu_lo``, ``mu_hi``)
    for the local operating-box constraints and performs a saddle-point
    (primal-descent / dual-ascent) update rather than a plain projected-gradient
    step. There is no global/coupled resource-balance dual and no dual-consensus
    message, so this is a LOCAL BOX-DUAL comparator, not a coupled economic
    primal-dual optimizer.
    """

    def __init__(
        self,
        problem: DSCEOSProblemData,
        config: DSCEOSConfig | None = None,
        primal_step: float = 0.06,          # DPD's own fixed primal step (no dedicated sweep); the
                                             # DPG comparator is tuned separately to its best step 0.10
        dual_step: float = 0.20,
        reference_tracking_gain: float = 0.55,
        velocity_damping: float = 1.50,
        consensus_gain: float | None = None,
    ) -> None:
        self.problem = problem
        self.config = config or DSCEOSConfig()
        self.primal_step = float(primal_step)
        self.dual_step = float(dual_step)
        self.reference_tracking_gain = float(reference_tracking_gain)
        self.velocity_damping = float(velocity_damping)
        self.consensus_gain = (
            float(consensus_gain)
            if consensus_gain is not None
            else float(self.config.aggregate_consensus_gain)
        )
        if self.config.adaptive_consensus_gain:
            self.consensus_gain = self._gershgorin_gain(self.problem.adjacency)
        self.z: Array | None = None
        # Per-coordinate box duals (private per-agent state, never transmitted).
        self.mu_lo: Array | None = None
        self.mu_hi: Array | None = None
        self.y_hat: Array | None = None
        self.previous_local_output: Array | None = None

    @staticmethod
    def _gershgorin_gain(W: Array, gamma: float = 0.95) -> float:
        deg = np.sum(np.asarray(W, dtype=float), axis=1)
        dmax = float(np.max(deg)) if deg.size else 0.0
        if dmax <= 1.0e-12:
            return 0.0
        return float(gamma / dmax)

    def reset(self, p0: Array) -> None:
        p0 = np.asarray(p0, dtype=float)
        self.z = np.minimum(np.maximum(p0.copy(), self.problem.lower), self.problem.upper)
        self.mu_lo = np.zeros_like(self.z)
        self.mu_hi = np.zeros_like(self.z)
        local = np.einsum("nmd,nd->nm", self.problem.aggregate_blocks, self.z)
        self.y_hat = local.copy()
        self.previous_local_output = local.copy()

    def _laplacian_smoothing(self, x: Array) -> Array:
        W = self.problem.adjacency
        deg = np.sum(W, axis=1)
        return W @ x - deg[:, None] * x

    def _aggregate_estimate_update(self) -> None:
        assert self.z is not None and self.y_hat is not None and self.previous_local_output is not None
        local = np.einsum("nmd,nd->nm", self.problem.aggregate_blocks, self.z)
        delta = local - self.previous_local_output
        consensus = self._laplacian_smoothing(self.y_hat)
        self.y_hat = self.y_hat + delta + self.consensus_gain * consensus
        self.previous_local_output = local

    def _utilization(self) -> Array:
        pr = self.problem
        return np.sum(pr.service_selector * self.z, axis=1) / pr.service_capacity

    def _primal_gradient(self, target_estimate: Array) -> Array:
        """Gradient of the augmented Lagrangian w.r.t. z (same CPES objective + box dual term)."""
        assert self.z is not None and self.y_hat is not None
        assert self.mu_lo is not None and self.mu_hi is not None
        pr, cfg = self.problem, self.config

        grad = cfg.loss_weight_scale * pr.loss_weight * (self.z - pr.rest)          # local loss
        aggregate_error = self.y_hat - target_estimate                             # tracking
        grad += cfg.aggregate_tracking_weight * np.einsum(
            "nmd,nm->nd", pr.aggregate_blocks, aggregate_error)

        rho = self._utilization()                                                  # sharing
        rho_diff = pr.adjacency @ rho - np.sum(pr.adjacency, axis=1) * rho
        lap_rho = -rho_diff
        grad += cfg.sharing_weight * (pr.service_selector / pr.service_capacity[:, None]) * lap_rho[:, None]

        ref_lap = -self._laplacian_smoothing(self.z - pr.rest)                      # internal reg.
        grad += cfg.internal_weight * ref_lap

        # Box-constraint dual term. g_lo = lower - z <= 0, g_hi = z - upper <= 0, so
        # d/dz [ mu_lo*(lower - z) + mu_hi*(z - upper) ] = -mu_lo + mu_hi.
        grad += (self.mu_hi - self.mu_lo)
        return grad

    def control(self, p: Array, v: Array, target_estimate: Array) -> tuple[Array, DPDDiagnostics]:
        p = np.asarray(p, dtype=float); v = np.asarray(v, dtype=float)
        target_estimate = np.asarray(target_estimate, dtype=float)
        if self.z is None or self.z.shape != p.shape:
            self.reset(p)
        assert self.z is not None and self.mu_lo is not None and self.mu_hi is not None
        self._aggregate_estimate_update()

        # --- primal descent (NO hard projection: the box is handled by the duals, Arrow-Hurwicz)
        grad = self._primal_gradient(target_estimate)
        self.z = self.z - self.primal_step * grad

        # --- dual ascent on the two box constraints g_lo = lower - z <= 0, g_hi = z - upper <= 0
        g_lo = self.problem.lower - self.z
        g_hi = self.z - self.problem.upper
        self.mu_lo = np.maximum(0.0, self.mu_lo + self.dual_step * g_lo)
        self.mu_hi = np.maximum(0.0, self.mu_hi + self.dual_step * g_hi)
        # a light safety clamp keeps the *reference* finite; the HOCBF filter enforces hard limits
        self.z = np.minimum(np.maximum(self.z, self.problem.lower - 0.5), self.problem.upper + 0.5)

        # --- PD tracking of the primal reference (identical to the DPG comparator)
        a_nom = self.reference_tracking_gain * (self.z - p) - self.velocity_damping * v
        u_nom = self.problem.masses[:, None] * a_nom + self.problem.dampings[:, None] * v
        u_nom = truncate_rows_by_norm(u_nom, self.problem.force_limits)

        est_spread = 0.0
        if self.y_hat is not None and self.y_hat.size:
            est_spread = float(np.max(row_norm(self.y_hat - np.mean(self.y_hat, axis=0, keepdims=True))))
        ref_spread = float(np.mean(row_norm(self.z - p)))
        mean_grad = float(np.mean(row_norm(grad)))
        force_margin = float(np.min(self.problem.force_limits - row_norm(u_nom)))
        return u_nom, DPDDiagnostics(
            estimate_spread=est_spread, mean_gradient_norm=mean_grad,
            reference_spread=ref_spread, force_margin=force_margin,
            mean_dual=float(np.mean(self.mu_lo + self.mu_hi)))
