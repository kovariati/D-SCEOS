"""
distributed_projected_gradient_controller.py
============================================

Optimization-aware diagnostic baseline for the D-SCEOS article.

The controller is intentionally not another D-SCEOS certificate. It is a
minimal peer-to-peer distributed projected-gradient reference generator that
uses the same fixed communication graph and gateway-injected target estimates
as the other controllers. Its output is then passed through the same HOCBF
safety filter used for the PD baselines.

Name used by the validation harness:
    projected_gradient_hocbf

Interpretation:
    DPG-HOCBF = distributed projected-gradient reference update + PD tracking
                + posthoc HOCBF safety filter.

The purpose is to provide one optimization-aware comparator without turning the
paper into an exhaustive distributed-optimization benchmark study.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dsceos_controller import DSCEOSConfig, DSCEOSProblemData, row_norm, truncate_rows_by_norm

Array = np.ndarray


@dataclass
class DPGDiagnostics:
    estimate_spread: float
    mean_gradient_norm: float
    reference_spread: float
    force_margin: float


class DistributedProjectedGradientController:
    """Peer-to-peer projected-gradient baseline.

    Each agent keeps a local reference ``z_i``. A dynamic-consensus estimate of
    the aggregate output of ``z`` is used to build an optimization-aware local
    gradient. The reference is projected back to the local operating box after
    each step. A standard local PD law tracks ``z_i``. Safety is not native to
    this baseline; the validation harness applies the common HOCBF safety
    filter to the nominal output.

    The local objective terms intentionally mirror the CPES objective used in
    the article, but the method lacks the D-SCEOS CLF/HOCBF arbitration and is
    therefore only an optimization-aware diagnostic comparator.
    """

    def __init__(
        self,
        problem: DSCEOSProblemData,
        config: DSCEOSConfig | None = None,
        # Selected by a coarse step-size sweep over {0.02..0.72} on the
        # stress-ladder probes and the realistic scenarios; the optimum is
        # flat around 0.06. In the documented one-at-a-time sweep larger steps DEGRADED the
        # objective and smaller steps converged more slowly; the sweep records the objective
        # only and does not demonstrate instability of the z / aggregate-estimate feedback loop.
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
        self.consensus_gain = (
            float(consensus_gain)
            if consensus_gain is not None
            else float(self.config.aggregate_consensus_gain)
        )
        if self.config.adaptive_consensus_gain:
            self.consensus_gain = self._gershgorin_gain(self.problem.adjacency)
        self.z: Array | None = None
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
        local = np.einsum("nmd,nd->nm", self.problem.aggregate_blocks, self.z)
        self.y_hat = local.copy()
        self.previous_local_output = local.copy()

    def _laplacian_smoothing(self, x: Array) -> Array:
        W = self.problem.adjacency
        deg = np.sum(W, axis=1)
        return W @ x - deg[:, None] * x

    def _aggregate_estimate_update(self) -> None:
        assert self.z is not None
        assert self.y_hat is not None
        assert self.previous_local_output is not None
        local = np.einsum("nmd,nd->nm", self.problem.aggregate_blocks, self.z)
        delta = local - self.previous_local_output
        consensus = self._laplacian_smoothing(self.y_hat)
        self.y_hat = self.y_hat + delta + self.consensus_gain * consensus
        self.previous_local_output = local

    def _gradient(self, target_estimate: Array) -> Array:
        assert self.z is not None
        assert self.y_hat is not None
        pr = self.problem
        cfg = self.config
        n = max(pr.lower.shape[0], 1)

        # Local cost-proportional quadratic loss.
        grad = cfg.loss_weight_scale * pr.loss_weight * (self.z - pr.rest)

        # Aggregate-service tracking through a local aggregate estimate.
        aggregate_error = self.y_hat - target_estimate
        # Size-consistent tracking gradient (matches the D-SCEOS convention).
        grad += cfg.aggregate_tracking_weight * np.einsum(
            "nmd,nm->nd", pr.aggregate_blocks, aggregate_error
        )

        # Capacity-normalized utilization sharing.
        rho = np.sum(pr.service_selector * self.z, axis=1) / pr.service_capacity
        rho_diff = self.problem.adjacency @ rho - np.sum(self.problem.adjacency, axis=1) * rho
        # Gradient of 0.5 * sum a_ij (rho_i-rho_j)^2 is L*rho.
        lap_rho = -rho_diff
        grad += cfg.sharing_weight * (
            pr.service_selector / pr.service_capacity[:, None]
        ) * lap_rho[:, None]

        # Mild graph-local smoothness of the reference, replacing the
        # D-SCEOS hidden-counter-action proxy by a standard projected-gradient
        # regularizer. This keeps the baseline in the distributed-optimization
        # family rather than in the D-SCEOS certificate family.
        ref_lap = -self._laplacian_smoothing(self.z - pr.rest)
        grad += cfg.internal_weight * ref_lap

        return grad

    def control(self, p: Array, v: Array, target_estimate: Array) -> tuple[Array, DPGDiagnostics]:
        p = np.asarray(p, dtype=float)
        v = np.asarray(v, dtype=float)
        target_estimate = np.asarray(target_estimate, dtype=float)
        if self.z is None or self.z.shape != p.shape:
            self.reset(p)

        # Keep the reference in the feasible box and update aggregate estimates.
        assert self.z is not None
        self.z = np.minimum(np.maximum(self.z, self.problem.lower), self.problem.upper)

        grad_norms = []
        for _ in range(self.optimizer_substeps):
            self._aggregate_estimate_update()
            grad = self._gradient(target_estimate)
            grad_norms.append(row_norm(grad))
            self.z = self.z - self.step_size * grad
            self.z = np.minimum(np.maximum(self.z, self.problem.lower), self.problem.upper)

        a_nom = self.reference_tracking_gain * (self.z - p) - self.velocity_damping * v
        u_nom = self.problem.masses[:, None] * a_nom + self.problem.dampings[:, None] * v
        u_nom = truncate_rows_by_norm(u_nom, self.problem.force_limits)

        estimate_spread = 0.0
        if self.y_hat is not None and self.y_hat.size:
            estimate_spread = float(np.max(row_norm(self.y_hat - np.mean(self.y_hat, axis=0, keepdims=True))))
        reference_spread = float(np.mean(row_norm(self.z - p))) if self.z is not None else 0.0
        mean_grad = float(np.mean(grad_norms[-1])) if grad_norms else 0.0
        force_margin = float(np.min(self.problem.force_limits - row_norm(u_nom)))
        return u_nom, DPGDiagnostics(
            estimate_spread=estimate_spread,
            mean_gradient_norm=mean_grad,
            reference_spread=reference_spread,
            force_margin=force_margin,
        )
