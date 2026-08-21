"""
dsceos_controller.py
========================

Reusable controller core for D-SCEOS: a gateway-injected peer-to-peer,
resource-aware swarm-coordination controller for CPES flexibility sharing,
implemented as a local CLF/HOCBF-constrained sampled controller with feasibility
diagnostics.

Terminology note: this module is NOT 'stability-certified' or 'safety-certified' in
the general sense. The stability argument in the paper is for a reduced continuous-time
surrogate, not for this sampled QCQP closed loop; and the operating-envelope property is
a per-sample admissibility of the applied input, not an intersample state-invariance
guarantee. The projection fallback additionally only returns a box-admissible point when
the HOCBF box and the actuator ball actually intersect; on an empty intersection it
returns a ball-admissible point with a positive box residual, which the diagnostics report.

Scope
-----
This file contains only controller/arbitration logic. It does not implement
scenario generation, plotting, file export or benchmark integration. The
controller uses a fixed sparse neighbour graph and no master/aggregator node.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence
import math
import time
import numpy as np
try:
    from scipy.optimize import minimize
except ImportError as _scipy_exc:  # pragma: no cover
    # Do NOT silently swap in the projection fallback: without SciPy the local program is solved by
    # a different algorithm, so the published procedure would change identity without any signal.
    # The controller raises at construction unless the caller explicitly opts in.
    minimize = None
    _SCIPY_IMPORT_ERROR = _scipy_exc
else:
    _SCIPY_IMPORT_ERROR = None

Array = np.ndarray


def row_norm(x: Array, eps: float = 0.0) -> Array:
    n = np.linalg.norm(np.asarray(x, dtype=float), axis=1)
    return np.maximum(n, eps) if eps else n


def truncate_rows_by_norm(u: Array, limits: Array, eps: float = 1.0e-12) -> Array:
    u = np.asarray(u, dtype=float).copy()
    norms = row_norm(u)
    limits = np.asarray(limits, dtype=float)
    scale = np.ones_like(norms)
    mask = norms > limits
    scale[mask] = limits[mask] / np.maximum(norms[mask], eps)
    return u * scale[:, None]


def _project_to_l2_ball(x: Array, radius: float, eps: float = 1.0e-12) -> Array:
    """Euclidean projection onto a centred l2 ball."""
    x = np.asarray(x, dtype=float)
    n = float(np.linalg.norm(x))
    if n <= radius + eps:
        return x.copy()
    return x * (radius / max(n, eps))


def project_to_box_ball(u_nom: Array, lb: Array, ub: Array, radius: float, *,
                        iterations: int = 60, eps: float = 1.0e-10,
                        return_status: bool = False):
    """Project a vector onto ``{u: lb <= u <= ub, ||u||_2 <= radius}``.

    Implemented by Dykstra alternating projections between the componentwise box and the Euclidean
    ball, so that the HOCBF bounds are not broken by a subsequent norm truncation.

    Infeasibility handling. Two distinct failure modes are reported rather than hidden:

      * an INCONSISTENT box, i.e. ``lb > ub`` on some coordinate, which makes the HOCBF set itself
        empty on that coordinate. The bounds are NOT reordered; the iteration runs on the midpoint
        surrogate so that a usable input is still produced, but the returned residual is measured
        against the ORIGINAL bounds and is therefore strictly positive;
      * an empty box-ball intersection with a consistent box, where the Dykstra iterate settles at a
        point with a positive residual.

    The returned residual is always measured against the caller's original ``lb``/``ub`` and
    ``radius``. With ``return_status=True`` the function additionally returns a status dictionary
    with the separate box, ball and inconsistency diagnostics.
    """
    x = np.asarray(u_nom, dtype=float).copy()
    lb = np.asarray(lb, dtype=float)
    ub = np.asarray(ub, dtype=float)

    inconsistent = np.any(lb > ub)
    if inconsistent:
        # Do NOT swap the bounds: swapping fabricates a non-empty interval and would report a zero
        # residual for a mathematically empty set. Iterate on the midpoint surrogate instead, and
        # score the result against the original bounds below.
        mid = 0.5 * (lb + ub)
        lb2 = np.where(lb > ub, mid, lb)
        ub2 = np.where(lb > ub, mid, ub)
    else:
        lb2, ub2 = lb, ub

    p = np.zeros_like(x)
    q = np.zeros_like(x)
    for _ in range(max(1, iterations)):
        y = np.minimum(np.maximum(x + p, lb2), ub2)
        p = x + p - y
        x_next = _project_to_l2_ball(y + q, radius)
        q = y + q - x_next
        if float(np.linalg.norm(x_next - x)) <= eps:
            x = x_next
            break
        x = x_next

    # Residuals against the ORIGINAL constraint data, never against the surrogate.
    box_res = max(float(np.max(lb - x)), float(np.max(x - ub)), 0.0)
    ball_res = max(float(np.linalg.norm(x) - radius), 0.0)
    residual = max(box_res, ball_res)
    if return_status:
        status = dict(box_residual=box_res, ball_residual=ball_res,
                      inconsistent_box=bool(inconsistent), feasible=bool(residual <= eps))
        return x, residual, status
    return x, residual


def project_rows_to_box_ball(u_nom: Array, lb: Array, ub: Array, limits: Array) -> tuple[Array, float]:
    u_nom = np.asarray(u_nom, dtype=float)
    out = np.zeros_like(u_nom)
    residual = 0.0
    for i in range(u_nom.shape[0]):
        out[i], ri = project_to_box_ball(u_nom[i], lb[i], ub[i], float(limits[i]))
        residual = max(residual, ri)
    return out, residual



def validate_communication_graph(W, *, context: str = "adjacency") -> None:
    """Validate every graph property the theory in this module relies on.

    The Laplacian gradient blocks, the Gershgorin consensus-gain rule and the consensus-convergence
    argument all assume a symmetric, non-negatively weighted, zero-diagonal and (for N>1) connected
    graph. This is called both at construction and on every runtime topology change, so a
    reconfiguration cannot silently install a graph the analysis does not cover.
    """
    W = np.asarray(W, dtype=float)
    if W.ndim != 2 or W.shape[0] != W.shape[1]:
        raise ValueError(f"{context} must be a square matrix.")
    n = W.shape[0]
    if not np.all(np.isfinite(W)):
        raise ValueError(f"{context} must be finite.")
    if not np.allclose(W, W.T, rtol=0.0, atol=1e-12):
        raise ValueError(f"{context} must be symmetric: the Laplacian gradient blocks and the "
                         "Gershgorin gain rule assume W = W^T.")
    if np.any(W < 0.0):
        raise ValueError(f"{context} weights must be non-negative.")
    if np.any(np.abs(np.diag(W)) > 1e-12):
        raise ValueError(f"{context} must have a zero diagonal (no self-loops).")
    if n > 1:
        seen, stack = {0}, [0]
        while stack:
            i = stack.pop()
            for j in np.nonzero(W[i] > 0.0)[0]:
                if int(j) not in seen:
                    seen.add(int(j)); stack.append(int(j))
        if len(seen) != n:
            raise ValueError(f"{context} must describe a connected graph: consensus convergence and "
                             "the aggregate-estimator argument assume connectivity.")


@dataclass(frozen=True)
class DSCEOSProblemData:
    """Agent-local and graph data for D-SCEOS.

    All arrays are expressed in normalized per-unit coordination variables.
    No row of the controller requires global all-to-all communication at run
    time; the validation harness may hold the arrays centrally only to simulate
    peer-to-peer execution.


    Mutability caveat. ``frozen=True`` prevents rebinding the attributes, but the NumPy arrays they
    hold remain writable, so the immutability contract is weaker than the decorator suggests.
    ``set_adjacency`` relies on this to update the topology in place and revalidates the graph
    first. Callers must not mutate any other array in place: the controller caches derived
    quantities that are refreshed only through the explicit topology-change API.
    """

    aggregate_blocks: Array       # shape (N, m, d), local A_i blocks
    lower: Array                  # shape (N, d)
    upper: Array                  # shape (N, d)
    rest: Array                   # shape (N, d)
    loss_weight: Array            # shape (N, d), diagonal local quadratic cost
    masses: Array                 # shape (N,)
    dampings: Array               # shape (N,)
    force_limits: Array           # shape (N,)
    service_selector: Array       # shape (N, d), g_i
    service_capacity: Array       # shape (N,), sbar_i
    adjacency: Array              # shape (N, N), fixed symmetric nonnegative neighbour weights

    def __post_init__(self) -> None:
        n = self.lower.shape[0]
        if self.upper.shape != self.lower.shape or self.rest.shape != self.lower.shape:
            raise ValueError("lower, upper and rest must share shape (N, d).")
        if self.aggregate_blocks.shape[0] != n:
            raise ValueError("aggregate_blocks must have one block per agent.")
        if self.loss_weight.shape != self.lower.shape:
            raise ValueError("loss_weight must have shape (N, d).")
        if self.service_selector.shape != self.lower.shape:
            raise ValueError("service_selector must have shape (N, d).")
        for name in ("masses", "dampings", "force_limits", "service_capacity"):
            if getattr(self, name).shape != (n,):
                raise ValueError(f"{name} must have shape (N,).")
        validate_communication_graph(self.adjacency, context="adjacency")
        if self.adjacency.shape != (n, n):
            raise ValueError("adjacency must have shape (N, N).")
        if np.any(self.masses <= 0) or np.any(self.dampings <= 0) or np.any(self.force_limits <= 0):
            raise ValueError("masses, dampings and force_limits must be positive.")
        if np.any(self.service_capacity <= 0):
            raise ValueError("service_capacity must be positive.")


@dataclass(frozen=True)
class DSCEOSConfig:
    optimizer_gain: float = 1.20
    velocity_damping: float = 1.40
    # Explicit-Euler consensus step. The default is used when adaptive
    # consensus tuning is disabled.
    aggregate_consensus_gain: float = 0.10
    # If True, the controller computes a Gershgorin upper bound on the largest
    # Laplacian eigenvalue and sets the aggregate-consensus gain to a certified
    # value inside the stability interval. For a symmetric nonnegative
    # adjacency matrix W and Laplacian L = D - W, Gershgorin gives
    # lambda_N(L) <= 2 * max_i d_i, where d_i = sum_j W_ij. The implemented
    # gain is
    #     K_y = gershgorin_safety_factor * 2 / lambda_N_bound
    #         = gershgorin_safety_factor / max_i d_i.
    # This is conservative compared with an eigenvalue-optimal gain, but it is a
    # hard, checkable graph-topological certificate and requires no iterative
    # eigenvalue estimator.
    adaptive_consensus_gain: bool = False
    gershgorin_safety_factor: float = 0.95
    # Deprecated compatibility alias is handled in external runners. Do not use
    # power-iteration spectral estimates in this version.
    # Weight of the normalized aggregate-tracking gradient. The exact gradient
    # carries a 1/N factor; this parameter plays the role of W_y in the paper
    # and prevents the aggregate target from being dominated by local costs.
    # Per-agent aggregate-tracking weight (size-consistent objective):
    # the tracking term of J is (N * aggregate_tracking_weight / 2) * ||y - y_T||^2,
    # so the per-agent tracking gradient carries no 1/N attenuation. Under a fleet
    # sequence with bounded average degree and a convergent empirical distribution
    # of unit parameters the terms of J are of the same extensive order and J/N has
    # a well-defined large-N limit (verified in the paper only for the two tested
    # fleet sizes). At the reference implementation value 2/3 the
    # N=15 study coincides numerically with the conventional w_y = 10 form.
    aggregate_tracking_weight: float = 2.0 / 3.0
    # Publication defaults calibrated for the normalized aggregate convention
    # y = mean_i A_i p_i. They are intentionally smaller than the centralized
    # source-integrity/source-integrity defaults, because otherwise local loss/sharing terms dominate the
    # 1/N-scaled aggregate pull.
    loss_weight_scale: float = 0.03
    sharing_weight: float = 0.15
    internal_weight: float = 0.03
    hocbf_alpha1: float = 2.0
    hocbf_alpha0: float = 1.0
    actuator_constraint: str = "norm"  # "norm" or "box"
    control_box_fraction: float = 1.0  # used only for box mode
    # Local CLF/HOCBF-QP parameters. These make the implementation match the
    # proof-bearing controller described in the article rather than only a
    # projected-gradient heuristic. The QP is solved independently per agent.
    use_local_qp: bool = True
    clf_rate: float = 0.80
    clf_position_gain: float = 1.20
    clf_cross_gain_factor: float = 0.35
    clf_slack_weight: float = 500.0
    # Hard-row feasibility tolerance used to accept a primary solver solution: a solver success
    # flag is not evidence of feasibility, so the returned point must actually satisfy the hard
    # rows to this tolerance or it is treated as a fallback.
    #
    # This is the RUNTIME acceptance gate and is deliberately distinct from the PUBLICATION quality
    # threshold that validate_results.py enforces on the released runs (2e-8 on the applied-force
    # residual). Both are recorded in the run summary. The runtime gate is set an order of magnitude
    # below the released residuals so that it never silently admits a point the publication gate
    # would reject: tightened from 1e-6 to 1e-9 for exactly that reason.
    feasibility_tolerance: float = 1.0e-9
    # Opt-in escape hatch for environments without SciPy. Leaving this False means the controller
    # refuses to run rather than silently solving the local program with the projection fallback.
    allow_projection_fallback_without_solver: bool = False

    def __post_init__(self) -> None:
        if self.optimizer_gain <= 0 or self.velocity_damping <= 0:
            raise ValueError("optimizer_gain and velocity_damping must be positive.")
        if self.aggregate_tracking_weight <= 0:
            raise ValueError("aggregate_tracking_weight must be positive.")
        if (self.aggregate_consensus_gain < 0 or self.loss_weight_scale < 0
                or self.sharing_weight < 0 or self.internal_weight < 0):
            raise ValueError("consensus, loss, sharing and internal weights must be nonnegative.")
        if not (0.0 < self.gershgorin_safety_factor < 1.0):
            raise ValueError("gershgorin_safety_factor must be in (0, 1).")
        if self.hocbf_alpha0 <= 0 or self.hocbf_alpha1 <= 0:
            raise ValueError("HOCBF gains must be positive.")
        # Admissibility of the relative-degree-two barrier: the formal statement uses the
        # double-root condition alpha_1^2 >= 4 alpha_0, which gives a real, non-oscillatory barrier
        # decay. Positivity alone admits gain pairs the analysis does not cover, so the condition is
        # enforced here rather than only reported in a diagnostic table.
        if self.hocbf_alpha1 ** 2 < 4.0 * self.hocbf_alpha0 - 1e-12:
            raise ValueError(
                f"HOCBF gains are not admissible: alpha1^2 = {self.hocbf_alpha1 ** 2:.6g} < "
                f"4*alpha0 = {4.0 * self.hocbf_alpha0:.6g}. Choose alpha1 >= 2*sqrt(alpha0); the "
                "released defaults sit on the double-root boundary.")
        if self.clf_rate <= 0 or self.clf_position_gain <= 0 or self.clf_slack_weight <= 0:
            raise ValueError("CLF rate, position gain and slack weight must be positive.")
        if not (0.0 <= self.clf_cross_gain_factor < 1.0):
            raise ValueError("clf_cross_gain_factor must be in [0, 1).")
        if self.actuator_constraint not in {"norm", "box"}:
            raise ValueError("actuator_constraint must be 'norm' or 'box'.")


@dataclass
class DSCEOSState:
    """Internal decentralized estimator state.

    y_hat[i] is agent i's local estimate of aggregate output. It is updated
    using a dynamic average-consensus step; only neighbour y_hat values and
    the local contribution change are needed.

    When adaptive_consensus_gain is enabled, the controller stores a
    Gershgorin upper bound on the largest graph-Laplacian eigenvalue and the
    resulting certified aggregate-consensus gain. These quantities depend only
    on the fixed communication graph and are diagnostic outputs for
    reproducibility.
    """

    y_hat: Array                  # shape (N, m)
    previous_local_output: Array  # shape (N, m)
    estimators_initialized: bool = False   # explicit lifecycle flag; see control()
    gershgorin_lambda_N_bound: float = 0.0
    gershgorin_max_weighted_degree: float = 0.0
    active_aggregate_consensus_gain: float = 0.0


@dataclass
class DSCEOSDiagnostics:
    aggregate_estimate_spread: float
    mean_capacity_margin: float
    force_margin: float
    mean_gradient_norm: float
    max_state_violation: float
    hocbf_projection_residual: float
    clf_value: float = math.nan
    clf_slack: float = math.nan
    clf_residual: float = math.nan   # DEPRECATED alias of max_qcqp_residual; not CLF-specific
    max_qcqp_residual: float = math.nan  # max(HOCBF-box, actuator-ball, CLF-row) residual
    qp_success_rate: float = math.nan


@dataclass(frozen=True)
class DSCEOSMessage:
    """Minimal peer-to-peer message exchanged between neighbouring agents.

    The message contains only quantities that an agent may broadcast locally: its
    current normalized operating coordinate, its rate, its capacity-normalized
    utilization, and its current dynamic-consensus estimate of the normalized
    aggregate service. No global state vector, global objective matrix, or
    master-node information is contained in this message.
    """

    agent: int
    p: Array
    v: Array
    rho: float
    y_hat: Array


class DistributedSCEOSController:
    """Fully peer-to-peer D-SCEOS controller.

    The controller is vectorized for simulation but follows a local information
    pattern: each agent uses its local data, its own aggregate estimate, and
    neighbour messages (rho_j, y_hat_j, p_j). No master node, central QP, or
    all-to-one aggregation is used in the control law. The messages() method
    exposes the peer-to-peer information interface explicitly.
    """

    def __init__(self, problem: DSCEOSProblemData, config: DSCEOSConfig | None = None):
        self.problem = problem
        self.config = config or DSCEOSConfig()
        n, m, _ = problem.aggregate_blocks.shape
        y0 = np.zeros((n, m), dtype=float)
        if minimize is None and not self.config.allow_projection_fallback_without_solver:
            raise RuntimeError(
                "scipy.optimize.minimize is unavailable, so the local CLF/HOCBF program cannot be "
                f"solved by the published procedure ({_SCIPY_IMPORT_ERROR}). Install the pinned SciPy "
                "(see requirements-lock.txt), or set "
                "DSCEOSConfig(allow_projection_fallback_without_solver=True) to run the deterministic "
                "projection instead -- which is a DIFFERENT algorithm and must be reported as such.")
        # Solver failure-mode counters. Each mode is counted separately so that a run summary can
        # distinguish 'the solver reported failure' from 'the solver reported success but returned an
        # infeasible point' from 'the solver raised' -- these have different implications.
        self._solver_rejects = 0        # success=True but hard rows violated beyond tolerance
        self._solver_not_success = 0    # solver reported success=False
        self._solver_nonfinite = 0      # solver returned a non-finite iterate
        self._solver_exceptions = 0     # solver raised
        self._solver_unavailable = 0    # no solver present (explicit opt-in path)
        self._last_solver_status = ""
        self._last_solver_error = ""
        self.state = DSCEOSState(
            y_hat=y0,
            previous_local_output=y0.copy(),
            active_aggregate_consensus_gain=float(self.config.aggregate_consensus_gain),
        )
        if self.config.adaptive_consensus_gain:
            self._tune_consensus_gain_from_gershgorin()

    # ----- Certified Gershgorin consensus-gain selection ------------------
    # The dynamic average-consensus update used in update_estimators() reads
    #
    #     y_hat_{k+1} = (I - K_y L) y_hat_k + Δy_local,
    #
    # where L = D - W is the symmetric graph Laplacian. On the
    # consensus-orthogonal subspace the homogeneous update is stable iff
    #
    #     0 < K_y < 2 / lambda_N(L).
    #
    # For a nonnegative symmetric adjacency matrix, Gershgorin's theorem gives
    # the checkable upper bound
    #
    #     lambda_N(L) <= 2 max_i d_i,    d_i = sum_j W_ij.
    #
    # Therefore K_y = gamma / max_i d_i with 0 < gamma < 1 satisfies
    # K_y < 2 / lambda_N(L) whenever the graph has at least one edge. This is
    # more conservative than an eigenvalue-optimal gain, but it is a hard
    # graph-topological certificate and requires no finite-time eigenvalue
    # estimator.

    def _laplacian(self) -> Array:
        W = self.problem.adjacency
        D = np.diag(np.sum(W, axis=1))
        return D - W

    def _gershgorin_laplacian_bound(self) -> tuple[float, float]:
        """Return (lambda_N_upper_bound, max_weighted_degree).

        The bound is λ_N(L) <= 2 max_i d_i for a symmetric nonnegative graph.
        It is computed from row sums and is therefore directly available from
        local degree information and one max-consensus round in a distributed
        implementation.
        """
        W = np.asarray(self.problem.adjacency, dtype=float)
        degrees = np.sum(W, axis=1)
        dmax = float(np.max(degrees)) if degrees.size else 0.0
        lam_bound = 2.0 * dmax
        return max(lam_bound, 0.0), dmax

    def _tune_consensus_gain_from_gershgorin(self) -> None:
        """Set K_y from the Gershgorin Laplacian bound.

        The applied value is
            K_y = safety_factor * 2 / lambda_N_bound
        with lambda_N_bound = 2 max_i d_i. If the graph has no edges, the
        configured static aggregate_consensus_gain is retained.
        """
        lam_bound, dmax = self._gershgorin_laplacian_bound()
        if lam_bound <= 1.0e-15:
            K_y = float(self.config.aggregate_consensus_gain)
        else:
            K_y = float(self.config.gershgorin_safety_factor * 2.0 / lam_bound)
        self.state = DSCEOSState(
            y_hat=self.state.y_hat,
            previous_local_output=self.state.previous_local_output,
            gershgorin_lambda_N_bound=lam_bound,
            gershgorin_max_weighted_degree=dmax,
            active_aggregate_consensus_gain=K_y,
        )

    # ---------------------------------------------------------------------

    def set_adjacency(self, adjacency: Array) -> None:
        """Replace the fixed peer-to-peer communication graph.

        In CPES applications the neighbour set is normally determined by
        physical, communication or contractual proximity, not by the current
        operating coordinates ``p``. This method is therefore intended for
        topology reconfiguration, commissioning changes, or agent-loss events.
        It should not be called each simulation step as a state-dependent
        dynamic operating-coordinate neighbour update.
        """
        W = np.asarray(adjacency, dtype=float)
        if W.shape != self.problem.adjacency.shape:
            raise ValueError("adjacency has incompatible shape")
        # Runtime topology changes must satisfy exactly the same preconditions as construction;
        # otherwise a reconfiguration could install a graph for which the Laplacian gradient, the
        # Gershgorin gain rule and the consensus argument are all invalid.
        validate_communication_graph(W, context="replacement adjacency")
        self.problem.adjacency[:] = W
        # When the topology changes we recompute the Gershgorin bound so the
        # consensus gain remains certified under the new fixed graph.
        if self.config.adaptive_consensus_gain:
            self._tune_consensus_gain_from_gershgorin()

    def reset_estimators(self, p: Array) -> None:
        local = self.local_outputs(p)
        # Each agent starts with its own scaled local contribution. Dynamic
        # average consensus then tracks the normalized aggregate service. In the
        # validation harness the local blocks are scaled so that
        # mean_i(A_i p_i) equals the reported aggregate output.
        # Preserve the graph-topological Gershgorin diagnostics (they depend
        # only on the communication graph, which has not changed at a reset).
        prev_bound = self.state.gershgorin_lambda_N_bound
        prev_degree = self.state.gershgorin_max_weighted_degree
        prev_Ky = self.state.active_aggregate_consensus_gain
        self.state = DSCEOSState(
            y_hat=local.copy(),
            previous_local_output=local.copy(),
            gershgorin_lambda_N_bound=prev_bound,
            gershgorin_max_weighted_degree=prev_degree,
            active_aggregate_consensus_gain=prev_Ky,
        )
        self.state.estimators_initialized = True

    def messages(self, p: Array, v: Array) -> list[DSCEOSMessage]:
        """Return the neighbour messages that an actual peer-to-peer
        implementation would broadcast locally.

        The vectorized simulator calls one controller object for convenience,
        but this method makes the information pattern explicit and reusable in
        agent-based simulators.
        """
        rho = self.utilization(p)
        return [DSCEOSMessage(i, p[i].copy(), v[i].copy(), float(rho[i]), self.state.y_hat[i].copy())
                for i in range(p.shape[0])]

    def local_outputs(self, p: Array) -> Array:
        Ablk = self.problem.aggregate_blocks
        return np.einsum("imd,id->im", Ablk, p)

    def utilization(self, p: Array) -> Array:
        return np.sum(self.problem.service_selector * p, axis=1) / self.problem.service_capacity

    def _as_per_agent_target(self, y_target: Array, n: int) -> Array:
        """Return one target estimate per agent.

        ``y_target`` may be either a global vector of shape ``(m,)`` for
        backwards-compatible experiments or a peer-to-peer estimate array
        of shape ``(N,m)``. The publication/fair-comparison path uses the latter.
        """
        y = np.asarray(y_target, dtype=float)
        if y.ndim == 1:
            return np.tile(y[None, :], (n, 1))
        if y.ndim == 2 and y.shape[0] == n:
            return y
        raise ValueError("y_target must have shape (m,) or (N,m).")

    def update_estimators(self, p: Array) -> None:
        cfg, pr = self.config, self.problem
        y_local = self.local_outputs(p)
        dy_local = y_local - self.state.previous_local_output
        W = pr.adjacency
        # Dynamic average-consensus style estimator. Each i uses only y_hat_j
        # from neighbours j with W_ij>0.
        consensus = W @ self.state.y_hat - np.sum(W, axis=1)[:, None] * self.state.y_hat
        # K_y is either the static config value or, when adaptive_consensus_gain
        # is on, the Gershgorin-tuned value stored in self.state.
        K_y = (self.state.active_aggregate_consensus_gain
               if cfg.adaptive_consensus_gain
               else cfg.aggregate_consensus_gain)
        y_hat_new = self.state.y_hat + K_y * consensus + dy_local
        self.state = DSCEOSState(
            y_hat=y_hat_new,
            previous_local_output=y_local.copy(),
            gershgorin_lambda_N_bound=self.state.gershgorin_lambda_N_bound,
            gershgorin_max_weighted_degree=self.state.gershgorin_max_weighted_degree,
            active_aggregate_consensus_gain=self.state.active_aggregate_consensus_gain,
        )

    def local_gradients(self, p: Array, y_target: Array) -> Array:
        pr, cfg = self.problem, self.config
        W = pr.adjacency
        n = p.shape[0]
        y_target_i = self._as_per_agent_target(y_target, n)
        rho = self.utilization(p)
        grad = cfg.loss_weight_scale * pr.loss_weight * (p - pr.rest)

        # Aggregate tracking gradient uses only the local normalized-aggregate
        # estimate. The validation convention is
        #   y = mean_i A_i p_i,
        # hence the exact gradient of 0.5||y-y_T||^2 with respect to p_i carries
        # the factor 1/N. The additional aggregate_tracking_weight is the
        # implemented W_y scalar. It is calibrated against the local loss,
        # sharing and internal-counter-action weights for the normalized
        # aggregate convention.
        # Fair decentralized target information: each agent compares its
        # aggregate-service estimate to its own locally available target
        # estimate. In the validation harness only gateway agents see the
        # supervisory target directly; all other agents obtain it by consensus.
        err_y = self.state.y_hat - y_target_i
        # Size-consistent tracking gradient: d/dp_i (N*w/2)||y-y_T||^2 = w * A_i^T (y - y_T).
        grad += cfg.aggregate_tracking_weight * np.einsum("imd,im->id", pr.aggregate_blocks, err_y)

        # Capacity-normalized utilization alignment, graph-local.
        diff_rho = np.sum(W * (rho[:, None] - rho[None, :]), axis=1)
        grad += cfg.sharing_weight * (diff_rho / pr.service_capacity)[:, None] * pr.service_selector

        # Graph-local hidden-counter-action term. Released formulation:
        # the internal penalty of the objective is the EDGE-DISAGREEMENT Laplacian
        # quadratic form
        #     Phi_int(delta) = (lam/2) delta^T L delta
        #                    = (lam/4) sum_ij w_ij ||delta_i - delta_j||^2 ,   L = D - W,
        # whose EXACT gradient block is
        #     grad_i Phi_int = lam (L delta)_i = lam ( d_i delta_i - sum_j w_ij delta_j )
        #                    = lam d_i ( delta_i - neighbour-average ).
        # The controller therefore applies the exact gradient, not a proxy: the
        # steady structural gradient-proxy bias of the previous formulation
        # (which used the non-symmetric operator B = I - D^{-1} W, i.e. D^{-1} L,
        # against an objective built from ||B delta||^2 whose exact gradient is
        # B^T B delta) is removed identically. The extra factor d_i is the LOCAL,
        # static weighted degree, so this needs no additional communication, no
        # second hop and no neighbour-degree exchange.
        degree = np.sum(W, axis=1)
        delta = p - pr.rest
        neigh_avg = np.zeros_like(delta)
        mask = degree > 1.0e-12
        neigh_avg[mask] = (W[mask] @ delta) / degree[mask, None]
        grad += cfg.internal_weight * degree[:, None] * (delta - neigh_avg)
        return grad

    def hocbf_force_bounds(self, p: Array, v: Array) -> tuple[Array, Array]:
        pr, cfg = self.problem, self.config
        m = pr.masses[:, None]
        d = pr.dampings[:, None]
        # lower state constraint h = p - lower
        lb = d * v - m * (cfg.hocbf_alpha1 * v + cfg.hocbf_alpha0 * (p - pr.lower))
        # upper state constraint h = upper - p
        ub = d * v + m * (cfg.hocbf_alpha0 * (pr.upper - p) - cfg.hocbf_alpha1 * v)
        if cfg.actuator_constraint == "box":
            dim = p.shape[1]
            box = (pr.force_limits[:, None] / math.sqrt(dim)) * cfg.control_box_fraction
            lb = np.maximum(lb, -box)
            ub = np.minimum(ub, box)
        return lb, ub


    def _local_clf_terms(self, e: Array, vel: Array, mass: float, damping: float) -> tuple[float, Array, float]:
        """Return V_i, control coefficient and non-control drift for the augmented CLF.

        The local reference is the gradient-implied target ``p_ref = p - g_i``;
        hence the local position error is ``e = p - p_ref = g_i``. During one
        sample this reference is frozen. The augmented quadratic CLF is

            V_i = .5*m*kp*||e||^2 + .5*m*||v||^2 + m*gamma*e^T v.

        Positive definiteness is guaranteed by ``0 <= gamma < sqrt(kp)``;
        the configuration uses gamma = clf_cross_gain_factor*sqrt(kp).
        """
        cfg = self.config
        kp = float(cfg.clf_position_gain)
        gamma = float(cfg.clf_cross_gain_factor * math.sqrt(kp))
        e = np.asarray(e, dtype=float)
        vel = np.asarray(vel, dtype=float)
        V = 0.5 * mass * kp * float(np.dot(e, e)) + 0.5 * mass * float(np.dot(vel, vel)) + mass * gamma * float(np.dot(e, vel))
        coeff_u = vel + gamma * e
        # dot V = coeff_u^T u + drift, with plant m vdot = u - damping*v.
        drift = mass * kp * float(np.dot(e, vel)) + (mass * gamma - damping) * float(np.dot(vel, vel)) - gamma * damping * float(np.dot(e, vel))
        return float(max(V, 0.0)), coeff_u, float(drift)

    def _solve_local_clf_hocbf_qp(self, u_nom: Array, lb: Array, ub: Array,
                                  radius: float, e: Array, vel: Array,
                                  mass: float, damping: float) -> tuple[Array, float, bool, float, float]:
        """Solve the per-agent CLF/HOCBF-QP.

        Decision variables are ``z = [u_1, ..., u_d, s]``. The HOCBF box and
        actuator norm are hard constraints, while the CLF inequality is relaxed
        through ``s >= 0``. If SLSQP fails, the function falls back to the joint
        HOCBF-box/actuator-ball projection and reports a positive residual.
        """
        cfg = self.config
        dim = int(u_nom.shape[0])
        V, coeff_u, drift = self._local_clf_terms(e, vel, mass, damping)
        # coeff_u^T u + drift <= -clf_rate*V + s
        # -> s - coeff_u^T u >= drift + clf_rate*V
        rhs = drift + float(cfg.clf_rate) * V
        lb = np.asarray(lb, dtype=float)
        ub = np.asarray(ub, dtype=float)
        u0, residual0 = project_to_box_ball(u_nom, lb, ub, radius)
        # If the projected point violates the CLF, start with enough slack.
        s0 = max(0.0, float(np.dot(coeff_u, u0) + drift + cfg.clf_rate * V))
        x0 = np.concatenate([u0, np.asarray([s0])])

        def obj(z: Array) -> float:
            u = z[:dim]
            s = float(z[dim])
            return 0.5 * float(np.dot(u - u_nom, u - u_nom)) + float(cfg.clf_slack_weight) * s * s

        def grad_obj(z: Array) -> Array:
            g = np.zeros_like(z)
            g[:dim] = z[:dim] - u_nom
            g[dim] = 2.0 * float(cfg.clf_slack_weight) * z[dim]
            return g

        constraints = []
        # CLF: s - coeff_u^T u - rhs >= 0
        constraints.append({
            'type': 'ineq',
            'fun': lambda z, cu=coeff_u, r=rhs: float(z[dim] - np.dot(cu, z[:dim]) - r),
            'jac': lambda z, cu=coeff_u, r=rhs: np.concatenate([-cu, np.asarray([1.0])]),
        })
        # actuator norm: radius^2 - ||u||^2 >= 0
        constraints.append({
            'type': 'ineq',
            'fun': lambda z, R=radius: float(R * R - np.dot(z[:dim], z[:dim])),
            'jac': lambda z, R=radius: np.concatenate([-2.0 * z[:dim], np.asarray([0.0])]),
        })
        bounds = [(float(lb[k]), float(ub[k])) for k in range(dim)] + [(0.0, None)]
        if minimize is None:
            clf_res = max(0.0, float(np.dot(coeff_u, u0) + drift + cfg.clf_rate * V))
            return u0, s0, False, max(residual0, clf_res), V
        try:
            res = minimize(obj, x0, jac=grad_obj, bounds=bounds, constraints=constraints,
                           method='SLSQP', options={'maxiter': 80, 'ftol': 1e-9, 'disp': False})
            if res.success and np.all(np.isfinite(res.x)):
                u = np.asarray(res.x[:dim], dtype=float)
                slack = float(max(0.0, res.x[dim]))
                box_res = max(float(np.max(lb - u)), float(np.max(u - ub)), 0.0)
                ball_res = max(float(np.linalg.norm(u) - radius), 0.0)
                clf_res = max(0.0, float(np.dot(coeff_u, u) + drift + cfg.clf_rate * V - slack))
                residual = max(box_res, ball_res, clf_res)
                # Accept the primary solution only if the hard rows actually hold to tolerance;
                # otherwise fall through to the deterministic projection and report a fallback.
                if residual <= cfg.feasibility_tolerance:
                    return u, slack, True, residual, V
                self._solver_rejects += 1
                self._last_solver_status = f"residual {residual:.3e} > tol {cfg.feasibility_tolerance:.1e}"
            elif not np.all(np.isfinite(getattr(res, 'x', np.array([np.nan])))):
                self._solver_nonfinite += 1
                self._last_solver_status = "non-finite iterate"
            else:
                self._solver_not_success += 1
                self._last_solver_status = f"success=False: {getattr(res, 'message', '')}"
        except (ValueError, FloatingPointError, np.linalg.LinAlgError, RuntimeError) as exc:
            # Record the failure rather than discarding it, so solver faults stay distinguishable
            # from numerical overflow and from genuine infeasibility.
            self._solver_exceptions += 1
            self._last_solver_error = f"{type(exc).__name__}: {exc}"
        clf_res = max(0.0, float(np.dot(coeff_u, u0) + drift + cfg.clf_rate * V - s0))
        return u0, s0, False, max(residual0, clf_res), V

    def local_clf_hocbf_qp(self, u_nom: Array, lb: Array, ub: Array, grad: Array, v: Array) -> tuple[Array, float, float, float, float]:
        """Solve all independent per-agent local CLF/HOCBF-QCQPs.

        PER-AGENT INSTRUMENTATION. If ``self.qcqp_trace`` is a list, each
        agent-step appends a record ``(agent, fallback, solve_time_s, residual, slack)``. The trace
        is OPT-IN and defaults to ``None``, so the released runs execute exactly the same arithmetic
        in the same order and remain byte-reproducible; only the timing calls are added when the
        trace is enabled, and wall-clock timing is reported as environment-specific.
        """
        pr = self.problem
        n = u_nom.shape[0]
        out = np.zeros_like(u_nom)
        slacks = []
        Vs = []
        residual = 0.0
        success = 0
        trace = getattr(self, "qcqp_trace", None)
        for i in range(n):
            if trace is None:
                u_i, s_i, ok_i, r_i, V_i = self._solve_local_clf_hocbf_qp(
                    u_nom[i], lb[i], ub[i], float(pr.force_limits[i]), grad[i], v[i],
                    float(pr.masses[i]), float(pr.dampings[i]))
            else:
                _t0 = time.perf_counter()
                u_i, s_i, ok_i, r_i, V_i = self._solve_local_clf_hocbf_qp(
                    u_nom[i], lb[i], ub[i], float(pr.force_limits[i]), grad[i], v[i],
                    float(pr.masses[i]), float(pr.dampings[i]))
                # Record the constraint residuals SEPARATELY (audit source-integrity, 5.2). ``r_i`` returned by
                # the solver is the MAXIMUM of the box, ball and relaxed-CLF residuals, not a CLF
                # residual: at a fallback the slack is set to s0 = max(0, a(u0) + cV), so the
                # relaxed CLF row holds by construction and its residual is identically zero. Only
                # the box and ball residuals carry independent information there, so they are
                # logged as their own fields and the composite is named for what it is.
                _box = float(max(np.max(lb[i] - u_i), np.max(u_i - ub[i]), 0.0))
                _ball = float(max(np.linalg.norm(u_i) - float(pr.force_limits[i]), 0.0))
                trace.append((int(i), int(not ok_i), float(time.perf_counter() - _t0),
                              float(r_i), float(s_i), _box, _ball))
            out[i] = u_i
            slacks.append(s_i)
            Vs.append(V_i)
            residual = max(residual, r_i)
            success += int(ok_i)
        return out, float(np.max(slacks) if slacks else math.nan), float(np.sum(Vs)), float(residual), float(success / max(n, 1))

    def control(self, p: Array, v: Array, y_target: Array) -> tuple[Array, DSCEOSDiagnostics]:
        pr, cfg = self.problem, self.config
        # Explicit lifecycle flag rather than inferring initialisation from numeric content: a
        # legitimate all-zero estimator state (a fleet exactly at rest) is indistinguishable from an
        # uninitialised one, so the previous allclose test could re-seed a correctly running estimator.
        if not self.state.estimators_initialized:
            self.reset_estimators(p)
        self.update_estimators(p)
        grad = self.local_gradients(p, y_target)
        desired_acc = -cfg.optimizer_gain * grad - cfg.velocity_damping * v
        u_nom = pr.masses[:, None] * desired_acc + pr.dampings[:, None] * v
        lb, ub = self.hocbf_force_bounds(p, v)
        if cfg.use_local_qp:
            u, clf_slack, clf_value, proj_residual, qp_success = self.local_clf_hocbf_qp(u_nom, lb, ub, grad, v)
        elif cfg.actuator_constraint == "norm":
            # Joint projection onto the HOCBF box and actuator ball.
            u, proj_residual = project_rows_to_box_ball(u_nom, lb, ub, pr.force_limits)
            clf_slack = math.nan; clf_value = math.nan; qp_success = math.nan
        else:
            u = np.minimum(np.maximum(u_nom, lb), ub)
            proj_residual = 0.0; clf_slack = math.nan; clf_value = math.nan; qp_success = math.nan
        cap_violation = float(max(np.max(pr.lower - p), np.max(p - pr.upper), 0.0))
        cap_margin = np.minimum(p - pr.lower, pr.upper - p)
        diag = DSCEOSDiagnostics(
            aggregate_estimate_spread=float(np.max(row_norm(self.state.y_hat - np.mean(self.state.y_hat, axis=0, keepdims=True)))),
            mean_capacity_margin=float(np.mean(cap_margin)),
            force_margin=float(np.min(pr.force_limits - row_norm(u))),
            mean_gradient_norm=float(np.mean(row_norm(grad))),
            max_state_violation=cap_violation,
            hocbf_projection_residual=float(proj_residual),
            clf_value=float(clf_value),
            clf_slack=float(clf_slack),
            # NOTE: the local solver returns ONE composite residual, the maximum of the HOCBF-box,
            # actuator-ball and CLF-row residuals. It is therefore reported under its accurate name
            # below; the legacy `clf_residual` field is kept as an alias for backward compatibility
            # but must NOT be read as a CLF-specific quantity.
            max_qcqp_residual=float(proj_residual),
            clf_residual=float(proj_residual),
            qp_success_rate=float(qp_success),
        )
        return u, diag

__all__ = [
    'DSCEOSProblemData', 'DSCEOSConfig', 'DSCEOSState', 'DSCEOSDiagnostics',
    'DSCEOSMessage', 'DistributedSCEOSController', 'row_norm',
    'truncate_rows_by_norm', 'project_rows_to_box_ball',
]
