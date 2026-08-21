"""
hocbf_safety_filter.py
======================

Posthoc HOCBF safety filter for baseline controllers.

Rationale (paper-level)
-----------------------
The D-SCEOS controller embeds the operating-envelope conditions in its local
CLF/HOCBF-QP. What this filter provides for the baselines is the same per-sample
property and no more: the applied input is admissible at the sampling instant. It does
NOT guarantee that the state stays inside the envelope between samples or at the next
sample -- the released Monte Carlo study records a small post-step capacity excursion of
order 1e-3 for the filtered PD baseline, which is exactly this distinction. The decentralized PD-style baselines
(``independent_tracking``, ``coherent_tracking``) do not provide this
property natively, so they routinely violate the operating envelopes
in borderline and infeasible regimes. The natural industrial practice is to
wrap a baseline controller in a Control-Barrier-Function safety filter, which
clips the requested control input to the largest HOCBF-consistent value
before sending it to the plant.

This module implements exactly that filter, reusing the same HOCBF gains
``alpha_0`` and ``alpha_1`` as the D-SCEOS controller, so that the safety
layer is identical across all controllers. The result is an apples-to-apples
energy comparison: every controller now respects ``cap_viol ~= 0``, and
the remaining differences in energy and aggregate-tracking error reflect
the actual nominal control strategy, not the absence of a safety net.

What this filter does
---------------------
Given a baseline's nominal force ``u_nom_i``, the filter solves the small
local QP

    u_i^* = argmin ||u_i - u_nom_i||^2
            s.t.   HOCBF box:  lb_i(p_i, v_i) <= u_i <= ub_i(p_i, v_i)
                   actuator:   ||u_i|| <= F_i

This is identical to the HOCBF + actuator constraints of the D-SCEOS QP,
but WITHOUT the CLF inequality and WITHOUT the optimization objective; the
baseline keeps its own logic for what u_nom should be.

The implementation uses the same closed-form per-row HOCBF-box-and-ball
projection as ``dsceos_controller.project_rows_to_box_ball``, so the filter
adds essentially zero computational overhead on top of the baseline call.
"""

from __future__ import annotations

import numpy as np

from dsceos_controller import (
    DSCEOSConfig,
    DSCEOSProblemData,
    project_rows_to_box_ball,
)

Array = np.ndarray


class HOCBFSafetyFilter:
    """Apply a posthoc HOCBF + actuator safety filter to any baseline force.

    The filter is parameterized by the same DSCEOSProblemData and DSCEOSConfig
    used by the D-SCEOS controller, so identical HOCBF gains are applied
    across all controllers in the validation harness.
    """

    def __init__(self, problem: DSCEOSProblemData, config: DSCEOSConfig | None = None) -> None:
        self.problem = problem
        self.config = config or DSCEOSConfig()

    def hocbf_force_bounds(self, p: Array, v: Array) -> tuple[Array, Array]:
        """Identical to DistributedSCEOSController.hocbf_force_bounds but
        without requiring a controller instance. We duplicate the small
        method here to keep the safety filter self-contained.
        """
        pr, cfg = self.problem, self.config
        m = pr.masses[:, None]
        d = pr.dampings[:, None]
        lb = d * v - m * (cfg.hocbf_alpha1 * v + cfg.hocbf_alpha0 * (p - pr.lower))
        ub = d * v + m * (cfg.hocbf_alpha0 * (pr.upper - p) - cfg.hocbf_alpha1 * v)
        return lb, ub

    def filter(self, u_nom: Array, p: Array, v: Array) -> tuple[Array, float]:
        """Project ``u_nom`` onto the HOCBF box intersected with the
        actuator ball. Returns (u_safe, residual) where ``residual`` is the
        maximum box/ball FEASIBILITY residual of the projected force across
        agents (how far ``u_safe`` lies outside the box-ball feasible set; zero
        when the intersection is non-empty). NOTE: this is the constraint
        feasibility residual, NOT the control modification ``||u_safe - u_nom||``
        (a different, generally larger quantity when a barrier is active).
        """
        lb, ub = self.hocbf_force_bounds(p, v)
        u_safe, projection_residual = project_rows_to_box_ball(
            np.asarray(u_nom, dtype=float),
            lb, ub,
            self.problem.force_limits,
        )
        return u_safe, float(projection_residual)
