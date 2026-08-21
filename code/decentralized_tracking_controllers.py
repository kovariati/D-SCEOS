"""
decentralized_tracking_controllers.py
======================================

Two simple decentralized PD-style tracking controllers as fair baselines for
D-SCEOS comparison. Both use a per-agent local target estimate obtained from
a fixed-graph dynamic average-consensus update (the same mechanism D-SCEOS
uses for its aggregate-output estimate).

This module is intentionally minimal:

* ``decentralized_independent_tracking``: each agent runs a PD loop on its
  own ``target_estimate_i - p_i``. No coordination between agents beyond the
  target estimate consensus.

* ``decentralized_coherent_tracking``: same PD plus a graph-Laplacian
  velocity-consensus term on the fixed CPES communication graph.

These are CENTRALIZED-INFORMATION-FREE: each agent only sees its own state,
the messages from its fixed CPES neighbours, and its own consensus-driven
estimate of y_T. There is no master node and no all-to-one aggregation.
The dynamic average-consensus update for the per-agent target estimate
itself is performed by the validation loop (function
``update_peer_estimate`` in ``dsceos_validation.py``), which also handles
gateway-injection of the supervisory uplink.
"""

from __future__ import annotations

import numpy as np

Array = np.ndarray


def decentralized_independent_tracking(
    p: Array,
    v: Array,
    target_estimate: Array,
    masses: Array,
    dampings: Array,
    force_limits: Array,
    kp: float = 1.25,
    kd: float = 1.6,
) -> Array:
    """Decentralized independent target tracking.

    Each agent runs the PD law

        a_i = kp (target_estimate_i - p_i) + kd (-v_i)

    using its locally maintained target estimate.

    Plant inversion: ``u_i = m_i a_i + μ_i v_i``, then ``||u_i|| <= F_i``.
    """
    a = kp * (target_estimate - p) + kd * (-v)
    u = np.asarray(masses)[:, None] * a + np.asarray(dampings)[:, None] * v
    return _truncate_rows_by_norm(u, force_limits)


def decentralized_coherent_tracking(
    p: Array,
    v: Array,
    target_estimate: Array,
    masses: Array,
    dampings: Array,
    force_limits: Array,
    adjacency: Array,
    kp: float = 1.25,
    kd: float = 1.6,
    velocity_consensus_gain: float = 0.15,
) -> Array:
    """Decentralized coherent target tracking.

    Decentralized independent tracking plus a graph-Laplacian velocity-
    consensus term on the fixed CPES communication graph:

        a_i = kp (target_estimate_i - p_i) + kd (-v_i)
              + velocity_consensus_gain * sum_j W_ij (v_j - v_i)

    The velocity consensus encourages the neighbours to move in a coordinated
    way, which can damp transient oscillations relative to the simpler
    decentralized independent baseline.
    """
    W = np.asarray(adjacency, dtype=float)
    deg = np.sum(W, axis=1)
    vel_consensus = W @ v - deg[:, None] * v
    a = kp * (target_estimate - p) + kd * (-v) + velocity_consensus_gain * vel_consensus
    u = np.asarray(masses)[:, None] * a + np.asarray(dampings)[:, None] * v
    return _truncate_rows_by_norm(u, force_limits)


def _truncate_rows_by_norm(u: Array, limits: Array) -> Array:
    norms = np.linalg.norm(u, axis=1)
    factor = np.where(norms > limits, limits / np.maximum(norms, 1e-12), 1.0)
    return u * factor[:, None]
