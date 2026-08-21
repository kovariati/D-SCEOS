"""
centralized_tracking_controller.py
===================================

Centralized-information PD tracking — a NON-OPERATIONAL upper benchmark
for the D-SCEOS comparison.

This controller assumes every agent KNOWS the global target y_T directly.
It implicitly requires a master-slave broadcast channel. In CPES practice
this is unrealistic and is precisely the architecture D-SCEOS replaces.

The controller is included here only as a measurement reference: it tells
us how well a controller could do under the centralized-information limit,
and therefore how much the peer-to-peer D-SCEOS gives up (or gains)
relative to that limit.
"""

from __future__ import annotations

import numpy as np

Array = np.ndarray


def centralized_tracking(
    p: Array,
    v: Array,
    target: Array,
    target_vel: Array | None,
    target_acc: Array | None,
    masses: Array,
    dampings: Array,
    force_limits: Array,
    kp: float = 1.25,
    kd: float = 1.6,
) -> Array:
    """Centralized PD tracking on the global target.

    Each agent uses the SAME global target ``y_T`` directly:

        a_i = target_acc + kp (target - p_i) + kd (target_vel - v_i)

    A non-operational benchmark only; this is the master-broadcast model.
    """
    n = p.shape[0]
    if target_vel is None:
        target_vel = np.zeros_like(target)
    if target_acc is None:
        target_acc = np.zeros_like(target)
    a = target_acc[None, :] + kp * (target[None, :] - p) + kd * (target_vel[None, :] - v)
    u = np.asarray(masses)[:, None] * a + np.asarray(dampings)[:, None] * v
    return _truncate_rows_by_norm(u, force_limits)


def _truncate_rows_by_norm(u: Array, limits: Array) -> Array:
    norms = np.linalg.norm(u, axis=1)
    factor = np.where(norms > limits, limits / np.maximum(norms, 1e-12), 1.0)
    return u * factor[:, None]
