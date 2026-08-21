"""
dsceos_validation.py
========================

Validation harness for the peer-to-peer D-SCEOS controller core. The controller
logic is in dsceos_controller.py; this file contains only benchmark setup,
integration, baselines and metrics. The communication graph is a fixed sparse
physical/communication-neighbour graph; it is not recomputed from the CPES
operating coordinates during the simulation.
"""
from __future__ import annotations

import argparse, csv, json, math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal
import numpy as np
try:
    from scipy.optimize import minimize
except Exception:  # pragma: no cover
    minimize = None

from dsceos_controller import (
    DSCEOSProblemData, DSCEOSConfig, DistributedSCEOSController,
    row_norm, truncate_rows_by_norm,
)
from decentralized_tracking_controllers import decentralized_independent_tracking, decentralized_coherent_tracking
from distributed_projected_gradient_controller import DistributedProjectedGradientController
from distributed_primal_dual_controller import DistributedPrimalDualController
from distributed_gradient_tracking_controller import DistributedGradientTrackingController
from hocbf_safety_filter import HOCBFSafetyFilter
from centralized_tracking_controller import centralized_tracking

Array = np.ndarray
ControllerName = Literal["dsceos", "coherent_tracking", "independent_tracking", "centralized_tracking", "projected_gradient_hocbf", "distributed_primal_dual_hocbf", "gradient_tracking_hocbf"]
# Single source of truth for the controllers this module can dispatch. ControllerName, the CLI
# choices and the dispatch in run_simulation are all derived from or checked against this.
CONTROLLER_REGISTRY = ("dsceos", "coherent_tracking", "independent_tracking", "centralized_tracking", "projected_gradient_hocbf", "distributed_primal_dual_hocbf", "gradient_tracking_hocbf")

ScenarioName = Literal["static_request", "moving_request", "step_request", "actuator_stress", "heterogeneous_cluster", "infeasible_request", "fault_tolerance"]

@dataclass(frozen=True)
class EnergyFlexibilityUnit:
    kind: str
    mass: float
    damping: float
    force_limit: float
    lower: tuple[float, float]
    upper: tuple[float, float]
    rest: tuple[float, float]
    loss_weight: tuple[float, float]
    aggregate_weight: tuple[float, float]
    service_capacity: float | None = None

@dataclass(frozen=True)
class ClusterConfig:
    n_thermal: int = 8
    n_storage: int = 6
    n_hydrogen: int = 5
    n_emobility: int = 5
    n_industrial: int = 6
    seed: int = 7
    # 0.0 = every unit starts at its nominal operating point with zero
    # velocity (the physical CPES pre-event state); matches the realistic
    # scenario protocol and removes a seeded random element.
    initial_spread: float = 0.0
    initial_speed_scale: float = 0.025
    communication_radius: float = 0.38
    neighbour_count: int = 4
    layout_spread: float = 1.0

@dataclass(frozen=True)
class SimulationConfig:
    cluster: ClusterConfig = ClusterConfig()
    controller: ControllerName = "dsceos"
    scenario: ScenarioName = "static_request"
    dsceos_config: DSCEOSConfig = DSCEOSConfig()
    dt: float = 0.04
    horizon: float = 20.0
    gateway_fraction: float = 0.15
    target_consensus_gain: float = 0.18
    target_gateway_gain: float = 0.85
    compute_reference_optimum: bool = False
    target_override: tuple[float, float] | None = None
    # When True, every baseline (coherent_tracking, independent_tracking,
    # centralized_tracking, projected_gradient_hocbf) is wrapped in a posthoc HOCBF + actuator safety
    # filter sharing the same HOCBF gains as D-SCEOS. This produces a
    # strictly apples-to-apples comparison in which every controller is
    # capacity-valid by construction. The D-SCEOS controller is unaffected
    # (its certificate already enforces capacity invariance).
    safety_filter: bool = False
    # Runtime options for the filter-masking experiment (defaults preserve
    # the main-run behaviour). dpg_filter_always=True keeps the DPG-HOCBF
    # comparator wrapped in the safety filter regardless of safety_filter;
    # set False to test DPG without the filter. containment_margin is the
    # numerical position-containment clip half-width (main runs use 0.05).
    dpg_filter_always: bool = True
    containment_margin: float = 0.05
    # Comparator tuning knobs. Defaults are the ORIGINAL nominal settings so that no run changes
    # behaviour unless it explicitly opts in. The source-integrity baseline retune passes the best-tested
    # settings from the N15/A sweep (reruns/baseline_tuning.json) -- DPG step 0.10, PD kp=0.75,
    # kd=2.5 -- explicitly in the main-table generators (batch_realistic, ladder_rerun and the
    # diagnostics that feed the published tables), leaving other experiments untouched.
    dpg_step_size: float = 0.06
    pd_kp: float = 1.25
    pd_kd: float = 1.6

@dataclass(frozen=True)
class TargetSignal:
    name: str
    position: callable
    velocity: callable
    acceleration: callable

@dataclass
class SimulationResult:
    time: Array
    positions: Array
    velocities: Array
    controls: Array
    metrics: dict[str, Array]
    summary: dict[str, float | int | str | bool]
    def save(self, outdir: str | Path) -> None:
        out = Path(outdir); out.mkdir(parents=True, exist_ok=True)
        with (out / "metrics.csv").open("w", newline="", encoding="utf-8") as f:
            keys = ["time"] + list(self.metrics)
            w = csv.writer(f); w.writerow(keys)
            for k, t in enumerate(self.time[:-1]):
                w.writerow([t] + [self.metrics[name][k] for name in self.metrics])
        with (out / "summary.json").open("w", encoding="utf-8") as f:
            json.dump(self.summary, f, indent=2, sort_keys=True)
        np.savez_compressed(out / "state_history.npz", time=self.time, positions=self.positions, velocities=self.velocities, controls=self.controls)


def make_units(cfg: ClusterConfig) -> list[EnergyFlexibilityUnit]:
    units: list[EnergyFlexibilityUnit] = []
    units += [EnergyFlexibilityUnit("thermal_storage_or_heat_pump_building_cluster", 1.60, 1.35, 0.34, (-0.55, -0.80), (0.55, 0.80), (0.00, 0.05), (2.20, 1.25), (0.85, 1.10)) for _ in range(cfg.n_thermal)]
    units += [EnergyFlexibilityUnit("electrochemical_or_hybrid_storage", 0.95, 1.10, 0.50, (-0.90, -0.75), (0.90, 0.75), (0.00, 0.00), (1.05, 1.15), (1.25, 0.95)) for _ in range(cfg.n_storage)]
    units += [EnergyFlexibilityUnit("hydrogen_or_power_to_x_buffer", 1.35, 0.95, 0.42, (-0.65, -0.90), (0.65, 0.90), (0.02, -0.03), (1.35, 0.90), (1.05, 1.20)) for _ in range(cfg.n_hydrogen)]
    units += [EnergyFlexibilityUnit("emobility_energy_hub", 0.85, 1.25, 0.46, (-0.75, -0.65), (0.75, 0.65), (0.00, 0.00), (1.25, 1.65), (1.15, 0.80)) for _ in range(cfg.n_emobility)]
    units += [EnergyFlexibilityUnit("flexible_industrial_process", 1.20, 0.85, 0.38, (-0.45, -0.70), (0.45, 0.70), (-0.02, 0.02), (2.60, 1.45), (0.75, 0.90)) for _ in range(cfg.n_industrial)]
    return units


def apply_heterogeneity(sim_cfg: SimulationConfig, units: list[EnergyFlexibilityUnit]) -> list[EnergyFlexibilityUnit]:
    if sim_cfg.scenario != "heterogeneous_cluster": return units
    rng = np.random.default_rng(sim_cfg.cluster.seed + 101)
    out = []
    for u in units:
        out.append(replace(u,
            mass=float(max(0.25, u.mass * rng.uniform(0.75, 1.35))),
            damping=float(max(0.15, u.damping * rng.uniform(0.75, 1.35))),
            force_limit=float(max(0.08, u.force_limit * rng.uniform(0.70, 1.25))),
            loss_weight=tuple(float(max(0.25, x * rng.uniform(0.75, 1.45))) for x in u.loss_weight),
            aggregate_weight=tuple(float(max(0.20, x * rng.uniform(0.80, 1.30))) for x in u.aggregate_weight)))
    return out


def arrays_from_units(units: list[EnergyFlexibilityUnit]) -> dict[str, Array]:
    lower = np.asarray([u.lower for u in units], dtype=float)
    upper = np.asarray([u.upper for u in units], dtype=float)
    inferred_cap = np.maximum(np.maximum(np.abs(lower[:, 0]), np.abs(upper[:, 0])), 1.0e-9)
    explicit_cap = np.asarray([np.nan if u.service_capacity is None else float(u.service_capacity) for u in units])
    service_capacity = np.where(np.isfinite(explicit_cap), explicit_cap, inferred_cap)
    return {
        "masses": np.asarray([u.mass for u in units], dtype=float),
        "dampings": np.asarray([u.damping for u in units], dtype=float),
        "force_limits": np.asarray([u.force_limit for u in units], dtype=float),
        "lower": lower,
        "upper": upper,
        "rest": np.asarray([u.rest for u in units], dtype=float),
        "loss_weight": np.asarray([u.loss_weight for u in units], dtype=float),
        # DEVIATION (released formulation): the aggregate is the NORMALIZED PHYSICAL FLEET SUM.
        # Uniform aggregate weights make A_i = I, so y = mean_i p_i and the DSO
        # un-scaling internal_to_dso(y) = sum_i p_i EXACTLY (physical GW delivery).
        # Capacity-awareness lives entirely in the utilization-sharing term (rho_i,
        # via service_capacity), not in the tracked aggregate.
        "aggregate_weight": np.ones((len(units), 2), dtype=float),
        "service_capacity": np.maximum(service_capacity, 1.0e-9),
    }


def make_aggregate_blocks(aggregate_weight: Array) -> Array:
    """Build local output blocks for normalized aggregate-service consensus.

    The D-SCEOS controller estimates the aggregate through dynamic average
    consensus. Therefore the local output blocks are scaled so that

        mean_i A_i p_i = sum_i (w_i / sum_j w_j) p_i,

    With the released formulation's uniform aggregate weights (A_i = I) this is the plain
    NORMALIZED PHYSICAL FLEET SUM: mean_i p_i, whose DSO un-scaling recovers
    sum_i p_i exactly (see compute_fleet_scaling / FleetScaling.internal_to_dso).
    Capacity heterogeneity is NOT carried here but by the utilization term rho_i.
    Keeping the mean form keeps the online estimator peer-to-peer: agents average
    local messages rather than sending them to a summing master node.
    """
    w = np.asarray(aggregate_weight, dtype=float)
    n, dim = w.shape
    denom = np.maximum(np.sum(w, axis=0), 1.0e-12)
    Ablk = np.zeros((n, dim, dim), dtype=float)
    for i in range(n):
        for k in range(dim):
            Ablk[i, k, k] = n * w[i, k] / denom[k]
    return Ablk


def aggregate_output(Ablk: Array, p: Array) -> Array:
    return np.mean(np.einsum("imd,id->im", Ablk, p), axis=0)


def make_physical_layout(n: int, cfg: ClusterConfig) -> Array:
    """Generate fixed exogenous communication coordinates.

    These coordinates are not CPES operating states. They represent installation,
    communication, feeder, building-campus, district-energy or contractual
    proximity. The neighbour graph is computed once from this layout and remains
    fixed during the simulation.
    """
    rng = np.random.default_rng(cfg.seed + 404)
    # A mildly perturbed grid gives reproducible locality without creating an
    # all-to-all graph.
    side = int(math.ceil(math.sqrt(n)))
    coords = []
    for idx in range(n):
        r = idx // side
        c = idx % side
        base = np.array([c, r], dtype=float) / max(side - 1, 1)
        jitter = rng.normal(0.0, 0.035, size=2)
        coords.append(base + jitter)
    layout = np.asarray(coords, dtype=float) * cfg.layout_spread
    return layout


def make_fixed_local_graph(layout: Array, radius: float, neighbour_count: int) -> Array:
    """Build a fixed sparse proximity graph from exogenous layout coordinates.

    Edges connect geographically or communicationally nearby units only. The
    graph is symmetrized and repaired with nearest-neighbour links if necessary
    so that the validation benchmark is connected, but it is never recomputed
    from the evolving operating-point coordinates ``p``.
    """
    layout = np.asarray(layout, dtype=float)
    n = layout.shape[0]
    W = np.zeros((n, n), dtype=float)
    if n <= 1:
        return W
    radius = max(float(radius), 1.0e-12)
    k = max(1, min(int(neighbour_count), n - 1))
    D = np.linalg.norm(layout[:, None, :] - layout[None, :, :], axis=2)
    for i in range(n):
        order = np.argsort(D[i])
        selected = [j for j in order if j != i and D[i, j] <= radius][:k]
        if not selected:
            selected = [int(order[1])]  # nearest neighbour repair
        for j in selected:
            w = math.exp(-(float(D[i, j]) / radius) ** 2)
            W[i, j] = max(W[i, j], w)
            W[j, i] = max(W[j, i], w)
    # connectivity repair by adding nearest links between components.
    while True:
        seen = np.zeros(n, dtype=bool)
        comps = []
        for start in range(n):
            if seen[start]:
                continue
            stack = [start]
            seen[start] = True
            comp = []
            while stack:
                a = stack.pop(); comp.append(a)
                for b in np.flatnonzero(W[a] > 0):
                    if not seen[b]:
                        seen[b] = True; stack.append(int(b))
            comps.append(comp)
        if len(comps) <= 1:
            break
        c0 = comps[0]
        best = None
        for a in c0:
            for comp in comps[1:]:
                for b in comp:
                    d = D[a, b]
                    if best is None or d < best[0]:
                        best = (d, a, b)
        _, a, b = best
        w = math.exp(-(float(D[a, b]) / radius) ** 2)
        W[a, b] = W[b, a] = max(w, 0.05)
    return W


def sample_initial_state(sim_cfg: SimulationConfig, arrays: dict[str, Array]) -> tuple[Array, Array]:
    """Sample the initial unit state for a simulation.

    Position randomisation follows an *inverse-capacity* rule: every
    unit is sampled from a Gaussian centred at its rest point with
    standard deviation
        σ_i = initial_spread × (capacity_box_i × P_max_min / P_max_i)
    so that large-capacity (typically slow base-load) units start
    close to rest, while small-capacity (typically fast peaker / DR)
    units have a wider initial spread. This reflects the physically
    plausible CPES baseline in which base-load is sitting at its
    nominal operating point and fast resources are already mildly
    perturbed by ambient stochastic effects.

    Setting initial_spread = 0 collapses every unit exactly onto its
    rest point (the rest-start initial condition).

    Velocities follow the same inverse-capacity rule.
    """
    rng = np.random.default_rng(sim_cfg.cluster.seed)
    lower, upper, rest = arrays["lower"], arrays["upper"], arrays["rest"]
    span = upper - lower  # (N, 2)
    spread = sim_cfg.cluster.initial_spread
    speed_scale = sim_cfg.cluster.initial_speed_scale

    if spread <= 0.0 and speed_scale <= 0.0:
        # Strict rest-start: every unit at its rest point with zero velocity.
        p = rest.copy()
        v = np.zeros_like(rest)
        return p, v

    # Inverse-capacity weighting: smaller units (small P_max ⇒ small span)
    # get a wider relative spread than large units.
    p_max = 0.5 * span[:, 0]                     # per-unit P_max (rest is centred)
    p_max_min = float(np.min(p_max[p_max > 0])) if np.any(p_max > 0) else 1.0
    inv_cap_weight = (p_max_min / np.maximum(p_max, 1e-9))[:, None]  # (N, 1), in [0, 1]

    # Per-unit Gaussian sample for position
    sigma_pos = spread * span * inv_cap_weight    # (N, 2)
    p = rest + rng.normal(0.0, 1.0, size=rest.shape) * sigma_pos
    # Keep inside the capacity box with a small safety margin
    p = np.minimum(np.maximum(p, lower + 0.05 * span), upper - 0.05 * span)

    # Per-unit Gaussian sample for velocity (zero-mean across the fleet
    # so that the initial aggregate velocity is approximately zero)
    sigma_vel = speed_scale * inv_cap_weight
    v = rng.normal(0.0, 1.0, size=p.shape) * sigma_vel
    v -= v.mean(axis=0, keepdims=True)

    if sim_cfg.scenario == "actuator_stress":
        v *= 1.8
    return p, v


def static_target(value=(0.30, 0.02)) -> TargetSignal:
    y = np.asarray(tuple(value), dtype=float); z = np.zeros_like(y)
    return TargetSignal("static_multi_energy_flexibility_request", lambda t: y.copy(), lambda t: z.copy(), lambda t: z.copy())


def moving_target(base=(0.18, 0.00), amplitude=(0.12, 0.05), omega=0.35) -> TargetSignal:
    base = np.asarray(base, dtype=float); amp = np.asarray(amplitude, dtype=float)
    def pos(t): return base + np.array([amp[0] * math.sin(omega * t), amp[1] * math.cos(omega * t)])
    def vel(t): return np.array([amp[0] * omega * math.cos(omega * t), -amp[1] * omega * math.sin(omega * t)])
    def acc(t): return np.array([-amp[0] * omega * omega * math.sin(omega * t), -amp[1] * omega * omega * math.cos(omega * t)])
    return TargetSignal("moving_multi_energy_request", pos, vel, acc)


def step_target() -> TargetSignal:
    z = np.zeros(2, dtype=float)
    def pos(t):
        if t < 0.35 * 20.0:
            return np.array([0.05, 0.00])
        if t < 0.70 * 20.0:
            return np.array([0.30, 0.02])
        return np.array([0.15, -0.08])
    return TargetSignal("stepwise_multi_energy_request", pos, lambda t: z.copy(), lambda t: z.copy())


def select_target(cfg: SimulationConfig) -> TargetSignal:
    if cfg.target_override is not None:
        return static_target(cfg.target_override)
    if cfg.scenario == "moving_request": return moving_target()
    if cfg.scenario == "actuator_stress": return moving_target(base=(0.20, -0.02), amplitude=(0.18, 0.07), omega=0.20)
    if cfg.scenario == "step_request": return step_target()
    if cfg.scenario == "infeasible_request": return static_target((0.85, 0.35))
    if cfg.scenario == "fault_tolerance": return static_target((0.25, 0.02))
    return static_target((0.30, 0.02))


def utilization(p, arrays):
    return p[:, 0] / arrays["service_capacity"]


def sharing_variance(p, arrays):
    r = utilization(p, arrays)
    return float(np.mean((r - np.mean(r)) ** 2))




def objective_terms(p: Array, arrays: dict[str, Array], W: Array, Ablk: Array, target: Array, cfg: DSCEOSConfig) -> dict[str, float]:
    """Compute the paper objective terms for diagnostics.

    This diagnostic is centralized only in the validation layer. It is not used
    by the D-SCEOS controller. It reports the same terms that appear in the
    article: aggregate tracking, local operating loss, capacity-normalized
    utilization sharing, and a graph-local internal counter-action proxy.
    """
    p = np.asarray(p, dtype=float)
    rest = arrays["rest"]
    loss_weight = arrays["loss_weight"]
    delta = p - rest
    y = aggregate_output(Ablk, p)
    # Size-consistent (extensive) tracking term: (N * w_bar / 2) * ||y - y_T||^2.
    agg = 0.5 * p.shape[0] * cfg.aggregate_tracking_weight * float(np.dot(y - target, y - target))
    local_loss = 0.5 * cfg.loss_weight_scale * float(np.sum(loss_weight * delta * delta))
    rho = utilization(p, arrays)
    # 0.5 * lambda * rho^T L rho = 0.25 * lambda * sum_ij w_ij (rho_i-rho_j)^2
    diff = rho[:, None] - rho[None, :]
    sharing = 0.25 * cfg.sharing_weight * float(np.sum(W * diff * diff))
    degree = np.sum(W, axis=1)
    neigh_avg = np.zeros_like(delta)
    mask = degree > 1.0e-12
    neigh_avg[mask] = (W[mask] @ delta) / degree[mask, None]
    internal_res = delta - neigh_avg
    # Released formulation: the internal counter-action penalty is the
    # SYMMETRIC edge-disagreement Laplacian quadratic form
    #     delta^T L delta = 0.5 * sum_ij w_ij ||delta_i - delta_j||^2,   L = D - W,
    # so that the controller's local term lam*(L delta)_i is its EXACT gradient.
    # The previous formulation used ||(I - D^{-1}W) delta||^2, whose exact gradient
    # is B^T B delta while the controller applied B delta -- a steady structural
    # bias. The reported "internal_proxy_norm" is now sqrt(delta^T L delta).
    Ldelta = degree[:, None] * delta - W @ delta
    internal_sq = float(np.sum(delta * Ldelta))
    internal = 0.5 * cfg.internal_weight * internal_sq
    return {
        "objective_value": agg + local_loss + sharing + internal,
        "objective_aggregate": agg,
        "objective_local_loss": local_loss,
        "objective_sharing": sharing,
        "objective_internal": internal,
        "internal_proxy_norm": math.sqrt(max(internal_sq, 0.0)),  # clamp: L is PSD, but delta^T L delta can round to -1e-18
    }


def solve_reference_optimum(p0: Array, arrays: dict[str, Array], W: Array, Ablk: Array, target: Array, cfg: DSCEOSConfig) -> tuple[float, Array | None, bool]:
    """Centralized diagnostic optimum for loss-gap reporting only.

    This function is not part of the decentralized controller. It computes a
    benchmark objective value used to report J(p)-J(p*) when requested.
    """
    if minimize is None:
        return math.nan, None, False
    lower = arrays["lower"].reshape(-1)
    upper = arrays["upper"].reshape(-1)
    x0 = np.minimum(np.maximum(p0.reshape(-1), lower), upper)
    shape = p0.shape
    def f(x):
        return objective_terms(x.reshape(shape), arrays, W, Ablk, target, cfg)["objective_value"]
    try:
        res = minimize(f, x0, method="L-BFGS-B", bounds=list(zip(lower, upper)), options={"maxiter": 150, "ftol": 1e-9})
        if res.success and np.isfinite(res.fun):
            return float(res.fun), res.x.reshape(shape), True
        return float(res.fun) if np.isfinite(res.fun) else math.nan, None, False
    except Exception:
        return math.nan, None, False

def capacity_violation(p, lower, upper):
    return float(max(np.max(lower - p), np.max(p - upper), 0.0))


def gateway_mask(n: int, fraction: float) -> Array:
    k = max(1, int(math.ceil(max(0.0, min(1.0, float(fraction))) * n)))
    idx = np.linspace(0, n - 1, k, dtype=int)
    mask = np.zeros(n, dtype=bool)
    mask[idx] = True
    return mask


def update_peer_estimate(est: Array, truth: Array, W: Array, gateways: Array, consensus_gain: float, gateway_gain: float) -> Array:
    est = np.asarray(est, dtype=float)
    truth = np.asarray(truth, dtype=float)
    if truth.ndim == 1:
        truth = truth[None, :]
    consensus = W @ est - np.sum(W, axis=1)[:, None] * est
    out = est + float(consensus_gain) * consensus
    if np.any(gateways):
        out[gateways] += float(gateway_gain) * (truth[0][None, :] - out[gateways])
    return out


def control_coherence(u, eps=1e-12):
    denom = float(np.sum(row_norm(u)))
    return 1.0 if denom <= eps else float(np.linalg.norm(np.sum(u, axis=0)) / denom)


def safe_nanmax(x: Array) -> float:
    x = np.asarray(x, dtype=float)
    return math.nan if x.size == 0 or np.all(np.isnan(x)) else float(np.nanmax(x))


def run_simulation(cfg: SimulationConfig) -> SimulationResult:
    if cfg.scenario == "fault_tolerance":
        return run_fault_tolerance_simulation(cfg)
    units = apply_heterogeneity(cfg, make_units(cfg.cluster))
    arrays = arrays_from_units(units)
    if cfg.scenario == "actuator_stress": arrays["force_limits"] = 0.60 * arrays["force_limits"]
    p, v = sample_initial_state(cfg, arrays)
    layout = make_physical_layout(len(units), cfg.cluster)
    W = make_fixed_local_graph(layout, cfg.cluster.communication_radius, cfg.cluster.neighbour_count)
    Ablk = make_aggregate_blocks(arrays["aggregate_weight"])
    problem = DSCEOSProblemData(
        aggregate_blocks=Ablk, lower=arrays["lower"], upper=arrays["upper"], rest=arrays["rest"], loss_weight=arrays["loss_weight"],
        masses=arrays["masses"], dampings=arrays["dampings"], force_limits=arrays["force_limits"], service_selector=np.tile(np.array([[1.0, 0.0]]), (len(units), 1)),
        service_capacity=arrays["service_capacity"], adjacency=W)
    dsceos = DistributedSCEOSController(problem, cfg.dsceos_config)
    dpg = DistributedProjectedGradientController(problem, cfg.dsceos_config, step_size=cfg.dpg_step_size)
    dpd = DistributedPrimalDualController(problem, cfg.dsceos_config)
    dgt = DistributedGradientTrackingController(problem, cfg.dsceos_config, step_size=cfg.dpg_step_size)
    safety = HOCBFSafetyFilter(problem, cfg.dsceos_config) if (cfg.safety_filter or (cfg.dpg_filter_always and cfg.controller in ("projected_gradient_hocbf", "distributed_primal_dual_hocbf", "gradient_tracking_hocbf"))) else None
    target = select_target(cfg)
    steps = int(round(cfg.horizon / cfg.dt)); n, dim = p.shape
    gateways = gateway_mask(n, cfg.gateway_fraction)
    y0 = target.position(0.0); yd0 = target.velocity(0.0); ydd0 = target.acceleration(0.0)
    # Information model: a non-gateway agent has NOT yet received the supervisory request at t=0 and
    # cannot evaluate any fleet-wide quantity, so its target estimate is initialised to the local
    # "no information yet" prior (zero requested deviation) -- the same local prior already used for the
    # velocity and acceleration estimates below. Only gateway agents are seeded with the true request.
    # (An earlier version seeded every agent with the true aggregate A p(0); that is a fleet-wide
    # quantity and is not locally computable once the initial state is not exactly at rest.)
    target_est = np.zeros((n, Ablk.shape[1]), dtype=float)
    target_vel_est = np.zeros_like(target_est)
    target_acc_est = np.zeros_like(target_est)
    target_est[gateways] = y0
    target_vel_est[gateways] = yd0
    target_acc_est[gateways] = ydd0
    time = np.arange(steps + 1) * cfg.dt
    P = np.zeros((steps + 1, n, dim)); Vv = np.zeros_like(P); U = np.zeros((steps, n, dim))
    P[0], Vv[0] = p, v
    keys = ["aggregate_error", "sharing_variance", "control_power", "cumulative_control_energy", "control_coherence", "capacity_violation", "force_margin", "estimator_spread", "target_estimate_spread", "gradient_norm", "hocbf_projection_residual", "clf_value", "clf_slack", "clf_residual", "local_qp_success_rate", "objective_value", "objective_aggregate", "objective_local_loss", "objective_sharing", "objective_internal", "internal_proxy_norm", "reference_objective", "loss_gap"]
    metrics = {k: [] for k in keys}
    cumulative_energy = 0.0
    for k in range(steps):
        t = float(time[k]); y = target.position(t); yd = target.velocity(t); ydd = target.acceleration(t)
        target_est = update_peer_estimate(target_est, y, W, gateways, cfg.target_consensus_gain, cfg.target_gateway_gain)
        target_vel_est = update_peer_estimate(target_vel_est, yd, W, gateways, cfg.target_consensus_gain, cfg.target_gateway_gain)
        target_acc_est = update_peer_estimate(target_acc_est, ydd, W, gateways, cfg.target_consensus_gain, cfg.target_gateway_gain)
        target_spread = float(np.max(row_norm(target_est - np.mean(target_est, axis=0, keepdims=True))))
        clf_value = math.nan; clf_slack = math.nan; clf_residual = math.nan; qp_rate = math.nan
        if cfg.controller == "dsceos":
            u, diag = dsceos.control(p, v, target_est)
            est_spread = diag.aggregate_estimate_spread; grad_norm = diag.mean_gradient_norm; force_margin = diag.force_margin; proj_resid = diag.hocbf_projection_residual; clf_value = diag.clf_value; clf_slack = diag.clf_slack; clf_residual = diag.clf_residual; qp_rate = diag.qp_success_rate
        elif cfg.controller == "coherent_tracking":
            u = decentralized_coherent_tracking(p, v, target_est, arrays["masses"], arrays["dampings"], arrays["force_limits"], adjacency=W)
            if safety is not None: u, proj_resid = safety.filter(u, p, v)
            else: proj_resid = math.nan
            est_spread = math.nan; grad_norm = math.nan; force_margin = float(np.min(arrays["force_limits"] - row_norm(u)))
        elif cfg.controller == "independent_tracking":
            u = decentralized_independent_tracking(p, v, target_est, arrays["masses"], arrays["dampings"], arrays["force_limits"], kp=cfg.pd_kp, kd=cfg.pd_kd)
            if safety is not None: u, proj_resid = safety.filter(u, p, v)
            else: proj_resid = math.nan
            est_spread = math.nan; grad_norm = math.nan; force_margin = float(np.min(arrays["force_limits"] - row_norm(u)))
        elif cfg.controller == "projected_gradient_hocbf":
            u_nom, dpg_diag = dpg.control(p, v, target_est)
            if safety is not None:
                u, proj_resid = safety.filter(u_nom, p, v)
            else:
                u, proj_resid = u_nom, math.nan
            est_spread = dpg_diag.estimate_spread
            grad_norm = dpg_diag.mean_gradient_norm
            force_margin = float(np.min(arrays["force_limits"] - row_norm(u)))
        elif cfg.controller == "gradient_tracking_hocbf":
            u_nom, gt_diag = dgt.control(p, v, target_est)
            if safety is not None:
                u, proj_resid = safety.filter(u_nom, p, v)
            else:
                u, proj_resid = u_nom, math.nan
            est_spread = gt_diag.estimate_spread
            grad_norm = gt_diag.mean_gradient_norm
            force_margin = float(np.min(arrays["force_limits"] - row_norm(u)))
        elif cfg.controller == "distributed_primal_dual_hocbf":
            u_nom, dpd_diag = dpd.control(p, v, target_est)
            if safety is not None:
                u, proj_resid = safety.filter(u_nom, p, v)
            else:
                u, proj_resid = u_nom, math.nan
            est_spread = dpd_diag.estimate_spread
            grad_norm = dpd_diag.mean_gradient_norm
            force_margin = float(np.min(arrays["force_limits"] - row_norm(u)))
        elif cfg.controller == "centralized_tracking":
            u = centralized_tracking(p, v, y, yd, ydd, arrays["masses"], arrays["dampings"], arrays["force_limits"])
            if safety is not None: u, proj_resid = safety.filter(u, p, v)
            else: proj_resid = math.nan
            est_spread = math.nan; grad_norm = math.nan; force_margin = float(np.min(arrays["force_limits"] - row_norm(u)))
        else:
            raise ValueError(f"Unknown controller: {cfg.controller}")
        acc = (u - arrays["dampings"][:, None] * v) / arrays["masses"][:, None]
        v = v + cfg.dt * acc
        p = p + cfg.dt * v
        raw_violation = capacity_violation(p, arrays["lower"], arrays["upper"])
        # Validation harness clips only for numerical containment, not as a hidden controller certificate.
        p = np.minimum(np.maximum(p, arrays["lower"] - cfg.containment_margin), arrays["upper"] + cfg.containment_margin)
        P[k + 1], Vv[k + 1], U[k] = p, v, u
        y_actual = aggregate_output(Ablk, p)
        terms = objective_terms(p, arrays, W, Ablk, y, cfg.dsceos_config)
        if cfg.compute_reference_optimum:
            ref_obj, _, ref_ok = solve_reference_optimum(p, arrays, W, Ablk, y, cfg.dsceos_config)
        else:
            ref_obj, ref_ok = math.nan, False
        loss_gap = terms["objective_value"] - ref_obj if ref_ok and np.isfinite(ref_obj) else math.nan
        power = float(np.sum(u * u)); cumulative_energy += cfg.dt * power
        metrics["aggregate_error"].append(float(np.linalg.norm(y_actual - y)))
        metrics["sharing_variance"].append(sharing_variance(p, arrays))
        metrics["control_power"].append(power)
        metrics["cumulative_control_energy"].append(cumulative_energy)
        metrics["control_coherence"].append(control_coherence(u))
        metrics["capacity_violation"].append(raw_violation)
        metrics["force_margin"].append(force_margin)
        metrics["estimator_spread"].append(est_spread)
        metrics["target_estimate_spread"].append(target_spread)
        metrics["gradient_norm"].append(grad_norm)
        metrics["hocbf_projection_residual"].append(proj_resid)
        metrics["clf_value"].append(clf_value)
        metrics["clf_slack"].append(clf_slack)
        metrics["clf_residual"].append(clf_residual)
        metrics["local_qp_success_rate"].append(qp_rate)
        for _name in ["objective_value", "objective_aggregate", "objective_local_loss", "objective_sharing", "objective_internal", "internal_proxy_norm"]:
            metrics[_name].append(terms[_name])
        metrics["reference_objective"].append(ref_obj)
        metrics["loss_gap"].append(loss_gap)
    metric_arrays = {k: np.asarray(v, dtype=float) for k, v in metrics.items()}
    summary = {
        "controller": cfg.controller, "scenario": cfg.scenario, "n_agents": int(n), "dt": float(cfg.dt), "horizon": float(cfg.horizon),
        "communication_graph": "fixed_exogenous_proximity",
        "communication_radius": float(cfg.cluster.communication_radius),
        "neighbour_count_design": int(cfg.cluster.neighbour_count),
        "aggregate_tracking_weight": float(cfg.dsceos_config.aggregate_tracking_weight),
        "loss_weight_scale": float(cfg.dsceos_config.loss_weight_scale),
        "sharing_weight": float(cfg.dsceos_config.sharing_weight),
        "internal_weight": float(cfg.dsceos_config.internal_weight),
        "aggregate_consensus_gain": float(cfg.dsceos_config.aggregate_consensus_gain),
        "adaptive_consensus_gain": bool(cfg.dsceos_config.adaptive_consensus_gain),
        "active_aggregate_consensus_gain": float(dsceos.state.active_aggregate_consensus_gain),
        "gershgorin_lambda_N_bound": float(dsceos.state.gershgorin_lambda_N_bound),
        "gershgorin_max_weighted_degree": float(dsceos.state.gershgorin_max_weighted_degree),
        "gershgorin_safety_factor": float(cfg.dsceos_config.gershgorin_safety_factor),
        "graph_edges": int(np.count_nonzero(np.triu(W > 0, 1))),
        "graph_density": float(np.count_nonzero(np.triu(W > 0, 1)) / max(n * (n - 1) / 2, 1)),
        "graph_min_degree": int(np.min(np.count_nonzero(W > 0, axis=1))) if n else 0,
        "gateway_count": int(np.sum(gateways)),
        "target_information_pattern": "fixed_graph_gateway_consensus",
        "final_aggregate_error": float(metric_arrays["aggregate_error"][-1]),
        "mean_sharing_variance": float(np.mean(metric_arrays["sharing_variance"])),
        "total_control_energy": float(metric_arrays["cumulative_control_energy"][-1]),
        "max_capacity_violation": float(np.max(metric_arrays["capacity_violation"])),
        "energy_comparison_valid": bool(float(np.max(metric_arrays["capacity_violation"])) <= 1e-6),
        "min_force_margin": float(np.nanmin(metric_arrays["force_margin"])),
        "max_hocbf_projection_residual": safe_nanmax(metric_arrays.get("hocbf_projection_residual", np.asarray([math.nan]))),
        "max_qcqp_hard_residual": safe_nanmax(metric_arrays.get("clf_residual", np.asarray([math.nan]))),
        # legacy alias; the value is the composite max(box, ball, CLF) residual, not CLF-only
        "max_clf_residual": safe_nanmax(metric_arrays.get("clf_residual", np.asarray([math.nan]))),
        "mean_clf_slack": float(np.nanmean(metric_arrays["clf_slack"])) if not np.all(np.isnan(metric_arrays["clf_slack"])) else math.nan,
        # PRIMARY-solver acceptance within tolerance, not a feasibility rate: after a fallback
        # the applied action may still be admissible. The complement is the fallback rate.
        "primary_solver_acceptance_rate": float(np.nanmean(metric_arrays["local_qp_success_rate"])) if not np.all(np.isnan(metric_arrays["local_qp_success_rate"])) else math.nan,
        "mean_local_qp_success_rate": float(np.nanmean(metric_arrays["local_qp_success_rate"])) if not np.all(np.isnan(metric_arrays["local_qp_success_rate"])) else math.nan,
        "final_objective_value": float(metric_arrays["objective_value"][-1]),
        "integrated_objective_value": float(cfg.dt * np.sum(metric_arrays["objective_value"])),
        "mean_objective_value": float(np.mean(metric_arrays["objective_value"])),
        "mean_objective_local_loss": float(np.mean(metric_arrays["objective_local_loss"])),
        "mean_objective_sharing": float(np.mean(metric_arrays["objective_sharing"])),
        "mean_objective_internal": float(np.mean(metric_arrays["objective_internal"])),
        "mean_objective_aggregate": float(np.mean(metric_arrays["objective_aggregate"])),
        "final_loss_gap": float(metric_arrays["loss_gap"][-1]) if np.isfinite(metric_arrays["loss_gap"][-1]) else math.nan,
        "mean_internal_proxy_norm": float(np.mean(metric_arrays["internal_proxy_norm"])),
    }
    return SimulationResult(time, P, Vv, U, metric_arrays, summary)



def run_fault_tolerance_simulation(cfg: SimulationConfig) -> SimulationResult:
    """Run a fixed-graph agent-loss scenario.

    The event removes one agent at mid-horizon and restarts the D-SCEOS
    controller on the remaining fixed communication graph. This is a validation
    harness feature only; it represents explicit CPES topology reconfiguration,
    not a state-dependent moving-neighbour update.
    """
    units0 = apply_heterogeneity(cfg, make_units(cfg.cluster))
    arrays0 = arrays_from_units(units0)
    p0, v0 = sample_initial_state(cfg, arrays0)
    layout0 = make_physical_layout(len(units0), cfg.cluster)
    active = np.arange(len(units0), dtype=int)
    p, v = p0.copy(), v0.copy()
    target = select_target(cfg)
    steps = int(round(cfg.horizon / cfg.dt)); n0, dim = p0.shape
    time = np.arange(steps + 1) * cfg.dt
    P = np.full((steps + 1, n0, dim), np.nan); Vv = np.full_like(P, np.nan); U = np.full((steps, n0, dim), np.nan)
    P[0, active], Vv[0, active] = p, v

    def build(active_idx, p_current):
        units = [units0[int(i)] for i in active_idx]
        arrays = arrays_from_units(units)
        if cfg.scenario == "actuator_stress": arrays["force_limits"] = 0.60 * arrays["force_limits"]
        layout = layout0[active_idx]
        W = make_fixed_local_graph(layout, cfg.cluster.communication_radius, cfg.cluster.neighbour_count)
        Ablk = make_aggregate_blocks(arrays["aggregate_weight"])
        problem = DSCEOSProblemData(
            aggregate_blocks=Ablk, lower=arrays["lower"], upper=arrays["upper"], rest=arrays["rest"], loss_weight=arrays["loss_weight"],
            masses=arrays["masses"], dampings=arrays["dampings"], force_limits=arrays["force_limits"],
            service_selector=np.tile(np.array([[1.0, 0.0]]), (len(units), 1)),
            service_capacity=arrays["service_capacity"], adjacency=W)
        ctl = DistributedSCEOSController(problem, cfg.dsceos_config)
        ctl.reset_estimators(p_current)
        dpg_ctl = DistributedProjectedGradientController(problem, cfg.dsceos_config, step_size=cfg.dpg_step_size)
        dpd_ctl = DistributedPrimalDualController(problem, cfg.dsceos_config)
        dpg_ctl.reset(p_current)
        saf = HOCBFSafetyFilter(problem, cfg.dsceos_config) if (cfg.safety_filter or (cfg.dpg_filter_always and cfg.controller in ("projected_gradient_hocbf", "distributed_primal_dual_hocbf", "gradient_tracking_hocbf"))) else None
        return units, arrays, W, Ablk, ctl, dpg_ctl, dpd_ctl, saf

    units, arrays, W, Ablk, dsceos, dpg, dpd, safety = build(active, p)
    gateways = gateway_mask(len(active), cfg.gateway_fraction)
    # Local "no information yet" prior for non-gateway agents (see the information-model note in
    # run_simulation); only gateways are seeded with the true request.
    target_est = np.zeros((len(active), Ablk.shape[1]), dtype=float)
    target_vel_est = np.zeros_like(target_est)
    target_acc_est = np.zeros_like(target_est)
    target_est[gateways] = target.position(0.0)
    target_vel_est[gateways] = target.velocity(0.0)
    target_acc_est[gateways] = target.acceleration(0.0)
    keys = ["aggregate_error", "sharing_variance", "control_power", "cumulative_control_energy", "control_coherence", "capacity_violation", "force_margin", "estimator_spread", "target_estimate_spread", "gradient_norm", "hocbf_projection_residual", "clf_value", "clf_slack", "clf_residual", "local_qp_success_rate", "objective_value", "objective_aggregate", "objective_local_loss", "objective_sharing", "objective_internal", "internal_proxy_norm", "reference_objective", "loss_gap", "active_agents"]
    metrics = {k: [] for k in keys}
    cumulative_energy = 0.0; dropped_agent = -1; drop_time = math.nan
    for k in range(steps):
        t = float(time[k])
        if dropped_agent < 0 and k >= steps // 2 and len(active) > 3:
            local_drop = len(active) // 2
            dropped_agent = int(active[local_drop]); drop_time = t
            keep = np.ones(len(active), dtype=bool); keep[local_drop] = False
            active = active[keep]; p = p[keep]; v = v[keep]
            units, arrays, W, Ablk, dsceos, dpg, dpd, safety = build(active, p)
            gateways = gateway_mask(len(active), cfg.gateway_fraction)
            # On reconfiguration the surviving agents restart the target estimate from the same local
            # prior; re-seeding it with the reduced fleet's true aggregate would require a fleet-wide
            # measurement mid-run, which the peer-to-peer information model does not provide.
            target_est = np.zeros((len(active), Ablk.shape[1]), dtype=float)
            target_vel_est = np.zeros_like(target_est)
            target_acc_est = np.zeros_like(target_est)
            target_est[gateways] = target.position(t)
            target_vel_est[gateways] = target.velocity(t)
            target_acc_est[gateways] = target.acceleration(t)
        y = target.position(t); yd = target.velocity(t); ydd = target.acceleration(t)
        target_est = update_peer_estimate(target_est, y, W, gateways, cfg.target_consensus_gain, cfg.target_gateway_gain)
        target_vel_est = update_peer_estimate(target_vel_est, yd, W, gateways, cfg.target_consensus_gain, cfg.target_gateway_gain)
        target_acc_est = update_peer_estimate(target_acc_est, ydd, W, gateways, cfg.target_consensus_gain, cfg.target_gateway_gain)
        target_spread = float(np.max(row_norm(target_est - np.mean(target_est, axis=0, keepdims=True))))
        clf_value = math.nan; clf_slack = math.nan; clf_residual = math.nan; qp_rate = math.nan
        if cfg.controller == "dsceos":
            u, diag = dsceos.control(p, v, target_est)
            est_spread = diag.aggregate_estimate_spread; grad_norm = diag.mean_gradient_norm; force_margin = diag.force_margin; proj_resid = diag.hocbf_projection_residual; clf_value = diag.clf_value; clf_slack = diag.clf_slack; clf_residual = diag.clf_residual; qp_rate = diag.qp_success_rate
        elif cfg.controller == "coherent_tracking":
            u = decentralized_coherent_tracking(p, v, target_est, arrays["masses"], arrays["dampings"], arrays["force_limits"], adjacency=W)
            if safety is not None: u, proj_resid = safety.filter(u, p, v)
            else: proj_resid = math.nan
            est_spread = math.nan; grad_norm = math.nan; force_margin = float(np.min(arrays["force_limits"] - row_norm(u)))
        elif cfg.controller == "independent_tracking":
            u = decentralized_independent_tracking(p, v, target_est, arrays["masses"], arrays["dampings"], arrays["force_limits"], kp=cfg.pd_kp, kd=cfg.pd_kd)
            if safety is not None: u, proj_resid = safety.filter(u, p, v)
            else: proj_resid = math.nan
            est_spread = math.nan; grad_norm = math.nan; force_margin = float(np.min(arrays["force_limits"] - row_norm(u)))
        elif cfg.controller == "projected_gradient_hocbf":
            u_nom, dpg_diag = dpg.control(p, v, target_est)
            if safety is not None:
                u, proj_resid = safety.filter(u_nom, p, v)
            else:
                u, proj_resid = u_nom, math.nan
            est_spread = dpg_diag.estimate_spread
            grad_norm = dpg_diag.mean_gradient_norm
            force_margin = float(np.min(arrays["force_limits"] - row_norm(u)))
        elif cfg.controller == "distributed_primal_dual_hocbf":
            u_nom, dpd_diag = dpd.control(p, v, target_est)
            if safety is not None:
                u, proj_resid = safety.filter(u_nom, p, v)
            else:
                u, proj_resid = u_nom, math.nan
            est_spread = dpd_diag.estimate_spread
            grad_norm = dpd_diag.mean_gradient_norm
            force_margin = float(np.min(arrays["force_limits"] - row_norm(u)))
        elif cfg.controller == "centralized_tracking":
            u = centralized_tracking(p, v, y, yd, ydd, arrays["masses"], arrays["dampings"], arrays["force_limits"])
            if safety is not None: u, proj_resid = safety.filter(u, p, v)
            else: proj_resid = math.nan
            est_spread = math.nan; grad_norm = math.nan; force_margin = float(np.min(arrays["force_limits"] - row_norm(u)))
        else:
            raise ValueError(f"Unknown controller: {cfg.controller}")
        acc = (u - arrays["dampings"][:, None] * v) / arrays["masses"][:, None]
        v = v + cfg.dt * acc; p = p + cfg.dt * v
        raw_violation = capacity_violation(p, arrays["lower"], arrays["upper"])
        p = np.minimum(np.maximum(p, arrays["lower"] - cfg.containment_margin), arrays["upper"] + cfg.containment_margin)
        P[k + 1, active], Vv[k + 1, active], U[k, active] = p, v, u
        y_actual = aggregate_output(Ablk, p)
        terms = objective_terms(p, arrays, W, Ablk, y, cfg.dsceos_config)
        if cfg.compute_reference_optimum:
            ref_obj, _, ref_ok = solve_reference_optimum(p, arrays, W, Ablk, y, cfg.dsceos_config)
        else:
            ref_obj, ref_ok = math.nan, False
        loss_gap = terms["objective_value"] - ref_obj if ref_ok and np.isfinite(ref_obj) else math.nan
        power = float(np.sum(u * u)); cumulative_energy += cfg.dt * power
        metrics["aggregate_error"].append(float(np.linalg.norm(y_actual - y)))
        metrics["sharing_variance"].append(sharing_variance(p, arrays))
        metrics["control_power"].append(power)
        metrics["cumulative_control_energy"].append(cumulative_energy)
        metrics["control_coherence"].append(control_coherence(u))
        metrics["capacity_violation"].append(raw_violation)
        metrics["force_margin"].append(force_margin)
        metrics["estimator_spread"].append(est_spread)
        metrics["target_estimate_spread"].append(target_spread)
        metrics["gradient_norm"].append(grad_norm)
        metrics["hocbf_projection_residual"].append(proj_resid)
        metrics["clf_value"].append(clf_value)
        metrics["clf_slack"].append(clf_slack)
        metrics["clf_residual"].append(clf_residual)
        metrics["local_qp_success_rate"].append(qp_rate)
        for _name in ["objective_value", "objective_aggregate", "objective_local_loss", "objective_sharing", "objective_internal", "internal_proxy_norm"]:
            metrics[_name].append(terms[_name])
        metrics["reference_objective"].append(ref_obj)
        metrics["loss_gap"].append(loss_gap)
        metrics["active_agents"].append(float(len(active)))
    metric_arrays = {k: np.asarray(v, dtype=float) for k, v in metrics.items()}
    summary = {
        "controller": cfg.controller, "scenario": cfg.scenario, "n_agents_initial": int(n0), "n_agents_final": int(len(active)), "dt": float(cfg.dt), "horizon": float(cfg.horizon),
        "communication_graph": "fixed_exogenous_proximity_reconfigured_after_agent_loss",
        "dropped_agent": int(dropped_agent), "drop_time": float(drop_time),
        "graph_edges_final": int(np.count_nonzero(np.triu(W > 0, 1))),
        "graph_min_degree_final": int(np.min(np.count_nonzero(W > 0, axis=1))) if len(active) else 0,
        "gateway_count_final": int(np.sum(gateways)),
        "target_information_pattern": "fixed_graph_gateway_consensus",
        "aggregate_tracking_weight": float(cfg.dsceos_config.aggregate_tracking_weight),
        "loss_weight_scale": float(cfg.dsceos_config.loss_weight_scale),
        "adaptive_consensus_gain": bool(cfg.dsceos_config.adaptive_consensus_gain),
        "active_aggregate_consensus_gain": float(dsceos.state.active_aggregate_consensus_gain),
        "gershgorin_lambda_N_bound": float(dsceos.state.gershgorin_lambda_N_bound),
        "gershgorin_max_weighted_degree": float(dsceos.state.gershgorin_max_weighted_degree),
        "gershgorin_safety_factor": float(cfg.dsceos_config.gershgorin_safety_factor),
        "final_aggregate_error": float(metric_arrays["aggregate_error"][-1]),
        "mean_sharing_variance": float(np.mean(metric_arrays["sharing_variance"])),
        "total_control_energy": float(metric_arrays["cumulative_control_energy"][-1]),
        "max_capacity_violation": float(np.max(metric_arrays["capacity_violation"])),
        "energy_comparison_valid": bool(float(np.max(metric_arrays["capacity_violation"])) <= 1e-6),
        "min_force_margin": float(np.nanmin(metric_arrays["force_margin"])),
        "max_hocbf_projection_residual": safe_nanmax(metric_arrays.get("hocbf_projection_residual", np.asarray([math.nan]))),
        "max_qcqp_hard_residual": safe_nanmax(metric_arrays.get("clf_residual", np.asarray([math.nan]))),
        # legacy alias; the value is the composite max(box, ball, CLF) residual, not CLF-only
        "max_clf_residual": safe_nanmax(metric_arrays.get("clf_residual", np.asarray([math.nan]))),
        "mean_clf_slack": float(np.nanmean(metric_arrays["clf_slack"])) if not np.all(np.isnan(metric_arrays["clf_slack"])) else math.nan,
        # PRIMARY-solver acceptance within tolerance, not a feasibility rate: after a fallback
        # the applied action may still be admissible. The complement is the fallback rate.
        "primary_solver_acceptance_rate": float(np.nanmean(metric_arrays["local_qp_success_rate"])) if not np.all(np.isnan(metric_arrays["local_qp_success_rate"])) else math.nan,
        "mean_local_qp_success_rate": float(np.nanmean(metric_arrays["local_qp_success_rate"])) if not np.all(np.isnan(metric_arrays["local_qp_success_rate"])) else math.nan,
        "final_objective_value": float(metric_arrays["objective_value"][-1]),
        "integrated_objective_value": float(cfg.dt * np.sum(metric_arrays["objective_value"])),
        "mean_objective_value": float(np.mean(metric_arrays["objective_value"])),
        "mean_objective_local_loss": float(np.mean(metric_arrays["objective_local_loss"])),
        "mean_objective_sharing": float(np.mean(metric_arrays["objective_sharing"])),
        "mean_objective_internal": float(np.mean(metric_arrays["objective_internal"])),
        "mean_objective_aggregate": float(np.mean(metric_arrays["objective_aggregate"])),
        "final_loss_gap": float(metric_arrays["loss_gap"][-1]) if np.isfinite(metric_arrays["loss_gap"][-1]) else math.nan,
        "mean_internal_proxy_norm": float(np.mean(metric_arrays["internal_proxy_norm"])),
    }
    return SimulationResult(time, P, Vv, U, metric_arrays, summary)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run gateway-injected peer-to-peer D-SCEOS validation.")
    # Kept in sync with ControllerName and the run_simulation dispatch; a registry test asserts it.
    p.add_argument("--controller", choices=list(CONTROLLER_REGISTRY), default="dsceos")
    p.add_argument("--scenario", choices=["static_request", "moving_request", "step_request", "actuator_stress", "heterogeneous_cluster", "infeasible_request", "fault_tolerance"], default="static_request")
    p.add_argument("--n-scale", type=float, default=1.0)
    p.add_argument("--horizon", type=float, default=None)
    p.add_argument("--dt", type=float, default=0.04)
    p.add_argument("--outdir", type=Path, default=Path("dsceos_results"))
    # Default matches the published configuration (DSCEOSConfig.aggregate_tracking_weight = 2/3).
    # The historical 10.0 predates the size-consistent objective and now denotes a different
    # controller, so running this CLI with defaults no longer silently diverges from the paper.
    p.add_argument("--aggregate-tracking-weight", type=float, default=2.0 / 3.0)
    p.add_argument("--loss-weight-scale", type=float, default=0.03)
    p.add_argument("--aggregate-consensus-gain", type=float, default=0.10)
    p.add_argument("--sharing-weight", type=float, default=0.15)
    p.add_argument("--internal-weight", type=float, default=0.03)
    p.add_argument("--communication-radius", type=float, default=0.38, help="Radius in the fixed exogenous communication layout, not in operating-coordinate space.")
    p.add_argument("--neighbour-count", type=int, default=4, help="Maximum number of nearest local neighbours selected before symmetrization.")
    p.add_argument("--gateway-fraction", type=float, default=0.15, help="Fraction of agents with direct supervisory target access; others use peer consensus.")
    p.add_argument("--target-consensus-gain", type=float, default=0.18)
    p.add_argument("--target-gateway-gain", type=float, default=0.85)
    p.add_argument("--compute-reference-optimum", action="store_true", help="compute centralized diagnostic objective optimum J(p*) for loss-gap reporting")
    p.add_argument("--safety-filter", action="store_true", help="wrap baseline controllers in posthoc HOCBF+actuator safety filter (D-SCEOS unaffected) so all controllers are capacity-valid; use this for an apples-to-apples energy comparison")
    p.add_argument("--target-x", type=float, default=None, help="override static target first coordinate")
    p.add_argument("--target-y", type=float, default=None, help="override static target second coordinate")
    p.add_argument("--dpg-step-size", type=float, default=0.06, help="projected-gradient step size of the DPG-HOCBF comparator")
    p.add_argument("--dpg-substeps", type=int, default=1, help="optimizer substeps per sampling instant of DPG-HOCBF")
    p.add_argument("--initial-spread", type=float, default=0.0, help="capacity-scaled random initial spread; 0 = nominal pre-event state (default)")
    return p.parse_args()


def build_config(args: argparse.Namespace) -> SimulationConfig:
    scale = max(float(args.n_scale), 0.05)
    cluster = ClusterConfig(
        n_thermal=max(1, int(round(8 * scale))), n_storage=max(1, int(round(6 * scale))), n_hydrogen=max(1, int(round(5 * scale))),
        n_emobility=max(1, int(round(5 * scale))), n_industrial=max(1, int(round(6 * scale))),
        communication_radius=float(args.communication_radius),
        neighbour_count=int(args.neighbour_count),
        initial_spread=float(args.initial_spread),
    )
    if args.dpg_step_size != 0.06 or args.dpg_substeps != 1:
        global DistributedProjectedGradientController
        _Base = DistributedProjectedGradientController
        _step, _sub = args.dpg_step_size, args.dpg_substeps
        class _TunedDPG(_Base):
            def __init__(self, problem, cfg, **kw):
                kw.setdefault("step_size", _step)
                kw.setdefault("optimizer_substeps", _sub)
                super().__init__(problem, cfg, **kw)
        DistributedProjectedGradientController = _TunedDPG
    ccfg = DSCEOSConfig(aggregate_tracking_weight=args.aggregate_tracking_weight, loss_weight_scale=args.loss_weight_scale, aggregate_consensus_gain=args.aggregate_consensus_gain, sharing_weight=args.sharing_weight, internal_weight=args.internal_weight)
    horizon = float(args.horizon) if args.horizon is not None else (60.0 if args.scenario == "moving_request" else 20.0)
    target_override = None if args.target_x is None or args.target_y is None else (float(args.target_x), float(args.target_y))
    return SimulationConfig(cluster=cluster, controller=args.controller, scenario=args.scenario, dsceos_config=ccfg, horizon=horizon, dt=args.dt, gateway_fraction=float(args.gateway_fraction), target_consensus_gain=float(args.target_consensus_gain), target_gateway_gain=float(args.target_gateway_gain), compute_reference_optimum=bool(args.compute_reference_optimum), target_override=target_override, safety_filter=bool(args.safety_filter))


def main() -> None:
    args = parse_args(); cfg = build_config(args); res = run_simulation(cfg); res.save(args.outdir)
    print(json.dumps(res.summary, indent=2, sort_keys=True))

if __name__ == "__main__":
    main()
