"""
realistic_cpes_catalog.py
==========================

Realistic-fictional CPES unit catalog for the application-example scenarios
described in Section 7 of the paper. Each unit is a 2-dimensional active +
reactive power flexibility provider, with parameters drawn from public
manufacturer datasheets at the order-of-magnitude level. The exact values
are illustrative — they are not the parameters of any specific product, but
they are within the plausible range for the listed equipment classes.

PHYSICAL INTERPRETATION OF THE OPERATING COORDINATE
----------------------------------------------------

  p_i ∈ R^2 = (P_i, Q_i)
  P_i = active-power deviation from the unit's rest operating point [GW]
  Q_i = reactive-power deviation from the unit's rest operating point [GVAR]

The DSO sends a single aggregate request y_T ∈ R^2 (one (GW, GVAR) value),
and the controllers must allocate it across the heterogeneous unit
population while respecting each unit's capability box and ramp limits.

UNIT-PARAMETER ↔ EnergyFlexibilityUnit MAPPING
-----------------------------------------------

The validation harness's EnergyFlexibilityUnit dataclass has six dynamical
parameters: mass, damping, force_limit, lower, upper, rest. These are
dimensionless in the original paper, so we use the following calibration:

  * lower / upper          = (P_min, Q_min), (P_max, Q_max)  [GW, GVAR]
                             directly, in physical units.
  * rest                   = baseline operating point relative to which the
                             flexibility deviation is measured.
  * mass m_i               = (τ_i / T63)²  with T63 ≈ 1.9991 the dimensionless
                             1/e response time of the ζ=0.9, k=1 damped second-
                             order plant. This places the OPEN-LOOP small-signal
                             1/e time constant of the reference unit-mass plant at
                             the documented tau_i; it is a reference inertia/damping
                             calibration proxy, NOT the nominal closed-loop response
                             time (the nominal closed loop is plant-inverted and
                             m_i,mu_i-independent, so tau_i sets only the relative
                             class ordering and the saturated ramp limit).
  * damping μ_i            = 2 ζ sqrt(m_i)  with ζ = 0.9 (near-critical, no
                             overshoot in CPES operation).
  * force_limit F_i        = μ_i * ramp_rate_per_minute, because the plant
                             has m_i vdot_i = u_i - μ_i v_i and therefore
                             the saturated steady ramp is v_ss = F_i/μ_i.
                             This maps sustained saturation to the documented
                             GW/min ramp limit.

LOSS AND REPORTED COST CALIBRATION
----------------------------------

  Internal controller loss:
      loss_C_i(p_i) = (1/2) * c_i * ||p_i||^2

  The internal loss is a smooth quadratic regularizer used by the CLF/QP
  layer. Its coefficients preserve a cost-proportional dispatch ordering but
  are not reported directly as monetary dispatch cost.

  The COST_NORMALIZATION constant is the c_n normaliser of the local
  loss term in the controller objective (see Section 7 of the paper).

  with dimensionless c_i, |Delta P_i| in GW, and E0 = 1 GW h. This reporting convention is
  implemented in run_realistic_scenario.py and in the Section 7 plotting
  scripts. The constant COST_NORMALIZATION is retained only for
  debug scaling of the internal quadratic objective.

The catalog includes 10 unit classes spanning the realistic CPES spectrum
from cheap-slow base-load generation (nuclear, lignite) through medium-
fast dispatchable (CCGT, pumped storage) to fast-expensive peaking
(gas engines, batteries) and flexibility-providing demand-side resources
(heat pump aggregators, EV hubs, industrial demand response, hydrogen P2X).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


# ---------------------------------------------------------------------------
# Physical parameter set per unit class. These values are illustrative but
# realistic at order-of-magnitude level.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PhysicalUnitParameters:
    """Per-unit-class physical parameters in SI / industry units."""
    name: str
    P_max_GW: float                # active power capability, symmetric
    Q_max_GVAR: float              # reactive power capability, symmetric
    P_min_GW: float                # asymmetric lower bound for prod-only / cons-only units
    Q_min_GVAR: float              # asymmetric lower bound for reactive
    ramp_GW_per_min: float         # active power ramp rate
    response_time_constant_s: float  # 1/e response time of the second-order dynamics
    cost_proportional_coeff: float  # dimensionless cost-proportional dispatch coefficient
    rest_P_GW: float = 0.0         # baseline P operating point
    rest_Q_GVAR: float = 0.0       # baseline Q operating point
    aggregate_weight_P: float = 1.0  # contribution to aggregate (identity A_i=I: exact physical fleet sum)
    aggregate_weight_Q: float = 1.0
    description: str = ""

    @property
    def time_constant_min(self) -> float:
        return self.response_time_constant_s / 60.0


# ---------------------------------------------------------------------------
# Catalog: 10 representative CPES unit classes
# ---------------------------------------------------------------------------
# Each entry follows the convention:
#   - Generators with hard lower bound > -P_max use the asymmetric P_min field
#   - Symmetric (bidirectional) units have P_min = -P_max
# ---------------------------------------------------------------------------

NUCLEAR_BLOCK = PhysicalUnitParameters(
    name="nuclear_baseload_block",
    P_max_GW=1.5, Q_max_GVAR=0.5,
    P_min_GW=-1.5, Q_min_GVAR=-0.5,
    ramp_GW_per_min=0.03,
    response_time_constant_s=480.0,   # 8 minutes — large thermal inertia
    cost_proportional_coeff=8.0,
    aggregate_weight_P=3.0, aggregate_weight_Q=2.0,
    description="Large baseload reactor block — cheapest energy, slowest "
                "ramp. Reactor xenon/fuel-temperature transients limit "
                "ramp; large generator mass dominates the inertia.",
)

LIGNITE_BLOCK = PhysicalUnitParameters(
    name="lignite_or_hard_coal_block",
    P_max_GW=1.0, Q_max_GVAR=0.4,
    P_min_GW=-1.0, Q_min_GVAR=-0.4,
    ramp_GW_per_min=0.10,
    response_time_constant_s=240.0,   # 4 minutes
    cost_proportional_coeff=35.0,
    aggregate_weight_P=2.0, aggregate_weight_Q=1.5,
    description="Solid-fuel thermal block — cheap but slow, boiler thermal "
                "limit dominates ramp rate.",
)

CCGT_BLOCK = PhysicalUnitParameters(
    name="combined_cycle_gas_turbine",
    P_max_GW=0.8, Q_max_GVAR=0.5,
    P_min_GW=-0.8, Q_min_GVAR=-0.5,
    ramp_GW_per_min=0.30,
    response_time_constant_s=120.0,   # 2 minutes
    cost_proportional_coeff=65.0,
    aggregate_weight_P=1.5, aggregate_weight_Q=1.5,
    description="Combined-cycle gas turbine — medium-fast, medium cost. "
                "Gas-turbine ramps quickly, steam recovery limits net ramp.",
)

GAS_ENGINE_PEAKER = PhysicalUnitParameters(
    name="reciprocating_gas_engine",
    P_max_GW=0.4, Q_max_GVAR=0.2,
    P_min_GW=-0.4, Q_min_GVAR=-0.2,
    ramp_GW_per_min=1.0,              # very fast
    response_time_constant_s=30.0,    # 30 seconds
    cost_proportional_coeff=180.0,
    aggregate_weight_P=0.8, aggregate_weight_Q=0.6,
    description="Reciprocating gas engine — fast peaker, expensive fuel. "
                "Used for short, sharp interventions.",
)

BATTERY_STORAGE = PhysicalUnitParameters(
    name="lithium_ion_battery_storage",
    P_max_GW=0.3, Q_max_GVAR=0.2,
    P_min_GW=-0.3, Q_min_GVAR=-0.2,
    ramp_GW_per_min=5.0,              # near-instantaneous
    response_time_constant_s=8.0,     # converter dynamics dominate
    cost_proportional_coeff=12.0,
    aggregate_weight_P=0.7, aggregate_weight_Q=0.7,
    description="Grid-scale Li-ion battery via four-quadrant inverter — "
                "fastest response, low cycling burden.",
)

PUMPED_HYDRO = PhysicalUnitParameters(
    name="pumped_hydro_storage",
    P_max_GW=0.6, Q_max_GVAR=0.3,
    P_min_GW=-0.6, Q_min_GVAR=-0.3,
    ramp_GW_per_min=0.4,
    response_time_constant_s=90.0,    # 90 seconds — penstock + machine
    cost_proportional_coeff=18.0,
    aggregate_weight_P=1.2, aggregate_weight_Q=1.0,
    description="Pumped-hydro plant — large, cheap, medium-slow. "
                "Penstock and synchronous machine dominate the time "
                "constant.",
)

HYDROGEN_P2X = PhysicalUnitParameters(
    name="hydrogen_electrolyzer_fuel_cell",
    P_max_GW=0.20, Q_max_GVAR=0.10,
    P_min_GW=-0.08, Q_min_GVAR=-0.10,   # asymmetric: electrolysis > fuel cell
    ramp_GW_per_min=0.5,
    response_time_constant_s=60.0,
    cost_proportional_coeff=200.0,
    aggregate_weight_P=0.5, aggregate_weight_Q=0.4,
    description="Hydrogen electrolyzer + fuel-cell reversibility — "
                "asymmetric: producing P (fuel cell) is small/expensive, "
                "consuming P (electrolysis) is larger but slower.",
)

HEAT_PUMP_AGGREGATOR = PhysicalUnitParameters(
    name="heat_pump_aggregator_cluster",
    P_max_GW=0.25, Q_max_GVAR=0.10,
    P_min_GW=-0.25, Q_min_GVAR=-0.10,
    ramp_GW_per_min=0.08,             # slow — thermal inertia of buildings
    response_time_constant_s=300.0,   # 5 minutes
    cost_proportional_coeff=80.0,
    aggregate_weight_P=0.6, aggregate_weight_Q=0.5,
    description="Heat-pump aggregator over residential clusters — limited "
                "by building thermal capacity. Indirect inverter-coupled "
                "reactive support possible.",
)

EV_V2G_HUB = PhysicalUnitParameters(
    name="ev_v2g_charging_hub",
    P_max_GW=0.20, Q_max_GVAR=0.15,
    P_min_GW=-0.20, Q_min_GVAR=-0.15,
    ramp_GW_per_min=2.0,              # power electronics, fast
    response_time_constant_s=15.0,
    cost_proportional_coeff=50.0,
    aggregate_weight_P=0.5, aggregate_weight_Q=0.5,
    description="Electric vehicle V2G hub — fast, medium cost. User-comfort "
                "penalty included in cost-proportional coefficient. Availability assumed 70%.",
)

INDUSTRIAL_DEMAND_RESPONSE = PhysicalUnitParameters(
    name="industrial_demand_response",
    P_max_GW=0.05, Q_max_GVAR=0.05,
    P_min_GW=-0.15, Q_min_GVAR=-0.05,   # asymmetric: shedding > shifting up
    ramp_GW_per_min=0.4,
    response_time_constant_s=180.0,   # 3 minutes — process inertia
    cost_proportional_coeff=150.0,
    aggregate_weight_P=0.5, aggregate_weight_Q=0.3,
    description="Industrial demand-response — process can be slowed (large "
                "negative P) or accelerated only modestly (small positive P). "
                "High cost reflects production-loss penalty.",
)


# ---------------------------------------------------------------------------
# Standard cluster definitions
# ---------------------------------------------------------------------------

ClusterName = Literal["realistic_15", "realistic_60"]


def cluster_15() -> list[PhysicalUnitParameters]:
    """N=15 cluster for the visualisation-focused scenarios.

    Composition:
      1 nuclear + 1 lignite + 1 CCGT + 2 gas-engine + 2 battery +
      1 pumped-hydro + 1 hydrogen + 2 heat-pump + 2 EV-hub + 2 industrial-DR
    """
    return (
        [NUCLEAR_BLOCK] * 1
        + [LIGNITE_BLOCK] * 1
        + [CCGT_BLOCK] * 1
        + [GAS_ENGINE_PEAKER] * 2
        + [BATTERY_STORAGE] * 2
        + [PUMPED_HYDRO] * 1
        + [HYDROGEN_P2X] * 1
        + [HEAT_PUMP_AGGREGATOR] * 2
        + [EV_V2G_HUB] * 2
        + [INDUSTRIAL_DEMAND_RESPONSE] * 2
    )


def cluster_60() -> list[PhysicalUnitParameters]:
    """N=60 cluster — 4x scaling of cluster_15 for the scalability table."""
    return cluster_15() * 4


def fleet_capacity_GW(cluster: list[PhysicalUnitParameters]) -> tuple[float, float]:
    """Total positive-direction P and Q capacity of the fleet."""
    p_cap = sum(u.P_max_GW for u in cluster)
    q_cap = sum(u.Q_max_GVAR for u in cluster)
    return p_cap, q_cap


# ---------------------------------------------------------------------------
# Calibration: physical parameters -> EnergyFlexibilityUnit values
# ---------------------------------------------------------------------------
#
# The validation harness uses dimensionless quantities. We pick a single
# calibration constant DT_MIN (the simulation step in minutes) that ties the
# integration time to physical wall-clock time. The simulation step is fixed at
# dt=0.05 simulation time units (paper default; ~3 s physical). For realistic
# CPES we slow the model so that one simulation time unit corresponds to one real
# minute (TIME_SCALE_S_PER_MIN). This lets us run a 30-minute physical
# scenario in 30 simulation time units, which is computationally tractable
# while preserving the physical ratios between unit response times.

TIME_SCALE_S_PER_MIN = 1.0     # 1 simulation time unit = 1 wall-clock minute

# Inertia calibration:
# The plant is a damped second-order system  m*ẍ + μ*ẋ + k*x = F  with k=1
# and damping ratio ζ=0.9 (near-critical, no overshoot in CPES operation).
# For a step input from rest the dimensionless 1/e response time, defined as
# the first time at which x(t) reaches (1 - 1/e) * x_ss, is a fixed constant
# T63_DIMENSIONLESS at ζ=0.9 with k=1, m=1. Numerically integrating the plant
# with dt=1e-6 gives T63_DIMENSIONLESS = 1.9991 (4 significant figures).
# Equivalently: for arbitrary m, T63 = √m * T63_DIMENSIONLESS, so requiring
# T63 = τ_doc (in sim-time units = minutes) gives
#     m = (τ_doc_min / T63_DIMENSIONLESS)²
# This is the physically correct OPEN-LOOP calibration: the open-loop small-signal
# 1/e response time of this reference plant matches the documented τ to the precision
# of the underlying integrator. NOTE: this is a reference inertia/damping calibration
# proxy for the open-loop plant; the nominal CLOSED loop is plant-inverted and
# m,μ-independent, so τ sets only the relative class ordering and the saturated ramp
# limit, not a reproduced closed-loop 1/e response time.
#
# (The earlier calibration m = (τ_doc / 2π)² was unsound: it equated τ_doc to
# the natural period 2π/ω_n rather than the 1/e response time, leaving the
# simulated transients 3.14× faster than the documented values. The steady-
# state ramp v_ss = F/μ was unaffected, so the present correction changes the
# transient time scales but preserves all steady-state quantities by design.)
#
# Stiffness calibration: k_proxy = 1, so the equilibrium is determined by the
# controller cost gradient, not by an artificial spring. Damping is
# μ = 2*ζ*sqrt(m*k) = 2*ζ*sqrt(m).

T63_DIMENSIONLESS = 1.999106   # 1/e response time at ζ=0.9, k=1, m=1
DAMPING_RATIO = 0.9            # near-critical, no overshoot


def calibrate_mass(phys: PhysicalUnitParameters) -> float:
    """Map a unit's documented 1/e response time to the simulator's mass.

    The plant is  m*ẍ + μ*ẋ + k*x = F  with k=1 and ζ=DAMPING_RATIO. The
    1/e response time of this plant is T63 = √m * T63_DIMENSIONLESS, so
        m = (τ_doc_min / T63_DIMENSIONLESS)²
    where τ_doc_min is the documented response_time_constant_s expressed in
    sim-time units (= minutes, since 1 simulation time unit = 1 wall-clock minute).

    Verified by numerical integration: with this m the OPEN-LOOP small-signal
    1/e response time of this reference plant matches τ_doc to four significant
    figures. This is a reference inertia/damping calibration proxy; the nominal
    CLOSED loop is plant-inverted and m,μ-independent, so τ_doc sets only the
    relative class ordering and the saturated ramp limit, not a reproduced
    closed-loop response time.
    """
    tau_sim_min = max(phys.time_constant_min / TIME_SCALE_S_PER_MIN, 0.01)
    return (tau_sim_min / T63_DIMENSIONLESS) ** 2


def calibrate_damping(mass: float, damping_ratio: float = DAMPING_RATIO) -> float:
    """Near-critical damping (ζ=0.9): μ = 2*ζ*sqrt(m*k) = 2*ζ*sqrt(m)."""
    return 2.0 * damping_ratio * np.sqrt(max(mass, 1.0e-12))


def calibrate_force_limit(phys: PhysicalUnitParameters, mass: float,
                          damping: float) -> float:
    """Map ramp rate (GW/min) to the simulator's force-limit (||u|| ≤ F).

    The damped second-order dynamics m * v_dot + μ * v = F has steady-state
    velocity v_ss = F/μ under sustained saturation. Requiring v_ss to equal
    the documented ramp rate (in GW/min = GW per sim-time unit) gives
    F = μ * ramp. Verified: the simulated saturated ramp matches the
    documented rate exactly.
    """
    return phys.ramp_GW_per_min * damping


def to_energy_flexibility_unit(phys: PhysicalUnitParameters,
                               cost_normalization: float = 50.0):
    """Build the EnergyFlexibilityUnit dataclass instance for the harness.

    The loss_weight in the harness is a dimensionless coefficient on |p|^2,
    weighted by loss_weight_scale (paper default 0.03). To keep the harness
    near the paper-validated configuration, we normalise the cost-proportional coefficient
    by `cost_normalization` so the resulting loss_weight has
    O(1) magnitude. This preserves the relative cost heterogeneity (cheap
    nuclear vs expensive gas peaker) while keeping the loss-vs-aggregate-
    tracking balance in the same regime as the validated paper-Table runs.

    The cost interpretation of the integrated J is then recovered by
    multiplying by `cost_normalization`:
        cost_dimless = J_T * cost_normalization
    """
    from dsceos_validation import EnergyFlexibilityUnit

    mass = calibrate_mass(phys)
    damping = calibrate_damping(mass)
    force_limit = calibrate_force_limit(phys, mass, damping)

    lower = (phys.P_min_GW, phys.Q_min_GVAR)
    upper = (phys.P_max_GW, phys.Q_max_GVAR)
    rest = (phys.rest_P_GW, phys.rest_Q_GVAR)

    # Normalised cost — O(1) magnitudes that preserve relative heterogeneity
    c_norm = phys.cost_proportional_coeff / cost_normalization
    loss_weight = (c_norm, c_norm)

    aggregate_weight = (phys.aggregate_weight_P, phys.aggregate_weight_Q)

    return EnergyFlexibilityUnit(
        kind=phys.name,
        mass=mass,
        damping=damping,
        force_limit=force_limit,
        lower=lower,
        upper=upper,
        rest=rest,
        loss_weight=loss_weight,
        aggregate_weight=aggregate_weight,
    )


# Numerical normalisation constant c_n for the cost coefficients: the local
# loss term of the controller objective uses c_i / COST_NORMALIZATION so the
# harness coefficients have O(1) magnitude while preserving the relative cost
# heterogeneity across the fleet (see the paper's Section 7 calibration text).
COST_NORMALIZATION = 50.0


# ---------------------------------------------------------------------------
# DSO ↔ controller-internal scaling layer (CPES aggregator architecture)
# ---------------------------------------------------------------------------
#
# Background:
# The validation harness uses the aggregate convention
#   y(t) = (1/N) Σ_i A_i p_i(t)
# with identity blocks A_i = I (uniform), so the internal-to-DSO un-scaling gives
# (1/N) Σ A_i = I. With identity blocks this is the PER-UNIT ARITHMETIC MEAN of
# the unit set-points (an identity aggregate, NOT a capacity-weighted average),
# and it is not the physical fleet sum Σ p_i.
#
# However, the DSO (or TSO) signals its flexibility request in PHYSICAL
# FLEET GW: "I need +3 GW of active power from the aggregator". The
# aggregator therefore needs an explicit scaling layer between the
# external DSO interface and the controller-internal convention.
#
# Scaling factor:
#   Let P_phys_max = Σ_i Pmax_i be the fleet's physical sum-of-caps.
#   Let y_internal_max = (1/N) Σ_i A_i Pmax_i be the controller-side max.
#   Then:
#     y_internal = y_DSO * (y_internal_max / P_phys_max)
#     y_DSO    = y_internal * (P_phys_max / y_internal_max)
#
# This is exactly what a real CPES aggregator does: receive a fleet-GW
# request, translate it to its internal allocation problem, run the
# controller, and report back the realised fleet-GW service.


@dataclass(frozen=True)
class FleetScaling:
    """Scaling factors between DSO physical GW and the controller's
    identity aggregate convention (A_i=I): the tracked aggregate is the exact normalized physical fleet sum."""
    P_phys_max_GW: float                # Σ_i Pmax_i (active)
    Q_phys_max_GVAR: float              # Σ_i Qmax_i (reactive)
    y_internal_P_max: float             # (1/N) Σ A_i[0,0] Pmax_i
    y_internal_Q_max: float             # (1/N) Σ A_i[1,1] Qmax_i

    @property
    def P_scale_DSO_GW_to_internal(self) -> float:
        """Multiply DSO-GW target by this to get controller-internal target."""
        return self.y_internal_P_max / max(self.P_phys_max_GW, 1.0e-12)

    @property
    def Q_scale_DSO_GVAR_to_internal(self) -> float:
        return self.y_internal_Q_max / max(self.Q_phys_max_GVAR, 1.0e-12)

    def dso_to_internal(self, y_dso_PQ: tuple[float, float]) -> np.ndarray:
        """Convert a (P_GW, Q_GVAR) DSO target to controller-internal units."""
        return np.array([
            y_dso_PQ[0] * self.P_scale_DSO_GW_to_internal,
            y_dso_PQ[1] * self.Q_scale_DSO_GVAR_to_internal,
        ], dtype=float)

    def internal_to_dso(self, y_internal_PQ: np.ndarray) -> np.ndarray:
        """Convert a controller-internal y(t) back to DSO-GW units."""
        return np.array([
            y_internal_PQ[0] / max(self.P_scale_DSO_GW_to_internal, 1.0e-12),
            y_internal_PQ[1] / max(self.Q_scale_DSO_GVAR_to_internal, 1.0e-12),
        ], dtype=float)


def compute_fleet_scaling(cluster: list[PhysicalUnitParameters]) -> FleetScaling:
    """Compute the DSO↔internal scaling factors for a given fleet.

    This needs the same A_i blocks the validation harness uses, computed
    from the per-unit aggregate_weight tuple.
    """
    n = len(cluster)
    # P-axis weights and capacity
    # DEVIATION (released formulation): uniform aggregate weights (A_i = I) so that the DSO
    # scaling gives internal_to_dso(mean_i p_i) = sum_i p_i exactly (physical fleet
    # sum in GW/GVAR). Consistent with arrays_from_units above.
    w_P = np.ones(n)
    w_Q = np.ones(n)
    Pmax = np.array([u.P_max_GW for u in cluster])
    Qmax = np.array([u.Q_max_GVAR for u in cluster])

    # A_i diagonal entries: N * w_i / Σ_j w_j
    A_diag_P = n * w_P / max(np.sum(w_P), 1.0e-12)
    A_diag_Q = n * w_Q / max(np.sum(w_Q), 1.0e-12)

    # Controller-side max: (1/N) Σ A_i Pmax_i
    y_internal_P_max = float(np.sum(A_diag_P * Pmax) / n)
    y_internal_Q_max = float(np.sum(A_diag_Q * Qmax) / n)

    # Physical fleet sum
    P_phys_max = float(np.sum(Pmax))
    Q_phys_max = float(np.sum(Qmax))

    return FleetScaling(
        P_phys_max_GW=P_phys_max,
        Q_phys_max_GVAR=Q_phys_max,
        y_internal_P_max=y_internal_P_max,
        y_internal_Q_max=y_internal_Q_max,
    )


def build_realistic_units(cluster_name: ClusterName = "realistic_15"):
    """Construct the list of EnergyFlexibilityUnit objects for a named cluster."""
    if cluster_name == "realistic_15":
        physical = cluster_15()
    elif cluster_name == "realistic_60":
        physical = cluster_60()
    else:
        raise ValueError(f"Unknown cluster: {cluster_name}")
    return [to_energy_flexibility_unit(p) for p in physical], physical


def print_catalog_summary(cluster: list[PhysicalUnitParameters]) -> None:
    """Print a per-unit summary table (useful for paper Section 7)."""
    print(f"{'Type':<35} {'Pmax':>6} {'Qmax':>6} {'ramp':>6} {'τ':>6} {'cost':>7}")
    print(f"{'':35} {'[GW]':>6} {'[GVAR]':>6} {'GW/min':>6} {'[s]':>6} {'cost coeff.':>7}")
    print("-" * 80)
    for u in cluster:
        print(f"{u.name:<35} "
              f"{u.P_max_GW:>6.2f} {u.Q_max_GVAR:>6.2f} "
              f"{u.ramp_GW_per_min:>6.2f} {u.response_time_constant_s:>6.0f} "
              f"{u.cost_proportional_coeff:>7.0f}")
    p_cap, q_cap = fleet_capacity_GW(cluster)
    print("-" * 80)
    print(f"{'TOTAL CAPACITY':<35} {p_cap:>6.2f} {q_cap:>6.2f}")


if __name__ == "__main__":
    print("=" * 80)
    print("REALISTIC CPES UNIT CATALOG — N=15 cluster")
    print("=" * 80)
    print_catalog_summary(cluster_15())
    print()
    print("=" * 80)
    print("REALISTIC CPES UNIT CATALOG — N=60 cluster (4x scaling)")
    print("=" * 80)
    p_cap, q_cap = fleet_capacity_GW(cluster_60())
    print(f"Total capacity: {p_cap:.2f} GW, {q_cap:.2f} GVAR across 60 units")
