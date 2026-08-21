"""
realistic_scenarios.py
=======================

Three realistic CPES application scenarios for Section 7 of the paper.

Each scenario defines:
  - a DSO request signal y_T_DSO(t) in PHYSICAL FLEET GW (GVAR)
  - the horizon (physical duration in minutes)

The signals here are expressed in PHYSICAL GW. The aggregator's CPES
internal target (in the controller's identity-aggregate convention, A_i=I)
is obtained via FleetScaling.dso_to_internal() — see realistic_cpes_catalog
for the architecture.

The three signal types follow common practice:

A) CONGESTION MANAGEMENT (step):  y_T_DSO = const for the full horizon.
   The DSO requests a sudden (P, Q) shift that must be held steady. Tests
   transient response, settling, and capacity invariance.

B) OFFSHORE WIND RAMP-DOWN (ramp):        y_T_DSO(t) varies linearly over horizon.
   Offshore wind generation ramps down over the event horizon; the TSO/DSO requests a continuously tracked balancing ramp. Tests continuous tracking.

C) BALANCING MARKET (mFRR steps): y_T_DSO(t) is a piecewise-constant
   15-minute schedule. Tests repeated transients.

In all three scenarios, the simulation time is mapped to wall-clock
minutes via TIME_SCALE_S_PER_MIN from realistic_cpes_catalog. The
the integrated objective J_T is the single reported objective metric.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from realistic_cpes_catalog import TIME_SCALE_S_PER_MIN, FleetScaling


RealisticScenarioName = Literal[
    "scenario_a_winter_morning_step",
    "scenario_b_wind_ramp_down_event",
    "scenario_c_winter_balancing_mfrr",
]


@dataclass(frozen=True)
class RealisticScenario:
    """A realistic CPES application scenario with physical-units metadata."""
    name: RealisticScenarioName
    description: str
    horizon_min: float                  # physical horizon in minutes
    target_P_max_GW: float              # peak |P| of the DSO signal [GW]
    target_Q_max_GVAR: float            # peak |Q| of the DSO signal [GVAR]
    signal_type: Literal["step", "ramp", "multi_ramp", "mfrr_blocks"]
    block_minutes: float = 15.0         # for mfrr_blocks signal

    @property
    def horizon_sim(self) -> float:
        """Convert physical minutes to simulation-time units."""
        return self.horizon_min * TIME_SCALE_S_PER_MIN


SCENARIO_A_WINTER_MORNING = RealisticScenario(
    name="scenario_a_winter_morning_step",
    description=(
        "Winter morning congestion management. A neighbouring transmission corridor "
        "trips at 07:15 due to an insulation fault. The DSO requests a "
        "sudden +2.0 GW active and +0.3 GVAR reactive flexibility from the "
        "regional VPP portfolio for 30 minutes, to "
        "keep the surviving feeder below its thermal limit until the "
        "faulted line is restored."
    ),
    horizon_min=30.0,
    target_P_max_GW=2.0,
    target_Q_max_GVAR=0.3,
    signal_type="step",
)


SCENARIO_B_WIND_RAMP_DOWN = RealisticScenario(
    name="scenario_b_wind_ramp_down_event",
    description=(
        "Realistic offshore wind ramp-down event, modelled after the "
        "documented Great Britain 3 Nov 2014 offshore-wind incident "
        "(Drew et al. 2015, doi:10.3390/resources4010155): the GB "
        "offshore fleet lost 1.2 GW over 1 h 50 min following a frontal "
        "subsidence. The regional VPP is contracted by the TSO to track "
        "a linear ramp from 0 to -1.2 GW over 110 min, compensating the "
        "loss of wind generation. Horizon 140 min (110 min ramp + 30 min settling); peak target change "
        "1.2 GW; ramp rate 0.655 GW/h, matching the GB 2014 event "
        "exactly, and consistent with the literature-documented "
        "intra-hour wind-ramp class (30-min to 2-h ramps, 30--70%% "
        "capacity change; Lochmann 2023, doi:10.1002/we.2816)."
    ),
    horizon_min=140.0,
    target_P_max_GW=1.2,
    target_Q_max_GVAR=0.0,
    signal_type="ramp",
)


SCENARIO_C_WINTER_BALANCING = RealisticScenario(
    name="scenario_c_winter_balancing_mfrr",
    description=(
        "Winter cold front with high wind forecast volatility. Over 60 "
        "minutes, the DSO passes through balancing energy in four "
        "consecutive 15-minute mFRR blocks dictated by the wind farm "
        "production forecast: +1.5, −0.8, +2.0, −1.2 GW (active), each "
        "with a coincident ±0.2 GVAR requirement. Fast units cycle through "
        "all four blocks; slow units cannot keep up and are best kept near "
        "rest."
    ),
    horizon_min=90.0,
    target_P_max_GW=2.0,
    target_Q_max_GVAR=0.2,
    signal_type="mfrr_blocks",
)


REALISTIC_SCENARIOS: dict[str, RealisticScenario] = {
    SCENARIO_A_WINTER_MORNING.name: SCENARIO_A_WINTER_MORNING,
    SCENARIO_B_WIND_RAMP_DOWN.name: SCENARIO_B_WIND_RAMP_DOWN,
    SCENARIO_C_WINTER_BALANCING.name: SCENARIO_C_WINTER_BALANCING,
}


# ---------------------------------------------------------------------------
# DSO signal generators — return a TargetSignal compatible with the
# existing dsceos_validation.py TargetSignal dataclass.
#
# Inputs are in PHYSICAL FLEET GW (DSO interface). Each generator takes a
# FleetScaling instance and internally translates to the controller-side
# identity-aggregate convention (A_i=I): exact normalized physical fleet sum.
# ---------------------------------------------------------------------------


def step_signal(P_GW: float, Q_GVAR: float, scaling: FleetScaling):
    """Step signal: y_T_DSO = (P, Q) GW constant for the entire horizon.

    The controller sees y_T = scaling.dso_to_internal((P, Q)).
    """
    from dsceos_validation import TargetSignal

    y_internal = scaling.dso_to_internal((P_GW, Q_GVAR))
    z = np.zeros_like(y_internal)
    return TargetSignal(
        "scenario_a_step_signal",
        lambda t: y_internal.copy(),
        lambda t: z.copy(),
        lambda t: z.copy(),
    )


def ramp_signal(P_initial_GW: float, P_final_GW: float,
                Q_GVAR: float, horizon_min: float,
                scaling: FleetScaling):
    """Linear ramp signal: y_T_DSO(t) goes from initial to final over
    the horizon. All inputs in DSO physical GW."""
    from dsceos_validation import TargetSignal

    horizon_sim = horizon_min * TIME_SCALE_S_PER_MIN
    y_initial = scaling.dso_to_internal((P_initial_GW, Q_GVAR))
    y_final = scaling.dso_to_internal((P_final_GW, Q_GVAR))
    slope = (y_final - y_initial) / horizon_sim
    z = np.zeros(2, dtype=float)

    def pos(t):
        t = float(np.minimum(t, horizon_sim))
        return y_initial + slope * t

    def vel(t):
        if t >= horizon_sim:
            return z.copy()
        return slope.copy()

    def acc(t):
        return z.copy()

    return TargetSignal("scenario_b_ramp_signal", pos, vel, acc)


def multi_ramp_signal(breakpoints_min_GW: list[tuple[float, float, float]],
                       scaling: FleetScaling):
    """Piecewise-linear multi-ramp signal: a list of (t_min, P_DSO_GW, Q_DSO_GVAR)
    breakpoints. Between consecutive breakpoints the target ramps linearly.

    Use case: wind-event reverse-balancing scenario with multiple direction
    changes within the horizon.
    """
    from dsceos_validation import TargetSignal

    n = len(breakpoints_min_GW)
    assert n >= 2, "multi_ramp_signal needs >= 2 breakpoints"
    times_sim = np.array([bp[0] * TIME_SCALE_S_PER_MIN
                          for bp in breakpoints_min_GW])
    targets_internal = np.array(
        [scaling.dso_to_internal((bp[1], bp[2])) for bp in breakpoints_min_GW]
    )  # shape (n, 2)
    z = np.zeros(2, dtype=float)

    def _segment(t):
        # Find the segment k such that times_sim[k] <= t < times_sim[k+1].
        # Clamp on both ends.
        if t <= times_sim[0]:
            return 0
        if t >= times_sim[-1]:
            return n - 2
        # Linear search (n is small, <10)
        for k in range(n - 1):
            if times_sim[k] <= t < times_sim[k + 1]:
                return k
        return n - 2

    def pos(t):
        t = float(t)
        k = _segment(t)
        t0, t1 = times_sim[k], times_sim[k + 1]
        y0, y1 = targets_internal[k], targets_internal[k + 1]
        if t1 == t0:
            return y0.copy()
        # Clamp to segment
        t_eff = float(np.clip(t, t0, t1))
        alpha = (t_eff - t0) / (t1 - t0)
        return (1.0 - alpha) * y0 + alpha * y1

    def vel(t):
        t = float(t)
        if t >= times_sim[-1] or t < times_sim[0]:
            return z.copy()
        k = _segment(t)
        t0, t1 = times_sim[k], times_sim[k + 1]
        y0, y1 = targets_internal[k], targets_internal[k + 1]
        if t1 == t0:
            return z.copy()
        return (y1 - y0) / (t1 - t0)

    def acc(t):
        return z.copy()

    return TargetSignal("scenario_b_multi_ramp_signal", pos, vel, acc)


def mfrr_block_signal(block_values_GW: list[tuple[float, float]],
                      block_minutes: float, scaling: FleetScaling):
    """mFRR-style piecewise-constant signal in 15-minute blocks.

    Each block specifies (P_DSO_GW, Q_DSO_GVAR) constant for block_minutes
    wall-clock minutes; transitions are instantaneous.
    """
    from dsceos_validation import TargetSignal

    block_sim = block_minutes * TIME_SCALE_S_PER_MIN
    blocks_internal = np.array(
        [scaling.dso_to_internal(b) for b in block_values_GW],
        dtype=float,
    )
    z = np.zeros(2, dtype=float)

    def pos(t):
        idx = int(min(t / block_sim, len(blocks_internal) - 1))
        return blocks_internal[idx].copy()

    def vel(t):
        return z.copy()

    def acc(t):
        return z.copy()

    return TargetSignal("scenario_c_mfrr_blocks", pos, vel, acc)


def build_signal_for(scenario: RealisticScenario, scaling: FleetScaling):
    """Build the TargetSignal for a named scenario."""
    if scenario.name == "scenario_a_winter_morning_step":
        return step_signal(P_GW=scenario.target_P_max_GW,
                           Q_GVAR=scenario.target_Q_max_GVAR,
                           scaling=scaling)
    elif scenario.name == "scenario_b_wind_ramp_down_event":
        # Realistic single-ramp profile, modelled on the GB 2014.11.03
        # offshore-wind incident: 0 -> -1.2 GW over 110 min, slope
        # 0.655 GW/h (matches the documented 1.2 GW / 1 h 50 min ramp-down).
        return ramp_signal(
            P_initial_GW=0.0,
            P_final_GW=-scenario.target_P_max_GW,
            Q_GVAR=scenario.target_Q_max_GVAR,
            horizon_min=110.0,
            scaling=scaling,
        )
    elif scenario.name == "scenario_c_winter_balancing_mfrr":
        # Block schedule normalised to peak |P|=1; scaled by target_P_max_GW
        # so the same recipe applies at different fleet sizes.
        norm_blocks = [(+0.75, +1.0), (-0.40, -1.0),
                       (+1.00, +1.0), (-0.60, -1.0)]
        Pscale = scenario.target_P_max_GW
        Qscale = scenario.target_Q_max_GVAR
        block_values = [(p * Pscale, q * Qscale) for (p, q) in norm_blocks]
        return mfrr_block_signal(
            block_values_GW=block_values,
            block_minutes=scenario.block_minutes,
            scaling=scaling,
        )
    raise ValueError(f"Unknown scenario {scenario.name}")


if __name__ == "__main__":
    from realistic_cpes_catalog import cluster_15, compute_fleet_scaling
    scaling = compute_fleet_scaling(cluster_15())

    print("=" * 80)
    print("REALISTIC CPES SCENARIO LIBRARY")
    print("=" * 80)
    print(f"\nFleet scaling: P_phys_max = {scaling.P_phys_max_GW:.2f} GW, "
          f"DSO->internal P-scale = {scaling.P_scale_DSO_GW_to_internal:.4f}")
    print(f"               Q_phys_max = {scaling.Q_phys_max_GVAR:.2f} GVAR, "
          f"DSO->internal Q-scale = {scaling.Q_scale_DSO_GVAR_to_internal:.4f}")

    for s in REALISTIC_SCENARIOS.values():
        print(f"\n>>> {s.name}")
        print(f"    horizon:       {s.horizon_min} min = {s.horizon_sim} sim units")
        print(f"    peak |P_DSO|:  {s.target_P_max_GW} GW (physical fleet)")
        print(f"    peak |Q_DSO|:  {s.target_Q_max_GVAR} GVAR (physical fleet)")
        print(f"    signal type:   {s.signal_type}")
        sig = build_signal_for(s, scaling)
        print(f"    signal samples (internal convention -> DSO GW/GVAR):")
        for frac, label in [(0.0, "t=0"), (0.25, "T/4"), (0.5, "T/2"),
                            (0.75, "3T/4"), (1.0, "T")]:
            t = frac * s.horizon_sim
            y_int = sig.position(t)
            y_dso = scaling.internal_to_dso(y_int)
            print(f"      {label:>5} ({t:5.1f} sim): y_internal=({y_int[0]:+.3f}, "
                  f"{y_int[1]:+.3f})  DSO=({y_dso[0]:+.3f} GW, {y_dso[1]:+.3f} GVAR)")
