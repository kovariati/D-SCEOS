"""
run_realistic_scenario.py
==========================

CLI runner for the three realistic CPES application scenarios (Section 7).
The runner constructs the realistic 15-unit (or 60-unit) cluster, builds
the DSO target signal for the chosen scenario, and runs the chosen
controller (D-SCEOS, DPG-HOCBF or one PD baseline) through the standard
dsceos_validation harness.

Outputs are placed in --outdir with the same layout as
run_dsceos_validation.py:
  metrics.csv          per-step time-series
  state_history.npz    arrays of time, positions, velocities, controls
  summary.json         scalar metrics


USAGE EXAMPLE
-------------

  python3 run_realistic_scenario.py \\
      --scenario scenario_a_winter_morning_step \\
      --controller dsceos \\
      --cluster realistic_15 \\
      --safety-filter \\
      --outdir results/scenario_a/dsceos
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from realistic_cpes_catalog import (
    TIME_SCALE_S_PER_MIN,
    build_realistic_units, compute_fleet_scaling, fleet_capacity_GW,
)
from realistic_scenarios import REALISTIC_SCENARIOS, build_signal_for


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scenario", required=True,
                   choices=list(REALISTIC_SCENARIOS.keys()))
    p.add_argument("--controller", required=True,
                   choices=["dsceos", "independent_tracking",
                            "coherent_tracking", "centralized_tracking",
                            "distributed_primal_dual_hocbf",
                            "projected_gradient_hocbf", "gradient_tracking_hocbf"])
    p.add_argument("--cluster", choices=["realistic_15", "realistic_60"],
                   default="realistic_15")
    p.add_argument("--target-multiplier", type=float, default=1.0,
                   help="Multiplier on the DSO target. Use 4.0 with "
                        "--cluster realistic_60 to keep the same fleet "
                        "load ratio as realistic_15.")
    p.add_argument("--outdir", required=True, type=Path)
    p.add_argument("--dt", type=float, default=0.05,
                   help="Integration step in simulation-time units. The "
                        "default 0.05 corresponds to 3-second physical "
                        "resolution at TIME_SCALE_S_PER_MIN=1.")
    p.add_argument("--safety-filter", action="store_true",
                   help="Wrap baseline controllers in the HOCBF safety "
                        "filter (apples-to-apples comparison).")
    p.add_argument("--aggregate-tracking-weight", type=float, default=2.0/3.0,
                   help="Per-agent tracking weight w_bar; the objective uses "
                        "(N*w_bar/2)*||y-y_T||^2 (size-consistent form). The "
                        "reference value 2/3 reproduces the conventional "
                        "w_y=10 normalization at N=15.")
    p.add_argument("--loss-weight-scale", type=float, default=0.03)
    p.add_argument("--sharing-weight", type=float, default=0.15)
    p.add_argument("--internal-weight", type=float, default=0.03)
    p.add_argument("--aggregate-consensus-gain", type=float, default=0.42)
    p.add_argument("--adaptive-consensus-gain", action="store_true",
                   help="Compute a Gershgorin upper bound on the graph "
                        "Laplacian largest eigenvalue and set a certified "
                        "K_y. Overrides --aggregate-consensus-gain.")
    p.add_argument("--gershgorin-safety-factor", type=float, default=0.95,
                   help="Safety factor for the Gershgorin K_y = factor / "
                        "max_degree rule (default 0.95).")
    p.add_argument("--gateway-fraction", type=float, default=0.15)
    p.add_argument("--target-consensus-gain", type=float, default=0.18)
    p.add_argument("--target-gateway-gain", type=float, default=0.85)
    p.add_argument("--initial-spread", type=float, default=0.0,
                   help="Initial-position spread around rest, applied with "
                        "inverse-capacity weighting: large-P_max units "
                        "(slow base-load) start very close to rest, while "
                        "small-P_max units (fast peakers, DR, EV) have a "
                        "wider random spread. The numerical value is the "
                        "Gaussian σ of the smallest unit, expressed as a "
                        "fraction of its own capacity-box width. Default "
                        "0.10 gives a non-trivial pre-event state with "
                        "the largest units within approximately one percent of rest. Set 0 to "
                        "force an exact rest start.")
    p.add_argument("--initial-speed-scale", type=float, default=0.0,
                   help="Initial-velocity Gaussian standard deviation, "
                        "also applied with inverse-capacity weighting. "
                        "Default 0 starts every unit at exactly zero "
                        "velocity (the physically meaningful CPES "
                        "before-event baseline).")
    p.add_argument("--dpg-step-size", type=float, default=0.06,
                   help="Projected-gradient step size of the DPG-HOCBF comparator.")
    p.add_argument("--dpg-substeps", type=int, default=1,
                   help="Optimizer substeps per sampling instant of DPG-HOCBF.")
    p.add_argument("--pd-kp", type=float, default=1.25,
                   help="Proportional gain of the PD baseline comparator.")
    p.add_argument("--pd-kd", type=float, default=1.6,
                   help="Derivative gain of the PD baseline comparator.")
    p.add_argument("--print-catalog", action="store_true")
    args = p.parse_args()

    scenario = REALISTIC_SCENARIOS[args.scenario]

    # Apply the target multiplier if requested (e.g. for scalability tests
    # that vary N while preserving fleet load ratio).
    if args.target_multiplier != 1.0:
        from dataclasses import replace
        scenario = replace(
            scenario,
            target_P_max_GW=scenario.target_P_max_GW * args.target_multiplier,
            target_Q_max_GVAR=scenario.target_Q_max_GVAR * args.target_multiplier,
        )

    # Import the validation harness lazily so the catalog test can run
    # without numpy import side-effects.
    from dsceos_validation import (
        ClusterConfig, SimulationConfig, aggregate_output, arrays_from_units,
        make_aggregate_blocks, make_fixed_local_graph, make_physical_layout,
        run_simulation, sample_initial_state,
    )
    from dsceos_controller import DSCEOSConfig

    # Build the realistic unit list and convert to validation-harness arrays
    units, physical = build_realistic_units(args.cluster)
    n = len(units)

    # Compute fleet scaling for the DSO ↔ controller-internal layer
    scaling = compute_fleet_scaling(physical)

    # The validation harness expects EnergyFlexibilityUnit instances and
    # internal helpers expect a ClusterConfig with the size breakdown. We
    # override make_units to inject our realistic list via monkey-patch.
    import dsceos_validation as dv

    original_make_units = dv.make_units
    original_select_target = dv.select_target

    def patched_make_units(cfg):
        return units

    signal = build_signal_for(scenario, scaling)

    def patched_select_target(cfg):
        return signal

    dv.make_units = patched_make_units
    dv.select_target = patched_select_target

    if args.print_catalog:
        print(f"Cluster: {args.cluster} (N={n})")
        for i, (u, ph) in enumerate(zip(units, physical)):
            print(f"  unit {i:2d}: {ph.name:<35} "
                  f"Pmax={ph.P_max_GW:+.2f} GW, ramp={ph.ramp_GW_per_min:.2f} GW/min, "
                  f"cost_coeff={ph.cost_proportional_coeff:.0f}")
        p_cap, q_cap = fleet_capacity_GW(physical)
        print(f"  fleet capacity: {p_cap:.2f} GW, {q_cap:.2f} GVAR")
        print(f"  DSO peak request: {scenario.target_P_max_GW} GW")
        print(f"  load ratio: {scenario.target_P_max_GW / p_cap * 100:.1f}%")
        print()

    # ClusterConfig dummy values (we monkey-patched make_units, so the
    # n_* fields are unused, but ClusterConfig is also queried for
    # initial_spread, communication_radius, etc.)
    # Communication-graph parameters come from the single authoritative source (graph_config.py),
    # so the runner, the validator and the regression tests cannot drift apart.
    from graph_config import (AUTHORITATIVE_COMMUNICATION_RADIUS,
                              AUTHORITATIVE_NEIGHBOUR_COUNT, AUTHORITATIVE_LAYOUT_SPREAD,
                              AUTHORITATIVE_SEED)
    cluster_cfg = ClusterConfig(
        n_thermal=3, n_storage=3, n_hydrogen=3, n_emobility=3, n_industrial=3,
        seed=AUTHORITATIVE_SEED,
        initial_spread=args.initial_spread,
        initial_speed_scale=args.initial_speed_scale,
        communication_radius=AUTHORITATIVE_COMMUNICATION_RADIUS,
        neighbour_count=AUTHORITATIVE_NEIGHBOUR_COUNT,
        layout_spread=AUTHORITATIVE_LAYOUT_SPREAD,
    )

    if args.dpg_substeps != 1:
        from distributed_projected_gradient_controller import DistributedProjectedGradientController as _DPGBase
        _sub = args.dpg_substeps
        class _TunedDPG(_DPGBase):
            def __init__(self, problem, cfg, **kw):
                kw.setdefault("optimizer_substeps", _sub)
                super().__init__(problem, cfg, **kw)
        dv.DistributedProjectedGradientController = _TunedDPG

    ccfg = DSCEOSConfig(
        aggregate_tracking_weight=args.aggregate_tracking_weight,
        loss_weight_scale=args.loss_weight_scale,
        sharing_weight=args.sharing_weight,
        internal_weight=args.internal_weight,
        aggregate_consensus_gain=args.aggregate_consensus_gain,
        adaptive_consensus_gain=args.adaptive_consensus_gain,
        gershgorin_safety_factor=args.gershgorin_safety_factor,
    )

    sim_cfg = SimulationConfig(
        cluster=cluster_cfg,
        controller=args.controller,
        scenario="static_request",   # internal harness mode; output summary is overwritten with the realistic scenario name
        dsceos_config=ccfg,
        dt=args.dt,
        horizon=float(scenario.horizon_sim),
        gateway_fraction=args.gateway_fraction,
        target_consensus_gain=args.target_consensus_gain,
        target_gateway_gain=args.target_gateway_gain,
        compute_reference_optimum=False,
        target_override=None,
        safety_filter=args.safety_filter,
        dpg_step_size=args.dpg_step_size,
        pd_kp=args.pd_kp,
        pd_kd=args.pd_kd,
    )

    print(f"Running: scenario={args.scenario}, controller={args.controller}")
    print(f"   cluster={args.cluster} (N={n})")
    print(f"   horizon={scenario.horizon_min:.0f} min = {scenario.horizon_sim} sim units")
    print(f"   dt={args.dt} (≈ {args.dt * 60:.1f} s physical resolution)")

    result = run_simulation(sim_cfg)
    args.outdir.mkdir(parents=True, exist_ok=True)

    # Augment the summary with realistic-CPES interpretation fields, so that
    # downstream tools (animation, plotting) can read them straight from
    # summary.json without re-running scaling computations.
    # The summary aggregate-error metric is in controller-internal units.
    # Convert the final error exactly by undoing the DSO-to-internal scaling
    # component-wise. This is more accurate than dividing by an averaged scale
    # when P and Q use different scaling factors.
    arrays_for_error = arrays_from_units(units)
    Ablk_for_error = make_aggregate_blocks(arrays_for_error["aggregate_weight"])
    final_y_internal = aggregate_output(Ablk_for_error, result.positions[-1])
    final_target_internal = signal.position(float(scenario.horizon_sim))
    internal_error_vec = final_y_internal - final_target_internal
    err_P_GW = internal_error_vec[0] / max(scaling.P_scale_DSO_GW_to_internal, 1.0e-12)
    err_Q_GVAR = internal_error_vec[1] / max(scaling.Q_scale_DSO_GVAR_to_internal, 1.0e-12)
    agg_err_DSO_GW = float(np.linalg.norm([err_P_GW, err_Q_GVAR]))

    result.summary.update({
        "scenario": scenario.name,
        "_realistic_scenario": scenario.name,
        "_dso_target_P_GW": scenario.target_P_max_GW,
        "_dso_target_Q_GVAR": scenario.target_Q_max_GVAR,
        "_fleet_P_phys_max_GW": scaling.P_phys_max_GW,
        "_fleet_Q_phys_max_GVAR": scaling.Q_phys_max_GVAR,
        "_P_scale_DSO_GW_to_internal": scaling.P_scale_DSO_GW_to_internal,
        "_Q_scale_DSO_GVAR_to_internal": scaling.Q_scale_DSO_GVAR_to_internal,
        "_horizon_min": scenario.horizon_min,
        "_agg_err_DSO_GW": agg_err_DSO_GW,
        "_per_agent_tracking_weight": args.aggregate_tracking_weight,
        "_agg_err_P_GW": float(err_P_GW),
        "_agg_err_Q_GVAR": float(err_Q_GVAR),
        "_final_y_internal_P": float(final_y_internal[0]),
        "_final_y_internal_Q": float(final_y_internal[1]),
        "_final_target_internal_P": float(final_target_internal[0]),
        "_final_target_internal_Q": float(final_target_internal[1]),
    })
    result.save(args.outdir)

    # Restore monkey-patches
    dv.make_units = original_make_units
    dv.select_target = original_select_target

    # Print summary in physical (DSO) units
    s = result.summary
    print()
    print("=" * 70)
    print("RESULT SUMMARY (physical DSO-side interpretation)")
    print("=" * 70)
    print(f"  Scenario:                  {scenario.name}")
    print(f"  Controller:                {args.controller}")
    print(f"  Horizon:                   {scenario.horizon_min:.0f} min")
    print(f"  DSO peak request:          ({scenario.target_P_max_GW:.2f} GW, "
          f"{scenario.target_Q_max_GVAR:.2f} GVAR)")
    print(f"  Fleet capacity:            ({scaling.P_phys_max_GW:.2f} GW, "
          f"{scaling.Q_phys_max_GVAR:.2f} GVAR)")
    print(f"  Load ratio:                "
          f"{scenario.target_P_max_GW/scaling.P_phys_max_GW*100:.1f}% of fleet")
    print(f"  ----------------------------------------------------------------")
    print(f"  Final aggregate error:     {agg_err_DSO_GW:.4f} GW-equivalent "
          f"(internal: {s['final_aggregate_error']:.5f})")
    print(f"  Max capacity violation:    {s['max_capacity_violation']:.5f} "
          f"(must be 0)")
    print(f"  Integrated J (internal):   {s['integrated_objective_value']:.4f}")
    print(f"  Total control effort:      {s['total_control_energy']:.4f} "
          f"(internal control-energy units)")
    print("=" * 70)


if __name__ == "__main__":
    main()
