"""
animate_realistic_scenario.py
===============================

Phase-portrait visualisation for the realistic CPES application scenarios
(Section 7). Differs from animate_phase_portrait.py in three ways:

  1. Physical units. Both axes are in physical GW / GVAR (the inverse-scaled
     internal coordinates), so the plot is directly readable by a power
     systems engineer.

  2. Unit-class colour coding. Each of the 10 unit classes (nuclear,
     lignite, CCGT, gas-engine, battery, pumped-hydro, hydrogen,
     heat-pump, EV, industrial-DR) has a distinct colour, so the
     heterogeneous behaviour predicted by the D-SCEOS controller (cheap
     units do more, expensive units do less) is visible at a glance.

  3. objective readout. The figure title reports the integrated CPES objective J_T
     integrated J(p) at the current frame.

USAGE EXAMPLE
-------------

  python3 animate_realistic_scenario.py \\
      --scenario scenario_a_winter_morning_step \\
      --runs-dir results/scenario_a \\
      --output-snapshot-png results/scenario_a/phase_portrait.png

The runs-dir must contain dsceos/, projected_gradient_hocbf/, independent_tracking/
subdirectories with state_history.npz and summary.json from
run_realistic_scenario.py.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter

from realistic_cpes_catalog import (
    build_realistic_units,
    compute_fleet_scaling,
)
from realistic_scenarios import REALISTIC_SCENARIOS, build_signal_for
from graph_config import (AUTHORITATIVE_COMMUNICATION_RADIUS as _R045,
                          AUTHORITATIVE_NEIGHBOUR_COUNT as _K4,
                          AUTHORITATIVE_LAYOUT_SPREAD as _LS,
                          AUTHORITATIVE_SEED as _SEED)  # single authoritative graph config


CONTROLLER_LABELS = {
    "dsceos": "D-SCEOS",
    "projected_gradient_hocbf": "DPG-HOCBF",
    "independent_tracking": "PD baseline",
}
CONTROLLER_ORDER = ["dsceos", "projected_gradient_hocbf", "independent_tracking"]


# Distinct colour per unit class. Chosen to be visually distinguishable
# under typical paper-printing conditions and to follow a rough mnemonic
# (browns/blacks for thermal generation, blues for storage, greens for
# zero-carbon flexibility, oranges for demand-side).
UNIT_CLASS_COLOR = {
    "nuclear_baseload_block":            "#444444",   # dark grey — slow, base
    "lignite_or_hard_coal_block":        "#7d5a3a",   # brown — fossil
    "combined_cycle_gas_turbine":        "#bf6f2a",   # bronze — gas
    "reciprocating_gas_engine":          "#d62728",   # red — fast peaker
    "lithium_ion_battery_storage":       "#1f77b4",   # blue — storage
    "pumped_hydro_storage":              "#17becf",   # cyan — hydro
    "hydrogen_electrolyzer_fuel_cell":   "#9467bd",   # purple — H2
    "heat_pump_aggregator_cluster":      "#2ca02c",   # green — heat-pump
    "ev_v2g_charging_hub":               "#bcbd22",   # olive — EV
    "industrial_demand_response":        "#ff7f0e",   # orange — DR
}


def load_run(run_dir: Path) -> dict:
    """Load state_history.npz and summary.json for a single run."""
    data = np.load(run_dir / "state_history.npz")
    with open(run_dir / "summary.json") as f:
        summary = json.load(f)
    return {
        "time": data["time"],
        "positions": data["positions"],
        "velocities": data["velocities"],
        "controls": data["controls"],
        "summary": summary,
    }


def get_unit_classes(cluster_name: str) -> list[str]:
    """Return the kind-string of each unit in the named cluster."""
    units, physical = build_realistic_units(cluster_name)
    return [p.name for p in physical]


def build_aggregate_blocks(cluster_name: str) -> np.ndarray:
    """Reconstruct the A_i aggregate blocks used by the controller."""
    from dsceos_validation import arrays_from_units, make_aggregate_blocks
    units, _ = build_realistic_units(cluster_name)
    arrays = arrays_from_units(units)
    return make_aggregate_blocks(arrays["aggregate_weight"])


def aggregate_internal(positions: np.ndarray, Ablk: np.ndarray) -> np.ndarray:
    """y_internal(t) = (1/N) Σ A_i p_i for a trajectory."""
    return np.mean(np.einsum("imd,tid->tim", Ablk, positions), axis=1)


def positions_to_DSO(positions: np.ndarray, scaling) -> np.ndarray:
    """Convert per-unit internal positions to DSO physical GW / GVAR.
    Per-unit positions p_i are in *internal* (capacity-normalized)
    coordinates. To plot them on the same DSO-GW axes as the target,
    we project them through the scaling layer:
        p_i_DSO = p_i / s_DSO
    so that a unit at the internal target value y_int = s_DSO * y_DSO
    appears at the DSO-target value on the axes, and the capacity-
    weighted mean over units (1/N) Σ A_i p_i_DSO equals y_DSO.
    """
    out = np.zeros_like(positions)
    for t in range(positions.shape[0]):
        for i in range(positions.shape[1]):
            out[t, i] = scaling.internal_to_dso(positions[t, i])
    return out


def aggregate_DSO(positions: np.ndarray, Ablk: np.ndarray, scaling) -> np.ndarray:
    """Compute the DSO-side aggregate trajectory in physical GW."""
    y_int = aggregate_internal(positions, Ablk)
    out = np.zeros_like(y_int)
    for t in range(y_int.shape[0]):
        out[t] = scaling.internal_to_dso(y_int[t])
    return out


def compute_J_optimum_aggregate(scenario_name: str, cluster_name: str,
                                target_internal: np.ndarray,
                                atw: float = 10.0) -> tuple[np.ndarray | None,
                                                            float | None]:
    """Compute the J(p)-globally-optimal aggregate y* for the realistic
    cluster + target. Returns (y*_internal, J*) or (None, None)."""
    try:
        from dsceos_validation import (
            arrays_from_units, make_aggregate_blocks,
            make_fixed_local_graph, make_physical_layout,
            solve_reference_optimum, ClusterConfig,
        )
        from dsceos_controller import DSCEOSConfig
    except ImportError:
        return None, None

    units, _ = build_realistic_units(cluster_name)
    arrays = arrays_from_units(units)
    n = len(units)
    cluster_cfg = ClusterConfig(
        n_thermal=3, n_storage=3, n_hydrogen=3, n_emobility=3,
        n_industrial=3, seed=_SEED, communication_radius=_R045, neighbour_count=_K4,
        layout_spread=_LS,
    )
    layout = make_physical_layout(n, cluster_cfg)
    W = make_fixed_local_graph(layout, cluster_cfg.communication_radius,
                               cluster_cfg.neighbour_count)
    Ablk = make_aggregate_blocks(arrays["aggregate_weight"])

    p0 = np.zeros((n, 2), dtype=float)  # start at rest (deviation = 0)
    ccfg = DSCEOSConfig(aggregate_tracking_weight=atw)
    J_star, p_star, ok = solve_reference_optimum(
        p0, arrays, W, Ablk, target_internal, ccfg,
    )
    if not ok or p_star is None:
        return None, None
    y_star = float(np.mean(np.einsum("imd,id->im", Ablk, p_star), axis=0)[0])
    y_star_full = np.mean(np.einsum("imd,id->im", Ablk, p_star), axis=0)
    return y_star_full, float(J_star)


def make_realistic_snapshot(
    runs: dict,
    scenario,
    scaling,
    Ablk: np.ndarray,
    unit_classes: list[str],
    physical: list,
    y_star_internal: np.ndarray | None,
    J_star_internal: float | None,
    output_path: Path,
    cluster_name: str = "realistic_15",
    compute_y_star_per_snapshot: bool = True,
):
    """Snapshot: 3 rows × 3 columns. Rows = controllers. Cols = (t=0, t=T/2, t=T).

    Per-snapshot J(p)-optimum y*(t) is recomputed against the target sampled at
    that snapshot's time, so the green-plus marker tracks the moving target
    (relevant for ramp / mFRR scenarios). For step signals the target is
    constant, so y*(t) is the same at all three snapshots.
    """

    # Compute DSO-side aggregate trajectories for all controllers
    dso_aggs = {c: aggregate_DSO(runs[c]["positions"], Ablk, scaling)
                for c in CONTROLLER_ORDER}

    # The target signal — sampled per-frame for non-static scenarios
    signal = build_signal_for(scenario, scaling)

    def target_at(t_sim):
        """Return DSO-GW target at simulation time t_sim."""
        y_int = signal.position(t_sim)
        return scaling.internal_to_dso(y_int)

    # Use the final-time target for axis-limits and J*-comparison reference
    target_final_internal = signal.position(scenario.horizon_sim)
    target_final_DSO = scaling.internal_to_dso(target_final_internal)
    target_initial_DSO = target_at(0)
    y_star_DSO_final = (scaling.internal_to_dso(y_star_internal)
                        if y_star_internal is not None else None)

    # Compute time-resolved y*(t) at the three snapshot timestamps so the
    # green-plus marker correctly tracks the moving target (for ramp / mFRR).
    # For step signals the result coincides with the final-time y* at all
    # three points. Falls back gracefully if the QP fails at any timestamp.

    # Per-unit capacity boxes from the catalog (used for hard-cap detection
    # in the snapshot rendering below). The box is (lower_P, upper_P,
    # lower_Q, upper_Q) per unit; an internal coordinate within
    # `cap_tol_frac × span` of either bound is flagged as at-cap.
    cap_lower = np.array([[p.P_min_GW, p.Q_min_GVAR] for p in physical])
    cap_upper = np.array([[p.P_max_GW, p.Q_max_GVAR] for p in physical])
    cap_span = cap_upper - cap_lower
    cap_tol_frac = 0.01  # 1% of span counts as "at cap"
    def y_star_at(t_sim: float):
        if not compute_y_star_per_snapshot:
            return y_star_DSO_final
        t_int = signal.position(t_sim)
        try:
            y_star_t, _ = compute_J_optimum_aggregate(
                scenario.name, cluster_name, t_int,
            )
        except Exception:
            return y_star_DSO_final
        if y_star_t is None:
            return y_star_DSO_final
        return scaling.internal_to_dso(y_star_t)

    # === percentile-based auto-scaling ===
    # The naive min/max scaling is dominated by the largest-capacity unit
    # (nuclear baseload reaches DSO-GW deviations up to 2.8 GW per-unit),
    # which compresses the inter-controller differences to invisible
    # sub-pixel range. Instead we use the 5th/95th percentiles of all
    # per-unit positions across all snapshot times to set the natural
    # zoom level, then ENSURE the target, aggregate and y* markers are
    # included by extending the range if needed.
    all_pos = np.concatenate(
        [positions_to_DSO(r["positions"], scaling).reshape(-1, 2)
         for r in runs.values()])
    all_agg = np.concatenate([dso_aggs[c] for c in CONTROLLER_ORDER])
    extra = [target_initial_DSO[None, :], target_final_DSO[None, :]]
    if y_star_DSO_final is not None:
        extra.append(y_star_DSO_final[None, :])

    # Percentile-based natural range (excludes the largest-capacity outliers)
    pos_xmin, pos_ymin = np.percentile(all_pos, 5, axis=0)
    pos_xmax, pos_ymax = np.percentile(all_pos, 95, axis=0)

    # Must-include points (target, aggregate, y*)
    important = np.vstack([all_agg] + extra)
    imp_xmin, imp_ymin = important.min(0)
    imp_xmax, imp_ymax = important.max(0)

    # Combined range: percentile of positions extended to fit important points
    xmin = min(pos_xmin, imp_xmin)
    xmax = max(pos_xmax, imp_xmax)
    ymin = min(pos_ymin, imp_ymin)
    ymax = max(pos_ymax, imp_ymax)

    pad_x = 0.10 * (xmax - xmin + 1.0e-6)
    pad_y = 0.10 * (ymax - ymin + 1.0e-6)
    xmin -= pad_x; xmax += pad_x
    ymin -= pad_y; ymax += pad_y

    # Avoid degenerate Y-axis (e.g. Scenario B has Q-target = 0 everywhere,
    # so the autoscaled span would be ~1e-7 and the units would visually
    # vanish off-axis). Enforce a minimum visible span proportional to
    # the X-span so the figure is still readable.
    min_y_span = 0.15 * (xmax - xmin)
    if (ymax - ymin) < min_y_span:
        y_center = 0.5 * (ymin + ymax)
        ymin = y_center - 0.5 * min_y_span
        ymax = y_center + 0.5 * min_y_span

    # === per-scenario manual override for ramp signals ===
    # For a monotonic ramp (Scenario B), the percentile auto-scaling can
    # crop the t=0 initial cluster (some small-coefficient units start
    # at deviations beyond +/-0.4 GW due to random stochastic initial
    # conditions). To ensure every initial unit position is visible AND
    # the target endpoint is comfortably contained, we override the
    # x-axis with a manual range tied to the target peak.
    if scenario.signal_type == "ramp":
        # x-range: enough to show initial scatter + full ramp trajectory
        # + small padding. P_max is positive; ramp goes 0 -> -P_max.
        # The initial scatter on the negative side can reach -0.5 GW DSO.
        # Include the full ramp endpoint (target_final_DSO[0]) on the
        # ramping side and at least +0.5 GW on the other side.
        P_max = float(scenario.target_P_max_GW)  # this is the absolute peak
        # The ramp goes from target_initial_DSO[0] (0) to target_final_DSO[0]
        # (-P_max). Allow ~0.1 GW padding on each side, and at least
        # +/-0.5 GW for initial-scatter visibility.
        if target_final_DSO[0] < target_initial_DSO[0]:
            # Down-ramp: target reaches a negative value
            xmin_manual = target_final_DSO[0] - 0.1
            xmax_manual = max(0.5, target_initial_DSO[0] + 0.1)
        else:
            # Up-ramp: target reaches a positive value
            xmin_manual = min(-0.5, target_initial_DSO[0] - 0.1)
            xmax_manual = target_final_DSO[0] + 0.1
        xmin = xmin_manual
        xmax = xmax_manual

    # Pick snapshot times: for mFRR-block scenarios we use a longer
    # "tracking phase" + "reached phase" pair per block, so the
    # block-by-block dynamics are visible. For step / ramp scenarios the
    # canonical (t=0, t=T/2, t=T) layout is sufficient.
    horizon_sim = float(scenario.horizon_sim)
    first_ctrl = CONTROLLER_ORDER[0]
    time_arr_ref = runs[first_ctrl]["time"]
    T_ref = runs[first_ctrl]["positions"].shape[0]

    # === scenario-specific snapshot time selection ===
    # Scenario A (step):    6 logarithmic columns concentrating on the
    #                       transient (0 -- 5 min) where all activity is.
    # Scenario B (ramp):    8 equally-spaced columns covering the linear
    #                       PV ramp 0 -- 60 min monotonically.
    # Scenario C (mFRR):    8 equally-spaced columns capturing the four
    #                       15-min mFRR block transitions.
    sig = scenario.signal_type
    if sig == "step":
        # Logarithmic-style sequence focused on the first 5 min transient
        snap_times = [0.0, 0.5, 1.5, 4.0, 12.0, horizon_sim]
        n_cols = len(snap_times)
    elif sig == "multi_ramp":
        # 8 columns aligned with ramp breakpoints: 0, T/6, T/3 (peak +),
        # T/2, 2T/3 (peak -), 5T/6, T - one extra at mid-points for clarity
        # Breakpoints at 0, T/3, 2T/3, T
        n_cols = 8
        snap_times = [k * horizon_sim / 7.0 for k in range(n_cols)]
    else:
        # Default: 8 uniformly-spaced timestamps
        n_cols = 8
        snap_times = [k * horizon_sim / 7.0 for k in range(n_cols)]
    # Label format: show time in physical minutes (= sim units under
    # TIME_SCALE_S_PER_MIN = 1 convention)
    snap_labels = []
    for k, t in enumerate(snap_times):
        if k == 0:
            snap_labels.append(f"t = 0 min")
        elif k == n_cols - 1:
            snap_labels.append(f"t = T = {t:.0f} min")
        else:
            # Format compactly; 1 decimal if needed, else integer
            if abs(t - round(t)) < 0.05:
                snap_labels.append(f"t = {t:.0f} min")
            else:
                snap_labels.append(f"t = {t:.1f} min")
    # Map to grid indices
    snap_idxs_ref = [int(np.argmin(np.abs(time_arr_ref - t)))
                     for t in snap_times]
    # Figure width scales with column count; each panel ~2.0 inches wide
    fig_w = 2.0 * n_cols
    fig, axes = plt.subplots(3, n_cols, figsize=(fig_w, 13.0),
                             sharex=True, sharey=True, squeeze=False)

    snapshot_labels = snap_labels  # local alias used below

    # Precompute y*(t) at each snapshot timestamp (independent of the
    # controller, so done once and cached).
    y_star_at_snap = []
    for s_idx in snap_idxs_ref:
        t_sim = float(time_arr_ref[s_idx])
        y_star_at_snap.append(y_star_at(t_sim))

    for row, ctrl in enumerate(CONTROLLER_ORDER):
        positions = runs[ctrl]["positions"]
        velocities = runs[ctrl]["velocities"]
        time_arr = runs[ctrl]["time"]
        # All controllers share the same time grid; reuse the
        # snap_idxs_ref indices computed once above.
        snap_idxs = snap_idxs_ref

        # Project per-unit positions to DSO-GW scale so that the units
        # and the target live on the same axes and a unit reaching the
        # target value visually overlaps with the target marker.
        positions_DSO = positions_to_DSO(positions, scaling)
        velocities_DSO = positions_to_DSO(velocities, scaling)

        for col, (idx, slabel) in enumerate(zip(snap_idxs, snapshot_labels)):
            ax = axes[row, col]
            pos = positions_DSO[idx]
            vel = velocities_DSO[idx]
            t = time_arr[idx]
            agg_internal = aggregate_internal(positions[idx:idx+1], Ablk)[0]
            agg_DSO = scaling.internal_to_dso(agg_internal)

            # Per-snapshot DSO target (varies for ramp / mFRR scenarios)
            target_DSO = target_at(t)

            # Per-snapshot J(p)-optimum y*(t) — recomputed against the target
            # at this snapshot time. The optimum moves with the target for
            # ramp / mFRR scenarios; for step signals it stays constant.
            y_star_DSO_t = y_star_at_snap[col]
            if y_star_DSO_t is not None:
                ax.scatter(*y_star_DSO_t, marker="P", s=320,
                           c="#2ca02c", edgecolor="black", lw=1.2,
                           zorder=5, alpha=0.85)

            # Per-unit trails leading up to this snapshot (short, recent history)
            trail_window = max(1, T_ref // 12)  # ~ T/12 of the trajectory
            start_t = max(0, idx - trail_window)
            for i, kind in enumerate(unit_classes):
                color = UNIT_CLASS_COLOR[kind]
                seg = positions_DSO[start_t:idx + 1, i, :]
                if seg.shape[0] > 1:
                    ax.plot(seg[:, 0], seg[:, 1], color=color,
                            alpha=0.40, lw=1.0, zorder=3)

            # Units, coloured by class — drawn last so they sit on top of trails
            # Add small jitter to overlapping units so all are visible
            from collections import defaultdict
            position_groups = defaultdict(list)
            for i in range(len(unit_classes)):
                key = (round(pos[i, 0], 3), round(pos[i, 1], 3))
                position_groups[key].append(i)

            # Aspect-aware jitter radii. Using a single circular jitter
            # in data units would visually stretch into an oval whenever
            # the X- and Y-axis spans differ — e.g. in Scenario C the
            # X-range is ~6.4 GW but the Y-range is ~0.8 GVAR, so a
            # 0.16-unit circle becomes 8× more visible in Y than in X.
            # Compute per-axis jitter radii so the cloud is visually
            # circular instead of vertically stretched.
            jitter_x = 0.025 * (xmax - xmin)
            jitter_y = 0.025 * (ymax - ymin)
            largest_overlap = 0
            total_overlapped = 0
            for key, idxs in position_groups.items():
                if len(idxs) == 1:
                    i = idxs[0]
                    color = UNIT_CLASS_COLOR[unit_classes[i]]
                    ax.scatter(pos[i, 0], pos[i, 1], c=color, s=110,
                               edgecolor="black", lw=0.7, zorder=4)
                else:
                    # Spread overlapping units in a small circle
                    n_overlap = len(idxs)
                    total_overlapped += n_overlap
                    if n_overlap > largest_overlap:
                        largest_overlap = n_overlap
                    angles = np.linspace(0, 2 * np.pi, n_overlap, endpoint=False)
                    for j, i in enumerate(idxs):
                        dx = jitter_x * np.cos(angles[j])
                        dy = jitter_y * np.sin(angles[j])
                        color = UNIT_CLASS_COLOR[unit_classes[i]]
                        ax.scatter(pos[i, 0] + dx, pos[i, 1] + dy,
                                   c=color, s=90, edgecolor="black",
                                   lw=0.7, zorder=4)

            # Single annotation for uniform-forcing condition: report the
            # SIZE OF THE LARGEST coinciding cluster, not the total of all
            # coinciding pairs. A baseline with 10 units at the target, 3
            # at a slightly displaced cap-point, and 2 industrial-DR units
            # stuck at their hard-cap shows up as "10/15 coincide" in the
            # largest cluster — still clearly uniform-forcing behaviour.
            if largest_overlap >= 0.6 * len(unit_classes):
                ax.text(0.98, 0.02,
                        f"uniform forcing\n{largest_overlap}/{len(unit_classes)} units coincide",
                        transform=ax.transAxes, ha="right", va="bottom",
                        fontsize=7.5, color="#444",
                        bbox=dict(boxstyle="round,pad=0.3",
                                  fc="#fff8e6", ec="#d4a017", alpha=0.9, lw=0.5),
                        zorder=10)
            elif idx == snap_idxs[-1] and largest_overlap == 0 and len(unit_classes) > 10:
                # Mark heterogeneous allocation in the final D-SCEOS snapshot
                ax.text(0.98, 0.02,
                        "heterogeneous\nallocation",
                        transform=ax.transAxes, ha="right", va="bottom",
                        fontsize=7.5, color="#0a5d2c",
                        bbox=dict(boxstyle="round,pad=0.3",
                                  fc="#e9f7e6", ec="#2ca02c", alpha=0.9, lw=0.5),
                        zorder=10)

            # Hard-cap annotations: flag units that have hit their own
            # capacity-box upper or lower bound. The baseline controllers
            # exhibit this in the realistic scenarios when their uniform-
            # forcing reference exceeds the small-capacity unit's headroom
            # (e.g. industrial DR with P_max = 0.05 GW asked to deliver
            # +0.22 GW per-unit). The flag explains the otherwise puzzling
            # presence of "stuck" units far from the uniform-forcing
            # cluster. The current internal positions are in positions[idx]
            # (per-unit GW deviation from nominal); a unit is "at cap" if
            # any component is within cap_tol_frac × span of either bound.
            p_int = positions[idx]
            for i in range(len(unit_classes)):
                at_upper = (np.abs(p_int[i] - cap_upper[i])
                            < cap_tol_frac * cap_span[i]).any()
                at_lower = (np.abs(p_int[i] - cap_lower[i])
                            < cap_tol_frac * cap_span[i]).any()
                if at_upper or at_lower:
                    # Build a compact label like "@cap +0.05" or "@cap -0.15".
                    # Use whichever component is at its bound; prefer P.
                    if abs(p_int[i, 0] - cap_upper[i, 0]) < cap_tol_frac * cap_span[i, 0]:
                        bound_val = cap_upper[i, 0]
                    elif abs(p_int[i, 0] - cap_lower[i, 0]) < cap_tol_frac * cap_span[i, 0]:
                        bound_val = cap_lower[i, 0]
                    elif abs(p_int[i, 1] - cap_upper[i, 1]) < cap_tol_frac * cap_span[i, 1]:
                        bound_val = cap_upper[i, 1]
                    else:
                        bound_val = cap_lower[i, 1]
                    label = f"@cap {bound_val:+.2f}"
                    # Position the label a little below the unit dot.
                    ax.annotate(
                        label,
                        xy=(pos[i, 0], pos[i, 1]),
                        xytext=(0, -12),
                        textcoords="offset points",
                        ha="center", va="top",
                        fontsize=6.5, color="#8b0000", zorder=8,
                        bbox=dict(boxstyle="round,pad=0.15",
                                  fc="#ffe6e6", ec="#8b0000",
                                  alpha=0.85, lw=0.5),
                    )

            # Velocity arrows — length is normalised to each unit's own
            # *time-constant-natural speed* P_max_i / τ_i. This is the
            # rate at which the unit could cross its own capacity range
            # within its own decision-time constant; it captures the
            # unit's intrinsic dynamic capability rather than its
            # hardware-gradient limit.
            #
            #   u_i = |dP/dt|_i / (P_max_i / τ_i)
            #
            # so that u_i ≈ 1 means the unit is moving fully through
            # its own range within its own response time — a vigorous
            # dynamic state. A slow base-load unit (nuclear, τ = 480 s,
            # P_max = 1.5 GW ⇒ natural speed 0.19 GW/min) running at
            # 0.008 GW/min appears as u ≈ 4%, correctly reflecting that
            # within its own slow decision-time it is only mildly
            # perturbed, not "thrashed about". velocities are stored in
            # per-unit GW per sim-minute and τ is documented in seconds.
            tau_s = np.array([p.response_time_constant_s for p in physical])
            p_max_arr = np.array([p.P_max_GW for p in physical])
            natural_speed = p_max_arr * 60.0 / np.maximum(tau_s, 1e-9)  # GW / sim-min
            vel_internal_raw = velocities[idx]  # per-unit GW / sim-min
            speed_internal = np.linalg.norm(vel_internal_raw, axis=1)
            # Skip arrows below 1% utilisation (visually invisible anyway)
            util = speed_internal / np.maximum(natural_speed, 1e-9)
            # Unit direction (handles zero-speed safely)
            with np.errstate(invalid="ignore", divide="ignore"):
                dirs = np.where(speed_internal[:, None] > 1e-9,
                                vel_internal_raw / speed_internal[:, None],
                                0.0)
            # Visual length: u = 1.0 (full natural speed) spans ~20% of x-range
            arrow_full_length = 0.20 * (xmax - xmin)
            for i in range(len(unit_classes)):
                if util[i] > 0.01:  # >= 1% utilisation
                    L = arrow_full_length * min(util[i], 1.0)
                    ax.annotate("", xy=(pos[i, 0] + dirs[i, 0] * L,
                                        pos[i, 1] + dirs[i, 1] * L),
                                xytext=(pos[i, 0], pos[i, 1]),
                                arrowprops=dict(arrowstyle="->", color="#222",
                                                lw=0.9, alpha=0.65),
                                zorder=5)

            # Aggregate marker (open black circle)
            ax.scatter(*agg_DSO, marker="o", s=240, facecolor="none",
                       edgecolor="black", lw=2.5, zorder=7)

            # Annotation box. All P, Q quantities are *deviations from the
            # unit/fleet nominal operating point*. Each unit is producing/
            # consuming its nominal output P_nom, Q_nom for t < 0; the
            # coordinate system here shows ΔP = P − P_nom and ΔQ = Q − Q_nom.
            # The DSO flexibility request is an instruction to deviate from
            # the nominal output by the target Δ(P,Q):
            # — agg(t)   = fleet-level ΔP, ΔQ aggregate reported to the DSO
            # — target(t)= DSO flexibility request (Δ-quantity)
            # — y*(t)    = J(p)-optimal fleet aggregate Δ that meets this request
            err_t = float(np.linalg.norm(agg_DSO - target_DSO))
            txt = (f"t = {t:5.2f} sim ({t * 60:6.1f} s phys)\n"
                   f"agg = ({agg_DSO[0]:+6.3f}, {agg_DSO[1]:+6.3f}) $\\Delta$GW/GVAR\n"
                   f"target = ({target_DSO[0]:+6.3f}, "
                   f"{target_DSO[1]:+6.3f}) $\\Delta$GW/GVAR\n"
                   f"$\\|y - y_T\\|$ = {err_t:.4f} GW-eq")
            if y_star_DSO_t is not None:
                err_s = float(np.linalg.norm(agg_DSO - y_star_DSO_t))
                txt += f"\n$\\|y - y^*\\|$ = {err_s:.4f} GW-eq"
            # For wide multi-block layouts we compress the annotation box
            # to fit; the t and target are already in the column title.
            if n_cols > 4:
                txt_short = (
                    f"$\\|y\\!-\\!y_T\\|$={err_t:.2f}\n"
                    f"agg=$({agg_DSO[0]:+.2f},{agg_DSO[1]:+.2f})$"
                )
                if y_star_DSO_t is not None:
                    err_s = float(np.linalg.norm(agg_DSO - y_star_DSO_t))
                    txt_short += f"\n$\\|y\\!-\\!y^*\\|$={err_s:.2f}"
                ax.text(0.02, 0.98, txt_short,
                        transform=ax.transAxes, ha="left", va="top",
                        fontsize=6.5,
                        bbox=dict(boxstyle="round,pad=0.2", fc="white",
                                  ec="gray", alpha=0.92))
            else:
                ax.text(0.02, 0.98, txt,
                        transform=ax.transAxes, ha="left", va="top",
                        fontsize=8.5,
                        bbox=dict(boxstyle="round,pad=0.3", fc="white",
                                  ec="gray", alpha=0.92))

            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)
            ax.grid(True, ls=":", alpha=0.4)
            if row == 2 and n_cols <= 4:
                ax.set_xlabel("$\\Delta P$ [GW] (deviation from nominal operating point)",
                              fontsize=10)
            if col == 0:
                ax.set_ylabel(f"{CONTROLLER_LABELS[ctrl]}\n$\\Delta Q$ [GVAR]",
                              fontsize=11)
            if row == 0:
                ax.set_title(slabel, fontsize=11 if n_cols <= 4 else 9)

    # Legend
    legend_handles = []
    for kind, color in UNIT_CLASS_COLOR.items():
        if kind in unit_classes:
            legend_handles.append(
                mlines.Line2D([], [], marker="o", color=color, mec="black",
                              mew=0.7, ms=9, lw=0, label=kind)
            )
    legend_handles.extend([
        mlines.Line2D([], [], marker="o", color="none", mec="black", mew=2.5,
                      ms=14, lw=0, label="aggregate $y(t)$ ($\\Delta$GW)"),
    ])
    if y_star_DSO_final is not None:
        legend_handles.append(
            mlines.Line2D([], [], marker="P", color="#2ca02c", mec="black",
                          mew=1.2, ms=15, lw=0, alpha=0.85,
                          label=r"$J(p)$-optimum $y^*$ ($\Delta$)")
        )
    fig.legend(handles=legend_handles, loc="lower center", ncols=4,
               fontsize=9, bbox_to_anchor=(0.5, -0.015), frameon=False)

    import json as _json
    jt_values = {}
    for c in CONTROLLER_ORDER:
        _s = runs[c].get("summary")
        if _s is None and "run_dir" in runs[c]:
            _s = _json.loads((runs[c]["run_dir"] / "summary.json").read_text())
        jt_values[c] = float(_s["integrated_objective_value"]) if _s else float("nan")
    # Title: describe the actual signal shape. The target is a *flexibility
    # request* — a deviation Δ(P,Q) from the fleet's nominal operating point.
    if scenario.signal_type == "step":
        target_desc = (f"DSO flexibility request = $\\Delta({target_initial_DSO[0]:+.2f}, "
                       f"{target_initial_DSO[1]:+.2f})$ GW/GVAR [step]")
    elif scenario.signal_type == "ramp":
        target_desc = (f"DSO flexibility request: $\\Delta({target_initial_DSO[0]:+.2f}, "
                       f"{target_initial_DSO[1]:+.2f})$ "
                       f"$\\to$ $\\Delta({target_final_DSO[0]:+.2f}, "
                       f"{target_final_DSO[1]:+.2f})$ GW/GVAR [ramp]")
    elif scenario.signal_type == "multi_ramp":
        target_desc = (f"DSO flexibility request: piecewise-linear "
                       f"$\\pm{scenario.target_P_max_GW:.1f}$ GW reverse-balancing schedule"
                       f" (3 ramp transitions)")
    else:
        target_desc = (f"DSO flexibility request: 15-min $\\Delta$-blocks "
                       f"$\\pm{scenario.target_P_max_GW:.1f}$ GW [mFRR]")

    jt_str = "  ".join(
        f"{CONTROLLER_LABELS[c]}: {jt_values[c]:.2f}"
        for c in CONTROLLER_ORDER)
    title = (
        f"Realistic CPES scenario: {scenario.name}\n"
        f"{target_desc}, horizon = {scenario.horizon_min:.0f} min, "
        f"N = {len(unit_classes)} units, fleet = {scaling.P_phys_max_GW:.1f} GW "
        f"({scenario.target_P_max_GW/scaling.P_phys_max_GW*100:.1f}% peak load)\n"
        f"Integrated CPES objective $J_T$: {jt_str}"
    )
    fig.suptitle(title, fontsize=11)

    # For wide multi-column layouts, add small x-axis labels under the
    # bottom-row subplots; otherwise add the standard x-axis label.
    if n_cols > 4:
        for col in range(n_cols):
            axes[2, col].set_xlabel("$\\Delta P$ [GW]", fontsize=8)

    plt.tight_layout(rect=[0, 0.05, 1, 0.92])
    plt.subplots_adjust(hspace=0.18, wspace=0.10)
    plt.savefig(str(output_path), dpi=140, bbox_inches="tight")
    # Vector companion for the two-column manuscript layout (review request). The raster save
    # above is unchanged, so the released PNG stays byte-reproducible; the PDF is written from the
    # same figure object and carries identical content.
    if str(output_path).lower().endswith(".png"):
        plt.savefig(str(output_path)[:-4] + ".pdf", bbox_inches="tight")
    print(f"  Saved: {output_path}")
    plt.close(fig)


def make_realistic_animation(
    runs: dict,
    scenario,
    scaling,
    Ablk: np.ndarray,
    unit_classes: list[str],
    physical: list,
    y_star_internal: np.ndarray | None,
    output_path: Path,
    fps: int = 25,
    trail_length: int = 12,
    frame_stride: int = 1,
    cluster_name: str | None = None,
):
    """Animated 3-panel side-by-side version of the realistic phase portrait.

    The red star is the (time-varying) DSO request y_T(t); the green plus is
    the J(p)-optimum y*(t) of the *current* request, recomputed whenever the
    request changes (cached per target value), so the star/plus pair moves
    together and their gap visualizes the deliberate cost-optimal deviation.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.6), sharex=True, sharey=True)

    signal = build_signal_for(scenario, scaling)

    def target_at(t_sim):
        y_int = signal.position(t_sim)
        return scaling.internal_to_dso(y_int)

    target_initial_DSO = target_at(0)
    target_final_DSO = target_at(scenario.horizon_sim)
    y_star_DSO = (scaling.internal_to_dso(y_star_internal)
                  if y_star_internal is not None else None)

    # Per-frame J(p)-optimum of the *current* request, cached per target
    # value (step/mFRR signals have only a few distinct targets; ramps
    # change per frame, but the N=15/N=60 optimization is fast).
    y_star_track = None
    if y_star_DSO is not None and cluster_name is not None:
        _cache: dict = {}
        def _ystar_for(t_sim):
            t_int = signal.position(t_sim)
            key = tuple(np.round(np.asarray(t_int, dtype=float), 6))
            if key not in _cache:
                try:
                    y_t, _ = compute_J_optimum_aggregate(
                        scenario.name, cluster_name, np.asarray(t_int, dtype=float))
                    _cache[key] = (scaling.internal_to_dso(y_t)
                                   if y_t is not None else y_star_DSO)
                except Exception:
                    _cache[key] = y_star_DSO
            return _cache[key]
        sample_T = min(r["positions"].shape[0] for r in runs.values())
        sample_time = next(iter(runs.values()))["time"]
        y_star_track = {}
        for fr in range(0, sample_T, max(1, int(frame_stride))):
            y_star_track[fr] = _ystar_for(float(sample_time[fr]))

    dso_aggs = {c: aggregate_DSO(runs[c]["positions"], Ablk, scaling)
                for c in CONTROLLER_ORDER}
    # Axis limits: use DSO-projected per-unit positions so units share
    # axes with the target.
    all_pos = np.concatenate(
        [positions_to_DSO(r["positions"], scaling).reshape(-1, 2)
         for r in runs.values()])
    all_agg = np.concatenate([dso_aggs[c] for c in CONTROLLER_ORDER])
    extra = [target_initial_DSO[None, :], target_final_DSO[None, :]]
    if y_star_DSO is not None:
        extra.append(y_star_DSO[None, :])
    if y_star_track:
        extra.append(np.vstack(list(y_star_track.values())))
    bound = np.vstack([all_pos, all_agg] + extra)
    xmin, ymin = bound.min(0)
    xmax, ymax = bound.max(0)
    pad_x = 0.10 * (xmax - xmin + 1.0e-6)
    pad_y = 0.10 * (ymax - ymin + 1.0e-6)
    xmin -= pad_x; xmax += pad_x
    ymin -= pad_y; ymax += pad_y
    # Pure-P scenarios (e.g. a wind ramp with zero GVAR request) would
    # otherwise auto-scale the Q axis to the ~1e-5 numerical-noise level,
    # blowing solver noise up to full-axis marker jumps. Enforce a
    # sensible minimum vertical span tied to the horizontal extent.
    min_y_span = max(0.12 * (xmax - xmin), 0.05)
    if (ymax - ymin) < min_y_span:
        yc = 0.5 * (ymax + ymin)
        ymin, ymax = yc - 0.5 * min_y_span, yc + 0.5 * min_y_span

    artists = {}
    for ax, ctrl in zip(axes, CONTROLLER_ORDER):
        positions = runs[ctrl]["positions"]
        velocities = runs[ctrl]["velocities"]
        time_arr = runs[ctrl]["time"]
        T, N, _ = positions.shape

        # Project per-unit positions and velocities to DSO-GW scale so
        # that they share axes with the target.
        positions_DSO = positions_to_DSO(positions, scaling)
        velocities_DSO = positions_to_DSO(velocities, scaling)

        # Ramp-normalised visual velocity for the quiver. Each per-unit
        # arrow length is proportional to |dP/dt|_i / ramp_max_i, the
        # Velocity quiver length normalised to each unit's *time-constant-
        # natural speed* P_max_i / τ_i (see snapshot rendering for the
        # rationale): u_i = |dP/dt|_i / (P_max_i / τ_i). A unit moving at
        # u = 1.0 traverses its own range within its own response time
        # and gets a full-length arrow (20% of x-axis). A slow base-load
        # unit (nuclear, τ = 480 s) running at 0.008 GW/min on natural
        # speed 0.19 GW/min appears at u ≈ 4%, correctly reflecting a
        # mild dynamic state within its own slow decision-time.
        tau_s = np.array([p.response_time_constant_s for p in physical])
        p_max_arr = np.array([p.P_max_GW for p in physical])
        natural_speed = p_max_arr * 60.0 / np.maximum(tau_s, 1e-9)
        arrow_full_length = 0.20 * (xmax - xmin)
        speed_int_all = np.linalg.norm(velocities, axis=2)  # (T, N) per-unit GW/sim-min
        with np.errstate(invalid="ignore", divide="ignore"):
            util_all = speed_int_all / np.maximum(natural_speed, 1e-9)[None, :]
            util_clip = np.clip(util_all, 0, 1)
            # Unit direction in DSO coords
            speed_safe = np.where(speed_int_all > 1e-9, speed_int_all, 1.0)
            dirs_DSO = velocities_DSO / speed_safe[:, :, None]
        # Visual velocity: dir × (arrow_full_length × util_clip), zeroed below 1%
        mask = util_all > 0.01
        velocities_visual = dirs_DSO * (arrow_full_length * util_clip)[:, :, None]
        velocities_visual = np.where(mask[:, :, None], velocities_visual, 0.0)

        # Reference points (the J(p)-optimum marker is updated per frame
        # to the optimum of the *current* request when tracking is on)
        ystar_scatter = None
        if y_star_DSO is not None:
            y0 = y_star_track[0] if y_star_track else y_star_DSO
            ystar_scatter = ax.scatter(*y0, marker="P", s=280,
                       c="#2ca02c", edgecolor="black", lw=1.2,
                       zorder=5, alpha=0.85)
        # Target scatter starts at the t=0 position; updated per-frame
        target_scatter = ax.scatter(*target_initial_DSO, marker="*", s=340,
                                     c="red", edgecolor="black", lw=1.2,
                                     zorder=6)

        # Per-unit scatter, coloured by class
        colors = [UNIT_CLASS_COLOR[k] for k in unit_classes]
        agents_scatter = ax.scatter(positions_DSO[0, :, 0], positions_DSO[0, :, 1],
                                     c=colors, s=95, edgecolor="black",
                                     lw=0.6, zorder=4)

        # Velocity quiver: visual length = ramp-utilisation × arrow_full_length
        # Use scale=1.0 with scale_units="xy" so the visual vector lengths
        # are taken directly in axes units (no automatic scaling).
        quiv = ax.quiver(positions_DSO[0, :, 0], positions_DSO[0, :, 1],
                         velocities_visual[0, :, 0], velocities_visual[0, :, 1],
                         color="#333333", angles="xy", scale_units="xy",
                         scale=1.0, width=0.004, alpha=0.7, zorder=5)

        # Aggregate marker
        agg_DSO_0 = dso_aggs[ctrl][0]
        agg_scatter = ax.scatter(*agg_DSO_0, marker="o", s=220,
                                  facecolor="none", edgecolor="black",
                                  lw=2.4, zorder=7)

        # Per-unit trails
        trails = []
        for i in range(N):
            line, = ax.plot([], [], color=colors[i], alpha=0.35,
                            lw=0.7, zorder=3)
            trails.append(line)

        text_box = ax.text(
            0.02, 0.98, "", transform=ax.transAxes,
            ha="left", va="top", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", fc="white",
                      ec="gray", alpha=0.92),
        )

        artists[ctrl] = {
            "agents": agents_scatter, "agg": agg_scatter, "quiv": quiv,
            "trails": trails, "text": text_box, "target": target_scatter,
            "ystar": ystar_scatter,
            "positions": positions_DSO, "velocities": velocities_visual,
            "time": time_arr, "N": N, "dso_agg": dso_aggs[ctrl],
        }

        ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)
        ax.grid(True, ls=":", alpha=0.4)
        ax.set_xlabel("$\\Delta P$ [GW] (deviation from rest)")
        ax.set_title(CONTROLLER_LABELS[ctrl], fontsize=12)

    axes[0].set_ylabel("$\\Delta Q$ [GVAR] (deviation from rest)")

    legend_handles = []
    for kind, color in UNIT_CLASS_COLOR.items():
        if kind in unit_classes:
            short = kind.split("_")[0]
            legend_handles.append(
                mlines.Line2D([], [], marker="o", color=color, mec="black",
                              mew=0.6, ms=9, lw=0, label=short)
            )
    legend_handles.extend([
        mlines.Line2D([], [], marker="o", color="none", mec="black", mew=2.5,
                      ms=14, lw=0, label="aggregate $y(t)$"),
        mlines.Line2D([], [], marker="*", color="red", mec="black",
                      mew=1.0, ms=18, lw=0,
                      label=r"DSO request $y_T(t)$ (time-varying)"),
    ])
    if y_star_DSO is not None:
        legend_handles.append(
            mlines.Line2D([], [], marker="P", color="#2ca02c", mec="black",
                          mew=1.2, ms=14, lw=0, alpha=0.85,
                          label=r"$J(p)$-optimum $y^*(t)$ of current request")
        )
    fig.legend(handles=legend_handles, loc="lower center", ncols=5,
               fontsize=8.5, bbox_to_anchor=(0.5, 0.0), frameon=False)
    fig.suptitle(
        f"Realistic CPES scenario: {scenario.name} — "
        f"DSO target ({scenario.target_P_max_GW:+.2f} GW, "
        f"{scenario.target_Q_max_GVAR:+.2f} GVAR), "
        f"horizon = {scenario.horizon_min:.0f} min, N = {len(unit_classes)}\n"
        "axes show deviations $\\Delta(P,Q)$ from each unit's own nominal "
        "operating point --- the pre-event state is $\\Delta = 0$ for every unit",
        fontsize=11,
    )

    T_total = min(artists[c]["positions"].shape[0] for c in CONTROLLER_ORDER)

    def animate(frame: int):
        out = []
        for ctrl in CONTROLLER_ORDER:
            a = artists[ctrl]
            pos = a["positions"][frame]; vel = a["velocities"][frame]
            t = a["time"][frame]
            a["agents"].set_offsets(pos)
            a["quiv"].set_offsets(pos)
            a["quiv"].set_UVC(vel[:, 0], vel[:, 1])

            agg = a["dso_agg"][frame]
            a["agg"].set_offsets(agg[None, :])

            # Per-frame target (DSO GW units, may move for ramp / mFRR)
            target_DSO_t = target_at(t)
            a["target"].set_offsets(target_DSO_t[None, :])
            y_star_now = (y_star_track.get(frame, y_star_DSO)
                          if y_star_track else y_star_DSO)
            if a.get("ystar") is not None and y_star_now is not None:
                a["ystar"].set_offsets(np.asarray(y_star_now)[None, :])

            start = max(0, frame - trail_length)
            for i in range(a["N"]):
                seg = a["positions"][start:frame + 1, i, :]
                a["trails"][i].set_data(seg[:, 0], seg[:, 1])

            err_t = float(np.linalg.norm(agg - target_DSO_t))
            txt = (f"t = {t:5.2f} sim\n"
                   f"agg = ({agg[0]:+6.3f}, {agg[1]:+6.3f}) GW/GVAR\n"
                   f"target = ({target_DSO_t[0]:+6.3f}, "
                   f"{target_DSO_t[1]:+6.3f}) GW/GVAR\n"
                   f"$\\|y - y_T\\|$ = {err_t:.3f} GW-eq")
            if y_star_now is not None:
                err_s = float(np.linalg.norm(agg - y_star_now))
                txt += f"\n$\\|y - y^*\\|$ = {err_s:.3f} GW-eq"
            a["text"].set_text(txt)
            out.extend([a["agents"], a["agg"], a["quiv"], a["text"],
                        a["target"]] + a["trails"])
            if a.get("ystar") is not None:
                out.append(a["ystar"])
        return out

    anim = FuncAnimation(fig, animate, frames=range(0, T_total, max(1, int(frame_stride))), interval=1000 // fps,
                         blit=False, repeat=False)
    plt.tight_layout(rect=[0, 0.10, 1, 0.93])

    suffix = output_path.suffix.lower()
    if suffix == ".mp4":
        try:
            writer = FFMpegWriter(fps=fps, codec="h264", bitrate=2400)
            anim.save(str(output_path), writer=writer, dpi=110)
        except (FileNotFoundError, RuntimeError) as e:
            print(f"  ffmpeg unavailable ({e}); falling back to GIF")
            gif_path = output_path.with_suffix(".gif")
            anim.save(str(gif_path), writer=PillowWriter(fps=fps), dpi=100)
            return gif_path
    elif suffix == ".gif":
        anim.save(str(output_path), writer=PillowWriter(fps=fps), dpi=100)
    print(f"  Saved: {output_path}")
    plt.close(fig)
    return output_path


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scenario", required=True,
                   choices=list(REALISTIC_SCENARIOS.keys()))
    p.add_argument("--cluster", choices=["realistic_15", "realistic_60"],
                   default="realistic_15")
    p.add_argument("--runs-dir", type=Path, required=True)
    p.add_argument("--output-snapshot-png", type=Path, default=None)
    p.add_argument("--output-mp4", type=Path, default=None)
    p.add_argument("--no-jstar", action="store_true",
                   help="Skip the J(p)-optimum marker (faster).")
    p.add_argument("--fps", type=int, default=25)
    p.add_argument("--frame-stride", type=int, default=1,
                   help="Render every k-th timestep (GIF size/time control).")
    args = p.parse_args()

    scenario = REALISTIC_SCENARIOS[args.scenario]
    units, physical = build_realistic_units(args.cluster)
    scaling = compute_fleet_scaling(physical)
    Ablk = build_aggregate_blocks(args.cluster)
    unit_classes = get_unit_classes(args.cluster)

    runs = {}
    for ctrl in CONTROLLER_ORDER:
        run_dir = args.runs_dir / ctrl
        if not (run_dir / "state_history.npz").exists():
            raise FileNotFoundError(
                f"Missing state_history.npz in {run_dir}. "
                f"Run all three controllers first.")
        runs[ctrl] = load_run(run_dir)

    y_star_internal = None
    J_star = None
    if not args.no_jstar:
        print("Computing J(p)-optimal aggregate y* ...")
        # For non-static signals, use the final-time target as the J*-reference,
        # since that is where the controller will settle.
        signal_obj = build_signal_for(scenario, scaling)
        if scenario.signal_type == "step":
            target_internal_for_jstar = signal_obj.position(0)
        else:
            target_internal_for_jstar = signal_obj.position(scenario.horizon_sim)
        y_star_internal, J_star = compute_J_optimum_aggregate(
            args.scenario, args.cluster, target_internal_for_jstar,
        )
        if y_star_internal is not None:
            y_star_DSO = scaling.internal_to_dso(y_star_internal)
            print(f"  y*_DSO = ({y_star_DSO[0]:.3f}, {y_star_DSO[1]:.3f}) GW/GVAR "
                  f"(computed against final-time target)")
            print(f"  J*     = {J_star:.4f} (internal)")
        else:
            print("  (J* computation failed)")

    if args.output_snapshot_png is not None:
        print("Rendering snapshot...")
        make_realistic_snapshot(runs, scenario, scaling, Ablk, unit_classes,
                                physical,
                                y_star_internal, J_star,
                                args.output_snapshot_png,
                                cluster_name=args.cluster,
                                compute_y_star_per_snapshot=False)

    if args.output_mp4 is not None:
        print("Rendering animation...")
        make_realistic_animation(runs, scenario, scaling, Ablk, unit_classes,
                                 physical,
                                 y_star_internal, args.output_mp4,
                                 fps=args.fps,
                                 frame_stride=args.frame_stride,
                                 cluster_name=args.cluster)


if __name__ == "__main__":
    main()
