from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np

from realistic_scenarios import REALISTIC_SCENARIOS

CONTROLLER_LABELS = {
    'dsceos': 'D-SCEOS',
    'projected_gradient_hocbf': 'DPG-HOCBF',
    'independent_tracking': 'PD baseline',
}
CONTROLLER_COLORS = {
    'dsceos': '#d62728',
    'projected_gradient_hocbf': '#1f77b4',
    'independent_tracking': '#2ca02c',
}
CONTROLLER_ORDER = ['dsceos', 'projected_gradient_hocbf', 'independent_tracking']
SCENARIO_PREFIX = {
    'scenario_a_winter_morning_step': 'N15_a',
    'scenario_b_wind_ramp_down_event': 'N15_b',
    'scenario_c_winter_balancing_mfrr': 'N15_c',
}


def trapz_cumulative(t: np.ndarray, x: np.ndarray) -> np.ndarray:
    if len(t) == 0:
        return np.array([])
    if len(t) == 1:
        return np.array([0.0])
    return np.concatenate([[0.0], np.cumsum(0.5 * (x[1:] + x[:-1]) * np.diff(t))])


def load_run(run_dir: Path) -> dict:
    rows = list(csv.DictReader(open(run_dir / 'metrics.csv', newline='')))
    t = np.array([float(r['time']) for r in rows], dtype=float)
    agg_error = np.array([float(r['aggregate_error']) for r in rows], dtype=float)
    loss = np.array([float(r.get('objective_local_loss', 0.0)) for r in rows], dtype=float)
    sharing = np.array([float(r.get('objective_sharing', 0.0)) for r in rows], dtype=float)
    internal = np.array([float(r.get('objective_internal', 0.0)) for r in rows], dtype=float)
    obj = np.array([float(r.get('objective_value', 0.0)) for r in rows], dtype=float)
    control_energy = np.array([float(r.get('cumulative_control_energy', 0.0)) for r in rows], dtype=float)
    with open(run_dir / 'summary.json', 'r') as f:
        summary = json.load(f)
    return {
        'time': t,
        'agg_error': agg_error,
        'loss': loss,
        'sharing': sharing,
        'internal': internal,
        'obj': obj,
        'cum_obj': (np.cumsum(obj) * float(summary.get('dt', (t[1]-t[0]) if len(t) > 1 else 0.0))
                    if len(t) else np.array([])),  # dt·Σ konvenció = summary integrated_objective_value
        'control_energy': control_energy,
        'summary': summary,
    }


def scenario_pretty_name(key: str) -> str:
    mapping = {
        'scenario_a_winter_morning_step': 'Scenario A',
        'scenario_b_wind_ramp_down_event': 'Scenario B',
        'scenario_c_winter_balancing_mfrr': 'Scenario C',
    }
    return mapping.get(key, key)


def generate_figure(scenario_key: str, results_dir: Path, output_png: Path, also_pdf: bool = False, portrait: bool = False) -> dict:
    scenario = REALISTIC_SCENARIOS[scenario_key]
    prefix = SCENARIO_PREFIX[scenario_key]
    runs = {}
    for ctrl in CONTROLLER_ORDER:
        run_dir = results_dir / f'{prefix}_{ctrl}'
        runs[ctrl] = load_run(run_dir)

    # Panel layout. The default 2x3 landscape grid reproduces the originally released PNGs
    # byte-for-byte. The optional 3x2 portrait grid keeps the same six panels and the same
    # per-panel size, but the narrower overall aspect makes each panel roughly 50% wider when
    # the figure is placed at single-column width in the two-column manuscript layout.
    n_rows, n_cols = (3, 2) if portrait else (2, 3)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=((10.4, 13.35) if portrait else (15.6, 8.9)))
    panels = [
        ('cum_obj', r'(a) cumulative full objective  $J_T(t)=\int_0^t J(p(\tau);y_T(\tau))\,d\tau$',
         'cumulative objective'),
        ('agg_error', r'(b) aggregate tracking error  $\|y(t)-y_T(t)\|$',
         'tracking error norm'),
        ('loss', r'(c) local loss term  $\sum_i C_i(p_i(t))$',
         'instantaneous local loss'),
        ('sharing', r'(d) utilization-sharing term  $\frac{\lambda_s}{2}\rho^\top L_s \rho$',
         'instantaneous sharing penalty'),
        ('internal', r'(e) internal counter-action term',
         'instantaneous internal penalty'),
        ('control_energy', r'(f) cumulative control energy  $\int_0^t \|u(\tau)\|^2 d\tau$',
         'cumulative control energy'),
    ]

    final_values = {key: {} for key, _, _ in panels}
    max_t = 0.0
    for idx, (series_key, title, ylabel) in enumerate(panels):
        ax = axes[idx // n_cols, idx % n_cols]
        ax.set_title(title, fontsize=10.2)
        for ctrl in CONTROLLER_ORDER:
            r = runs[ctrl]
            t = r['time']
            x = r[series_key]
            max_t = max(max_t, float(t[-1]) if len(t) else 0.0)
            ax.plot(t, x, lw=1.8, color=CONTROLLER_COLORS[ctrl], label=CONTROLLER_LABELS[ctrl])
            final_values[series_key][ctrl] = float(x[-1]) if len(x) else float('nan')
        ax.set_xlim(0, max_t)
        ax.set_xlabel(r'$t$ [min] (physical wall-clock)')
        ax.set_ylabel(ylabel, fontsize=9.2)
        ax.grid(True, ls=':', alpha=0.35)
        # annotate only cumulative panels to avoid clutter
        if series_key in {'cum_obj', 'control_energy'}:
            for ctrl in CONTROLLER_ORDER:
                y = final_values[series_key][ctrl]
                ax.annotate(f'{y:.2f}', xy=(max_t, y), xytext=(7, 0), textcoords='offset points',
                            color=CONTROLLER_COLORS[ctrl], fontsize=8.8, fontweight='bold', va='center')

    handles = [mlines.Line2D([], [], color=CONTROLLER_COLORS[c], lw=2.0, label=CONTROLLER_LABELS[c])
               for c in CONTROLLER_ORDER]
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.5, -0.02), ncol=3,
               frameon=False, fontsize=10.4)

    ds = runs['dsceos']['summary']
    dpg = runs['projected_gradient_hocbf']['summary']
    pd = runs['independent_tracking']['summary']
    ds_j = final_values['cum_obj']['dsceos']
    dpg_j = final_values['cum_obj']['projected_gradient_hocbf']
    pd_j = final_values['cum_obj']['independent_tracking']
    dpg_gap = ((dpg_j - ds_j) / dpg_j * 100.0) if dpg_j > 0 else 0.0
    pd_gap = ((pd_j - ds_j) / pd_j * 100.0) if pd_j > 0 else 0.0
    fig.suptitle(
        f"{scenario_pretty_name(scenario_key)} objective-oriented comparison for {scenario.name}\n"
        f"DSO target = ({scenario.target_P_max_GW:+.2f}, {scenario.target_Q_max_GVAR:+.2f}) GW/GVAR, "
        f"horizon T = {scenario.horizon_min:.0f} min   •   "
        f"D-SCEOS lowers final $J_T$ by {dpg_gap:.0f}% vs DPG-HOCBF and {pd_gap:.0f}% vs PD baseline",
        fontsize=11.3,
    )

    plt.tight_layout(rect=[0, 0.045, 1, 0.92])
    plt.subplots_adjust(wspace=0.28, hspace=0.42)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=160, bbox_inches='tight')
    # Optional vector companion (review request: vector PDF for the two-column layout).
    # The raster path above is left byte-identical, so the released PNGs remain reproducible;
    # the PDF is an additional artefact drawn from exactly the same figure object.
    if also_pdf:
        fig.savefig(output_png.with_suffix('.pdf'), bbox_inches='tight')
    plt.close(fig)

    return {
        'scenario': scenario_key,
        'output_png': str(output_png),
        'final_cumulative_JT': final_values['cum_obj'],
        'final_control_energy': final_values['control_energy'],
        'final_tracking_error': final_values['agg_error'],
        'final_local_loss': final_values['loss'],
        'final_sharing_penalty': final_values['sharing'],
        'final_internal_penalty': final_values['internal'],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results-dir', type=Path, required=True)
    ap.add_argument('--output-dir', type=Path, required=True)
    ap.add_argument('--scenario', action='append', default=list(REALISTIC_SCENARIOS.keys()), choices=list(REALISTIC_SCENARIOS.keys()))
    ap.add_argument('--vector-pdf', action='store_true', help='additionally write a vector PDF next to each PNG')
    ap.add_argument('--portrait', action='store_true', help='use the 3x2 portrait panel grid (larger panels at column width)')
    args = ap.parse_args()
    summaries = []
    seen = []
    for sc in args.scenario:
        if sc not in seen:
            seen.append(sc)
    for sc in seen:
        out = args.output_dir / (scenario_pretty_name(sc).lower().replace(' ', '_') + '_objective_main.png')
        summaries.append(generate_figure(sc, args.results_dir, out, also_pdf=args.vector_pdf, portrait=args.portrait))
        print(f'Saved {out}')
    with open(args.output_dir / 'objective_main_figure_summary.json', 'w') as f:
        json.dump(summaries, f, indent=2)


if __name__ == '__main__':
    main()
