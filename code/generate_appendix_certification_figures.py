from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np

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
CONTROLLER_LINESTYLES = {'dsceos': '-', 'projected_gradient_hocbf': '--', 'independent_tracking': ':'}
SCENARIO_PREFIX = {
    'scenario_a': ('N15_a', 'Scenario A', 'winter morning step'),
    'scenario_b': ('N15_b', 'Scenario B', 'offshore wind ramp-down'),
    'scenario_c': ('N15_c', 'Scenario C', 'winter balancing mFRR'),
}


def load_metrics(run_dir: Path) -> dict[str, np.ndarray]:
    rows = list(csv.DictReader(open(run_dir / 'metrics.csv', newline='')))
    out: dict[str, list[float]] = {k: [] for k in rows[0].keys()}
    for row in rows:
        for key, value in row.items():
            try:
                out[key].append(float(value))
            except Exception:
                out[key].append(float('nan'))
    return {key: np.asarray(value, dtype=float) for key, value in out.items()}


def load_hmin(run_dir: Path, hmin_clip: float = 0.25) -> np.ndarray:
    """Compute a normalized operating-envelope margin from saved states.

    The saved simulator summary records capacity violation but not the positive
    distance to the nearest operating-envelope face. For the appendix diagnostic,
    h_min(t) is reconstructed directly from state_history.npz. It is the minimum
    signed distance from any unit coordinate to its lower/upper local operating
    bound. Positive values mean all unit-level box constraints are strictly
    satisfied; zero means at least one coordinate lies on the certified boundary.
    """
    data = np.load(run_dir / 'state_history.npz')
    positions = data['positions']
    summary = json.loads((run_dir / 'summary.json').read_text())
    cluster = 'realistic_60' if positions.shape[1] >= 60 else 'realistic_15'

    # Import here so the script can still be inspected without importing the
    # whole validation stack.
    from realistic_cpes_catalog import build_realistic_units
    units, _ = build_realistic_units(cluster)
    lower = np.asarray([u.lower for u in units], dtype=float)
    upper = np.asarray([u.upper for u in units], dtype=float)
    dist_lower = positions - lower[None, :, :]
    dist_upper = upper[None, :, :] - positions
    hmin = np.minimum(dist_lower, dist_upper).min(axis=(1, 2))
    t_npz = data['time']
    return {'time': t_npz, 'hmin': np.clip(hmin, -hmin_clip, hmin_clip)}


def generate_one(scenario_key: str, results_dir: Path, output_dir: Path) -> dict:
    prefix, pretty, subtitle = SCENARIO_PREFIX[scenario_key]
    runs = {ctrl: load_metrics(results_dir / f'{prefix}_{ctrl}') for ctrl in CONTROLLER_ORDER}
    hmins = {ctrl: load_hmin(results_dir / f'{prefix}_{ctrl}') for ctrl in CONTROLLER_ORDER}

    fig, axes = plt.subplots(1, 3, figsize=(15.6, 4.7))
    panels = [
        ('hmin', r'(a) minimum operating-envelope margin  $h_{\min}(t)$', 'minimum margin'),
        ('clf_slack', r'(b) native CLF slack  $s_D(t)$ --- D-SCEOS only', 'CLF slack'),
        ('estimator_spread', r'(c) aggregate-estimator disagreement --- estimator-based controllers', 'estimator disagreement'),
    ]
    max_t = max(float(runs[ctrl]['time'][-1]) for ctrl in CONTROLLER_ORDER)
    summaries: dict[str, dict[str, float | None]] = {}

    undefined: dict = {}
    for ax, (key, title, ylabel) in zip(axes, panels):
        ax.set_title(title, fontsize=10.5)
        summaries[key] = {}
        for ctrl in CONTROLLER_ORDER:
            t = runs[ctrl]['time']
            if key == 'hmin':
                hpack = hmins[ctrl]
                y_raw = hpack['hmin']
                t_raw = hpack['time']
                y = np.interp(t, t_raw, y_raw) if len(y_raw) != len(t) or not np.allclose(t_raw[:len(t)], t[:len(t)]) else y_raw[:len(t)]
            else:
                y = runs[ctrl].get(key, np.full_like(t, np.nan, dtype=float))
            if np.all(np.isnan(y)):
                summaries[key][ctrl] = None
                undefined.setdefault(key, []).append(CONTROLLER_LABELS[ctrl])
                continue
            ax.plot(t, y, color=CONTROLLER_COLORS[ctrl], lw=1.8,
                    ls=CONTROLLER_LINESTYLES[ctrl], label=CONTROLLER_LABELS[ctrl])
            valid = y[~np.isnan(y)]
            summaries[key][ctrl] = float(valid[-1]) if valid.size else None
        if key in {'hmin', 'clf_slack'}:
            ax.axhline(0.0, color='black', lw=0.8, ls='--', alpha=0.60)
        ax.set_xlim(0, max_t)
        ax.set_xlabel(r'$t$ [min] (physical wall-clock)')
        ax.set_ylabel(ylabel, fontsize=9.2)
        ax.grid(True, ls=':', alpha=0.35)
        ax.legend(loc='upper right', fontsize=8.2, frameon=True, framealpha=0.85)
        if key in undefined:
            note = 'not defined for: ' + ', '.join(undefined[key])
            if key == 'clf_slack':
                note += ' (they use the common posthoc HOCBF filter)'
            elif key == 'estimator_spread':
                note += ' (no aggregate-consensus estimator)'
            ax.text(0.50, 0.55, note, transform=ax.transAxes, fontsize=7.8,
                    va='center', ha='center',
                    bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='0.75', alpha=0.9))

    fig.suptitle(f'{pretty} certification-oriented appendix diagnostics for {subtitle}', fontsize=11.5)
    fig.tight_layout(rect=[0, 0.02, 1, 0.90])
    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / f'{scenario_key}_appendix_certification.png'
    fig.savefig(out, dpi=180, bbox_inches='tight')
    # Vector companion for the manuscript (review request). The raster save above is unchanged,
    # so the released PNG remains byte-reproducible.
    if str(out).lower().endswith('.png'):
        fig.savefig(str(out)[:-4] + '.pdf', bbox_inches='tight')
    plt.close(fig)
    return {'scenario': scenario_key, 'file': str(out), 'final_values': summaries}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--results-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    all_summaries = []
    for scenario_key in SCENARIO_PREFIX:
        summary = generate_one(scenario_key, args.results_dir, args.output_dir)
        all_summaries.append(summary)
        print(f"Saved {summary['file']}")
    (args.output_dir / 'appendix_certification_summary.json').write_text(json.dumps(all_summaries, indent=2))


if __name__ == '__main__':
    main()
