"""dpg_n60_retune_robustness.py
==============================
Robustness of the N=60 ordering to comparator retuning.

The main tables use a single global DPG-HOCBF step (0.10) selected as the best-tested value on the
N15/A reference scenario and transferred without retuning. At N=60 that transferred step overshoots,
so part of the headline N=60 gap reflects imperfect transfer rather than the control law alone.

To show the N=60 ordering does NOT depend on that transfer, this script runs the FULL documented
DPG-HOCBF step grid {0.02, 0.04, 0.06, 0.10, 0.20, 0.40, 0.72} on each of the three N=60 scenarios,
identifies the *best-tested* step per scenario (the grid minimiser), and compares that best DPG J_T
against the unchanged D-SCEOS trajectory. On this grid the best-tested N=60 step is 0.04 for all three
scenarios (NOT 0.06); D-SCEOS still attains the lower objective in every scenario, though the
Scenario-B margin is only ~2%. Releasing the full grid (not a single alternative step) makes the
evidence tamper-evident: the published best step and best J_T are recomputed here and re-asserted by
validate_results.py.

Output: dpg_n60_retune_robustness.json
Run from the package root:  python reruns/dpg_n60_retune_robustness.py
"""
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
[_sys.path.insert(0, _p) for _p in (_ROOT, _os.path.join(_ROOT, "code")) if _p not in _sys.path]
_os.chdir(_ROOT)
import json, subprocess
from pathlib import Path

SCEN = {"a": "scenario_a_winter_morning_step",
        "b": "scenario_b_wind_ramp_down_event",
        "c": "scenario_c_winter_balancing_mfrr"}
DPG_GRID = [0.02, 0.04, 0.06, 0.10, 0.20, 0.40, 0.72]   # the documented baseline-sweep grid
TARGET_MULTIPLIER = "4.0"   # design-matched N=60 request (same relative loading as N=15)
_PY = _sys.executable       # Windows-portable: current interpreter, not a literal "python3"

rows = []
for sk, sname in SCEN.items():
    ds = json.load(open(f"results/N60_{sk}_dsceos/summary.json"))["integrated_objective_value"]
    grid = {}
    for step in DPG_GRID:
        out = Path(f"results/_robustness_N60_{sk}_dpg{str(step).replace('.', '')}")
        subprocess.run([_PY, "code/run_realistic_scenario.py", "--controller", "projected_gradient_hocbf",
                        "--scenario", sname, "--cluster", "realistic_60", "--target-multiplier", TARGET_MULTIPLIER,
                        "--safety-filter", "--adaptive-consensus-gain", "--outdir", str(out),
                        "--dpg-step-size", str(step), "--pd-kp", "0.75", "--pd-kd", "2.5"],
                       capture_output=True, check=True)
        grid[str(step)] = round(float(json.load(open(out / "summary.json"))["integrated_objective_value"]), 4)
    best_step = min(grid, key=grid.get)
    best_jt = grid[best_step]
    rows.append(dict(scenario=sk.upper(), J_T_dsceos=round(float(ds), 4),
                     dpg_grid=grid, best_dpg_step=float(best_step), best_dpg_J_T=best_jt,
                     dsceos_lower=bool(ds < best_jt),
                     ratio_dpg_best=round(best_jt / float(ds), 3)))

allwin = all(r["dsceos_lower"] for r in rows)
best_steps = sorted(set(r["best_dpg_step"] for r in rows))
print(f"{'scenario':<10}{'D-SCEOS':>10}{'best step':>11}{'best DPG':>10}{'ratio':>8}   lower?")
for r in rows:
    print(f"{r['scenario']:<10}{r['J_T_dsceos']:>10.4f}{r['best_dpg_step']:>11}{r['best_dpg_J_T']:>10.4f}"
          f"{r['ratio_dpg_best']:>7.3f}x   {'YES' if r['dsceos_lower'] else 'NO'}")
print(f"\nBest-tested N=60 DPG step(s) on the grid: {best_steps}")
print(f"D-SCEOS wins all three N=60 scenarios against the best-tested DPG: {allwin}")
json.dump(dict(note="N=60 ordering robustness to DPG retuning. Full documented step grid run on each "
                    "N=60 scenario; the best-tested step is the grid minimiser (0.04 here, not 0.06). "
                    "Main tables use the N15/A-best step 0.10, transferred. D-SCEOS is compared against "
                    "the best-tested N=60 DPG per scenario.",
               dpg_grid_steps=DPG_GRID, target_multiplier=float(TARGET_MULTIPLIER),
               results=rows, dsceos_wins_all=allwin, best_steps=best_steps),
          open("dpg_n60_retune_robustness.json", "w"), indent=2)
print("-> dpg_n60_retune_robustness.json")
