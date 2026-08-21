"""test_comm_physical_units.py
=============================
Regression guard for the physical-time conversion of the communication rate.

The sampling step is dt = 0.05 SIMULATION units, and one simulation time unit is one wall-clock
MINUTE (Section 7), so one sampling instant is 3 PHYSICAL seconds. The per-agent communication rate
must therefore be

    R_i = 9 scalars * 8 bytes * |N_i| / 3 s

and NOT 9*8*|N_i| / 0.05 (which would treat the step as 0.05 s and overstate the rate by 60x).

This test pins:
  1. the named constants DT_SIM_MIN and DT_PHYS_S in the profiler;
  2. the 60x relationship between them;
  3. the released fleet_solver_profile.json values, which must equal 9*8*mean_neighbours/3
     (physical), i.e. ~106-122 B/s, not ~6-7 kB/s.
"""
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "reruns"))

FAILS = []


def check(name, cond, got=None, exp=None):
    if cond:
        print(f"PASS: {name}")
    else:
        print(f"FAIL: {name}  (got={got!r} exp={exp!r})")
        FAILS.append(name)


# 1) named constants exist and encode the minute->second conversion
import fleet_scaling_profile as fsp  # noqa: E402

check("DT_SIM_MIN == 0.05", abs(fsp.DT_SIM_MIN - 0.05) < 1e-15, fsp.DT_SIM_MIN, 0.05)
check("DT_PHYS_S == 3.0 (60 * DT_SIM_MIN)", abs(fsp.DT_PHYS_S - 3.0) < 1e-12, fsp.DT_PHYS_S, 3.0)
check("DT_PHYS_S == 60 * DT_SIM_MIN", abs(fsp.DT_PHYS_S - 60.0 * fsp.DT_SIM_MIN) < 1e-12,
      fsp.DT_PHYS_S, 60.0 * fsp.DT_SIM_MIN)

# 2) the released profile artefact uses the physical-second rate
_path = os.path.join(_HERE, "fleet_solver_profile.json")
if os.path.exists(_path):
    prof = json.load(open(_path, encoding="utf-8"))
    for row in prof["results"]:
        nbr = row.get("mean_neighbours")
        cm = row.get("comm_bytes_per_s_per_agent", {})
        expected = 9 * 8 * nbr / 3.0
        check(f"profile N={row['N']}: comm mean == 9*8*neighbours/3s",
              abs(cm.get("mean", -1) - expected) <= 1.0, cm.get("mean"), round(expected, 2))
        # sanity: physical rate is ~100-130 B/s, definitely under 1 kB/s (not the 6-7 kB/s bug)
        check(f"profile N={row['N']}: comm mean < 1000 B/s (physical, not 60x inflated)",
              cm.get("mean", 1e9) < 1000.0, cm.get("mean"), "< 1000 B/s")
else:
    print("WARN: fleet_solver_profile.json not present; skipping artefact check")

if FAILS:
    print(f"\n{len(FAILS)} FAILED")
    sys.exit(1)
print("\nALL PASSED")
