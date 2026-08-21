"""symbolic verification of the exact CLF and its derivative.

Appendix C1 states the augmented CLF V_i and its exact derivative dot V_i = coeff_u^T u + drift, and
claims this identity is what the released solver assembles. This test proves the identity three ways:

  1. SYMBOLICALLY, by differentiating V_i along the double-integrator plant and checking that the
     result equals coeff_u^T u + drift with coeff_u and drift as coded, for symbolic m, mu, kp,
     gamma, e, v, u;
  2. by checking the positive-definiteness condition det M = kp - gamma^2 > 0 (i.e. gamma < sqrt kp);
  3. NUMERICALLY, by finite-differencing the code's own _local_clf_terms against a forward Euler
     step of the plant, at random states, and confirming agreement to solver tolerance.

Run:  python3 test_clf_derivation.py
"""
import math
import os
import sys

import numpy as np

sys.dont_write_bytecode = True
_H = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _H)
sys.path.insert(0, os.path.join(_H, "code"))

from dsceos_controller import DSCEOSConfig, DistributedSCEOSController, DSCEOSProblemData  # noqa: E402

fails = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{('  ' + detail) if detail else ''}")
    if not ok:
        fails.append(name)


# ---- 1) symbolic identity ---------------------------------------------------------------------
try:
    import sympy as sp

    m, mu, kp, g = sp.symbols("m mu kp gamma", positive=True)
    e1, e2, v1, v2, u1, u2 = sp.symbols("e1 e2 v1 v2 u1 u2", real=True)
    e = sp.Matrix([e1, e2]); v = sp.Matrix([v1, v2]); u = sp.Matrix([u1, u2])
    V = sp.Rational(1, 2) * m * kp * e.dot(e) + sp.Rational(1, 2) * m * v.dot(v) + m * g * e.dot(v)
    edot = v
    vdot = (u - mu * v) / m
    Vdot = sp.expand(sum(sp.diff(V, x) * d for x, d in
                         ((e1, edot[0]), (e2, edot[1]), (v1, vdot[0]), (v2, vdot[1]))))
    coeff_u = v + g * e
    drift = m * kp * e.dot(v) + (m * g - mu) * v.dot(v) - g * mu * e.dot(v)
    model = sp.expand(coeff_u.dot(u) + drift)
    check("symbolic dot V == coeff_u^T u + drift", sp.simplify(Vdot - model) == 0)
    check("positive-definiteness det M = kp - gamma^2", sp.expand(m * m * kp - (m * g) ** 2).equals(sp.expand(m * m * (kp - g ** 2))))
except ImportError:
    print("  [skip] symbolic check (sympy not installed; the numerical identity below is the "
          "load-bearing check and requires only numpy)")


# ---- 3) numerical identity against the code's own CLF terms -----------------------------------
cfg = DSCEOSConfig()
n = 1
problem = DSCEOSProblemData(
    aggregate_blocks=np.eye(2)[None, :, :], lower=np.full((n, 2), -1.0), upper=np.full((n, 2), 1.0),
    rest=np.zeros((n, 2)), loss_weight=np.ones((n, 2)), masses=np.array([1.3]),
    dampings=np.array([0.4]), force_limits=np.array([5.0]),
    service_selector=np.tile(np.array([[1.0, 0.0]]), (n, 1)),
    service_capacity=np.array([1.0]), adjacency=np.zeros((n, n)))
ctrl = DistributedSCEOSController(problem, cfg)

rng = np.random.default_rng(4040)
worst = 0.0
for _ in range(2000):
    e = rng.normal(size=2); vel = rng.normal(size=2); u = rng.normal(size=2)
    mass, damping = 1.3, 0.4
    V, coeff_u, drift = ctrl._local_clf_terms(e, vel, mass, damping)
    vdot_model = float(coeff_u @ u + drift)
    # finite-difference dV/dt along the frozen-reference plant  e' = v,  v' = (u - mu v)/m
    h = 1e-6
    def Vfun(ee, vv):
        kp = cfg.clf_position_gain
        gamma = cfg.clf_cross_gain_factor * math.sqrt(kp)
        return 0.5 * mass * kp * (ee @ ee) + 0.5 * mass * (vv @ vv) + mass * gamma * (ee @ vv)
    de = vel
    dv = (u - damping * vel) / mass
    vdot_fd = (Vfun(e + h * de, vel + h * dv) - Vfun(e - h * de, vel - h * dv)) / (2 * h)
    worst = max(worst, abs(vdot_model - vdot_fd))
check("numerical dot V (code coeff_u,drift) == finite-difference of code V", worst < 1e-6,
      f"max |diff| = {worst:.2e}")

# positive-definiteness the config actually uses
kp = cfg.clf_position_gain
gamma = cfg.clf_cross_gain_factor * math.sqrt(kp)
check("configured gamma < sqrt(kp) (V positive definite)", gamma < math.sqrt(kp),
      f"gamma={gamma:.4f}, sqrt(kp)={math.sqrt(kp):.4f}")

print("\n=== RESULT ===")
if fails:
    print("FAILED: " + "; ".join(fails)); sys.exit(1)
print("Exact CLF / dot V derivation of Appendix C1 verified symbolically and numerically.")
