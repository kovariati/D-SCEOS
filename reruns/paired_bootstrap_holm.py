#!/usr/bin/env python3
"""
Paired bootstrap confidence intervals and multiple-comparison control.

This is a pure RE-ANALYSIS of the seed-level records already stored in the Monte Carlo artefacts;
it runs no simulation. It addresses two statistical requests:

  * report a distribution-free paired interval alongside the Student-t interval, and
  * control the family-wise error rate across the many simultaneous paired comparisons.

For each campaign, scenario and comparator it computes the paired difference
d_s = J_T(comparator, seed s) - J_T(D-SCEOS, seed s) and reports:
  - the Student-t 95% interval (as already published),
  - a BCa-free percentile bootstrap 95% interval over 20000 paired resamples,
  - the exact two-sided paired sign test p-value,
  - the Holm-Bonferroni adjusted significance decision over the whole family of comparisons.
"""
from __future__ import annotations
import json, math, os, sys
from itertools import product
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.abspath(os.path.join(HERE, ".."))
CAMPAIGNS = ["monte_carlo_A_N15.json", "monte_carlo_B_N15.json", "monte_carlo_A_N60.json"]
BOOT = 20000
SEED = 20260728


def paired_diffs(records, scenario, comparator, baseline="dsceos"):
    by = {}
    for r in records:
        if r.get("scenario") != scenario:
            continue
        by.setdefault(r["seed"], {})[r["controller"]] = float(r["J_T"])
    seeds = sorted(s for s, v in by.items() if comparator in v and baseline in v)
    return np.array([by[s][comparator] - by[s][baseline] for s in seeds]), seeds


def student_ci(d, alpha=0.05):
    n = d.size
    m = d.mean()
    se = d.std(ddof=1) / math.sqrt(n)
    # two-sided t quantile without scipy: use the normal quantile corrected by a small-sample factor
    from statistics import NormalDist
    z = NormalDist().inv_cdf(1 - alpha / 2)
    # Welch-free approximation is inadequate for small n; use scipy when available
    try:
        from scipy import stats
        tq = stats.t.ppf(1 - alpha / 2, n - 1)
    except Exception:
        tq = z
    return m, m - tq * se, m + tq * se


def bootstrap_ci(d, n_boot=BOOT, alpha=0.05, seed=SEED):
    rng = np.random.default_rng(seed)
    n = d.size
    idx = rng.integers(0, n, size=(n_boot, n))
    means = d[idx].mean(axis=1)
    return float(np.percentile(means, 100 * alpha / 2)), float(np.percentile(means, 100 * (1 - alpha / 2)))


def sign_test_p(d):
    """Exact two-sided paired sign test (ties dropped)."""
    pos = int((d > 0).sum())
    neg = int((d < 0).sum())
    n = pos + neg
    if n == 0:
        return 1.0
    k = min(pos, neg)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def holm(pvals, alpha=0.05):
    order = sorted(range(len(pvals)), key=lambda i: pvals[i])
    m = len(pvals)
    reject = [False] * m
    for rank, i in enumerate(order):
        if pvals[i] <= alpha / (m - rank):
            reject[i] = True
        else:
            break
    return reject


def main():
    rows = []
    for fname in CAMPAIGNS:
        path = os.path.join(PKG, fname)
        if not os.path.exists(path):
            print(f"  [skip] {fname} not present")
            continue
        data = json.load(open(path))
        recs = data["records"]
        scenarios = sorted({r["scenario"] for r in recs})
        comparators = sorted({r["controller"] for r in recs} - {"dsceos"})
        for sc, cmp_ in product(scenarios, comparators):
            d, seeds = paired_diffs(recs, sc, cmp_)
            if d.size < 3:
                continue
            m, tlo, thi = student_ci(d)
            blo, bhi = bootstrap_ci(d)
            rows.append(dict(campaign=fname.replace("monte_carlo_", "").replace(".json", ""),
                             scenario=sc, comparator=cmp_, n_seeds=int(d.size),
                             mean_diff=float(m), t_ci=[float(tlo), float(thi)],
                             boot_ci=[blo, bhi], sign_p=sign_test_p(d),
                             win_rate=float((d > 0).mean())))
    if not rows:
        print("no campaigns available"); return 1
    reject = holm([r["sign_p"] for r in rows])
    for r, rej in zip(rows, reject):
        r["holm_significant"] = bool(rej)

    print(f"{'campaign':<8}{'scn':<4}{'comparator':<14}{'n':>3}  {'mean':>8}  "
          f"{'t-CI':>20}  {'bootstrap-CI':>20}  {'sign p':>8}  Holm")
    for r in rows:
        print(f"{r['campaign']:<8}{r['scenario']:<4}{r['comparator']:<14}{r['n_seeds']:>3}  "
              f"{r['mean_diff']:>8.4f}  [{r['t_ci'][0]:>8.4f},{r['t_ci'][1]:>8.4f}]  "
              f"[{r['boot_ci'][0]:>8.4f},{r['boot_ci'][1]:>8.4f}]  {r['sign_p']:>8.2e}  "
              f"{'yes' if r['holm_significant'] else 'NO'}")

    n_pos_t = sum(1 for r in rows if r["t_ci"][0] > 0)
    n_pos_b = sum(1 for r in rows if r["boot_ci"][0] > 0)
    n_holm = sum(1 for r in rows if r["holm_significant"])
    print(f"\ncomparisons: {len(rows)} | t-CI strictly positive: {n_pos_t} | "
          f"bootstrap-CI strictly positive: {n_pos_b} | Holm-significant: {n_holm}")

    out = os.path.join(PKG, "results", "paired_bootstrap_holm.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(dict(n_bootstrap=BOOT, rng_seed=SEED, comparisons=rows,
                   n_t_ci_positive=n_pos_t, n_bootstrap_ci_positive=n_pos_b,
                   n_holm_significant=n_holm), open(out, "w"), indent=2)
    print(f"saved results/paired_bootstrap_holm.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
