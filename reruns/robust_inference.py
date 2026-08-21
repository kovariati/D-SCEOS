#!/usr/bin/env python3
"""
Distribution-free paired inference and multiplicity control for the Monte Carlo campaigns.

This is a RE-ANALYSIS of the seed-level records already stored in the monte_carlo_*.json
artefacts; it runs no new simulation. For every (campaign, scenario, comparator) family it
recomputes the paired contrast against D-SCEOS with

  * a paired percentile bootstrap confidence interval on the mean difference
    (resampling seeds, not runs, so the pairing is preserved),
  * an exact two-sided paired sign test, and
  * a Holm-Bonferroni adjustment of the resulting p-values across the whole family of
    comparisons, so the "strictly positive interval" statements are not read as a set of
    independent tests.

Output: monte_carlo_robust_inference.json
"""
from __future__ import annotations
import json, glob, os, math
import numpy as np

B = 20000          # bootstrap resamples
SEED = 20240517    # fixed so the interval endpoints are reproducible
ALPHA = 0.05


def paired_table(records):
    """{(scenario, controller): {seed: J_T}} -> paired arrays against D-SCEOS."""
    by = {}
    for r in records:
        by.setdefault((r["scenario"], r.get("label") or r["controller"]), {})[r["seed"]] = float(r["J_T"])
    return by


def bootstrap_ci(diff, rng, B=B, alpha=ALPHA):
    n = diff.size
    idx = rng.integers(0, n, size=(B, n))
    means = diff[idx].mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def sign_test_p(diff):
    """Exact two-sided sign test (zeros dropped)."""
    nz = diff[diff != 0.0]
    n = nz.size
    if n == 0:
        return 1.0
    k = int((nz > 0).sum())
    # two-sided exact binomial p at p0 = 0.5
    from math import comb
    tail = sum(comb(n, i) for i in range(0, min(k, n - k) + 1)) / (2.0 ** n)
    return float(min(1.0, 2.0 * tail))


def holm(pvals):
    """Holm-Bonferroni adjusted p-values, order preserved."""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        val = (m - rank) * pvals[i]
        running = max(running, val)
        adj[i] = min(1.0, running)
    return adj


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.abspath(os.path.join(here, ".."))
    files = sorted(glob.glob(os.path.join(root, "monte_carlo_*_N*.json")))
    files = [f for f in files if "partial" not in f]
    rng = np.random.default_rng(SEED)

    rows, pvals = [], []
    for f in files:
        d = json.load(open(f))
        recs = d.get("records") or []
        if not recs:
            continue
        campaign = os.path.basename(f).replace("monte_carlo_", "").replace(".json", "")
        by = paired_table(recs)
        scenarios = sorted({s for s, _ in by})
        for sc in scenarios:
            base = by.get((sc, "D-SCEOS"))
            if not base:
                continue
            for ctrl in sorted({c for s, c in by if s == sc and c != "D-SCEOS"}):
                other = by[(sc, ctrl)]
                seeds = sorted(set(base) & set(other))
                if len(seeds) < 3:
                    continue
                diff = np.array([other[s] - base[s] for s in seeds], dtype=float)
                lo, hi = bootstrap_ci(diff, rng)
                p = sign_test_p(diff)
                rows.append(dict(campaign=campaign, scenario=sc, comparator=ctrl,
                                 n_seeds=len(seeds), mean_diff=float(diff.mean()),
                                 boot_ci95_low=lo, boot_ci95_high=hi,
                                 boot_ci_excludes_zero=bool(lo > 0.0),
                                 win_rate_dsceos=float((diff > 0).mean()),
                                 sign_test_p=p))
                pvals.append(p)

    adj = holm(pvals) if pvals else []
    for r, a in zip(rows, adj):
        r["sign_test_p_holm"] = a
        r["significant_after_holm"] = bool(a < ALPHA)

    out = dict(method=dict(bootstrap_resamples=B, rng_seed=SEED, alpha=ALPHA,
                           bootstrap="paired percentile over seeds",
                           test="exact two-sided paired sign test",
                           multiplicity="Holm-Bonferroni across all reported comparisons"),
               n_comparisons=len(rows),
               all_boot_ci_exclude_zero=bool(rows) and all(r["boot_ci_excludes_zero"] for r in rows),
               all_significant_after_holm=bool(rows) and all(r["significant_after_holm"] for r in rows),
               comparisons=rows)
    path = os.path.join(root, "monte_carlo_robust_inference.json")
    json.dump(out, open(path, "w"), indent=2)

    print(f"{'campaign':<10} {'scn':<4} {'comparator':<14} {'mean diff':>10} "
          f"{'boot CI95':>22} {'win':>5} {'p_holm':>9}")
    for r in rows:
        print(f"{r['campaign']:<10} {r['scenario']:<4} {r['comparator']:<14} "
              f"{r['mean_diff']:>10.4f} [{r['boot_ci95_low']:>9.4f},{r['boot_ci95_high']:>9.4f}] "
              f"{r['win_rate_dsceos']:>5.2f} {r['sign_test_p_holm']:>9.2e}")
    print()
    print("all bootstrap CIs exclude zero :", out["all_boot_ci_exclude_zero"])
    print("all significant after Holm     :", out["all_significant_after_holm"])
    print("saved monte_carlo_robust_inference.json")


if __name__ == "__main__":
    main()
