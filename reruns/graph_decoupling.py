#!/usr/bin/env python3
"""
Point 9: separate the coordination (objective) graph from the communication overlay.

In the released design the two coincide, so removing a communication link also changes the objective
and its optimum, and an agent-loss or topology experiment mixes an algorithmic transient with a
change of the optimisation problem. This script separates them.

The objective graph G_o is held FIXED at the released topology, so the optimisation problem, its
reference allocation and J_T are unchanged and remain directly comparable. Only the communication
overlay G_c used by the estimators is degraded, by deleting a fraction of its edges (keeping the
remainder connected). The resulting change in J_T is therefore attributable to the information flow
alone, not to a redefinition of the objective.

Output: graph_decoupling.json
"""
from __future__ import annotations
import os, sys, json
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "code"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import dsceos_controller as dc
import dsceos_validation as dv
from dsceos_validation import run_simulation
from qcqp_crosscheck_run import build_realistic_cfg


def connected(W):
    n = W.shape[0]
    seen, stack = {0}, [0]
    while stack:
        i = stack.pop()
        for j in np.nonzero(W[i] > 0.0)[0]:
            if int(j) not in seen:
                seen.add(int(j)); stack.append(int(j))
    return len(seen) == n


def thin_edges(W, frac, rng, tries=400):
    """Delete a fraction of the edges while keeping the graph connected."""
    iu = [(i, j) for i in range(W.shape[0]) for j in range(i + 1, W.shape[0]) if W[i, j] > 0]
    n_del = int(round(frac * len(iu)))
    if n_del == 0:
        return W.copy(), 0
    for _ in range(tries):
        Wc = W.copy()
        order = rng.permutation(len(iu))
        removed = 0
        for idx in order:
            if removed >= n_del:
                break
            i, j = iu[idx]
            w = Wc[i, j]
            Wc[i, j] = Wc[j, i] = 0.0
            if connected(Wc):
                removed += 1
            else:
                Wc[i, j] = Wc[j, i] = w
        if removed == n_del and connected(Wc):
            return Wc, removed
    return Wc, removed


def run(cluster, sk, frac, seed):
    """Objective graph fixed; only the estimator communication overlay is thinned."""
    scfg = build_realistic_cfg(cluster, sk)
    rng = np.random.default_rng(seed)
    holder = {}

    orig_update = dv.update_peer_estimate
    orig_ctrl_init = dc.DistributedSCEOSController.__init__

    def patched_init(self, problem, config=None):
        orig_ctrl_init(self, problem, config)
        # The controller keeps the FIXED objective graph in problem.adjacency (used by the sharing and
        # internal gradient blocks). The thinned overlay is stored separately for the estimators only.
        W_obj = np.asarray(problem.adjacency, dtype=float)
        Wc, removed = thin_edges(W_obj, frac, rng)
        holder["W_comm"] = Wc
        holder["removed"] = removed
        holder["edges"] = int((W_obj > 0).sum() // 2)
        self._comm_overlay = Wc

    def patched_update(est, truth, W, gateways, gain, gw_gain):
        # estimators run on the thinned overlay
        return orig_update(est, truth, holder.get("W_comm", W), gateways, gain, gw_gain)

    dc.DistributedSCEOSController.__init__ = patched_init
    dv.update_peer_estimate = patched_update
    try:
        res = run_simulation(scfg)
    finally:
        dc.DistributedSCEOSController.__init__ = orig_ctrl_init
        dv.update_peer_estimate = orig_update

    s = res.summary
    return dict(J_T=float(s["integrated_objective_value"]),
                final_aggregate_error=float(s["final_aggregate_error"]),
                max_capacity_violation=float(s["max_capacity_violation"]),
                comm_edges_removed=holder.get("removed", 0),
                objective_edges=holder.get("edges", 0))


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", default="N15_a,N15_c")
    ap.add_argument("--fracs", default="0.0,0.1,0.2,0.3")
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()

    cfg_map = {"N15_a": ("realistic_15", "a"), "N15_b": ("realistic_15", "b"),
               "N15_c": ("realistic_15", "c"), "N60_a": ("realistic_60", "a"),
               "N60_b": ("realistic_60", "b"), "N60_c": ("realistic_60", "c")}
    fracs = [float(x) for x in args.fracs.split(",")]
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    path = os.path.join(root, "graph_decoupling.json")
    out = {}
    if os.path.exists(path):
        try: out = json.load(open(path))
        except Exception: out = {}

    print(f"{'config':<8} {'frac':>6} {'removed':>8} {'J_T':>10} {'dJ_T %':>9} {'cap':>9}")
    for name in [c.strip() for c in args.configs.split(",")]:
        cl, sk = cfg_map[name]
        base = None
        out.setdefault(name, {})
        for f in fracs:
            r = run(cl, sk, f, args.seed)
            if f == 0.0:
                base = r["J_T"]
            rel = 100.0 * (r["J_T"] - base) / max(abs(base), 1e-12) if base else 0.0
            r["rel_delta_pct_vs_full_overlay"] = rel
            out[name][f"{f:.2f}"] = r
            print(f"{name:<8} {f:6.2f} {r['comm_edges_removed']:8d} {r['J_T']:10.5f} "
                  f"{rel:+9.3f} {r['max_capacity_violation']:9.1e}")
            json.dump(out, open(path, "w"), indent=2)
    print("saved graph_decoupling.json")


if __name__ == "__main__":
    main()
