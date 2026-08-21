"""test_topology_reproduction.py
================================
Independent reproduction of the Monte-Carlo topology edge-hashes.

The MC artefacts publish a per-seed canonical edge-list SHA (`edge_sha`) and a distinct-topology
count, and `validate_results.py` checks the hash *set* for uniqueness. That guards integrity of the
released hash list, but on its own it does not prove the hashes correspond to the graphs the seed
actually generates. This test closes that gap: it regenerates each seed's graph from the SAME seed and
generator used by `reruns/monte_carlo.py` (`grng = default_rng(2_000_000 + seed)`,
`random_connected_graph(...)` with the authoritative radius / neighbour cap) and recomputes the
canonical edge-list SHA, then asserts it matches the published `edge_sha`. This makes the topology
claim independently reproducible from the seed alone, without shipping the full edge lists.
"""
import hashlib
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "code"))
sys.path.insert(0, os.path.join(_HERE, "reruns"))

import numpy as np  # noqa: E402
from graph_config import (authoritative_cluster_kwargs,  # noqa: E402,F401
                          AUTHORITATIVE_COMMUNICATION_RADIUS, AUTHORITATIVE_NEIGHBOUR_COUNT)

# reuse the exact generator from the MC module
import importlib.util  # noqa: E402
_spec = importlib.util.spec_from_file_location("mc", os.path.join(_HERE, "reruns", "monte_carlo.py"))
_mc = importlib.util.module_from_spec(_spec)
sys.argv = ["x"]
_spec.loader.exec_module(_mc)

FAILS = []


def _edge_sha_from_W(W):
    iu, ju = np.where(np.triu(W > 0, k=1))
    edge_str = ";".join(f"{int(a)}-{int(b)}" for a, b in zip(iu, ju))
    return hashlib.sha256(edge_str.encode()).hexdigest()[:16]


def check(name, cond, got=None, exp=None):
    if cond:
        print(f"PASS: {name}")
    else:
        print(f"FAIL: {name}  (got={got!r} exp={exp!r})")
        FAILS.append(name)


# fleet sizes per campaign file. REQUIRED campaigns: a missing or edge_sha-less artefact FAILS the
# test (fail-closed) rather than being silently skipped. MC-B N15 is included so the
# consensus-impaired campaign's topologies are also independently reproduced (audit Sig#3).
CAMPAIGNS = {
    "monte_carlo_A_N15.json": ("realistic_15", 15),
    "monte_carlo_B_N15.json": ("realistic_15", 15),
    "monte_carlo_A_N60.json": ("realistic_60", 60),
}

for fname, (cluster, n) in CAMPAIGNS.items():
    path = os.path.join(_HERE, fname)
    if not os.path.exists(path):
        check(f"{fname}: REQUIRED campaign artefact present", False, "absent", "present")
        continue
    d = json.load(open(path, encoding="utf-8"))
    gr = d["graph_realisations"]
    published = gr.get("edge_sha", [])
    if not published:
        check(f"{fname}: REQUIRED edge_sha present for independent reproduction", False, "absent", "present")
        continue
    # one graph is drawn per seed (shared across scenarios); the published list repeats it per record,
    # so regenerate per seed and compare against the first occurrence for that seed.
    seeds = sorted(set(r["seed"] for r in d["records"] if r["controller"] == "dsceos"))
    # map seed -> published edge_sha (take the dsceos records, in order)
    ds_records = [r for r in d["records"] if r["controller"] == "dsceos"]
    pub_by_seed = {}
    for r, h in zip(ds_records, published):
        pub_by_seed.setdefault(r["seed"], h)

    # fail-closed: require the expected number of seeds (30) actually present before asserting
    check(f"{fname}: at least 30 seeds available to reproduce", len(seeds) >= 30, len(seeds), ">= 30")

    n_ok = 0
    for seed in seeds:
        grng = np.random.default_rng(2_000_000 + seed)
        W, _layout, _l2, _lN = _mc.random_connected_graph(
            grng, n, AUTHORITATIVE_COMMUNICATION_RADIUS, AUTHORITATIVE_NEIGHBOUR_COUNT)
        recomputed = _edge_sha_from_W(W)
        if recomputed == pub_by_seed.get(seed):
            n_ok += 1
        else:
            check(f"{fname} seed {seed}: recomputed edge_sha matches published",
                  False, recomputed, pub_by_seed.get(seed))
    check(f"{fname}: all {len(seeds)} seed topologies reproduce from seed alone",
          n_ok == len(seeds) and len(seeds) >= 30, f"{n_ok}/{len(seeds)}", f"{len(seeds)}/{len(seeds)} (>=30)")

if FAILS:
    print(f"\n{len(FAILS)} FAILED")
    sys.exit(1)
print("\nALL PASSED")
