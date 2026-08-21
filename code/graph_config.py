"""Single authoritative graph configuration for the released realistic study.

This module is the ONE place that states the communication-graph parameters and the spectral
fingerprints of the graphs the released results were produced on. The regression test
(``test_graph_config.py``) imports these constants instead of hard-coding them in several places,
and checks the runner's EFFECTIVE configuration against them (see the dry-run test there).

Scope note: these are the parameters of the released study configuration, not a general
recommendation. The per-agent communication volume on these graphs is 9*d_i scalars per sampling
step, which is set by the local degree (topology), not by the fleet size N.
"""

# --- released communication-graph parameters -------------------------------------------------
AUTHORITATIVE_COMMUNICATION_RADIUS = 0.45
AUTHORITATIVE_NEIGHBOUR_COUNT = 4
AUTHORITATIVE_LAYOUT_SPREAD = 1.0
AUTHORITATIVE_SEED = 7

# --- spectral fingerprints of the released graphs (radius 0.45, neighbour_count 4) ------------
# Rebuilt and checked by test_graph_config.py; a changed EFFECTIVE graph will not match these.
GRAPH_FINGERPRINTS = {
    15: {"edges": 26, "d_max": 2.8405, "lambda_N": 4.2500,
         "adjacency_sha256": "8a5ff728b95fcade575d5f29e20ea48f8be77aa497d8197fc4e350b65984fcfd"},
    60: {"edges": 143, "d_max": 6.9827, "lambda_N": 8.0163,
         "adjacency_sha256": "72bf5e8af1764767b38d2c16063712bc084bf176fd39cdb9cf0775fbceaee552"},
}

# fingerprint of the 30-unit STRESS-LADDER graph (radius 0.38, neighbour_count 4, spread 1.0, seed 7)
LADDER_FINGERPRINT = {"n": 30, "edges": 70, "d_max": 5.1097, "lambda_N": 6.0979,
                      "adjacency_sha256": "9979f2a6b47ab1d3f62a45140831b17cb9a62770fa17d84143fab79c03e9d355"}


def adjacency_hash(W):
    """Stable SHA-256 of a weighted adjacency matrix: sorted (i, j, w) triples with the weight
    rounded to 12 decimals. Exact edge-set identity, not just a spectral summary, so a graph change
    that happens to preserve d_max / lambda_N is still detected."""
    import hashlib
    import numpy as np
    idx = np.argwhere(np.asarray(W) > 0)
    rows = sorted((int(i), int(j), format(round(float(W[i, j]), 12), ".12f")) for i, j in idx)
    return hashlib.sha256("|".join(f"{i},{j},{w}" for i, j, w in rows).encode()).hexdigest()

# tolerance used when comparing rebuilt graph spectra against the fingerprints
FINGERPRINT_TOL = 5e-2


# --- stress-ladder graph (a DELIBERATELY different configuration, single-sourced here too) ------
# The seven-regime stress ladder runs on a denser 30-unit fleet at a smaller radius; it is NOT the
# released realistic graph and must not be silently unified with it.
LADDER_COMMUNICATION_RADIUS = 0.38
LADDER_NEIGHBOUR_COUNT = 4
LADDER_LAYOUT_SPREAD = 1.0
LADDER_SEED = 7
# unit mix of the 30-unit ladder fleet (kept here so the ladder graph is fully single-sourced)
LADDER_UNIT_MIX = dict(n_thermal=8, n_storage=6, n_hydrogen=5, n_emobility=5, n_industrial=6)


def ladder_cluster_kwargs():
    """Graph-related ClusterConfig keyword arguments of the stress ladder. The ladder deliberately
    uses a different radius from the realistic study; every ladder script must splat this instead of
    relying on any dataclass default (source-integrity audit major 1)."""
    return dict(seed=LADDER_SEED,
                communication_radius=LADDER_COMMUNICATION_RADIUS,
                neighbour_count=LADDER_NEIGHBOUR_COUNT,
                layout_spread=LADDER_LAYOUT_SPREAD)


def authoritative_cluster_kwargs():
    """The graph-related ClusterConfig keyword arguments of the released realistic study.
    Every released run path should splat this instead of repeating literals."""
    return dict(seed=AUTHORITATIVE_SEED,
                communication_radius=AUTHORITATIVE_COMMUNICATION_RADIUS,
                neighbour_count=AUTHORITATIVE_NEIGHBOUR_COUNT,
                layout_spread=AUTHORITATIVE_LAYOUT_SPREAD)


def build_graph_from_cluster(make_fixed_local_graph, layout, cluster_cfg):
    """Build the communication graph from the EFFECTIVE cluster configuration, never from literals,
    so a configuration change propagates everywhere instead of drifting silently."""
    return make_fixed_local_graph(layout, cluster_cfg.communication_radius,
                                  cluster_cfg.neighbour_count)


def assert_authoritative_cluster(cluster_cfg, who="cluster config"):
    """Raise AssertionError unless an ALREADY-CONSTRUCTED ClusterConfig carries ALL released graph
    parameters (radius, neighbour count, layout spread, seed). This inspects the effective object,
    so a runtime replacement is caught, unlike a source-literal check."""
    checks = [("communication_radius", float(cluster_cfg.communication_radius),
               float(AUTHORITATIVE_COMMUNICATION_RADIUS), 1e-9),
              ("neighbour_count", int(cluster_cfg.neighbour_count),
               int(AUTHORITATIVE_NEIGHBOUR_COUNT), 0),
              ("layout_spread", float(cluster_cfg.layout_spread),
               float(AUTHORITATIVE_LAYOUT_SPREAD), 1e-9),
              ("seed", int(cluster_cfg.seed), int(AUTHORITATIVE_SEED), 0)]
    for name, got, want, tol in checks:
        ok = (got == want) if tol == 0 else (abs(got - want) < tol)
        assert ok, f"{who}: effective {name} {got} != authoritative {want}"
