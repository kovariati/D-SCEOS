"""Regression test (m-03): the tracked aggregate un-scales to the physical fleet sum."""
import sys as _sys
_sys.dont_write_bytecode = True  # never leave __pycache__ inside the released package (source-integrity Design note); also avoids stale-bytecode confusion when a module is
# edited and restored within the same second during fault injection
import os,sys
_H=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,_H); sys.path.insert(0,os.path.join(_H,"code"))
import numpy as np
from realistic_cpes_catalog import build_realistic_units, compute_fleet_scaling
from dsceos_validation import arrays_from_units, make_aggregate_blocks, aggregate_output
def test_internal_to_dso_equals_sum():
    for cl in ["realistic_15","realistic_60"]:
        u,phys=build_realistic_units(cl); arr=arrays_from_units(u)
        Ablk=make_aggregate_blocks(arr["aggregate_weight"]); sc=compute_fleet_scaling(phys)
        rng=np.random.default_rng(0)
        for _ in range(100):
            p=rng.uniform(arr["lower"],arr["upper"]); y=aggregate_output(Ablk,p)
            dso=np.array([y[0]/sc.P_scale_DSO_GW_to_internal, y[1]/sc.Q_scale_DSO_GVAR_to_internal])
            assert np.max(np.abs(dso-p.sum(axis=0)))<1e-12
    print("PASS: internal_to_dso(aggregate) == sum_i p_i to 1e-12")
if __name__=="__main__": test_internal_to_dso_equals_sum()
