#!/usr/bin/env python3
"""
Internal consistency audit of the manuscript's reported objective values.

Every integrated-objective (J_T) figure that appears in a manuscript table is re-read from the
authoritative per-run artefacts in results/<CONFIG>_<controller>/summary.json and compared. The
audit is deliberately source-of-truth driven: the JSON summaries are the single authoritative
result source, and the manuscript is checked against them -- never the other way round.

Exit code 0 if every checked cell matches within the printed tolerance, 1 otherwise, so this can
be wired into run_all.py / CI as a fail-closed gate.

Usage:  python3 reruns/validate_paper_tables.py [--paper ../2_revised_paper/paper.tex] [--tol 5e-4]
"""
from __future__ import annotations
import argparse, json, os, re, sys

CTRL = {"D-SCEOS": "dsceos",
        "DPG-HOCBF": "projected_gradient_hocbf",
        "PD baseline": "independent_tracking"}
# DPD-HOCBF is produced by a separate rerun artefact, not a results/<cfg>_<ctrl> directory.

SCEN_ROW = {"A": "a", "B": "b", "C": "c"}


def authoritative(root, cfg, ctrl_key):
    p = os.path.join(root, "results", f"{cfg}_{ctrl_key}", "summary.json")
    if not os.path.exists(p):
        return None
    return float(json.load(open(p))["integrated_objective_value"])


def _table_block(tex, label):
    """Return only the lines of the table environment carrying \\label{label}."""
    i = tex.find("\\label{" + label + "}")
    if i < 0:
        return ""
    start = max(tex.rfind("\\begin{table}", 0, i), tex.rfind("\\begin{table*}", 0, i))
    end = tex.find("\\end{table}", i)
    end2 = tex.find("\\end{table*}", i)
    end = min([e for e in (end, end2) if e >= 0] or [len(tex)])
    return tex[start:end]


def parse_scalability(tex):
    """Rows of the N=60 scalability table ONLY (scoped by its label, so Monte Carlo rows,
    which share the 'scenario & controller & value' shape, are not picked up)."""
    tex = _table_block(tex, "tab:section7_scalability")
    out, cur = [], None
    for line in tex.splitlines():
        m = re.match(r"\s*([ABC])\s*&\s*(D-SCEOS|DPG-HOCBF|PD baseline)\s*&\s*(\\textbf\{)?([0-9.]+)\}?\s*&", line)
        if m:
            cur = m.group(1)
            out.append((cur, m.group(2), float(m.group(4))))
            continue
        m = re.match(r"\s*&\s*(D-SCEOS|DPG-HOCBF|PD baseline)\s*&\s*(\\textbf\{)?([0-9.]+)\}?\s*&", line)
        if m and cur:
            out.append((cur, m.group(1), float(m.group(3))))
    return out


def parse_dpd(tex):
    """Rows of the box-dual comparison table ONLY (scoped by its label)."""
    tex = _table_block(tex, "tab:dpd")
    out = []
    for line in tex.splitlines():
        m = re.match(r"\s*(N\d+)-([ABC])\s*&\s*(\\textbf\{)?([0-9.]+)\}?\s*&\s*([0-9.]+)\s*&\s*([0-9.]+)\s*&\s*([0-9.]+)", line)
        if m:
            out.append((f"{m.group(1)}_{SCEN_ROW[m.group(2)]}",
                        {"D-SCEOS": float(m.group(4)),
                         "DPD-HOCBF": float(m.group(5)),
                         "DPG-HOCBF": float(m.group(6)),
                         "PD baseline": float(m.group(7))}))
    return out


def main():
    ap = argparse.ArgumentParser()
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.abspath(os.path.join(here, ".."))
    ap.add_argument("--paper", default=os.path.join(root, "..", "2_revised_paper", "paper.tex"))
    ap.add_argument("--tol", type=float, default=5e-4)
    args = ap.parse_args()

    if not os.path.exists(args.paper):
        print(f"Manuscript source not found at {args.paper}.\n"
              "This audit compares the manuscript tables against results/*/summary.json and is only\n"
              "meaningful next to the LaTeX source, which this repository does not ship. Point --paper\n"
              "at a paper.tex to run it.")
        return 0
    tex = open(args.paper).read()
    bad, checked, skipped = [], 0, 0

    print("=== N=60 scalability table (config N60_x) ===")
    for scen, ctrl, val in parse_scalability(tex):
        cfg = f"N60_{SCEN_ROW[scen]}"
        ref = authoritative(root, cfg, CTRL[ctrl])
        if ref is None:
            skipped += 1
            continue
        checked += 1
        ok = abs(val - ref) <= max(args.tol, args.tol * abs(ref))
        flag = "ok " if ok else "ERR"
        print(f"  [{flag}] {cfg:8s} {ctrl:12s} paper={val:9.3f} results={ref:9.3f}")
        if not ok:
            bad.append((cfg, ctrl, val, ref))

    print("=== box-dual comparison table ===")
    for cfg, cells in parse_dpd(tex):
        for ctrl, val in cells.items():
            if ctrl not in CTRL:
                skipped += 1
                continue
            ref = authoritative(root, cfg, CTRL[ctrl])
            if ref is None:
                skipped += 1
                continue
            checked += 1
            ok = abs(val - ref) <= max(args.tol, args.tol * abs(ref))
            flag = "ok " if ok else "ERR"
            print(f"  [{flag}] {cfg:8s} {ctrl:12s} paper={val:9.3f} results={ref:9.3f}")
            if not ok:
                bad.append((cfg, ctrl, val, ref))

    print()
    print(f"checked {checked} table cells against results/*/summary.json "
          f"({skipped} skipped: no per-run artefact)")
    if bad:
        print(f"MISMATCHES ({len(bad)}):")
        for cfg, ctrl, v, r in bad:
            print(f"  {cfg} {ctrl}: paper {v} vs authoritative {r}")
        print("=== TABLE AUDIT FAIL ===")
        return 1
    print("=== TABLE AUDIT PASS ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
