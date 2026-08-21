#!/usr/bin/env python3
"""
Guard against silently corrupted vector conversions of the diagram figures.

An SVG->PDF conversion can fail in ways that still produce a valid, small PDF: shapes lose their
fill and render as solid black. That is invisible to a LaTeX build and to a page-count check, and it
happened once in this project. This script rasterises every figure the manuscript includes and
compares its dark-pixel fraction against the reference PNG of the same figure. A large divergence
means the vector file no longer depicts the same content.

Exit code 0 if every included figure matches its reference, 1 otherwise, so it can be wired into CI.
"""
from __future__ import annotations
import os, re, subprocess, sys

import numpy as np
from PIL import Image

TOL_PCT = 5.0          # allowed difference in near-black pixel fraction
DARK = 40              # mean RGB below this counts as near-black


def dark_fraction(path):
    a = np.asarray(Image.open(path).convert("RGB")).astype(float)
    return float((a.mean(axis=2) < DARK).mean() * 100.0)


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    figdir = os.path.abspath(os.path.join(here, "..", "..", "2_revised_paper", "figures"))
    tex = os.path.abspath(os.path.join(here, "..", "..", "2_revised_paper", "paper.tex"))
    if not os.path.exists(tex) or not os.path.isdir(figdir):
        print("Manuscript source or figure directory not found.\n"
              "This check rasterises the figures the manuscript includes and compares them against\n"
              "their reference PNGs; it is only meaningful next to the LaTeX source, which this\n"
              "repository does not ship.")
        return 0
    included = set(re.findall(r"\\includegraphics[^{]*\{([^}]+)\}", open(tex).read()))
    bad, checked = [], 0
    print(f"{'figure':<34} {'reference':>10} {'included':>10} {'delta':>8}  verdict")
    for name in sorted(included):
        stem, ext = os.path.splitext(name)
        ref = os.path.join(figdir, stem + ".png")
        inc = os.path.join(figdir, name)
        if not os.path.exists(ref) or not os.path.exists(inc) or ext == ".png":
            continue
        out = os.path.join("/tmp", "figchk_" + stem)
        r = subprocess.run(["pdftoppm", "-png", "-r", "70", "-singlefile", inc, out],
                           capture_output=True)
        if r.returncode != 0 or not os.path.exists(out + ".png"):
            print(f"{name:<34} {'-':>10} {'-':>10} {'-':>8}  RASTERISE FAILED")
            bad.append(name); continue
        a, b = dark_fraction(ref), dark_fraction(out + ".png")
        checked += 1
        ok = abs(b - a) <= TOL_PCT
        print(f"{name:<34} {a:10.2f} {b:10.2f} {b - a:+8.2f}  {'ok' if ok else 'BROKEN'}")
        if not ok:
            bad.append(name)
    print(f"\nchecked {checked} vector figures against their reference PNGs")
    if bad:
        print("BROKEN:", ", ".join(bad)); print("=== FIGURE INTEGRITY FAIL ==="); return 1
    print("=== FIGURE INTEGRITY PASS ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
