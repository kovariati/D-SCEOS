"""Path setup: makes the bundled fixed code and sweep_driver importable
regardless of the current working directory. Import this first."""
import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, os.path.join(_HERE, "code")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
RESULTS_DIR = os.path.join(_HERE, "results")
