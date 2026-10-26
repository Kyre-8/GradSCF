"""Check the standalone example's converged energies and analytic BSSE gradient."""
from functools import lru_cache
from pathlib import Path
import runpy

import jax
import numpy as np


@lru_cache(maxsize=1)
def run_example():
    path = Path(__file__).resolve().parents[1] / "examples/basis/optimize_bsse_contractions.py"
    return runpy.run_path(str(path), run_name="__main__")


def test_bsse_optimization_preserves_fragment_energy_budget():
    ns = run_example()
    assert ns["fit"].success
    before, after = np.asarray(ns["before"]), np.asarray(ns["after"])
    np.testing.assert_allclose(before, [
        -149.935375926426, -74.963402136324, -74.963160069924,
        -74.963543825991, -74.969599187153,
    ], atol=1e-8, rtol=0)
    weights = np.array([0.,1.,1.,-1.,-1.])
    assert weights @ after < weights @ before - 1e-4 / ns["KCAL"]
    assert np.max(after[1:3]-before[1:3]) <= ns["ENERGY_BUDGET"] + 1e-8
    assert np.min(after[1:3]-after[3:5]) >= -1e-8


def test_implicit_bsse_gradient_matches_reconverged_finite_difference():
    ns = run_example()
    x = ns["x0"]
    direction = np.array([.2,-.3,.4,-.1,.15,-.25])
    direction /= np.linalg.norm(direction)
    step = 1e-4
    _, gradient, _, _ = ns["evaluate"](tuple(x))
    value_plus = ns["evaluate"](tuple(x+step*direction))[0]
    value_minus = ns["evaluate"](tuple(x-step*direction))[0]
    fd = (value_plus-value_minus)/(2*step)
    np.testing.assert_allclose(gradient @ direction, fd, atol=2e-4, rtol=2e-4)
