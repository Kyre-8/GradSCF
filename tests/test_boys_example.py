"""Checks for the ordinary GradSCF ethylene Boys example."""
import ast
from pathlib import Path
import runpy

import jax.numpy as jnp
import numpy as np
import pytest


@pytest.fixture(scope="module")
def example():
    from gradscf.scf.facade import RKS

    calls = []
    kernel = RKS.kernel

    def counted_kernel(self):
        calls.append(self)
        return kernel(self)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(RKS, "kernel", counted_kernel)
        result = runpy.run_path("examples/localization/ethylene_boys.py")
    assert len(calls) == 1
    return result


def test_example_is_gradscf_only_and_has_no_cli(example):
    from gradscf.scf.facade import RKS

    assert isinstance(example["mf"], RKS)
    tree = ast.parse(Path("examples/localization/ethylene_boys.py").read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            assert all(not item.name.startswith(("pyscf", "argparse")) for item in node.names)
        if isinstance(node, ast.ImportFrom):
            assert not (node.module or "").startswith(("pyscf", "argparse"))


def test_localized_occupied_space_and_derivatives(example):
    assert example["nocc"] == 8
    assert example["orth_error"] < 1e-10
    assert example["density_error"] < 1e-10
    assert jnp.linalg.norm(example["local_gradient"]) < 1e-8
    assert example["score_final"] > example["score_initial"] + 1.
    np.testing.assert_allclose(example["local_ad"], example["analytic"], atol=1e-10)
    np.testing.assert_allclose(example["grad"] @ example["vector"], example["fd"], atol=2e-8)
    np.testing.assert_allclose(example["hvp"], example["fd_hvp"], atol=2e-7)


def test_optimization_trajectory_contains_every_accepted_iteration(example):
    rotations = np.asarray(example["rotation_history"])
    stages = example["iteration_stages"]
    best = example["best"]
    polished = example["polished"]
    assert len(rotations) == 1 + best.nit + polished.nit
    assert len(stages) == len(rotations)
    assert stages[0] == "Initial"
    assert stages.count("L-BFGS-B") == best.nit
    assert stages.count("Newton-CG") == polished.nit
    np.testing.assert_allclose(rotations[-1], example["u_opt"], atol=1e-12)
    np.testing.assert_allclose(
        rotations.transpose(0, 2, 1) @ rotations,
        np.broadcast_to(np.eye(example["nocc"]), rotations.shape), atol=1e-12,
    )
    dipole = np.asarray(example["dipole_mo"])
    scores = [np.sum(np.einsum("pi,xpq,qi->xi", u, dipole, u)**2) for u in rotations]
    assert np.min(np.diff(scores)) > -1e-10


def test_boys_origin_invariance(example):
    d = example["dipole_mo"]
    shifted = d - jnp.array([.2, -.4, .1])[:, None, None] * jnp.eye(example["nocc"])
    gradient = example["gradient"]
    theta = example["theta"]
    np.testing.assert_allclose(gradient(theta, d), gradient(theta, shifted), atol=1e-11)


def test_selected_solution_has_no_negative_local_curvature(example):
    from gradscf.solvers import EigenSolverConfig, LinearOperator, solve_hermitian

    size = example["nparam"]
    zero = jnp.zeros(size)
    operator = LinearOperator(
        (size, size), zero.dtype,
        matvec=lambda v: example["hessp"](zero, v, example["dipoles_local"]),
        diagonal=jnp.ones(size),
    )
    result = solve_hermitian(
        operator, config=EigenSolverConfig(nroots=1, atol=1e-7),
        initial_vectors=jnp.asarray(np.random.default_rng(0).normal(size=(size, 3))),
    )
    assert np.all(result.converged)
    assert result.values[0] > -1e-6


def test_independent_pyscf_comparison(example, monkeypatch):
    pytest.importorskip("pyscf")
    from pyscf.scf import hf

    def forbidden_scf(*args, **kwargs):
        pytest.fail("The comparison must not run PySCF SCF")

    monkeypatch.setattr(hf, "kernel", forbidden_scf)
    compare = runpy.run_path("tests/comparisons/compare_ethylene_boys.py")["compare"]
    _, overlaps, difference = compare(example)
    assert difference >= -1e-7
    assert np.all(overlaps <= 1. + 1e-10)
