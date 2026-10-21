"""Small H2 gates before using CIS second response for molecular spectra."""
import importlib.util
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gradscf import ci, gto, scf
from gradscf.ao2mo import transform_integrals
from gradscf.solvers import EigenResponseConfig, LinearSolverConfig
from gradscf_tools.excited import CISGeometry


@pytest.fixture(scope="module")
def hydrogen():
    pytest.importorskip("pyscf")
    path = Path(__file__).resolve().parents[1]/"examples/tdscf/pyscf_cis_reference.py"
    spec = importlib.util.spec_from_file_location("cis_higher_reference", path)
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    coordinates = np.array([[0., 0., -.7], [0., 0., .7]])
    mol = gto.M(atom=list(zip(("H", "H"), coordinates)), basis="6-31g*", unit="Bohr", cart=True)
    mf = scf.RHF(mol, integral_backend="native", conv_tol=1e-12,
                 conv_tol_density=1e-11, conv_tol_grad=1e-10, max_cycle=180).run()
    assert mf.converged
    return coordinates, mf, reference.evaluate_cis


def test_h2_fixed_mo_cis_h1_hessian_matches_independent_dense_reference(hydrogen):
    coordinates, mf, oracle = hydrogen
    model = CISGeometry(mf, nroots=2)
    inputs = model.energy.inputs(jnp.asarray(coordinates))
    coefficients = jnp.asarray(mf.mo_coeff)
    h1, eri = transform_integrals(inputs["hcore"], coefficients, eri=inputs["eri"])
    reference = oracle(("H", "H"), coordinates, "6-31g*", nroots=2, with_gradient=False)
    # Rotate the independent PySCF CIS matrix into the native virtual frame.
    # H2 has one occupied orbital. Its occupied sign cancels in this similarity.
    rotation = np.asarray(coefficients[:, 1:]).T@reference["overlap"]@reference["coefficients"][:, 1:]
    np.testing.assert_allclose(rotation@rotation.T, np.eye(3), atol=1e-9)
    matrix = jnp.asarray(rotation@reference["cis_matrix"]@rotation.T)
    b = jnp.array([[.2, .3, -.1], [.3, -.1, .2], [-.1, .2, .1]])
    c = jnp.array([[.1, -.2, .2], [-.2, .3, .1], [.2, .1, -.1]])
    d = jnp.diag(jnp.array([.1, .2, -.1]))
    shift = lambda p: p[0]*b+p[1]*c+p[0]*p[1]*d
    weights = jnp.array([.7, -.3])
    config = ci.CIConfig(nroots=2, solver="dense", conv_tol=1e-11)

    def energy(p):
        # A virtual-only h1 perturbation preserves the occupied-virtual
        # stationarity condition and adds the same one-body block to CIS A.
        perturbed = h1.at[1:, 1:].add(shift(p))
        return weights@ci.solve_cis(perturbed, eri, nocc=1, config=config).excitation_energies

    expected = lambda p: weights@jnp.linalg.eigvalsh(matrix+shift(p))[:2]
    point = jnp.array([.13, -.07])
    np.testing.assert_allclose(energy(point), expected(point), atol=2e-8, rtol=0)
    actual = jax.jit(jax.hessian(energy))(point)
    np.testing.assert_allclose(actual, jax.hessian(expected)(point), atol=2e-7, rtol=0)
    step = 1e-4
    first = jax.jit(jax.grad(energy))
    finite = np.column_stack([(first(point+step*v)-first(point-step*v))/(2*step)
                              for v in jnp.eye(2)])
    np.testing.assert_allclose(actual, finite, atol=2e-7, rtol=0)


@pytest.mark.parametrize("response_method", [None, "direct"])
def test_h2_relaxed_cis_geometry_hvp_matches_gradient_differences(hydrogen, response_method):
    coordinates, mf, oracle = hydrogen
    response = (None if response_method is None else EigenResponseConfig(
        linear_config=LinearSolverConfig(method=response_method, max_dense=8, rtol=1e-10)))
    model = CISGeometry(mf, nroots=2, response=response)
    energy, gradient, state = model.total_gradient(coordinates, 0)
    anchor = model.anchor(coordinates, state, 0)
    direction = np.array([[.2, -.1, -.7], [-.2, .1, .7]])
    direction /= np.linalg.norm(direction)
    product = model.total_hessian_vector(coordinates, 0, direction)
    np.testing.assert_allclose(energy, oracle(("H", "H"), coordinates, "6-31g*", nroots=2)["total_energy"],
                               atol=2e-8, rtol=0)
    errors = []
    for step in (1e-3, 3e-4):
        gradients, independent = [], []
        for sign in (1, -1):
            moved = coordinates+sign*step*direction
            _, g, displaced_state = model.total_gradient(moved, 0)
            assert model.match(moved, displaced_state, anchor)["root"] == 0
            gradients.append(g)
            independent.append(oracle(("H", "H"), moved, "6-31g*", nroots=2)["gradient"])
        finite = (gradients[0]-gradients[1])/(2*step)
        reference = (independent[0]-independent[1])/(2*step)
        errors.append(float(np.max(np.abs(product-finite))))
        np.testing.assert_allclose(product, reference, atol=3e-6, rtol=0)
    assert errors[-1] < 2e-6
