"""Moving native integrals -> stationary HF -> GW/BSE, CPU float64."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gradscf import bse, gto, gw, scf
from gradscf_tools.gw_response import MolecularGW


@pytest.fixture(scope="module")
def reference():
    mol = gto.M(atom="H 0 0 -.7; H 0 0 .7", basis="sto-3g", unit="Bohr", cart=True)
    return scf.RHF(mol, conv_tol=1e-12, conv_tol_grad=1e-10,
                   integral_backend="native").run()


@pytest.mark.parametrize("method", ["g0w0", "evgw0", "evgw"])
def test_relaxed_qp_and_weight_geometry_response(reference, method):
    model = MolecularGW(reference, method=method, nw=48, eta=1e-5)
    coordinates = model.coordinates
    direction = jnp.array([[0., 0., -.5], [0., 0., .5]])
    def values(r):
        result = model(r)
        return jnp.concatenate([result.mo_energy, result.qp_weight, result.screening_energy])
    evaluate = jax.jit(values)
    primal, tangent = jax.jit(lambda r: jax.jvp(values, (r,), (direction,)))(coordinates)
    assert np.isfinite(primal).all() and np.isfinite(tangent).all()
    for step in (1e-3, 3e-4, 1e-4):
        finite = (evaluate(coordinates+step*direction)-evaluate(coordinates-step*direction))/(2*step)
        np.testing.assert_allclose(tangent, finite, atol=2e-6, rtol=1e-5)
    weights = jnp.arange(1, primal.size+1, dtype=float)/primal.size
    reverse = jax.jit(jax.grad(lambda r: weights@values(r)))(coordinates)
    np.testing.assert_allclose(jnp.sum(reverse*direction), weights@tangent, atol=2e-8, rtol=2e-7)
    np.testing.assert_allclose(jnp.sum(reverse, axis=0), 0., atol=1e-8, rtol=0)
    independent_path = gw.GW(reference, method=method, nw=48, eta=1e-5,
                              max_cycle=100, conv_tol=1e-10, damp=.3).run()
    assert independent_path.converged
    np.testing.assert_allclose(primal[:2], independent_path.mo_energy, atol=2e-8, rtol=0)
    if method == "evgw":
        np.testing.assert_allclose(primal[:2], primal[-2:], atol=1e-12, rtol=0)
    if method == "evgw0":
        # Fixed W0 in the outer cycle still responds to the input geometry.
        assert np.linalg.norm(np.asarray(tangent[-2:])) > 1e-3


@pytest.mark.parametrize("tda", [True, False])
def test_complete_bse_geometry_chain_and_transition_operator(reference, tda):
    model = MolecularGW(reference, nw=48, eta=1e-5)
    config = bse.BSEConfig(nroots=1, tda=tda, solver="dense", conv_tol=1e-10)
    direction = jnp.array([[0., 0., -.5], [0., 0., .5]])
    def observable(r):
        _, result, dipole, space = model.optical(r, config=config)
        strength = bse.oscillator_strengths(result, dipole, space)
        static = bse.polarizability(result, dipole, space)[2, 2].real
        return jnp.concatenate([result.excitation_energies, strength, jnp.atleast_1d(static)])
    evaluate = jax.jit(observable)
    value, derivative = jax.jit(lambda r: jax.jvp(observable, (r,), (direction,)))(model.coordinates)
    assert np.isfinite(derivative).all() and np.all(np.asarray(value) > 0)
    for step in (1e-3, 3e-4, 1e-4):
        finite = (evaluate(model.coordinates+step*direction)-evaluate(model.coordinates-step*direction))/(2*step)
        np.testing.assert_allclose(derivative, finite, atol=3e-6, rtol=2e-5)


def test_molecular_gw_rejects_unvalidated_reference(reference):
    from copy import copy
    other = copy(reference)
    other.xc = "pbe"
    with pytest.raises(NotImplementedError, match="RHF"):
        MolecularGW(other)


def test_native_pole_seeds_feed_the_same_finite_eta_equation(reference):
    model = MolecularGW(reference, nw=48, eta=1e-5)
    seeds = model.pole_seeds(model.coordinates)
    assert np.all(seeds.valid) and np.all(seeds.weights > .5)
    continued = MolecularGW(reference, nw=48, eta=1e-5, qp_solver="newton", qp_initial=seeds.seeds)
    result = jax.jit(continued)(model.coordinates)
    assert bool(result.converged)
    np.testing.assert_allclose(result.mo_energy, model(model.coordinates).mo_energy, atol=2e-9, rtol=0)
    with pytest.raises(ValueError, match="max_dense"):
        model.pole_seeds(model.coordinates, max_dense=2)
    with pytest.raises(TypeError, match="host-only"):
        jax.jit(model.pole_seeds)(model.coordinates)
    restricted = MolecularGW(reference, g_orbitals=(0,))
    with pytest.raises(NotImplementedError, match="full"):
        restricted.pole_seeds(restricted.coordinates)


def test_evgw_numerical_damping_preserves_outer_response(reference):
    """Numerical fixed-point controls must not redefine the converged tangent."""
    direction = jnp.array([[0., 0., -.5], [0., 0., .5]])
    results = []
    for damp in (.3, .7):
        model = MolecularGW(reference, method="evgw0", nw=24, eta=1e-5,
                            max_cycle=600, damp=damp)
        def values(r):
            state = model(r)
            return jnp.concatenate([state.mo_energy, state.qp_weight, state.screening_energy])
        results.append(jax.jit(lambda r: jax.jvp(values, (r,), (direction,)))(model.coordinates))
    for first, second in zip(results[0], results[1]):
        np.testing.assert_allclose(first, second, atol=3e-9, rtol=2e-8)
