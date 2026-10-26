"""Host-only pole seeds: physical rational roots, projector groups and bounds."""
import importlib

import jax
import jax.numpy as jnp
import numpy as np
from numpy.polynomial import Polynomial
import pytest


def _module():
    try:
        return importlib.import_module("gradscf.gw.branches")
    except ModuleNotFoundError as error:
        raise AssertionError("The shared-solver pole_seeds API is not implemented") from error


def _inputs():
    return dict(base_energy=np.array([.2, .7]), mo_energy=np.array([-.3, .7]), nocc=1,
                poles=np.array([1.]), couplings=np.sqrt(2.)*np.array([[[.3, .4], [.4, .1]]]))


def test_two_pole_polynomial_roots_and_weights():
    data = _inputs()
    result = _module().pole_seeds(**data, orbs=(0,))
    bath, vertices = np.array([-1.3, 1.7]), np.array([.3, .4])
    polynomial = Polynomial([-.2, 1.])*Polynomial([-bath[0], 1.])*Polynomial([-bath[1], 1.])
    polynomial -= vertices[0]**2*Polynomial([-bath[1], 1.])+vertices[1]**2*Polynomial([-bath[0], 1.])
    roots = polynomial.roots()
    weights = 1/(1+np.sum(vertices[None, :]**2/(roots[:, None]-bath)**2, axis=1))
    order = np.argsort(-weights)
    assert result.orbs == (0,) and result.valid.tolist() == [True]
    np.testing.assert_allclose(result.seeds, roots[order[:1]], atol=2e-12, rtol=0)
    np.testing.assert_allclose(result.weights, weights[order[:1]], atol=2e-12, rtol=0)
    np.testing.assert_allclose(result.candidates.energies[0, :3], roots[order], atol=2e-12, rtol=0)
    np.testing.assert_allclose(result.candidates.weights[0, :3], weights[order], atol=2e-12, rtol=0)
    np.testing.assert_array_equal(result.candidates.sizes, [[1, 1, 1, 0]])
    assert np.isnan(result.candidates.energies[0, 3])
    np.testing.assert_allclose(result.weight_gap, [weights[order[0]]-weights[order[1]]], atol=2e-12)
    np.testing.assert_allclose(result.pole_gap, [np.min(abs(result.seeds[0]-bath))], atol=2e-12)
    assert result.residual_norms[0] < 1e-11


def test_rpa_degenerate_rotation_and_mo_phases_are_invariant():
    data = _inputs()
    data.update(poles=np.array([1., 1.]), couplings=np.array([
        [[.3, .4], [.4, .1]], [[.1, -.2], [-.2, .3]]]))
    reference = _module().pole_seeds(**data)
    angle = .37
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    phases = np.array([-1., 1.])
    changed = dict(data, couplings=np.einsum("kl,lmp,m,p->kmp", rotation, data["couplings"], phases, phases))
    actual = _module().pole_seeds(**changed)
    assert reference.valid.all() and actual.valid.all()
    np.testing.assert_allclose(actual.seeds, reference.seeds, atol=3e-12, rtol=0)
    np.testing.assert_allclose(actual.weights, reference.weights, atol=3e-12, rtol=0)
    np.testing.assert_allclose(actual.candidates.weights, reference.candidates.weights, atol=3e-12, rtol=0)


def test_degenerate_primary_uses_group_weight_and_is_invalid(monkeypatch):
    module = _module()
    shared = module.solve_hermitian
    calls = []

    def rotated_shared(matrix, **kwargs):
        calls.append(kwargs)
        result = shared(matrix, **kwargs)
        values = np.asarray(result.raw_values)
        vectors = np.array(result.raw_vectors)
        if abs(values[1]-values[0]) < 1e-12:
            vectors[:, :2] = vectors[:, :2]@np.array([[1., -1.], [1., 1.]])/np.sqrt(2.)
        return result._replace(raw_vectors=jnp.asarray(vectors))

    monkeypatch.setattr(module, "solve_hermitian", rotated_shared)
    result = module.pole_seeds(np.array([-2., 0.]), np.array([-1., 1.]), 1,
                              np.array([1.]), np.zeros((1, 2, 2)), orbs=(1, 0))
    assert result.orbs == (1, 0) and result.valid.tolist() == [True, False]
    assert result.seeds[0] == 0. and np.isnan(result.seeds[1])
    np.testing.assert_allclose(result.weights, [1., 1.], atol=2e-14, rtol=0)
    assert result.candidates.sizes[1, 0] == 2
    assert all(call["config"].method == "dense" and call["config"].nroots == 3 for call in calls)
    assert all(call["response"].target == "subspace" for call in calls)


def test_equal_primary_weights_are_invalid():
    # One coupled resonant bath state gives equal 1/2 weights at -3 and -1.
    result = _module().pole_seeds(np.array([-2., 0.]), np.array([-1., 1.]), 1,
                                 np.array([1.]), np.array([[[np.sqrt(2.), 0.], [0., 0.]]]), orbs=(0,))
    assert not result.valid[0] and np.isnan(result.seeds[0])
    np.testing.assert_allclose(result.candidates.weights[0, :2], [.5, .5], atol=2e-14)
    assert result.weight_gap[0] < 1e-13


def test_dense_capacity_is_checked_before_solver(monkeypatch):
    module = _module()
    def unexpected(*args, **kwargs):
        pytest.fail("Dense capacity must be rejected before calling the shared solver")
    monkeypatch.setattr(module, "solve_hermitian", unexpected)
    with pytest.raises(ValueError, match="max_dense"):
        module.pole_seeds(**_inputs(), max_dense=2)


def test_unconverged_shared_spectrum_does_not_produce_a_seed(monkeypatch):
    module = _module()
    shared = module.solve_hermitian
    def failed_spectrum(*args, **kwargs):
        return shared(*args, **kwargs)._replace(converged=jnp.asarray(False))
    monkeypatch.setattr(module, "solve_hermitian", failed_spectrum)
    result = module.pole_seeds(**_inputs(), orbs=(0,))
    assert not result.valid[0] and np.isnan(result.seeds[0])


def test_numpy_integer_controls_are_accepted():
    result = _module().pole_seeds(**dict(_inputs(), nocc=np.int64(1)),
                                 max_dense=np.int64(3), orbs=(np.int64(0),))
    assert result.valid[0] and result.orbs == (0,)


@pytest.mark.parametrize("name,value", [
    ("nocc", True), ("nocc", 0), ("nocc", 2), ("nocc", 1.5),
    ("max_dense", True), ("max_dense", 0), ("max_dense", 3.5),
    ("orbs", ()), ("orbs", (0, 0)), ("orbs", (-1,)), ("orbs", (2,)),
    ("orbs", (True,)), ("orbs", (0.5,)),
    ("gap_tol", 0.), ("gap_tol", np.nan), ("weight_tol", -1.), ("weight_tol", True),
])
def test_static_parameter_boundaries(name, value):
    data = _inputs()
    data[name] = value
    with pytest.raises(ValueError, match=name):
        _module().pole_seeds(**data)


@pytest.mark.parametrize("name,value", [
    ("base_energy", np.zeros((2, 1))), ("base_energy", np.array([np.nan, 1.])),
    ("mo_energy", np.zeros(3)), ("mo_energy", np.array([0., np.inf])),
    ("poles", np.zeros(1)), ("poles", -np.ones(1)), ("poles", np.ones((1, 1))),
    ("poles", np.zeros(0)),
    ("couplings", np.zeros((1, 2, 3))), ("couplings", np.ones((1, 2, 2), complex)),
    ("couplings", np.array([[[0., .2], [.1, 0.]]])),
])
def test_array_boundaries(name, value):
    data = _inputs()
    data[name] = value
    with pytest.raises(ValueError, match=name):
        _module().pole_seeds(**data)


def test_host_only_rejects_jit_and_ad():
    function = _module().pole_seeds
    data = _inputs()
    def selected(base):
        return function(**dict(data, base_energy=base)).seeds[0]
    with pytest.raises(TypeError, match="host-only"):
        jax.jit(selected)(jnp.asarray(data["base_energy"]))
    with pytest.raises(TypeError, match="host-only"):
        jax.grad(selected)(jnp.asarray(data["base_energy"]))


def test_float64_mode_is_required():
    function = _module().pole_seeds
    enabled = jax.config.x64_enabled
    try:
        jax.config.update("jax_enable_x64", False)
        with pytest.raises(ValueError, match="float64"):
            function(**_inputs())
    finally:
        jax.config.update("jax_enable_x64", enabled)


def test_native_h2_matches_independent_scipy_arrowhead():
    from scipy.linalg import eigh
    from gradscf import gto, scf
    from gradscf.gw.g0w0 import _mo_factors
    from gradscf.gw.poles import rpa_pole_expansion
    from gradscf.integrals.molecular.factorization import eri_pair_matrix_to_df_factors_traceable

    function = _module().pole_seeds
    mf = scf.RHF(gto.M(atom="H 0 0 0; H 0 0 .74", basis="sto-3g", cart=True),
                 conv_tol=1e-12, conv_tol_grad=1e-10, integral_backend="native").run()
    assert mf.converged
    factors = eri_pair_matrix_to_df_factors_traceable(mf._scf_inputs.eri_pair_matrix, nao=2, tol=1e-12)
    mo = _mo_factors(factors, mf.mo_coeff)
    poles, couplings = map(np.asarray, rpa_pole_expansion(mf.mo_energy, mo[:, :1, 1:], mo))
    result = function(mf.mo_energy, mf.mo_energy, 1, poles, couplings)
    for p in range(2):
        matrix = np.diag(np.r_[mf.mo_energy[p], np.asarray(mf.mo_energy)+np.array([-1., 1.])*poles[0]])
        matrix[0, 1:] = matrix[1:, 0] = couplings[0, :, p]/np.sqrt(2*poles[0])
        values, vectors = eigh(matrix)
        chosen = np.argmax(vectors[0]**2)
        assert result.valid[p]
        np.testing.assert_allclose(result.seeds[p], values[chosen], atol=2e-12, rtol=0)
        np.testing.assert_allclose(result.weights[p], vectors[0, chosen]**2, atol=2e-12, rtol=0)
