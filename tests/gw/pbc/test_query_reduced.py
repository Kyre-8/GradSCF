"""Reduced-dimensional query domain and omitted bare exchange head."""
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

from gradscf.gw.pbc import query
from gradscf.integrals.periodic.coulomb import ReducedBoundary


def model(dimension):
    lattice = np.eye(3)*6.
    reciprocal = 2*np.pi*np.linalg.inv(lattice).T
    fractions = np.array([[0., 0., 0.], [1/3, 0., 0.], [-1/3, 0., 0.]])
    inputs = SimpleNamespace(boundary=ReducedBoundary(jnp.asarray(lattice), dimension, (3, 3, 3)),
        lattice=lattice, kpoints=fractions@reciprocal, coords=np.zeros((27, 3)),
        ao=np.zeros((3, 4, 27, 2)))
    result = SimpleNamespace(mo_energy_spin=jnp.tile(jnp.array([-.5, .5]), (2, 3, 1)),
                             mo_occ_spin=jnp.tile(jnp.array([1., 0.]), (2, 3, 1)))
    return SimpleNamespace(cell=SimpleNamespace(dimension=dimension), inputs=inputs,
        result=result, unrestricted=False, xc='pbe', exxdiv=None), reciprocal


@pytest.mark.parametrize('dimension', [1, 2])
def test_reduced_reference_optin_uses_none_exchange(dimension, monkeypatch):
    mf, _ = model(dimension)
    calls = []
    monkeypatch.setattr(query, 'validate_reference', lambda *a, **k: calls.append(k))
    source, nocc, _, table = query._source(mf, False)
    assert source.shape == (3, 2) and nocc == 1 and table.shape == (3, 3)
    assert calls[0]['low_dimensional'] is True
    mf.exxdiv = 'ewald'
    with pytest.raises(NotImplementedError, match='exxdiv=None'):
        query._source(mf, False)


@pytest.mark.parametrize('dimension', [1, 2])
@pytest.mark.parametrize('component', [.02, 1.])
def test_raw_vacuum_query_is_rejected_before_wrapping(dimension, component, monkeypatch):
    mf, reciprocal = model(dimension)
    monkeypatch.setattr(query, 'validate_reference', lambda *a, **k: None)
    def forbidden(*args, **kwargs):
        pytest.fail('A raw vacuum-direction query must be rejected before get_bands')
    mf.get_bands = forbidden
    point = np.zeros((1, 3))
    point[0, dimension] = component
    with pytest.raises(ValueError, match='nonperiodic'):
        query.g0w0_bands(mf, point@reciprocal, [0])


@pytest.mark.parametrize('dimension', [1, 2])
def test_raw_source_vacuum_reciprocal_vector_is_rejected(dimension, monkeypatch):
    mf, reciprocal = model(dimension)
    monkeypatch.setattr(query, 'validate_reference', lambda *a, **k: None)
    mf.inputs.kpoints[0] += reciprocal[dimension]
    with pytest.raises(ValueError, match='nonperiodic'):
        query._source(mf, False)


def test_omitted_head_uses_actual_cross_overlap():
    # Each frame is orthonormal. Their occupied subspaces are different, so
    # cross-frame overlaps are not Kronecker deltas, even at equal k points.
    left = jnp.eye(4, dtype=complex)
    right = left.at[:2, :2].set(jnp.array([[.6, -.8], [.8, .6]]))
    weights = jnp.ones(4)
    expected = .1*np.array([.36, .64, 0., 0.])
    actual = query._negative_head_exchange(-.2, left, right, weights, 1, 2)
    np.testing.assert_allclose(actual, expected, atol=2e-15, rtol=0)
    for head in (0., .2):
        np.testing.assert_array_equal(query._negative_head_exchange(head, left, right, weights, 1, 2),
                                      np.zeros(4))
    phases = jnp.exp(1j*jnp.array([.2, -.3, .4, .6]))
    phased = query._negative_head_exchange(-.2, left*phases, right*phases[::-1], weights, 1, 2)
    np.testing.assert_allclose(phased, expected, atol=2e-15, rtol=0)


@pytest.mark.parametrize('head', [-.2, .2])
def test_query_restores_negative_head_and_keeps_positive_head_once(head, monkeypatch):
    """A head-only bare model has analytic QP shifts and zero correlation."""
    from gradscf.solvers.nonlinear import ScalarRootConfig
    lattice = np.eye(3)*6.
    inputs = SimpleNamespace(boundary=ReducedBoundary(jnp.asarray(lattice), 1, (2, 1, 1)),
        lattice=lattice, weights=jnp.ones(2), gvectors=jnp.array([[0., 0., 0.], [1., 0., 0.]]),
        kpoints=jnp.array([[0., 0., 0.], [-.5, 0., 0.]]))
    rotation = jnp.array([[.6, -.8], [.8, .6]])
    coefficients = jnp.tile(jnp.eye(2)[None, None], (2, 1, 1, 1))
    mf = SimpleNamespace(inputs=inputs, cell=SimpleNamespace(mesh=(2, 1, 1)), exxdiv=None,
        result=SimpleNamespace(mo_coeff_spin=jnp.tile(jnp.eye(2), (2, 2, 1, 1))))
    data = dict(energy=jnp.array([[-.5, .5]]), selected=np.array([0, 1]), nocc=1,
        source_energy=jnp.array([[-.5, .5], [-.5, .5]]),
        coefficients=jnp.eye(2)[None], target_values=rotation[None],
        shifted_energy=jnp.array([[[-.5, .5]], [[-.5, .5]]]),
        shifted_coeff=coefficients, shifted=np.array([[[0., 0., 0.]], [[-.5, 0., 0.]]]),
        canonical=np.zeros((1, 3)), vxc=jnp.zeros((1, 2)), ef=0., table=np.array([[0, 1], [1, 0]]))
    monkeypatch.setattr(query, '_prepare', lambda *a, **k: data)
    monkeypatch.setattr(query, 'finite_head', lambda inputs: jnp.array(head), raising=False)
    monkeypatch.setattr(query, 'build_band_inputs', lambda *a, **k:
        SimpleNamespace(ao=jnp.eye(2)[None, None]))
    monkeypatch.setattr(query, 'kpoint_product_factor_builder', lambda *a, **k: None)
    monkeypatch.setattr(query, 'kpoint_transition_block', lambda *a, **k:
        dict(factors=jnp.zeros((2, 2), complex), gram=jnp.zeros((2, 2), complex),
             differences=jnp.array([-1., 1.]), signs=jnp.array([1., -1.])))
    def mixed(*args, q_cart, **kwargs):
        values = jnp.zeros((2, 4), complex)
        if np.all(np.asarray(q_cart) == 0.):
            values = values.at[0].set(np.sqrt(max(head, 0.))*rotation.reshape(-1))
        return values
    monkeypatch.setattr(query, 'mixed_product_factors', mixed)
    result = query.g0w0_bands(mf, np.zeros((1, 3)), [0, 1], nw=8,
        self_energy='diagonal',
        root_config=ScalarRootConfig(ftol=1e-10, xtol=1e-10))
    expected = np.array([[-.5, .5]])-head/2*np.array([[.36, .64]])
    assert bool(result.converged)
    np.testing.assert_allclose(result.mo_energy, expected, atol=2e-13, rtol=0)
    np.testing.assert_allclose(result.sigma_qp, 0., atol=2e-13, rtol=0)
