"""CI spin diagnostics share the caller budget without exporting a full RDM."""
import importlib
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gradscf import ci
from gradscf._memory import tree_storage_bytes


@pytest.mark.parametrize('unrestricted,dtype', [(False, jnp.float64),
                                              (True, jnp.float64), (True, jnp.float32)])
def test_spin_contracts_without_exporting_two_body_density(unrestricted, dtype, monkeypatch):
    module = importlib.import_module('gradscf.ci.spin')
    space = ci.make_uci_space(3, (2, 1)) if unrestricted else ci.make_ci_space(3, 1)
    spin_space = space if unrestricted else ci.make_uci_space(3, (1, 1))
    assert spin_space.determinants == space.determinants
    rng = np.random.default_rng(747)
    vector = jnp.asarray(rng.normal(size=space.size), dtype=dtype)
    overlap = jnp.asarray(rng.normal(scale=.2, size=(3, 3))) if unrestricted else jnp.eye(3)
    gamma = ci.make_rdm2(vector, spin_space)[1]
    na, nb = spin_space.nocc
    expected = .25*(na-nb)**2+.5*(na+nb)-jnp.einsum('pqrs,ps,qr->', gamma, overlap, overlap)

    def forbidden(*args, **kwargs):
        pytest.fail('Scalar spin diagnostic exported a complete two-body density')

    monkeypatch.setattr(module, 'make_rdm2', forbidden, raising=False)
    monkeypatch.setattr(importlib.import_module('gradscf.ci.properties'), 'make_rdm2', forbidden)
    options = dict(overlap_ab=overlap) if unrestricted else {}
    actual = jax.jit(lambda c: ci.spin_square(c, space, **options))(vector)
    assert np.isfinite(np.asarray(actual)).all() and np.isfinite(expected)
    assert actual[0].dtype == jnp.float64
    atol, rtol = (2e-6, 2e-5) if dtype == jnp.float32 else (2e-12, 2e-11)
    np.testing.assert_allclose(actual[0], expected, atol=atol, rtol=rtol)


def test_spin_rejects_caller_exhaustion_before_contraction(monkeypatch):
    module = importlib.import_module('gradscf.ci.spin')
    space = ci.make_uci_space(3, (2, 1))

    def forbidden(*args, **kwargs):
        pytest.fail('Exhausted spin budget reached a contraction')

    monkeypatch.setattr(module, '_budgeted_columns', forbidden, raising=False)
    with pytest.raises(ValueError, match='memory|budget|resident'):
        ci.spin_square(jnp.ones(space.size), space, overlap_ab=jnp.eye(3),
                       max_memory=.001, reserved_bytes=1000)
    with pytest.raises(ValueError, match='reserved_bytes'):
        ci.spin_square(jnp.ones(space.size), space, overlap_ab=jnp.eye(3),
                       max_memory=32., reserved_bytes=-1)


def test_spin_facade_forwards_budget_and_retained_owners(monkeypatch):
    module = importlib.import_module('gradscf.ci.api')
    source = SimpleNamespace(max_memory=32., held=jnp.ones(13))
    method = ci.CISD(source)
    vector = jnp.ones(4)
    method.reference = SimpleNamespace(nocc=1, fock=jnp.ones((2, 2)))
    method._integrals = SimpleNamespace(kernel=jnp.ones(9))
    method.result = SimpleNamespace(coefficients=vector, other=jnp.ones(5))
    method._density_vector = lambda root: vector
    method.space = ci.make_ci_space(2, 1)
    expected = tree_storage_bytes(source, method.reference, method._integrals,
                                  method.result, exclude=(vector,))

    def checked(c, space, **kwargs):
        assert c is vector and space is method.space
        assert kwargs == dict(overlap_ab=None, max_memory=32., reserved_bytes=expected)
        return 'contracted'

    monkeypatch.setattr(module, 'spin_square', checked)
    assert method.spin_square() == 'contracted'


def test_spin_joint_coefficient_overlap_hvp_matches_density_contraction():
    space = ci.make_uci_space(4, (2, 1), frozen=([0], [3]))
    rng = np.random.default_rng(748)
    point = (jnp.asarray(rng.normal(size=space.size)),
             jnp.asarray(np.eye(4)+rng.normal(scale=.1, size=(4, 4))))
    direction = jax.tree.map(lambda x: jnp.asarray(rng.normal(scale=.03, size=x.shape)), point)

    def reference(values):
        c, overlap = values
        gamma = ci.make_rdm2(c, space)[1]
        return 1.75-jnp.einsum('pqrs,ps,qr->', gamma, overlap, overlap, precision='highest')

    def contracted(values):
        c, overlap = values
        return ci.spin_square(c, space, overlap_ab=overlap, max_memory=32., reserved_bytes=512)[0]

    def bundle(function):
        return (jax.value_and_grad(function)(point),
                jax.jvp(jax.grad(function), (point,), (direction,))[1])

    actual, expected = jax.jit(lambda: bundle(contracted))(), jax.jit(lambda: bundle(reference))()
    for a, e in zip(jax.tree.leaves(actual), jax.tree.leaves(expected)):
        assert np.isfinite(np.asarray(a)).all() and np.isfinite(np.asarray(e)).all()
        np.testing.assert_allclose(a, e, atol=3e-10, rtol=3e-9)
    derivative = jax.jit(jax.grad(contracted))
    for step in (3e-4, 1e-4):
        plus = jax.tree.map(lambda x, v: x+step*v, point, direction)
        minus = jax.tree.map(lambda x, v: x-step*v, point, direction)
        finite = jax.tree.map(lambda a, b: (a-b)/(2*step), derivative(plus), derivative(minus))
        for a, e in zip(jax.tree.leaves(actual[1]), jax.tree.leaves(finite)):
            assert np.isfinite(np.asarray(a)).all() and np.isfinite(np.asarray(e)).all()
            np.testing.assert_allclose(a, e, atol=2e-7, rtol=2e-6)
