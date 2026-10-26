"""A small full export uses the original formula only when its table fits."""
import importlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gradscf import ci


def test_small_density_prefers_connections_but_keeps_a_budgeted_sigma_fallback(monkeypatch):
    properties = importlib.import_module('gradscf.ci.properties')
    space = ci.make_ci_space(6, 2)
    vector = jnp.asarray(np.random.default_rng(771).normal(size=space.size))

    def forbidden(*args, **kwargs):
        pytest.fail('An affordable small density export used the expensive sigma derivative')

    with monkeypatch.context() as patch:
        patch.setattr(properties, 'build_direct_hamiltonian', forbidden)
        patch.setattr(properties, 'build_one_body', forbidden)
        high = jax.jit(lambda c: ci.make_rdm2(c, space, max_memory=64.))
        expected = high(vector)
        assert np.isfinite(expected).all()
        assert np.isnan(high(jnp.zeros_like(vector))).all()
        assert np.isnan(high(vector.at[0].set(jnp.inf))).all()
    connections = importlib.import_module('gradscf.ci._rdm')

    def no_table(*args, **kwargs):
        pytest.fail('An unaffordable density connection table was built')

    monkeypatch.setattr(connections, '_density_connections', no_table)
    for maximum, reserved in ((1., 0), (64., 48_000_000)):
        result = jax.jit(lambda c: ci.make_rdm2(c, space, max_memory=maximum,
                                               reserved_bytes=reserved))(vector)
        np.testing.assert_allclose(result, expected, atol=2e-12, rtol=2e-12)
    monkeypatch.setattr(properties, 'build_direct_hamiltonian', no_table)
    with pytest.raises(ValueError, match='memory|budget|resident'):
        ci.make_rdm2(vector, space, max_memory=.001)
