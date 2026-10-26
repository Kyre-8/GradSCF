"""Reuse a dynamic numerical cache without changing the shared QP selector."""
from dataclasses import replace
import importlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gradscf.gw.qp import solve_quasiparticle_roots
from gradscf.solvers.nonlinear import ScalarRootConfig
from test_self_consistency import _model


def _six_band_model():
    model = _model(madelung=0.)
    energy = jnp.array([[-.9, -.7, -.7, .4, .4, .4],
                        [-.85, -.65, -.65, .45, .45, .45]])
    rng = np.random.default_rng(97)
    zero = .025*(rng.normal(size=(2, 2, 6, 6))+1j*rng.normal(size=(2, 2, 6, 6)))
    zero = (zero+zero.conj().swapaxes(-1, -2))/2
    cross = .025*(rng.normal(size=(2, 6, 6))+1j*rng.normal(size=(2, 6, 6)))
    factors = (jnp.asarray(zero), jnp.asarray(np.stack((cross, cross.conj().swapaxes(-1, -2)))))
    diagonal = jax.vmap(jnp.diag)(energy)
    return replace(model, energy=energy, coeff=jnp.tile(jnp.eye(6), (2, 1, 1)),
        hcore=diagonal, reference_fock=diagonal, reference_potential=jnp.zeros_like(diagonal),
        nocc=3, factors=factors)


@pytest.mark.parametrize('group', [(0,), (1, 2), (3, 4, 5)])
@pytest.mark.parametrize('seeded', [False, True])
def test_dynamic_diagonal_roots_match_existing_selector_and_reuse_group_size(group, seeded):
    module = importlib.import_module('gradscf.gw.pbc.evgw')
    source = importlib.import_module('gradscf.gw.pbc.source')
    helper = module._diagonal_roots
    helper.clear_cache()
    model = _six_band_model()
    cache = source.build_correlation(model, model.energy, nw=4, eta=.003)
    numerical = {name: value for name, value in cache.items() if name not in ('nk', 'linear_config')}
    config = ScalarRootConfig(method='hybrid', ftol=1e-10, xtol=1e-10, maxiter=60)
    compiled = None
    for k, indices in ((0, group), (1, tuple((i+1) % 6 for i in group))):
        # The second index array deliberately changes its values as well as k.
        # The helper's scalar trace equation is independently checked below;
        # the public driver still supplies complete reference KS clusters.
        bands = jnp.asarray(indices)
        reference = jnp.mean(model.energy[k, bands])[None]
        shift = jnp.asarray(.002+.001*k)
        guess = reference+.02 if seeded else None
        correlation = lambda omega: source.correlation_diagonal(cache, k, omega, bands)
        equation = lambda omega: omega-reference-shift-correlation(omega[0]).real
        expected = solve_quasiparticle_roots(equation, reference, initial_guess=guess,
                                            target=guess, config=config)
        actual, sigma = helper(numerical, reference, shift, bands, jnp.asarray(k), guess,
            nk=cache['nk'], linear_config=cache['linear_config'], root_config=config)
        jax.block_until_ready((actual, sigma))
        for name, left, right in zip(actual._fields, actual, expected):
            if name in ('converged', 'candidate_count', 'branch_valid'):
                np.testing.assert_array_equal(left, right)
            else:
                np.testing.assert_allclose(left, right, atol=2e-11, rtol=2e-11)
        np.testing.assert_allclose(sigma, correlation(expected.roots[0]), atol=2e-12, rtol=2e-11)
        if compiled is None:
            compiled = helper._cache_size()
            assert compiled == 1
        else:
            assert helper._cache_size() == compiled


def test_correlation_kernel_reuses_executable_when_source_k_changes():
    source = importlib.import_module('gradscf.gw.pbc.source')
    source._correlation_kernel.clear_cache()
    model = _model(scale=.4)
    cache = source.build_correlation(model, model.energy, nw=4, eta=.003)
    first = source.correlation_at(cache, 0, jnp.asarray(.9))
    first.block_until_ready()
    compiled = source._correlation_kernel._cache_size()
    second = source.correlation_at(cache, 1, jnp.asarray(.9))
    second.block_until_ready()
    assert source._correlation_kernel._cache_size() == compiled
    assert np.max(abs(np.asarray(second-first))) > 1e-10
