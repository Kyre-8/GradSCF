"""CI observables reuse sigma actions; dense spatial exports are budgeted."""
import importlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gradscf import ci


@pytest.mark.parametrize('unrestricted,frozen', [(False, None), (False, [0]),
                                                (True, None), (True, ([0], []))])
def test_sigma_density_exports_do_not_build_connection_tables(unrestricted, frozen, monkeypatch):
    from pyscf import fci
    n, occ = 3, (2, 1) if unrestricted else (1, 1)
    space = (ci.make_uci_space(n, occ, frozen=frozen) if unrestricted else
             ci.make_ci_space(n, occ[0], frozen=frozen))
    vector = np.random.default_rng(731).normal(size=space.size)
    strings = [list(map(int, fci.cistring.make_strings(range(n), no))) for no in occ]
    full = np.zeros(tuple(map(len, strings)))
    for c, det in zip(vector, space.determinants):
        full[strings[0].index(det & ((1 << n)-1)), strings[1].index(det >> n)] = c
    full /= np.linalg.norm(full)
    expected = (fci.direct_spin1.make_rdm12s(full, n, occ) if unrestricted else
                fci.direct_spin1.make_rdm12(full, n, occ))
    module = importlib.import_module('gradscf.ci._rdm')
    monkeypatch.setattr(module, 'affordable', lambda *a, **kw: False)
    monkeypatch.setattr(module, '_density_connections', lambda *a: pytest.fail(
        'Low-budget CI property materialized a density connection table'))
    actual = jax.jit(lambda c: ci.make_rdm12(c, space, max_memory=32.))(jnp.asarray(vector))
    for a, b in zip(jax.tree.leaves(actual), jax.tree.leaves(expected)):
        np.testing.assert_allclose(a, b, atol=2e-12, rtol=2e-12)


def test_observable_contracts_without_density_export_and_retains_second_response(monkeypatch):
    module = importlib.import_module('gradscf.ci.properties')
    expect = getattr(module, 'expectation_value', None)
    assert callable(expect), 'Missing direct CI observable contraction'
    space = ci.make_ci_space(3, 1)
    rng = np.random.default_rng(732)
    c = jnp.asarray(rng.normal(size=space.size))
    h = jnp.asarray(rng.normal(size=(3, 3)))
    g = jnp.asarray(rng.normal(size=(3,)*4))
    dm1, dm2 = ci.make_rdm12(c, space)
    expected = jnp.sum(h*dm1.T)+.5*jnp.sum(g*dm2)
    monkeypatch.setattr(module, 'make_rdm2', lambda *a, **kw: pytest.fail('Observable exported a 2-RDM'))
    actual = jax.jit(lambda x: expect(x, space, h, g, max_memory=32.))(c)
    np.testing.assert_allclose(actual, expected, atol=2e-12)
    direction = .03*jnp.sin(c)
    gradient = jax.jit(jax.grad(lambda x: expect(x, space, h, g, max_memory=32.)))
    hvp = jax.jvp(gradient, (c,), (direction,))[1]
    step = 1e-4
    finite = (gradient(c+step*direction)-gradient(c-step*direction))/(2*step)
    np.testing.assert_allclose(hvp, finite, atol=2e-7, rtol=2e-6)


def test_dense_export_rejects_known_output_before_sigma(monkeypatch):
    module = importlib.import_module('gradscf.ci.properties')
    space = ci.make_ci_space(4, 1)
    monkeypatch.setattr(module, 'build_direct_hamiltonian',
        lambda *a, **kw: pytest.fail('Insufficient output budget reached sigma construction'), raising=False)
    with pytest.raises(ValueError, match='memory|budget|resident'):
        ci.make_rdm2(jnp.ones(space.size), space, max_memory=.001)
