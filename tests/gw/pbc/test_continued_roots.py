"""One compiled continued matrix QP selector for changing query data."""
import importlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gradscf.gw.pbc.query import _solve_subspace_qp
from gradscf.solvers.nonlinear import ScalarRootConfig


def _queries(eta):
    rng = np.random.default_rng(121)
    amplitude = .025*(rng.normal(size=(3, 3, 3))+1j*rng.normal(size=(3, 3, 3)))
    residues = amplitude@amplitude.conj().swapaxes(-1, -2)
    poles = np.array([-1.1, .9, 1.7])
    reference = np.array([-.6, -.2, .4])
    fock = np.array([[-.58, .018+.01j, -.009j],
                     [.018-.01j, -.18, .012-.008j], [.009j, .012+.008j, .42]])
    yield tuple(map(jnp.asarray, (reference, fock, residues, poles, -.4, eta, 1)))
    # All numerical arguments change while their shapes/dtypes stay fixed.
    yield tuple(map(jnp.asarray, (reference+.07, fock+.04*np.eye(3), 1.12*residues,
                                  poles+.03, .1, eta+.005, 2)))


@pytest.mark.parametrize('eta', [0., .01])
def test_continued_query_reuses_executable_and_matches_complete_old_matrix_selector(eta):
    module = importlib.import_module('gradscf.gw.pbc.spectrum')
    helper = module._continued_query
    helper.clear_cache()
    config = ScalarRootConfig(method='hybrid', ftol=1e-9, xtol=1e-9, maxiter=80)
    compiled = None
    for reference, fock, residues, poles, mu, broadening, nocc in _queries(eta):
        def correlation(omega):
            return jnp.einsum('l,lij->ij', 1/(omega-mu+1j*broadening-poles), residues)
        diagonal = jnp.diag(reference)
        expected = _solve_subspace_qp(reference, fock-diagonal, jnp.zeros_like(diagonal),
            correlation, config=config, occupied=jnp.arange(len(reference)) < nocc)
        actual = helper(reference, fock, residues, poles, mu, broadening, nocc,
                        root_config=config)
        jax.block_until_ready(actual)
        assert np.asarray(actual['converged']).all()
        for name in expected:
            if name in ('converged', 'candidate_count', 'branch_valid'):
                np.testing.assert_array_equal(actual[name], expected[name])
            elif name == 'vectors':
                # Individual complex eigenvector phases are not observables.
                projector = lambda v: jnp.einsum('ai,bi->iab', v, v.conj())
                np.testing.assert_allclose(projector(actual[name]), projector(expected[name]),
                                           atol=3e-10, rtol=3e-10)
            else:
                np.testing.assert_allclose(actual[name], expected[name], atol=3e-10, rtol=3e-10)
        if compiled is None:
            compiled = helper._cache_size()
            assert compiled == 1
        else:
            assert helper._cache_size() == compiled
