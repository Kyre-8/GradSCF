"""Full TD must honor the existing facade's initial-space size control."""
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

from gradscf import tdscf


@pytest.mark.parametrize("guess_count", [3, 10, np.int64(3)])
def test_full_td_initial_guess_reaches_disconnected_lowest_root(monkeypatch, guess_count):
    # The orbital-gap ordering differs from the screened excitation ordering.
    # A one-vector seed is an invariant subspace with a perfectly converged
    # higher root. Expanding the seed must expose the other symmetry sectors.
    a = jnp.array([1.4, .9, .3])
    b = jnp.array([.1, .1, .05])
    reference = SimpleNamespace(mo_coeff=jnp.eye(4), mo_occ=jnp.array([2., 0., 0., 0.]),
                                mo_energy=jnp.array([0., .2, .4, .6]), nocc=1)

    def operator(rows):
        x, y = rows[:, :3], rows[:, 3:]
        return jnp.concatenate([a*x+b*y, -(b*x+a*y)], axis=1)

    monkeypatch.setattr("gradscf.tddft.casida.gen_tdhf_vind", lambda *args, **kwargs: operator)
    narrow = tdscf.TDDFT(reference, nstates=1, davidson_tol=1e-11).kernel()
    expanded = tdscf.TDDFT(reference, nstates=1, davidson_tol=1e-11,
                          davidson_initial_guess_count=guess_count).kernel()
    assert bool(narrow.converged) and bool(expanded.converged)
    np.testing.assert_allclose(narrow.excitation_energies, np.sqrt(1.4**2-.1**2), atol=1e-11)
    np.testing.assert_allclose(expanded.excitation_energies, np.sqrt(.3**2-.05**2), atol=1e-11)
    norm = 2*jnp.sum(expanded.x_amplitudes**2-expanded.y_amplitudes**2)
    np.testing.assert_allclose(norm, 1., atol=1e-12)


@pytest.mark.parametrize("guess_count", [0, -1, 2.5, True, False, np.float64(3), "3"])
def test_full_td_initial_guess_count_rejects_nonpositive_or_noninteger_values(guess_count):
    from gradscf.solvers.eigen.rpa import implicit_differential_davidson_lowest_tdhf

    def operator(rows):
        return jnp.concatenate([rows[:, :1], -rows[:, 1:]], axis=1)

    with pytest.raises(ValueError, match="initial_guess_count must be a positive integer"):
        implicit_differential_davidson_lowest_tdhf(
            operator, nroots=1, size=1, diag=jnp.ones(1),
            initial_guess_count=guess_count,
        )
