"""Dense BSE root capacity is independent of the Davidson subspace limit."""

import jax.numpy as jnp
import numpy as np
import pytest

from gradscf import bse


def independent_transitions():
    occupied = -np.arange(1, 9)[::-1] * 0.131
    virtual = np.arange(1, 7) * 0.277
    energies = jnp.asarray(np.r_[occupied, virtual])
    factors = jnp.zeros((1, 14, 14))
    space = bse.make_bse_space(14, 8)
    expected = np.sort((virtual[None, :] - occupied[:, None]).ravel())
    return energies, factors, space, expected


@pytest.mark.parametrize("tda", [True, False])
def test_dense_all_roots_exceed_default_davidson_subspace(tda):
    energies, factors, space, expected = independent_transitions()
    cfg = bse.BSEConfig(tda=tda, solver="dense", nroots=48)
    result = bse.run_bse(energies, energies, factors, space, config=cfg)

    np.testing.assert_allclose(result.excitation_energies, expected, atol=1e-12, rtol=0)
    assert np.all(result.converged & result.stable)
    assert np.max(result.residual_norms) < 1e-12


@pytest.mark.parametrize("tda", [True, False])
def test_dense_all_roots_preserve_dense_capacity_guard(tda):
    energies, factors, space, _ = independent_transitions()
    cfg = bse.BSEConfig(tda=tda, solver="dense", nroots=48, max_dense=47)
    with pytest.raises(ValueError, match="max_dense"):
        bse.run_bse(energies, energies, factors, space, config=cfg)


@pytest.mark.parametrize("tda", [True, False])
def test_davidson_all_roots_still_require_sufficient_subspace(tda):
    energies, factors, space, _ = independent_transitions()
    cfg = bse.BSEConfig(tda=tda, solver="davidson", nroots=48, max_space=40)
    with pytest.raises(ValueError, match="max_subspace must be at least nroots"):
        bse.run_bse(energies, energies, factors, space, config=cfg)
