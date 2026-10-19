"""Fixed occupations must remain compatible with molecular GW pole spectra."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gradscf.gw.evgw import _evgw_loop
from gradscf.gw.qsgw import qsgw_cd_restricted
from gradscf.gw.types import GWResult
from gradscf.scf.autodiff import SCFDifferentiationConfig


_AD = SCFDifferentiationConfig(tolerance=1e-10, max_iter=30, restart=10)


def _constant_static_driver(target):
    """An exact diagonal Dyson model with a frequency-independent correction."""
    def driver(mo_energy_poles, **kwargs):
        energy = jnp.asarray(mo_energy_poles)
        residual = energy - target
        return GWResult(mo_energy=energy, mo_coeff=jnp.eye(energy.shape[-1]),
                        converged=jnp.all(jnp.abs(residual) < 1e-9),
                        qp_residual=residual)
    return driver


@pytest.mark.parametrize("configuration", [None, _AD])
@pytest.mark.parametrize("history", [0, 4])
def test_evgw_rejects_occupation_inversion_at_an_outer_root(configuration, history):
    initial = jnp.array([-.5, .5])
    target = jnp.array([.4, -.4])
    with pytest.raises(Exception, match="occupation gap"):
        _evgw_loop(_constant_static_driver(target), initial, nocc=1,
                   max_iter=8, tol=1e-9, diis_space=history,
                   differentiation=configuration)


@pytest.mark.parametrize("initial_guess", [jnp.array([.5, -.5]), jnp.array([0., 0.])])
def test_evgw_rejects_invalid_occupation_seed(initial_guess):
    initial = jnp.array([-.5, .5])
    with pytest.raises(Exception, match="occupation gap"):
        _evgw_loop(_constant_static_driver(initial), initial, nocc=1,
                   initial_guess=initial_guess, max_iter=8, tol=1e-9)


def test_evgw_checks_global_gap_without_requiring_same_sector_order():
    # Intrasector degeneracy and orbital permutations preserve occupations.
    energy = jnp.array([-.3, -.8, .9, .9])
    result = _evgw_loop(_constant_static_driver(energy), energy, nocc=2,
                       max_iter=2, tol=1e-9)
    assert result.converged
    np.testing.assert_array_equal(result.mo_energy, energy)


def test_unrestricted_evgw_checks_both_spin_occupation_sectors():
    initial = jnp.array([[-.8, -.3, .8], [-.6, .4, .9]])
    target = initial.at[1, 0].set(.7)
    with pytest.raises(Exception, match="occupation gap"):
        _evgw_loop(_constant_static_driver(target), initial, nocc=(2, 1),
                   max_iter=8, tol=1e-9)


@pytest.mark.parametrize("configuration", [None, _AD])
def test_evgw_cannot_overwrite_a_negative_fixed_gw_weight(configuration):
    # Sigma(omega; E) = 2*omega - 2*E has a trivial on-shell outer root,
    # but the fixed-G/W frequency slope gives Z = 1/(1-2) = -1.
    energy = jnp.array([-.5, .5])

    def driver(mo_energy_poles, **kwargs):
        return GWResult(mo_energy=mo_energy_poles, mo_coeff=jnp.eye(2),
                        converged=True, qp_residual=mo_energy_poles-energy,
                        qp_weight=-jnp.ones(2))

    with pytest.raises(Exception, match="quasiparticle slope"):
        _evgw_loop(driver, energy, nocc=1, max_iter=4, tol=1e-9,
                   differentiation=configuration)


def _qsgw_zero_interaction():
    energy = jnp.array([-.5, .7])
    return dict(mo_energy=energy, mo_coeff=jnp.eye(2), nocc=1,
                df_factors=jnp.zeros((1, 2, 2)), hcore_matrix=jnp.diag(energy),
                nw=8, max_iter=4, tol=1e-10, tol_density=1e-10)


@pytest.mark.parametrize("nocc", [0, 2])
def test_qsgw_requires_occupied_and_virtual_orbitals(nocc):
    with pytest.raises(ValueError, match="occupied and virtual"):
        qsgw_cd_restricted(**{**_qsgw_zero_interaction(), "nocc": nocc})


@pytest.mark.parametrize("configuration", [None, _AD])
def test_qsgw_rejects_an_inverted_starting_occupation(configuration):
    kw = _qsgw_zero_interaction()
    with pytest.raises(Exception, match="occupation gap"):
        qsgw_cd_restricted(**{**kw, "mo_energy": kw["mo_energy"][::-1]},
                          differentiation=configuration)


@pytest.mark.parametrize("configuration", [None, _AD])
def test_qsgw_records_the_final_screening_spectrum(configuration):
    result = qsgw_cd_restricted(**_qsgw_zero_interaction(), differentiation=configuration)
    assert result.converged
    assert result.method == "qsgw"
    np.testing.assert_allclose(result.screening_energy, result.mo_energy, atol=0, rtol=0)
