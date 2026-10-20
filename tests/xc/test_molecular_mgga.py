"""Molecular tau-dependent XC must use the same energy, Fock and Hessian."""
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gradscf.scf import RKSConfig, UKSConfig, run_rks_from_integrals, run_uks_from_integrals
from gradscf.scf.rks import _fock_components_for_density
from gradscf.scf.uks import _point_unrestricted_xc_energy_kernel, _raw_fock_and_energy_for_state
from gradscf.tools.features import _spin_density_and_gradient, _spin_tau


def _grid():
    ao = jnp.asarray([[1., .2], [.8, -.1], [.4, .9]])
    derivative = jnp.asarray([ao, [[.1, .03], [.04, .2], [-.1, .05]],
                              [[.02, .1], [.2, -.03], [.05, -.1]],
                              [[-.05, .02], [.01, -.05], [.1, .1]]])
    return ao, derivative, jnp.asarray([.5, .7, .6])


def _spin_fields(density):
    density = .5*(density+jnp.swapaxes(density, -1, -2))
    ao, derivative, _ = _grid()
    ra, ga = _spin_density_and_gradient(ao, derivative, density[0])
    rb, gb = _spin_density_and_gradient(ao, derivative, density[1])
    return jnp.concatenate([ra[:, None], rb[:, None], ga, gb,
                            _spin_tau(derivative, density[0])[:, None],
                            _spin_tau(derivative, density[1])[:, None]], axis=-1)


def _xc_energy(density):
    return jnp.dot(_grid()[2], _point_unrestricted_xc_energy_kernel('r2scan', 'MGGA')(
        _spin_fields(density)))


def _uks_energy_fock(density):
    ao, derivative, weights = _grid()
    energy, _, fa, fb = _raw_fock_and_energy_for_state(
        density_a=density[0], density_b=density[1], ao=ao, ao_deriv1=derivative,
        weights=weights, h=jnp.zeros((2, 2)), eri=jnp.zeros((2, 2, 2, 2)), df_factors=None,
        enuc=jnp.asarray(0.), alpha=jnp.asarray(0.), cfg=UKSConfig(xc_spec='r2scan'),
        xc_kind='MGGA')
    return energy, jnp.stack([fa, fb])


@pytest.mark.parametrize('spin', ['restricted', 'unrestricted'])
def test_mgga_fock_and_fock_jvp_are_energy_derivatives(spin):
    """The tau AO projector is required even for a two-orbital grid."""
    density = jnp.asarray([[[.7, .08], [.08, .4]], [[.4, -.03], [-.03, .3]]])
    tangent = jnp.asarray([[[.1, -.02], [-.02, -.03]], [[-.04, .03], [.03, .07]]])
    if spin == 'restricted':
        density, tangent = density.sum(axis=0), tangent.sum(axis=0)
        ao, derivative, weights = _grid()

        def energy(d):
            return _xc_energy(jnp.stack([d/2, d/2]))

        def fock(d):
            _, _, _, result = _fock_components_for_density(
                density=d, mo_coeff=jnp.eye(2), mo_occ=jnp.asarray([2., 0.]),
                mo_energy=jnp.asarray([-1., 1.]),
                jk_builder=lambda *_: (jnp.zeros((2, 2)), jnp.zeros((2, 2))),
                ao=ao, ao_deriv1=derivative, weights=weights, h=jnp.zeros((2, 2)),
                alpha=jnp.asarray(0.), cfg=RKSConfig(xc_spec='r2scan'), xc_kind='MGGA')
            return result
    else:
        energy = _xc_energy
        fock = lambda d: _uks_energy_fock(d)[1]
    expected = jax.grad(energy)(density)
    actual = fock(density)
    np.testing.assert_allclose(actual, expected, rtol=2e-10, atol=2e-11)
    actual_hvp = jax.jvp(fock, (density,), (tangent,))[1]
    expected_hvp = jax.jvp(jax.grad(energy), (density,), (tangent,))[1]
    np.testing.assert_allclose(actual_hvp, expected_hvp, rtol=2e-9, atol=2e-10)
    step = 1e-4
    finite = (fock(density+step*tangent)-fock(density-step*tangent))/(2*step)
    np.testing.assert_allclose(actual_hvp, finite, rtol=2e-6, atol=2e-9)


@pytest.mark.parametrize('spin', ['restricted', 'unrestricted'])
def test_mgga_scf_map_accepts_tau_dependent_functional(spin):
    ao, derivative, weights = _grid()
    kwargs = dict(overlap=jnp.eye(2), hcore=jnp.diag(jnp.asarray([-1., .5])),
                  eri=jnp.zeros((2, 2, 2, 2)), nuclear_repulsion=0.,
                  ao=ao, ao_deriv1=derivative, grid_weights=weights)
    if spin == 'restricted':
        result = run_rks_from_integrals(**kwargs, nelectron=2,
                    config=RKSConfig(xc_spec='r2scan', max_cycle=3))
    else:
        result = run_uks_from_integrals(**kwargs, nalpha=1, nbeta=1,
                    config=UKSConfig(xc_spec='r2scan', max_cycle=3))
    assert np.isfinite(result.total_energy)
    assert int(result.cycles) > 0


def test_unrestricted_mgga_td_projection_matches_ao_fock_hessian():
    from gradscf.tddft._unrestricted_semilocal_response import (
        UnrestrictedSemilocalResponseFunctional, build_unrestricted_semilocal_response_action)

    ao, derivative, weights = _grid()
    density = jnp.asarray([[[.7, .08], [.08, .4]], [[.4, -.03], [-.03, .3]]])
    molecule = SimpleNamespace(ao=ao, ao_deriv1=derivative,
                              rdm1=density, grid=SimpleNamespace(weights=weights))
    functional = UnrestrictedSemilocalResponseFunctional('r2scan')
    occupied, virtual = jnp.eye(2)[:, :1], jnp.eye(2)[:, 1:]
    action = build_unrestricted_semilocal_response_action(
        molecule, occupied, virtual, occupied, virtual, functional.spin_grid_response_hvp,
        feature_kind='MGGA', dtype=jnp.float64)
    xa, xb = jnp.asarray([[[.13]]]), jnp.asarray([[[-.07]]])
    actual = jnp.stack(action(xa, xb), axis=1)
    tangent = jnp.stack([occupied@xa[0]@virtual.T, occupied@xb[0]@virtual.T])
    hessian_action = jax.jvp(jax.grad(_xc_energy), (density,), (tangent,))[1]
    expected = jnp.einsum('pi,spq,qa->sia', occupied, hessian_action, virtual)[None]
    np.testing.assert_allclose(actual, expected, rtol=2e-9, atol=2e-10)


def test_inactive_mgga_grid_has_zero_energy_potential_and_higher_derivatives():
    """Mask before evaluating invalid density tails; retain the complete tau map."""
    density = jnp.zeros((2, 2, 2))
    tangent = jnp.ones_like(density)*1e-15
    energy, fock = _uks_energy_fock(density)
    np.testing.assert_array_equal(energy, 0.)
    np.testing.assert_array_equal(fock, 0.)
    first = jax.jvp(lambda d: _uks_energy_fock(d)[1], (density,), (tangent,))[1]
    second = jax.jvp(lambda d: jax.jvp(lambda x: _uks_energy_fock(x)[1],
                                    (d,), (tangent,))[1], (density,), (tangent,))[1]
    np.testing.assert_array_equal(first, 0.)
    np.testing.assert_array_equal(second, 0.)
