"""Built-in SCF XC reductions consume AO blocks without dense grid inputs."""
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gradscf.scf import rks, uks


@dataclass
class ArrayBlocks:
    """Test double for the grid provider; production AO evaluation is tested separately."""

    ao: object
    derivative: object
    weights: object
    requests: list

    def reduce(self, function, initial, *, deriv):
        self.requests.append(deriv)
        carry = initial
        for start in range(0, len(self.weights), 2):
            count = min(2, len(self.weights) - start)
            ao = jnp.pad(self.ao[start:start + count], ((0, 2 - count), (0, 0)))
            derivative = None if deriv == 0 else jnp.pad(
                self.derivative[:, start:start + count], ((0, 0), (0, 2 - count), (0, 0)))
            weights = jnp.pad(self.weights[start:start + count], (0, 2 - count))
            carry = function(carry, ao, derivative, None, jnp.zeros((2, 3)), weights)
        return carry


def grid(scale=1.):
    ao = jnp.asarray([[1., .2], [.8, -.1], [.4, .9]]) * scale
    derivative = jnp.stack([ao,
        jnp.asarray([[.1, .03], [.04, .2], [-.1, .05]]) * scale,
        jnp.asarray([[.02, .1], [.2, -.03], [.05, -.1]]) * scale,
        jnp.asarray([[-.05, .02], [.01, -.05], [.1, .1]]) * scale])
    return ao, derivative, jnp.asarray([.5, .7, .6]) * (1. + .1 * (scale - 1.))


def evaluate(density, *, xc, unrestricted, blocked, scale=1., compute_potential=True):
    ao, derivative, weights = grid(scale)
    requests = []
    provider = ArrayBlocks(ao, derivative, weights, requests) if blocked else None
    kwargs = dict(ao=jnp.zeros((0, 2)) if blocked else ao,
                  ao_deriv1=jnp.zeros((4, 0, 2)) if blocked else derivative,
                  weights=jnp.zeros((0,)) if blocked else weights,
                  grid_ao=provider, xc_kind='LDA' if xc == 'svwn' else 'MGGA' if xc == 'r2scan' else 'GGA')
    if unrestricted:
        result = uks._unrestricted_xc_energy_and_matrices(
            density_a=density[0], density_b=density[1], cfg=uks.UKSConfig(xc_spec=xc),
            compute_potential=compute_potential, **kwargs)
    else:
        result = rks._restricted_xc_energy_and_matrix(
            density=density, cfg=rks.RKSConfig(xc_spec=xc), **kwargs)
    return result, requests


@pytest.mark.parametrize('unrestricted', [False, True], ids=['rks', 'uks'])
@pytest.mark.parametrize('xc', ['svwn', 'pbe', 'pbe0', 'r2scan'])
def test_blocked_builtin_xc_matches_dense(unrestricted, xc):
    density = jnp.asarray([[[.7, .08], [.08, .4]], [[.4, -.03], [-.03, .3]]])
    if not unrestricted:
        density = density.sum(axis=0)
    expected, _ = evaluate(density, xc=xc, unrestricted=unrestricted, blocked=False)
    actual, requests = evaluate(density, xc=xc, unrestricted=unrestricted, blocked=True)
    for value, reference in zip(actual, expected, strict=True):
        np.testing.assert_allclose(value, reference, rtol=3e-12, atol=3e-12)
    assert requests == [0 if xc == 'svwn' else 1]
    if unrestricted:
        energy_only, _ = evaluate(density, xc=xc, unrestricted=True, blocked=True,
                                  compute_potential=False)
        np.testing.assert_allclose(energy_only[0], actual[0], rtol=3e-12, atol=3e-12)


@pytest.mark.parametrize('unrestricted', [False, True], ids=['rks', 'uks'])
def test_blocked_builtin_xc_preserves_density_and_grid_derivatives(unrestricted):
    density = jnp.asarray([[[.7, .08], [.08, .4]], [[.4, -.03], [-.03, .3]]])
    if not unrestricted:
        density = density.sum(axis=0)
    tangent = jnp.linspace(-.03, .04, density.size).reshape(density.shape)
    tangent = .5 * (tangent + tangent.swapaxes(-1, -2))

    def energy(dm, scale, blocked):
        # SCF densities and their tangents are real symmetric (Hermitian).
        # Project the independent matrix variables onto that physical domain.
        dm = .5 * (dm + dm.swapaxes(-1, -2))
        return evaluate(dm, xc='pbe', unrestricted=unrestricted, blocked=blocked, scale=scale)[0][0]

    dense = lambda dm: energy(dm, 1., False)
    blocked = lambda dm: energy(dm, 1., True)
    potential = evaluate(density, xc='pbe', unrestricted=unrestricted, blocked=True)[0][1]
    np.testing.assert_allclose(potential, jax.grad(blocked)(density), rtol=2e-10, atol=2e-11)
    np.testing.assert_allclose(jax.grad(blocked)(density), jax.grad(dense)(density), rtol=2e-10, atol=2e-11)
    np.testing.assert_allclose(jax.jvp(jax.grad(blocked), (density,), (tangent,))[1],
                               jax.jvp(jax.grad(dense), (density,), (tangent,))[1], rtol=2e-9, atol=2e-10)
    for order in (1, 2):
        derivatives = []
        for stream in (False, True):
            function = lambda scale: energy(density, scale, stream)
            for _ in range(order):
                function = jax.grad(function)
            derivatives.append(function(jnp.asarray(1.)))
        np.testing.assert_allclose(*derivatives, rtol=2e-9, atol=2e-10)


def test_hf_grid_contribution_does_not_request_ao():
    density = jnp.eye(2)
    provider = ArrayBlocks(*grid(), requests=[])
    energy, potential = rks._restricted_xc_energy_and_matrix(
        density=density, ao=jnp.zeros((0, 2)), ao_deriv1=jnp.zeros((4, 0, 2)),
        weights=jnp.zeros((0,)), cfg=rks.RKSConfig(xc_spec='hf'), xc_kind='HF', grid_ao=provider)
    assert not provider.requests
    assert float(energy) == 0.
    np.testing.assert_array_equal(potential, np.zeros((2, 2)))


@pytest.mark.parametrize('unrestricted', [False, True], ids=['rks', 'uks'])
def test_builtin_fock_consumes_provider_with_empty_ao(unrestricted):
    density = jnp.asarray([[[.7, .08], [.08, .4]], [[.4, -.03], [-.03, .3]]])
    if not unrestricted:
        density = density.sum(axis=0)
    expected, _ = evaluate(density, xc='pbe', unrestricted=unrestricted, blocked=False)
    provider = ArrayBlocks(*grid(), requests=[])
    kwargs = dict(ao=jnp.zeros((0, 2)), ao_deriv1=jnp.zeros((4, 0, 2)),
                  weights=jnp.zeros((0,)), h=jnp.zeros((2, 2)),
                  alpha=jnp.asarray(0.), xc_kind='GGA', grid_ao=provider)
    if unrestricted:
        energy, _, fa, fb = uks._raw_fock_and_energy_for_state(
            density_a=density[0], density_b=density[1],
            eri=jnp.zeros((2, 2, 2, 2)), df_factors=None, enuc=jnp.asarray(0.),
            cfg=uks.UKSConfig(xc_spec='pbe'), **kwargs)
        actual = energy, jnp.stack([fa, fb])
    else:
        _, _, contribution, fock = rks._fock_components_for_density(
            density=density, mo_coeff=jnp.eye(2), mo_occ=jnp.asarray([2., 0.]),
            mo_energy=jnp.asarray([-1., 1.]),
            jk_builder=lambda *_: (jnp.zeros((2, 2)), jnp.zeros((2, 2))),
            cfg=rks.RKSConfig(xc_spec='pbe'), **kwargs)
        actual = contribution.energy, fock
    for value, reference in zip(actual, expected, strict=True):
        np.testing.assert_allclose(value, reference, rtol=3e-12, atol=3e-12)
    assert provider.requests == [1]
