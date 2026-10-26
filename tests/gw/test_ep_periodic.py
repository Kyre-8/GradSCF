"""Periodic Fan Matsubara reference, momentum blocks and complex response."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gradscf.gw.matsubara import matsubara_grid


def inputs(nw=12):
    from gradscf.gw.ep_coupling import PeriodicPhononModel
    rng = np.random.default_rng(610)
    f = jnp.array([[[-.6, .04j], [-.04j, .5]], [[-.3, .06], [.06, .8]]])
    g = jnp.asarray(rng.normal(size=(2, 2, 2, 2, 2)) + 1j * rng.normal(size=(2, 2, 2, 2, 2))) * .03
    model = PeriodicPhononModel(jnp.array([[.08, .13], [.11, .2]]), g,
                               jnp.array([[0, 1], [1, 0]]), jnp.array([.3, .7]))
    grid = matsubara_grid(nw=nw, beta=8.)
    e, u = jnp.linalg.eigh(f)
    weight = -jnp.exp(-grid.tau[:, None, None] * e - jnp.logaddexp(0., -grid.beta * e))
    green = jnp.einsum('kim,tkm,kjm->tkij', u, weight, u.conj())
    return model, grid, f, green


def oracle(model, grid, f):
    e, u = np.linalg.eigh(f)
    result = np.zeros((2 * grid.nw, len(f), 2, 2), dtype=complex)
    for q in range(2):
        for k in range(2):
            j = model.k_plus_q[q, k]
            for l, omega in enumerate(model.energies[q]):
                v = np.asarray(model.couplings[q, k, l]) @ u[j]
                nb = 1 / np.expm1(grid.beta * omega)
                for m, energy in enumerate(e[j]):
                    nf = 1 / (1 + np.exp(grid.beta * energy))
                    weight = ((nb + 1 - nf) / (1j * np.asarray(grid.fermion) - energy - omega)
                              + (nb + nf) / (1j * np.asarray(grid.fermion) - energy + omega))
                    result[:, k] += model.q_weights[q] * weight[:, None, None] * np.outer(v[:, m], v[:, m].conj())
    return result


def test_periodic_reference_poles_and_molecular_limit():
    from gradscf.gw.ep_coupling import periodic_fan_self_energy, fan_self_energy, PhononModel
    model, grid, f, green = inputs()
    out = jax.jit(periodic_fan_self_energy)(green, f, 0., model, grid)
    np.testing.assert_allclose(out['sigma_iw'], oracle(model, grid, f), atol=2e-13)
    # Gamma real symmetric one-q/one-k representation must reduce to molecular.
    from dataclasses import replace
    g = model.couplings[:1, :1].real
    g = (g + g.swapaxes(-1, -2)) / 2
    gamma = replace(model, couplings=g, energies=model.energies[:1],
                    k_plus_q=jnp.zeros((1, 1), int), q_weights=jnp.ones(1))
    matrix = f[:1].real
    e, u = jnp.linalg.eigh(matrix[0])
    gtau = (u[None] * (-jnp.exp(-grid.tau[:, None] * e - jnp.logaddexp(0., -grid.beta*e)))[:, None, :]) @ u.T
    periodic = periodic_fan_self_energy(gtau[:, None], matrix, 0., gamma, grid)
    molecular = fan_self_energy(gtau, matrix[0], 0., PhononModel(gamma.energies[0], g[0, 0]), grid)
    for key in ('sigma_iw', 'sigma_moment'):
        np.testing.assert_allclose(periodic[key].squeeze(axis=1 if key == 'sigma_iw' else 0), molecular[key], atol=2e-13)


def test_periodic_parameter_jvp_vjp_and_covariance():
    from dataclasses import replace
    from gradscf.gw.ep_coupling import periodic_fan_self_energy
    model, grid, f, green = inputs()
    def objective(t):
        changed = replace(model, energies=model.energies.at[1, 0].add(.1*t),
                          couplings=model.couplings * (1 + .2*t))
        out = periodic_fan_self_energy(green, f, 0., changed, grid)
        return jnp.sum(jnp.abs(out['sigma_iw'])**2) + jnp.trace(out['sigma_moment'], axis1=-2, axis2=-1).real.sum()
    actual = jax.jit(jax.grad(objective))(0.)
    fd = (objective(1e-4) - objective(-1e-4)) / 2e-4
    np.testing.assert_allclose(actual, fd, atol=2e-8)
    np.testing.assert_allclose(jax.jvp(objective, (0.,), (1.,))[1], actual, atol=1e-12)
    rng = np.random.default_rng(66)
    rotations = jnp.asarray(np.linalg.qr(rng.normal(size=(2, 2, 2))
                             + 1j * rng.normal(size=(2, 2, 2)))[0])
    rotate = lambda x: rotations.swapaxes(-1, -2).conj() @ x @ rotations
    vertices = jnp.einsum('kia,qklij,qkjb->qklab', rotations.conj(), model.couplings,
                          rotations[model.k_plus_q])
    expected = periodic_fan_self_energy(green, f, 0., model, grid)
    actual = periodic_fan_self_energy(rotate(green), rotate(f), 0., replace(model, couplings=vertices), grid)
    for key in expected:
        np.testing.assert_allclose(actual[key], rotate(expected[key]), atol=2e-12)


def test_periodic_dressed_green_quadrature_converges():
    from gradscf.gw.ep_coupling import periodic_fan_self_energy
    errors = []
    for nw in (16, 32, 64):
        model, grid, f1, _ = inputs(nw)
        f2 = f1 + jnp.array([[[.2, .03j], [-.03j, -.1]], [[-.1, .04], [.04, .25]]])
        mixed = .4 * f1 + .6 * f2
        green = 0.
        for weight, fock in ((.4, f1), (.6, f2)):
            e, u = jnp.linalg.eigh(fock)
            gt = -jnp.exp(-grid.tau[:, None, None]*e - jnp.logaddexp(0., -grid.beta*e))
            green = green + weight * jnp.einsum('kim,tkm,kjm->tkij', u, gt, u.conj())
        exact = .4 * oracle(model, grid, f1) + .6 * oracle(model, grid, f2)
        got = periodic_fan_self_energy(green, mixed, 0., model, grid)['sigma_iw']
        errors.append(np.max(abs(got[nw:nw+3] - exact[nw:nw+3])))
    assert errors[-1] < errors[-2] / 3.5 < errors[0] / 12.25
    assert errors[-1] < 1e-5


def test_periodic_invalid_reference_rejected():
    from dataclasses import replace
    from gradscf.gw.ep_coupling import periodic_fan_self_energy
    model, grid, f, green = inputs()
    with pytest.raises(ValueError, match='Hermitian'):
        periodic_fan_self_energy(green, f.at[0, 0, 1].add(.1), 0., model, grid)
    with pytest.raises(Exception, match='positive'):
        jax.jit(lambda x: periodic_fan_self_energy(green, f, 0.,
            replace(model, energies=model.energies.at[0, 0].set(x)), grid)['sigma_iw'])(-.1).block_until_ready()


def test_q_accumulation_preserves_mixed_precision_promotion():
    from gradscf.gw.ep_coupling import periodic_fan_self_energy_tau
    green = jnp.ones((3, 1, 1, 1), dtype=jnp.complex64)
    g = jnp.ones((2, 1, 1, 1, 1), dtype=jnp.complex64)
    d = -jnp.ones((3, 2, 1), dtype=jnp.float32)
    result = periodic_fan_self_energy_tau(green, g, d, jnp.zeros((2, 1), int),
                                          jnp.array([.3, .7], dtype=jnp.float64))
    assert result.dtype == jnp.complex128
    np.testing.assert_allclose(result, 1., atol=1e-13)
