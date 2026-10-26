"""Nested-JIT BSE response keeps static indices and physical guards traceable."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gradscf import bse
from gradscf.solvers import LinearSolverConfig


def model(t):
    energy = jnp.array([-.8, .35, .7])
    factors = jnp.array([
        [[.2, .08, .03], [.08, .13, .04], [.03, .04, .15]],
        [[.1, .03, -.04], [.03, .09, .02], [-.04, .02, .12]],
    ])
    qp = energy + t * jnp.array([.03, -.02, .04])
    screening = energy + jnp.array([-.1, -.02, .03])
    screening += t * jnp.array([.02, -.01, .03])
    return qp, screening, factors * (1 + .1 * t)


def configuration(*, tda=True, solver="dense", energy_only=False, screening="direct"):
    return bse.BSEConfig(
        tda=tda, solver=solver, nroots=1, conv_tol=1e-11, block_size=1,
        gradient_mode="eigenvalue_only" if energy_only else "implicit_eigenvector",
        response_config=LinearSolverConfig(method="direct", rtol=1e-11, max_dense=8),
        screening_config=LinearSolverConfig(method=screening, rtol=1e-12,
                                            atol=1e-14, restart=2),
    )


def observable(result, *, energy_only=False):
    value = jnp.sum(result.excitation_energies)
    if not energy_only:
        amplitudes = jax.tree.leaves((result.x_amplitudes, result.y_amplitudes))
        for amplitude in amplitudes:
            # A sign-invariant observable with a nonzero orbital-frame response.
            value += .2 * jnp.sum(jnp.arange(amplitude.size).reshape(amplitude.shape)
                                  * amplitude**2)
    return value


def check_nested_response(value):
    t = .13
    inner = jax.jit(value)
    expected_value, raw = jax.jit(jax.value_and_grad(value))(t)
    actual_value, reverse = jax.jit(jax.value_and_grad(inner))(t)
    forward = jax.jit(lambda x: jax.jvp(inner, (x,), (jnp.ones_like(x),))[1])(t)
    finite = (inner(t + 1e-4) - inner(t - 1e-4)) / 2e-4
    assert np.isfinite(expected_value) and np.isfinite(raw)
    np.testing.assert_allclose(actual_value, expected_value, atol=2e-12, rtol=0)
    np.testing.assert_allclose([forward, reverse], raw, atol=2e-10, rtol=0)
    np.testing.assert_allclose(raw, finite, atol=2e-8, rtol=0)


@pytest.mark.parametrize("solver", ["dense", "davidson"])
@pytest.mark.parametrize("energy_only", [False, True])
@pytest.mark.parametrize("screening", ["direct", "gmres"])
def test_tda_nested_jvp_vjp(solver, energy_only, screening):
    space = bse.make_bse_space(3, 1)
    config = configuration(solver=solver, energy_only=energy_only, screening=screening)
    check_nested_response(lambda t: observable(
        bse.run_bse(*model(t), space, config=config), energy_only=energy_only))


@pytest.mark.parametrize("screening", ["direct", "gmres"])
def test_full_davidson_nested_jvp_vjp(screening):
    space = bse.make_bse_space(3, 1)
    config = configuration(tda=False, solver="davidson", screening=screening)
    check_nested_response(lambda t: observable(bse.run_bse(*model(t), space, config=config)))


@pytest.mark.parametrize("tda", [True, False])
def test_spin_conserving_nested_jvp_vjp(tda):
    from dataclasses import replace

    space = bse.make_bse_space(3, (1, 1))
    config = replace(configuration(tda=tda, solver="davidson"), singlet=None)

    def value(t):
        qp, screening, factors = model(t)
        qp = jnp.stack((qp, qp + jnp.array([.02, .09, -.04])))
        screening = jnp.stack((screening, screening + jnp.array([.03, .07, -.02])))
        factors = jnp.stack((factors, factors * .9))
        return observable(bse.run_bse(qp, screening, factors, space, config=config))

    check_nested_response(value)


@pytest.mark.parametrize("failure", ["gap", "factors", "solve"])
def test_invalid_screening_still_rejects_nested_jvp_vjp(failure):
    from dataclasses import replace

    space = bse.make_bse_space(3, 1)
    config = configuration()
    if failure == "solve":
        config = replace(config, screening_config=LinearSolverConfig(
            method="gmres", rtol=1e-12, atol=1e-14, restart=1, maxiter=1))

    def solve(t):
        qp, screening, factors = model(t)
        if failure == "gap":
            screening = screening.at[1].set(screening[0])
        if failure == "factors":
            factors = factors.at[0, 0, 1].add(.01)
        return bse.run_bse(qp, screening, factors, space, config=config)

    result = jax.jit(solve)(.13)
    assert not np.any(result.converged | result.response_valid)
    inner = jax.jit(lambda t: solve(t).excitation_energies[0])
    forward = jax.jit(lambda t: jax.jvp(inner, (t,), (jnp.ones_like(t),))[1])(.13)
    reverse = jax.jit(jax.grad(inner))(.13)
    assert not np.isfinite(forward) and not np.isfinite(reverse)
