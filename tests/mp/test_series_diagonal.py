"""Canonical Taylor Jacobians have exact structure; noncanonical charts do not."""
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gradscf import mp
from gradscf.mp import series
from test_series_staging import model


@pytest.fixture(autouse=True)
def cold_kernel():
    series._run_series.clear_cache()
    yield
    series._run_series.clear_cache()


@pytest.mark.parametrize('unrestricted', [False, True], ids=['rhf', 'uhf'])
@pytest.mark.parametrize('algorithm', ['series', 'tensor'])
def test_canonical_coefficients_and_raw_chart_response_match_generic_solver(unrestricted, algorithm, monkeypatch):
    h, g, no = model(unrestricted)
    cfg = mp.MPConfig(order=4, algorithm=algorithm, with_coefficients=True)
    point = (h, g)
    rng = np.random.default_rng(722)
    direction = jax.tree.map(lambda x: jnp.asarray(rng.normal(scale=.02, size=x.shape)), point)
    original, observed = series.solve_linear, []

    def generic(operator, rhs, **kwargs):
        return original(replace(operator, exact_diagonal=False), rhs, **kwargs)

    def recorded(operator, rhs, **kwargs):
        observed.append(operator.exact_diagonal)
        return original(operator, rhs, **kwargs)

    def value(values):
        result = mp.run_mp(*values, nocc=no, config=cfg)
        return result.corrections, result.t2, result.wavefunction_coefficients

    def bundle(values):
        result, tangent = jax.jvp(value, (values,), (direction,))
        loss = lambda p: sum((i+1)*jnp.sum(x*x) for i, x in enumerate(jax.tree.leaves(value(p))))
        gradient = jax.grad(loss)(values)
        hvp = jax.jvp(jax.grad(loss), (values,), (direction,))[1]
        return result, tangent, gradient, hvp

    monkeypatch.setattr(series, 'solve_linear', generic)
    expected = jax.jit(lambda p: bundle(p))(point)
    series._run_series.clear_cache()
    monkeypatch.setattr(series, 'solve_linear', recorded)
    actual = jax.jit(lambda p: bundle(p))(point)
    assert observed and all(observed), 'Canonical operator was not declared exactly diagonal'
    for got, want in zip(jax.tree.leaves(actual), jax.tree.leaves(expected)):
        np.testing.assert_allclose(got, want, atol=3e-11, rtol=3e-9)


def test_noncanonical_diagonal_primal_keeps_offdiagonal_fock_response(monkeypatch):
    h, g, no = model()
    mean_field = jnp.diag(jnp.linspace(-1., 1., h.shape[0]))-h
    fock = h+mean_field
    perturbation = jnp.array([[0., 0., 0.], [0., 0., .15], [0., .15, 0.]])
    cfg = mp.MPConfig(order=2, algorithm='series', orbital_basis='noncanonical', with_t2=False)
    original, observed = series.solve_linear, []

    def recorded(operator, rhs, **kwargs):
        observed.append(operator.exact_diagonal)
        return original(operator, rhs, **kwargs)

    monkeypatch.setattr(series, 'solve_linear', recorded)

    def energy(t):
        return mp.run_mp(h+t*perturbation, g, nocc=no, config=cfg).correlation_energy

    def oracle(t):
        values, vectors = jnp.linalg.eigh((fock+t*perturbation)[1:, 1:])
        interaction = vectors.T@g[0, 1:, 0, 1:]@vectors
        return jnp.sum((2*interaction-interaction.T)*interaction/
                       (2*fock[0, 0]-values[:, None]-values[None, :]))

    actual = jax.jit(lambda t: (energy(t), jax.grad(energy)(t),
        jax.jvp(jax.grad(energy), (t,), (jnp.ones_like(t),))[1]))(0.)
    expected = oracle(0.), jax.grad(oracle)(0.), jax.hessian(oracle)(0.)
    np.testing.assert_allclose(actual, expected, atol=3e-12, rtol=3e-9)
    assert observed and not any(observed), 'A diagonal primal is not a diagonal operator family'
    assert abs(float(actual[1])) > 1e-9
    # An incorrect numerical-only declaration fails away from the diagonal point.
    series._run_series.clear_cache()
    monkeypatch.setattr(series, 'solve_linear', lambda operator, rhs, **kwargs:
        original(replace(operator, exact_diagonal=True), rhs, **kwargs))
    wrong = mp.run_mp(h+.2*perturbation, g, nocc=no, config=cfg)
    assert not wrong.valid and np.isnan(wrong.correlation_energy)
