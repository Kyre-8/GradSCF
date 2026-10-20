"""Invalid streamed triples must reject active JVPs and VJPs with NaNs."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from test_streamed_triples_response import moments


def _derivative(scalar, operation):
    if operation == "jvp":
        return jax.jit(lambda x: jax.jvp(scalar, (x,), (jnp.ones_like(x),))[1])(1.)
    return jax.jit(jax.grad(scalar))(1.)


@pytest.mark.parametrize("operation", ["jvp", "vjp"])
def test_private_singular_triples_reject_active_derivatives(moments, operation):
    from gradscf.cc._streamed_triples import _streamed_general_correction

    ints, t1, t2 = moments
    bad = ints._replace(fock=jnp.zeros_like(ints.fock))

    def scalar(scale):
        connected, singles, _, _ = _streamed_general_correction(
            bad, t1, scale * t2, denominator_tol=1e-10)
        return connected + singles

    _, _, minimum, valid = _streamed_general_correction(
        bad, t1, t2, denominator_tol=1e-10)
    assert minimum == 0. and not valid
    assert np.isnan(_derivative(scalar, operation))


@pytest.fixture(scope="module", params=["restricted", "unrestricted"])
def public_inputs(request):
    from gradscf.cc.ground import METHODS
    from gradscf.cc.integrals import prepare_integrals
    from gradscf.cc.types import CCResult
    from gradscf.cc.uccsd import prepare_ucc_integrals

    rng = np.random.default_rng(842)
    n = 4
    factors = rng.normal(size=(4, n, n)) * .05
    factors += factors.transpose(0, 2, 1)
    eri = jnp.asarray(np.einsum("Lpq,Lrs->pqrs", factors, factors))
    if request.param == "restricted":
        nocc = 2
        mean_field = prepare_integrals(jnp.zeros((n, n)), eri, nocc=nocc).fock
        h = jnp.diag(jnp.array([-1.2, -.9, .5, .8])) - mean_field
        singular_h = -mean_field
        t1 = jnp.asarray(rng.normal(size=(2, 2)) * .03)
        t2 = jnp.asarray(rng.normal(size=(2, 2, 2, 2)) * .02)
        t2 = .5 * (t2 + t2.transpose(1, 0, 3, 2))
    else:
        nocc = (2, 1)
        eri = (eri, eri, eri)
        mean_field = prepare_ucc_integrals(
            (jnp.zeros((n, n)),) * 2, eri, nocc=nocc)[0].fock
        # prepare_ucc_integrals orders active occupied before active virtual.
        active = jnp.array([0, 1, 4, 2, 3, 5, 6, 7])
        original = jnp.zeros((2 * n, 2 * n)).at[jnp.ix_(active, active)].set(mean_field)
        fields = (original[:n, :n], original[n:, n:])
        levels = (jnp.array([-1.2, -.9, .5, .8]), jnp.array([-1.1, .4, .7, .9]))
        h = tuple(jnp.diag(e) - f for e, f in zip(levels, fields))
        singular_h = tuple(-f for f in fields)
        t1 = tuple(jnp.asarray(rng.normal(size=(o, n-o)) * .03) for o in nocc)
        blocks = [jnp.asarray(rng.normal(size=shape) * .02) for shape in
                  ((2, 2, 2, 2), (2, 1, 2, 3), (1, 1, 3, 3))]
        for i in (0, 2):
            blocks[i] = blocks[i] - blocks[i].swapaxes(0, 1)
            blocks[i] = blocks[i] - blocks[i].swapaxes(2, 3)
        t2 = tuple(blocks)
    zero = jnp.asarray(0.)
    state = CCResult(zero, zero, zero, t1, t2, zero, zero, jnp.asarray(0),
                     jnp.asarray(True), jnp.asarray(1.), jnp.asarray(METHODS.index("ccsd")))
    return h, singular_h, eri, nocc, state


@pytest.mark.parametrize("reason", ["state", "denominator"])
@pytest.mark.parametrize("operation", ["jvp", "vjp"])
def test_public_streamed_invalid_triples_reject_active_derivatives(public_inputs, reason, operation):
    from gradscf import cc

    h, singular_h, eri, nocc, state = public_inputs
    if reason == "denominator":
        h = singular_h
    else:
        state = state._replace(converged=jnp.asarray(False))

    def evaluate(scale):
        amplitudes = jax.tree.map(lambda a: scale * a, state.t2)
        changed = state._replace(t2=amplitudes)
        # A loose residual bound isolates the failed state flag or spectrum.
        return cc.evaluate_triples(h, eri, changed, nocc=nocc, algorithm="streamed",
                                   orbital_basis="semicanonical", residual_tol=1e6)

    result = evaluate(1.)
    assert not result.valid
    assert all(np.isnan(x) for x in
               (result.energy, result.connected_component, result.singles_component))
    if reason == "denominator":
        assert result.min_abs_denominator < 1e-10
    else:
        assert result.min_abs_denominator > 1e-10
    assert np.isnan(_derivative(lambda x: evaluate(x).energy, operation))


def test_public_streamed_invalid_state_second_derivative_is_nan(public_inputs):
    from gradscf import cc

    h, _, eri, nocc, state = public_inputs
    state = state._replace(converged=jnp.asarray(False))

    def scalar(scale):
        changed = state._replace(t2=jax.tree.map(lambda a: scale * a, state.t2))
        return cc.evaluate_triples(h, eri, changed, nocc=nocc, algorithm="streamed",
                                   orbital_basis="semicanonical", residual_tol=1e6).energy

    assert np.isnan(jax.jit(jax.grad(jax.grad(scalar)))(1.))
