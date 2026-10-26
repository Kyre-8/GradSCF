"""Dense Taylor checkpoint policies retain all coefficients and outer response."""
from math import ceil

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gradscf import mp
from gradscf._memory import workspace_bytes
from gradscf.mp import _series_memory, series
from gradscf.mp.integrals import _reference_data
from gradscf.solvers.nonlinear import taylor
from test_series_staging import model


def checkpoint_case(with_coefficients):
    """Derive budgets from the real traced planner, including requested outputs."""
    h, g, nocc = model()
    config = mp.MPConfig(order=4, algorithm="tensor", with_coefficients=with_coefficients)
    records = []

    def probe(hh, gg):
        hs, gs, focks, _ = _reference_data(hh, gg, nocc)
        space = series.make_series_space(h.shape[0], nocc, config=config, packed_only=True)
        dense_bytes = 32 * space.dense_elements * h.dtype.itemsize
        full = _series_memory.plan_series(hs, gs, focks, space, nocc, (), config, 1.)
        assert not full.packed and not full.checkpoint
        # The planner reserves 20% of decimal MB. Stay inside each policy's
        # interval rather than relying on a rounded, fixture-specific MB value.
        budgets = {checkpoint: (ceil((full.resident_bytes + scratch) * 5 / 4) + 1) / 1e6
                   for checkpoint, scratch in ((False, 3 * dense_bytes),
                                               (True, 3 * dense_bytes // 2))}
        plans = {}
        for checkpoint, maximum in budgets.items():
            plan = _series_memory.plan_series(hs, gs, focks, space, nocc, (), config, maximum)
            available = workspace_bytes(maximum, resident_bytes=plan.resident_bytes)
            assert not plan.packed and plan.checkpoint == checkpoint
            assert available >= dense_bytes
            assert (available < 2 * dense_bytes) == checkpoint
            plans[checkpoint] = plan
        records.append((budgets, plans))
        return hh

    # Trace the same h/g owners as _run_series; eager array aliasing must not
    # choose a budget that accidentally takes another policy after tracing.
    jax.eval_shape(probe, h, g)
    budgets, plans = records[0]
    return h, g, nocc, config, budgets, plans


def test_shared_normalized_taylor_helper_preserves_outer_ad():
    helper = getattr(taylor, "_jet", None)
    assert callable(helper), "The normalized Taylor helper belongs in solvers/nonlinear/taylor.py"
    assert series._jet is helper, "MP and root Taylor propagation must share coefficient normalization"

    def coefficients(a):
        _, terms = helper(lambda x: x**3, (a,),
                          ((jnp.ones_like(a), jnp.zeros_like(a), jnp.zeros_like(a)),))
        return jnp.stack(terms)

    np.testing.assert_allclose(jax.jit(coefficients)(2.), [12., 6., 1.], atol=1e-12)
    np.testing.assert_allclose(jax.jacfwd(coefficients)(2.), [12., 3., 0.], atol=1e-12)
    np.testing.assert_allclose(jax.jacrev(coefficients)(2.), [12., 3., 0.], atol=1e-12)
    _, second = jax.jvp(jax.jacrev(coefficients), (2.,), (1.,))
    np.testing.assert_allclose(second, [6., 0., 0.], atol=1e-12)


@pytest.mark.parametrize("with_coefficients", [False, True])
def test_dense_checkpoint_preserves_complete_history_and_response(with_coefficients, monkeypatch):
    h, g, nocc, config, budgets, plans = checkpoint_case(with_coefficients)
    _, _, focks, _ = _reference_data(h, g, nocc)
    mean_field = focks[0] - h
    original = _series_memory.plan_series
    observed = []

    def record(*args, **kwargs):
        plan = original(*args, **kwargs)
        observed.append(plan)
        return plan

    monkeypatch.setattr(_series_memory, "plan_series", record)

    def evaluate(checkpoint):
        def loss(parameter):
            # This is the existing fixture's canonical physical input chart:
            # scale the interaction while preserving its reference Fock.
            result = mp.run_mp(h - (parameter - 1.) * mean_field, parameter * g,
                               nocc=nocc, config=config, max_memory=budgets[checkpoint])
            return result.correlation_energy, result

        def bundle(parameter):
            ((_, result), gradient), ((jvp, response), hvp) = jax.jvp(
                jax.value_and_grad(loss, has_aux=True), (parameter,), (.2,))
            return (result, gradient, jvp, hvp, response.corrections,
                    response.wavefunction_coefficients, response.t2)

        series._run_series.clear_cache()
        observed.clear()
        result = jax.jit(bundle)(1.)
        assert observed, "Observe the policy used by the actual differentiated MP call"
        assert all(not p.packed and p.checkpoint == checkpoint for p in observed)
        assert all(p.resident_bytes == plans[checkpoint].resident_bytes for p in observed)
        return result

    expected, actual = evaluate(False), evaluate(True)
    assert bool(expected[0].valid) and bool(actual[0].valid)
    assert actual[0].corrections.shape == (config.order - 1,)
    assert actual[0].t2 is not None
    if with_coefficients:
        assert actual[0].wavefunction_coefficients.shape[0] == config.order // 2 + 1
        np.testing.assert_array_equal(actual[0].wavefunction_coefficients[:, 0], [1., 0., 0.])
    else:
        assert actual[0].wavefunction_coefficients is None
    for a, e in zip(jax.tree.leaves(actual), jax.tree.leaves(expected)):
        assert np.isfinite(a).all() and np.isfinite(e).all()
        np.testing.assert_allclose(a, e, atol=3e-11, rtol=3e-10)
