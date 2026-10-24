"""Selected native LCCD/LCCSD under one finite budget, including implicit HVP."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gradscf import cc
from gradscf.cc.integrals import from_orbitals
from test_native_virtual_action import _problem as native_problem
from test_prepared_properties import _config


@pytest.mark.parametrize('model', ['lccd', 'lccsd'])
def test_selected_native_budget_preserves_lcc_energy_gradient_and_hvp(model):
    # The same weak five-MO frame used by test_caller_memory._native_frame.
    _, basis, _, _ = native_problem()
    c = jnp.asarray(np.random.default_rng(6561).normal(scale=.08, size=(basis.nao, 5)))
    fock = jnp.diag(jnp.asarray([-1.3, -.8, .3, .6, .9]))
    dc = .02*jnp.cos(c)
    probe = jnp.arange(25., dtype=c.dtype).reshape(5, 5)/250
    df = .5*(probe+probe.T)
    config = _config(model)
    point = jnp.asarray(.02)

    def reference(x):
        return -2.+.007*x+.0004*x*x

    def evaluate(x, mode):
        data = from_orbitals(c+x*dc, fock+x*df, 2, reference(x),
                             direct_basis=basis, max_memory=32., vvvv=mode)
        assert (type(data.vvvv).__name__ == 'VirtualERI') == (mode == 'direct')
        state = cc.run_cc_from_integrals(data, config=config, max_memory=32.)
        return state.total_energy, (state.correlation_energy, state.converged)

    def bundle(x, mode):
        function = lambda y: evaluate(y, mode)
        primal, tangent = jax.jvp(jax.value_and_grad(function, has_aux=True),
                                  (x,), (jnp.asarray(1.),))
        (energy, (correlation, converged)), gradient = primal
        return energy, gradient, tangent[1], correlation, converged

    direct = jax.jit(lambda x: bundle(x, 'direct'))(point)
    dense = jax.jit(lambda x: bundle(x, 'dense'))(point)
    assert bool(direct[-1]) and bool(dense[-1])
    assert abs(float(direct[-2])) > 1e-12
    reference_terms = jnp.array([reference(point), jax.grad(reference)(point),
                                 jax.grad(jax.grad(reference))(point)])
    for result in (direct, dense):
        assert np.isfinite(np.asarray(result[:4])).all()
    np.testing.assert_allclose(direct[:4], dense[:4], atol=3e-11, rtol=3e-8)
    # A live reference must not hide a missing native coefficient/Fock response.
    correlated = jnp.array(direct[:3])-reference_terms
    assert np.all(np.abs(np.asarray(correlated)) > 1e-12)
    np.testing.assert_allclose(correlated, jnp.array(dense[:3])-reference_terms,
                               atol=3e-12, rtol=3e-7)
