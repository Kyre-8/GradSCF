"""The shared tensor inverse budgets its arrays before spectral work."""
import importlib
import inspect

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gradscf.solvers import solve_tensor_sum


def _entry():
    assert 'max_memory' in inspect.signature(solve_tensor_sum).parameters
    return solve_tensor_sum


def test_tensor_sum_rejects_exhausted_budget_before_eigh(monkeypatch):
    entry = _entry()
    factors, rhs = (jnp.eye(3), jnp.eye(2)), jnp.ones((3, 2))
    monkeypatch.setattr(jnp.linalg, 'eigh', lambda *a, **k: pytest.fail('Budget reached eigensolve'))
    with pytest.raises(ValueError, match='memory|budget|resident'):
        entry(factors, rhs, max_memory=.0001, reserved_bytes=512)
    with pytest.raises(ValueError, match='reserved_bytes'):
        entry(factors, rhs, max_memory=1., reserved_bytes=-1)


def test_tensor_sum_keeps_original_and_promoted_inputs_charged(monkeypatch):
    entry = _entry()
    module = importlib.import_module('gradscf.solvers.linear.tensor_sum')
    rhs = jnp.ones((3, 2), jnp.float32)
    factors = jnp.eye(3, dtype=jnp.float64), jnp.eye(2, dtype=jnp.float32)
    calls = []

    def record(maximum, **kwargs):
        calls.append((maximum, kwargs['resident_bytes']))
        return 1_000_000

    monkeypatch.setattr(module, 'workspace_bytes', record)
    result = entry(factors, rhs, max_memory=2., reserved_bytes=4096)
    assert result.converged and result.solution.dtype == jnp.float64
    originals = rhs.nbytes+sum(f.nbytes for f in factors)
    promoted = (rhs.size+factors[1].size)*8
    assert calls and all(m == 2. and r >= 4096+originals+promoted for m, r in calls)


def test_tensor_sum_budget_preserves_repeated_eigenvalue_second_response():
    entry = _entry()
    base = jnp.eye(2)*2
    direction = jnp.asarray([[.1, .3], [.3, -.1]])
    rhs = jnp.asarray([[.2, -.4], [.7, .1]])

    def loss(x, budgeted):
        options = dict(max_memory=2., reserved_bytes=4096) if budgeted else {}
        solved = entry((base+x*direction, jnp.diag(jnp.array([.7, 1.1]))),
                       rhs*(1+.2*x), **options)
        return jnp.sum(solved.solution**2)

    def bundle(budgeted):
        return jax.jit(lambda x: jax.jvp(jax.value_and_grad(lambda y: loss(y, budgeted)),
                                        (x,), (1.,)))(0.)

    for actual, expected in zip(jax.tree.leaves(bundle(True)), jax.tree.leaves(bundle(False))):
        np.testing.assert_allclose(actual, expected, atol=2e-13, rtol=2e-12)
