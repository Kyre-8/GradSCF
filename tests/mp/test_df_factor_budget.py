"""Three-index AO2MO retains raw nonsymmetric inputs under a byte budget."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gradscf import ao2mo


def _entry():
    function = getattr(ao2mo, 'transform_df_factors', None)
    assert callable(function), 'Missing shared three-index DF transform'
    return function


@pytest.mark.parametrize('factor_dtype', [jnp.float32, jnp.float64])
def test_df_factor_blocks_keep_raw_chart_and_joint_second_response(factor_dtype):
    transform = _entry()
    rng = np.random.default_rng(3917)
    factors = jnp.asarray(rng.normal(size=(7, 3, 3)), factor_dtype)
    left, right = (jnp.asarray(rng.normal(size=(3, n))) for n in (2, 1))
    point = factors, left, right
    tangent = jax.tree.map(lambda a: jnp.full_like(a, .03), point)

    def loss(values, blocked):
        b, c, d = values
        value = (transform(b, c, d, max_memory=.0015) if blocked else
                 jnp.einsum('Qpq,pi,qa->Qia', b, c, d, precision='highest'))
        return jnp.sum(jnp.sin(value)**2)

    bundle = lambda blocked: jax.jit(lambda values: jax.jvp(
        jax.value_and_grad(lambda x: loss(x, blocked)), (values,), (tangent,)))(point)
    for a, b in zip(jax.tree.leaves(bundle(True)), jax.tree.leaves(bundle(False))):
        np.testing.assert_allclose(a, b, atol=3e-12 if a.dtype == jnp.float64 else 3e-6,
                                   rtol=3e-6)
    actual = transform(*point, max_memory=.0015)
    assert actual.dtype == jnp.result_type(*point) and actual.shape == (7, 2, 1)


def test_df_factor_empty_axes_do_not_call_a_contraction(monkeypatch):
    transform = _entry()
    monkeypatch.setattr(jnp, 'einsum', lambda *a, **k: pytest.fail('Empty transform contracted'))
    for shapes in (((0, 3, 3), (3, 2), (3, 1)), ((7, 3, 3), (3, 0), (3, 2))):
        actual = transform(*(jnp.zeros(s) for s in shapes), max_memory=1.)
        assert actual.shape == (shapes[0][0], shapes[1][1], shapes[2][1])


def test_df_factor_budget_fails_before_contraction(monkeypatch):
    transform = _entry()
    monkeypatch.setattr(jnp, 'einsum', lambda *a, **k: pytest.fail('Budget reached contraction'))
    with pytest.raises(ValueError, match='memory|budget|resident'):
        transform(jnp.ones((7, 3, 3)), jnp.ones((3, 2)), jnp.ones((3, 1)),
                  max_memory=.0001, reserved_bytes=4096)
