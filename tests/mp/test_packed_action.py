"""Packed target tiles retain the exact normal-ordered tensor action."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gradscf.mp._tensor import TensorSpace, build_tensor_action, make_tensor_space


def _inputs(nmo, nocc, frozen):
    rng = np.random.default_rng(2421)
    # Deliberately raw tensors exercise the existing input chart, without
    # relying on extra ERI permutation symmetry in the packed contraction.
    g = jnp.asarray(rng.normal(size=(nmo,) * 4) * .02)
    f = jnp.asarray(rng.normal(size=(nmo, nmo)) * .1)
    unrestricted = isinstance(nocc, tuple)
    eri = (g, g * .97, g * 1.03) if unrestricted else g
    focks = (f, f * 1.07) if unrestricted else (f,)
    rank = sum(nocc) if unrestricted else 2 * nocc
    space = make_tensor_space(nmo, nocc, frozen=frozen, max_excitation=rank)
    vector = jnp.asarray(rng.normal(size=space.size))
    return eri, focks, vector, space


@pytest.mark.parametrize("nmo,nocc,frozen,source,target", [
    (4, 2, None, 4, 4),
    (4, (2, 1), None, 3, 2),
    (5, (2, 2), ([0], [4]), 3, 3),
    (6, 3, None, 6, 2),
    (6, 3, None, 6, 6),
], ids=["quadruples", "open-shell-target", "frozen-spins", "sextuple-source", "sextuple-full"])
def test_packed_action_matches_dense_without_expanding(
        monkeypatch, nmo, nocc, frozen, source, target):
    eri, focks, vector, space = _inputs(nmo, nocc, frozen)
    expected = jax.jit(lambda g, f, c: build_tensor_action(g, f, space).apply(
        c, max_input_rank=source, max_output_rank=target))(eri, focks, vector)
    packed = make_tensor_space(nmo, nocc, frozen=frozen,
                              max_excitation=max(space.ranks), packed_only=True)

    def no_expansion(*args, **kwargs):
        raise AssertionError("Packed actions must not expand antisymmetric tensors")

    monkeypatch.setattr(TensorSpace, "expand", no_expansion)
    actual = jax.jit(lambda g, f, c: build_tensor_action(g, f, packed).apply_packed(
        c, max_input_rank=source, max_output_rank=target,
        max_tile_elements=128))(eri, focks, vector)
    np.testing.assert_allclose(actual, expected, atol=3e-11, rtol=2e-12)
    np.testing.assert_array_equal(np.asarray(actual)[np.asarray(space.ranks) > target], 0.)


@pytest.mark.parametrize("unrestricted", [False, True])
def test_joint_integral_fock_state_response_matches_dense(unrestricted):
    eri, focks, vector, space = _inputs(4, (2, 1) if unrestricted else 2, None)
    rng = np.random.default_rng(2422)
    tangents = jax.tree.map(lambda x: jnp.asarray(rng.normal(size=x.shape)) * .03,
                            (eri, focks, vector))

    def evaluate(packed):
        def action(*values):
            obj = build_tensor_action(values[0], values[1], space)
            return (obj.apply_packed(values[2], max_tile_elements=128) if packed
                    else obj(values[2]))
        def scalar(*values):
            value = action(*values)
            return jnp.vdot(value, value)
        def bundle(*values):
            (_, gradient), (_, hvp) = jax.jvp(
                jax.value_and_grad(scalar, argnums=(0, 1, 2)), values, tangents)
            return (action(*values), jax.jvp(action, values, tangents)[1],
                    gradient, hvp)
        print("Joint tensor response: packed=%s unrestricted=%s" %
              (packed, unrestricted), flush=True)
        return jax.jit(bundle)(eri, focks, vector)

    expected, actual = evaluate(False), evaluate(True)
    for a, b in zip(jax.tree.leaves(actual), jax.tree.leaves(expected)):
        np.testing.assert_allclose(a, b, atol=3e-10, rtol=3e-11)


def test_packed_tile_budget_is_checked_before_contraction(monkeypatch):
    from gradscf.mp import _tensor_packed
    eri, focks, vector, space = _inputs(4, 2, None)
    action = build_tensor_action(eri, focks, space)

    def no_allocation(*args, **kwargs):
        raise AssertionError("Minimum tile capacity must precede kernel compression")

    monkeypatch.setattr(_tensor_packed, "_compressed", no_allocation)
    for limit in (0, 1):
        with pytest.raises(ValueError, match="max_tile_elements"):
            action.apply_packed(vector, max_tile_elements=limit)


@pytest.mark.parametrize("fock_dtype", [jnp.float32, jnp.float64])
def test_packed_action_preserves_mixed_dtype_promotion(fock_dtype):
    eri, focks, vector, space = _inputs(4, 2, None)
    focks = tuple(x.astype(fock_dtype) for x in focks)
    vector = vector.astype(jnp.float32)
    action = build_tensor_action(eri, focks, space)
    expected = jax.jit(lambda c: action.apply(c, max_output_rank=0))(vector)
    actual = jax.jit(lambda c: action.apply_packed(c, max_output_rank=0,
                                                 max_tile_elements=128))(vector)
    assert actual.dtype == expected.dtype == jnp.float64
    # Reordering contractions containing float32 Fock kernels changes their
    # float32 accumulation rounding, even though the final ERI sum is float64.
    tolerance = 4 * np.finfo(np.float32).eps if fock_dtype == jnp.float32 else 2e-13
    np.testing.assert_allclose(actual, expected, atol=tolerance, rtol=tolerance)
