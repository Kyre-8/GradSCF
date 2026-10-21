"""Budgeted AO->MO blocks retain their exact values and nested JAX response.

The dense AO ERIs below are tiny test oracles. A DF input remains resident;
the budget limits additional auxiliary-axis transformation intermediates.
DF accumulation reserves three full output buffers; dense and s4 reserve two.
It plans forward arrays, not the full AD process or its required cotangents;
caller and derivative-budget integration remains a separate WF-29 task.
"""
import inspect

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gradscf._memory import DEFAULT_MAX_MEMORY_MB
from gradscf.integrals.molecular import ao2mo


def _problem(dtype=np.float64, columns=(2, 3, 2, 1)):
    rng = np.random.default_rng(812)
    factors = rng.normal(scale=.3, size=(17, 4, 4))
    factors = factors + factors.transpose(0, 2, 1)
    coefficients = tuple(jnp.asarray(rng.normal(scale=.4, size=(4, n)), dtype=dtype)
                         for n in columns)
    return jnp.asarray(factors, dtype=dtype), coefficients


def _full_axis_block(factors, coefficients):
    left = jnp.einsum("Lpq,pi,qj->Lij", factors, *coefficients[:2], precision="highest")
    right = jnp.einsum("Lpq,pi,qj->Lij", factors, *coefficients[2:], precision="highest")
    return jnp.einsum("Lij,Lkl->ijkl", left, right, precision="highest")


def _dense_block(factors, coefficients):
    eri = jnp.einsum("Lpq,Lrs->pqrs", factors, factors, precision="highest")
    return jnp.einsum("pqrs,pi,qj,rk,sl->ijkl", eri, *coefficients,
                      precision="highest", optimize="optimal")


def _tight_budget(dtype):
    # 4.5 decimal kB for float64: live inputs and three DF output buffers fit,
    # while transforming all 17 auxiliary rows into MO factors does not.
    return .0045 * np.dtype(dtype).itemsize / 8


def _equation_shapes(value):
    """Visit nested scan/checkpoint jaxprs without depending on JAX internals."""
    value = getattr(value, "jaxpr", value)
    if hasattr(value, "eqns"):
        for equation in value.eqns:
            for variable in equation.outvars:
                shape = getattr(getattr(variable, "aval", None), "shape", None)
                if shape is not None:
                    yield tuple(shape)
            yield from _equation_shapes(equation.params)
    elif isinstance(value, dict):
        for child in value.values():
            yield from _equation_shapes(child)
    elif isinstance(value, (tuple, list)):
        for child in value:
            yield from _equation_shapes(child)


def _assert_tree_close(actual, expected, *, atol, rtol):
    assert jax.tree.structure(actual) == jax.tree.structure(expected)
    for observed, reference in zip(jax.tree.leaves(actual), jax.tree.leaves(expected)):
        np.testing.assert_allclose(observed, reference, atol=atol, rtol=rtol)


@pytest.mark.parametrize("dtype", [np.float32, np.float64], ids=["float32", "float64"])
def test_df_auxiliary_tiles_match_full_axis_oracle(dtype):
    factors, coefficients = _problem(dtype)
    expected = _full_axis_block(factors, coefficients)
    actual = jax.jit(lambda b, cs: ao2mo.transform_eri_block(
        cs, df_factors=b, max_memory=_tight_budget(dtype)))(factors, coefficients)
    assert actual.shape == (2, 3, 2, 1)
    assert actual.dtype == jnp.dtype(dtype)
    tolerance = 3e-6 if dtype == np.float32 else 2e-12
    np.testing.assert_allclose(actual, expected, atol=tolerance, rtol=tolerance)


def test_df_auxiliary_tiles_do_not_form_full_axis_mo_factors():
    factors, coefficients = _problem()
    traced = jax.make_jaxpr(lambda b, cs: ao2mo.transform_eri_block(
        cs, df_factors=b, max_memory=_tight_budget(np.float64)))(factors, coefficients)
    # The supplied [naux,nao,nao] factors are legitimate resident data.
    # A new full-naux MO factor or one-leg MO intermediate defeats the budget.
    forbidden = [shape for shape in _equation_shapes(traced)
                 if factors.shape[0] in shape and shape != factors.shape]
    assert not forbidden, f"Full auxiliary-axis intermediates escaped chunking: {forbidden}"


def test_ao2mo_uses_shared_default_memory_budget():
    parameter = inspect.signature(ao2mo.transform_eri_block).parameters["max_memory"]
    assert parameter.default == DEFAULT_MAX_MEMORY_MB


def test_empty_auxiliary_space_returns_zero_block():
    factors, coefficients = _problem()
    actual = jax.jit(lambda b, cs: ao2mo.transform_eri_block(
        cs, df_factors=b, max_memory=_tight_budget(np.float64)))(factors[:0], coefficients)
    assert actual.shape == (2, 3, 2, 1)
    assert actual.dtype == jnp.float64
    np.testing.assert_array_equal(actual, np.zeros((2, 3, 2, 1)))


def test_empty_mo_space_retains_shape_and_dtype():
    factors, coefficients = _problem(np.float32)
    coefficients = (coefficients[0][:, :0], *coefficients[1:])
    actual = jax.jit(lambda b, cs: ao2mo.transform_eri_block(
        cs, df_factors=b, max_memory=_tight_budget(np.float32)))(factors, coefficients)
    assert actual.shape == (0, 3, 2, 1)
    assert actual.dtype == jnp.float32
    assert actual.size == 0


def test_mixed_dtype_df_block_matches_promoted_dense_oracle():
    factors, _ = _problem(np.float32)
    _, coefficients = _problem(np.float64)
    result_dtype = jnp.result_type(factors, *coefficients)
    expected = _dense_block(factors.astype(result_dtype), coefficients)
    actual = jax.jit(lambda b, cs: ao2mo.transform_eri_block(
        cs, df_factors=b, max_memory=_tight_budget(np.float64)))(factors, coefficients)
    assert actual.dtype == result_dtype == jnp.float64
    np.testing.assert_allclose(actual, expected, atol=2e-12, rtol=2e-12)


@pytest.mark.parametrize("representation", ["df", "dense", "packed"])
def test_block_memory_limit_counts_live_inputs_and_output(representation, monkeypatch):
    factors, coefficients = _problem(columns=(7, 6, 5, 4))
    if representation == "df":
        source, keyword = factors, "df_factors"
    else:
        source = jnp.einsum("Lpq,Lrs->pqrs", factors, factors, precision="highest")
        keyword = "eri"
        if representation == "packed":
            rows, columns = np.tril_indices(factors.shape[1])
            source = source[rows, columns][:, rows, columns]
            keyword = "eri_pair_matrix"
    resident = source.nbytes + sum(c.nbytes for c in coefficients)
    output = np.prod([c.shape[1] for c in coefficients]) * np.dtype(source.dtype).itemsize
    buffers = 3 if representation == "df" else 2
    # Shared workspace_bytes reserves 20%. Leave 64 bytes less than the
    # required live inputs plus live outputs, even before tile scratch.
    # DF tail addition retains the accumulator, its contribution and their sum.
    max_memory = (resident + buffers * output - 64) / .8 / 1_000_000

    def forbidden(*args, **kwargs):
        pytest.fail("The memory budget must be checked before AO->MO contraction")

    monkeypatch.setattr(ao2mo.jnp, "einsum", forbidden)
    with pytest.raises(ValueError, match="(?i)(memory|budget|resident|minimum)"):
        ao2mo.transform_eri_block(coefficients, max_memory=max_memory, **{keyword: source})


@pytest.mark.parametrize("max_memory", [0., -1., np.inf, np.nan, True])
def test_block_memory_limit_rejects_invalid_values(max_memory, monkeypatch):
    factors, coefficients = _problem()

    def forbidden(*args, **kwargs):
        pytest.fail("An invalid budget must fail before AO->MO contraction")

    monkeypatch.setattr(ao2mo.jnp, "einsum", forbidden)
    with pytest.raises(ValueError, match="max_memory"):
        ao2mo.transform_eri_block(coefficients, df_factors=factors, max_memory=max_memory)


@pytest.mark.parametrize("representation", ["dense", "packed"])
def test_existing_ao_representations_accept_budget(representation):
    factors, coefficients = _problem()
    eri = jnp.einsum("Lpq,Lrs->pqrs", factors, factors, precision="highest")
    if representation == "dense":
        kwargs = dict(eri=eri)
    else:
        rows, columns = np.tril_indices(factors.shape[1])
        kwargs = dict(eri_pair_matrix=eri[rows, columns][:, rows, columns])
    expected = _dense_block(factors, coefficients)
    observed = jax.jit(lambda cs: ao2mo.transform_eri_block(
        cs, max_memory=1., **kwargs))(coefficients)
    np.testing.assert_allclose(observed, expected, atol=2e-12, rtol=2e-12)


def test_df_chunked_coefficient_and_factor_response_matches_dense():
    factors, coefficients = _problem()
    point = factors, *coefficients
    rng = np.random.default_rng(813)
    direction = tuple(jnp.asarray(rng.normal(scale=.07, size=x.shape)) for x in point)
    cotangent = jnp.linspace(-.3, .4, 12).reshape(2, 3, 2, 1)

    def actual_block(args):
        return ao2mo.transform_eri_block(
            args[1:], df_factors=args[0], max_memory=_tight_budget(np.float64))

    def reference_block(args):
        return _dense_block(args[0], args[1:])

    actual_jvp = jax.jit(lambda p, d: jax.jvp(actual_block, (p,), (d,))[1])(point, direction)
    expected_jvp = jax.jit(lambda p, d: jax.jvp(reference_block, (p,), (d,))[1])(point, direction)
    np.testing.assert_allclose(actual_jvp, expected_jvp, atol=2e-12, rtol=2e-11)

    def pullback(block, args):
        _, transpose = jax.vjp(block, args)
        return transpose(cotangent)[0]

    actual_vjp = jax.jit(lambda p: pullback(actual_block, p))(point)
    expected_vjp = jax.jit(lambda p: pullback(reference_block, p))(point)
    _assert_tree_close(actual_vjp, expected_vjp, atol=2e-11, rtol=2e-10)

    def loss(block, args):
        value = block(args)
        return jnp.sum(jnp.sin(value) + .3 * value**2)

    actual_loss = jax.jit(lambda p: loss(actual_block, p))
    actual_gradient = jax.jit(jax.grad(lambda p: loss(actual_block, p)))
    reference_gradient = jax.jit(jax.grad(lambda p: loss(reference_block, p)))
    actual_grad = actual_gradient(point)
    _assert_tree_close(actual_grad, reference_gradient(point), atol=2e-11, rtol=2e-10)
    actual_hvp = jax.jit(lambda p, d: jax.jvp(actual_gradient, (p,), (d,))[1])(point, direction)
    expected_hvp = jax.jit(lambda p, d: jax.jvp(reference_gradient, (p,), (d,))[1])(point, direction)
    _assert_tree_close(actual_hvp, expected_hvp, atol=2e-10, rtol=2e-9)
    directional_gradient = sum(jnp.vdot(g, d) for g, d in zip(actual_grad, direction))
    for step in (3e-4, 1e-4):
        plus = jax.tree.map(lambda p, d: p + step * d, point, direction)
        minus = jax.tree.map(lambda p, d: p - step * d, point, direction)
        finite_difference = (actual_loss(plus) - actual_loss(minus)) / (2 * step)
        np.testing.assert_allclose(directional_gradient, finite_difference, atol=2e-8, rtol=2e-6)
        finite_difference_hvp = jax.tree.map(
            lambda a, b: (a - b) / (2 * step), actual_gradient(plus), actual_gradient(minus))
        _assert_tree_close(actual_hvp, finite_difference_hvp, atol=2e-8, rtol=2e-6)
