"""Default Hermitian searches keep work arrays bounded for a few roots."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gradscf.solvers import EigenSolverConfig, LinearOperator, solve_hermitian
from gradscf.solvers.eigen.davidson import _davidson_lowest_symmetric
from gradscf.solvers.eigen.primal import solve_ritz


def _operator(size):
    diagonal = jnp.linspace(1.0, 8.0, size)
    direction = jnp.linspace(0.2, 0.8, size) / jnp.sqrt(float(size))

    def apply(vector):
        return diagonal * vector + 0.02 * direction * (direction @ vector)

    def apply_block(vectors):
        return (
            diagonal[:, None] * vectors
            + 0.02 * direction[:, None] * (direction @ vectors)[None, :]
        )

    return LinearOperator(
        (size, size),
        diagonal.dtype,
        apply,
        diagonal=diagonal + 0.02 * direction**2,
        matmat=apply_block,
    )


def _jaxpr_shapes(value):
    shapes = set()
    if hasattr(value, "jaxpr"):
        shapes.update(_jaxpr_shapes(value.jaxpr))
    elif hasattr(value, "eqns"):
        for variable in (*value.constvars, *value.invars, *value.outvars):
            shape = getattr(variable.aval, "shape", None)
            if shape is not None:
                shapes.add(shape)
        for equation in value.eqns:
            for variable in (*equation.invars, *equation.outvars):
                shape = getattr(variable.aval, "shape", None)
                if shape is not None:
                    shapes.add(shape)
            shapes.update(_jaxpr_shapes(equation.params))
    elif isinstance(value, dict):
        for item in value.values():
            shapes.update(_jaxpr_shapes(item))
    elif isinstance(value, (tuple, list)):
        for item in value:
            shapes.update(_jaxpr_shapes(item))
    return shapes


@pytest.mark.parametrize(
    "size,nroots,max_subspace,width",
    [(128, 2, None, 40), (128, 12, None, 52), (24, 2, None, 24),
     (128, 2, 19, 19), (128, 2, 200, 128)],
)
def test_low_level_search_basis_has_bounded_default(
    size, nroots, max_subspace, width
):
    diagonal = jax.ShapeDtypeStruct((size,), jnp.float64)

    def solve(diag):
        return _davidson_lowest_symmetric(
            lambda x: diag[:, None] * x,
            size=size,
            diag=diag,
            nroots=nroots,
            max_subspace=max_subspace,
            return_basis=True,
        )

    basis, applied, active = jax.eval_shape(solve, diagonal)
    assert basis.shape == applied.shape == (size, width)
    assert active.shape == (width,)


def test_low_level_default_initial_guess_does_not_form_a_full_identity():
    size = 128

    def solve(diag):
        return _davidson_lowest_symmetric(
            lambda x: diag[:, None] * x,
            size=size,
            diag=diag,
            nroots=2,
            max_subspace=40,
        )

    traced = jax.make_jaxpr(solve)(jnp.linspace(1.0, 8.0, size))
    assert (size, size) not in _jaxpr_shapes(traced)


@pytest.mark.parametrize("nroots,width", [(2, 40), (11, 52)])
def test_public_ritz_default_includes_boundary_root_without_full_basis(nroots, width):
    size = 128
    config = EigenSolverConfig(nroots=nroots)
    traced = jax.make_jaxpr(lambda: solve_ritz(_operator(size), config))()
    shapes = _jaxpr_shapes(traced)
    assert (size, width) in shapes
    assert (size, size) not in shapes


def test_public_ritz_keeps_explicit_subspace_validation():
    with pytest.raises(ValueError, match="boundary root"):
        solve_ritz(_operator(128), EigenSolverConfig(nroots=2, max_subspace=4))


@pytest.mark.parametrize("max_subspace", [None, 48])
def test_bounded_default_preserves_roots_residuals_and_eigenvalue_response(max_subspace):
    operator = _operator(128)
    config = EigenSolverConfig(
        nroots=2, max_subspace=max_subspace, atol=1e-10, maxiter=100
    )
    reference = jnp.linalg.eigvalsh(operator.apply(jnp.eye(128)))[:2]

    def solve(scale):
        scaled = LinearOperator(
            operator.shape,
            operator.dtype,
            lambda v: scale * operator.apply(v),
            diagonal=scale * operator.diagonal,
            matmat=lambda v: scale * operator.apply(v),
        )
        return solve_hermitian(scaled, config=config)

    result = jax.jit(solve)(1.0)
    assert np.all(result.converged)
    assert result.raw_values.shape == (3,)
    np.testing.assert_allclose(result.values, reference, atol=2e-10, rtol=0.0)
    np.testing.assert_allclose(
        jax.jit(jax.grad(lambda t: jnp.sum(solve(t).values)))(1.0),
        jnp.sum(reference),
        atol=2e-10,
        rtol=0.0,
    )
