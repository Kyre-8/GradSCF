"""Matrix-free operator convention: vectors (n,), block columns (n, nvec)."""
from dataclasses import dataclass
from typing import Callable, Any

import jax
import jax.numpy as jnp
from jaxtyping import Array


@dataclass(frozen=True)
class LinearOperator:
    """Construct inside transformed functions, or close over static callables.

    Numerical parameters captured by the callables remain differentiable. This
    container itself is not a dynamic JIT argument. T is an algebraic transpose,
    not a conjugate transpose. The initial public solver API accepts real data.
    ``exact_diagonal`` declares the complete operator family to be diagonal;
    an approximate preconditioner or a numerically diagonal point is insufficient.
    """
    shape: tuple[int, int]
    dtype: Any
    matvec: Callable[[Array], Array]
    diagonal: Array | None = None
    matmat: Callable[[Array], Array] | None = None
    transpose_matvec: Callable[[Array], Array] | None = None
    exact_diagonal: bool = False

    def __post_init__(self):
        if len(self.shape) != 2 or min(self.shape) < 0:
            raise ValueError("Invalid operator shape")
        if self.diagonal is not None and self.diagonal.shape != (min(self.shape),):
            raise ValueError("Operator diagonal shape mismatch")
        if not isinstance(self.exact_diagonal, bool):
            raise TypeError("exact_diagonal must be a static bool")
        if self.exact_diagonal and (self.shape[0] != self.shape[1] or self.diagonal is None):
            raise ValueError("An exact diagonal declaration requires a square operator and diagonal")

    def apply(self, values: Array) -> Array:
        values = jnp.asarray(values)
        if values.ndim not in (1, 2) or values.shape[0] != self.shape[1]:
            raise ValueError("Operator input shape mismatch")
        if values.ndim == 1:
            result = self.matvec(values)
        elif self.matmat is not None:
            result = self.matmat(values)
        else:
            result = jax.vmap(self.matvec, in_axes=1, out_axes=1)(values)
        result = jnp.asarray(result)
        if result.shape != (self.shape[0],) + values.shape[1:]:
            raise ValueError("Operator output shape mismatch")
        return result

    @property
    def T(self):
        transpose = self.transpose_matvec
        if transpose is None:
            def transpose(v):
                return jax.linear_transpose(self.matvec, jnp.zeros(self.shape[1], self.dtype))(v)[0]
        return LinearOperator(self.shape[::-1], self.dtype, transpose,
                              diagonal=self.diagonal, transpose_matvec=self.matvec,
                              exact_diagonal=self.exact_diagonal)


def as_operator(matrix_or_operator) -> LinearOperator:
    if isinstance(matrix_or_operator, LinearOperator):
        return matrix_or_operator
    matrix = jnp.asarray(matrix_or_operator)
    if matrix.ndim != 2:
        raise ValueError("Expected a matrix or LinearOperator")
    return LinearOperator(matrix.shape, matrix.dtype, lambda v: matrix @ v,
                          diagonal=jnp.diag(matrix), matmat=lambda v: matrix @ v,
                          transpose_matvec=lambda v: matrix.T @ v)


def validate_real_square(operator):
    if operator.shape[0] != operator.shape[1]:
        raise ValueError("Solver requires a square operator")
    dtype = jnp.dtype(operator.dtype)
    if not jnp.issubdtype(dtype, jnp.floating):
        raise NotImplementedError("Public solvers currently require real floating-point data")
