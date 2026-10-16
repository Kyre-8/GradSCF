"""Shared solvers: matrix-free eigenvector response and a nonsymmetric solve."""
import json
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from gradscf.solvers import LinearOperator, EigenSolverConfig, solve_hermitian, solve_linear


def main():
    cfg = EigenSolverConfig(nroots=1, gradient_mode="implicit_eigenvector", atol=1e-10)
    def observable(t):
        a = jnp.array([[1.0, t], [t, 2.0]])
        op = LinearOperator(a.shape, a.dtype, lambda v: a @ v, diagonal=jnp.diag(a))
        result = solve_hermitian(op, config=cfg)
        return result.values[0] + .2*result.vectors[0, 0]**2
    derivative = jax.jit(jax.grad(observable))(.1)
    finite_difference = (observable(.10001)-observable(.09999))/2e-5
    result = solve_linear(jnp.array([[2., .3], [-.1, 1.]]), jnp.array([1., -.5]))
    print(json.dumps({
        "devices": [str(d) for d in jax.devices()], "float64": True,
        "observable_derivative": float(derivative),
        "finite_difference": float(finite_difference),
        "linear_solution": result.solution.tolist(),
        "linear_residual": float(result.residual_norm),
        "linear_converged": bool(result.converged),
    }, indent=2))


if __name__ == "__main__":
    main()
