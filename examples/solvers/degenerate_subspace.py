"""An internally degenerate spectral projector and its first-order response.

Run: PYTHONPATH=src JAX_PLATFORMS=cpu python examples/solvers/degenerate_subspace.py
The toy matrix and coupling parameter are dimensionless; no chemical units.
"""

from dataclasses import replace
import json
import platform
import time

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from gradscf.solvers import (
    EigenSolverConfig,
    EigenResponseConfig,
    LinearSolverConfig,
    LinearOperator,
    solve_hermitian,
)


def main():
    start = time.perf_counter()
    diagonal = jnp.array([1.0, 1.0, 3.0, 5.0])
    perturbation = jnp.array(
        [
            [0.2, 0.6, 0.4, 0.1],
            [0.6, -0.3, 0.2, 0.3],
            [0.4, 0.2, 0.1, -0.2],
            [0.1, 0.3, -0.2, 0.4],
        ]
    )
    probe = jnp.array([0.3, 0.4, 0.5, 0.2])
    config = EigenSolverConfig(nroots=2, atol=1e-11)

    def solve(t, cfg=config):
        op = LinearOperator(
            (4, 4),
            diagonal.dtype,
            lambda x: diagonal * x + t * (perturbation @ x),
            diagonal=diagonal + t * jnp.diag(perturbation),
        )
        return solve_hermitian(
            op,
            probes=probe,
            config=cfg,
            response=EigenResponseConfig(
                target="subspace", linear_config=LinearSolverConfig(rtol=1e-11)
            ),
        )

    def loss(t):
        return probe @ solve(t).projection

    result = solve(0.0)
    assert result.converged and result.response_valid
    value, backward = jax.jit(jax.value_and_grad(loss))(0.0)
    forward = jax.jvp(loss, (0.0,), (1.0,))[1]
    step = 1e-4

    # Independent dense oracle, using no eigendecomposition AD at degeneracy.
    def oracle(t):
        _, x = np.linalg.eigh(np.diag(diagonal) + t * np.asarray(perturbation))
        return np.sum((x[:, :2].T @ np.asarray(probe)) ** 2)

    finite_difference = (oracle(step) - oracle(-step)) / (2 * step)
    np.testing.assert_allclose(backward, finite_difference, atol=2e-9, rtol=0)
    np.testing.assert_allclose(forward, backward, atol=1e-11, rtol=0)
    cut = solve(0.0, replace(config, nroots=1))
    assert cut.converged and not cut.response_valid
    print(
        json.dumps(
            dict(
                machine=platform.machine(),
                jax=jax.__version__,
                devices=[str(d) for d in jax.devices()],
                dtype=str(diagonal.dtype),
                dimension=4,
                rank=2,
                eigenvalue_sum=float(result.eigenvalue_sum),
                boundary_gap=float(result.boundary_gap),
                residual_norms=np.asarray(result.residual_norms).tolist(),
                loss=float(value),
                jvp=float(forward),
                vjp=float(backward),
                finite_difference=float(finite_difference),
                step=step,
                absolute_error=float(abs(backward - finite_difference)),
                cut_cluster_status=int(cut.status),
                elapsed_seconds=time.perf_counter() - start,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
