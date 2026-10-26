"""An internally degenerate spectral projector and its first-order response.

Run: PYTHONPATH=src JAX_PLATFORMS=cpu python examples/solvers/degenerate_subspace.py
The toy matrix and coupling parameter are dimensionless; no chemical units.
"""

from dataclasses import replace
import hashlib
import json
from pathlib import Path
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

    def solve(t, cfg=config, probes=probe):
        op = LinearOperator(
            (4, 4),
            diagonal.dtype,
            lambda x: diagonal * x + t * (perturbation @ x),
            diagonal=diagonal + t * jnp.diag(perturbation),
        )
        return solve_hermitian(
            op,
            probes=probes,
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
    report = dict(
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
            )

    # A03: cross the two internal roots while retaining the whole isolated pair.
    # Sorted-root slopes below use independent NumPy eigenvectors only away
    # from the crossing. No derivative is assigned to either sorted root at 0.
    projector = jax.jit(lambda t: solve(t, probes=jnp.eye(4)).projection)
    trace = lambda t: solve(t).eigenvalue_sum
    responses = jax.jit(lambda t: jnp.array([jax.grad(loss)(t), jax.grad(trace)(t)]))
    projector_response = jax.jit(lambda t: jax.jvp(projector, (t,), (1.,))[1])
    matrix0, b = np.diag(diagonal), np.asarray(perturbation)
    scan = []
    for point in np.linspace(-.15, .15, 61):
        state = solve(float(point))
        assert state.converged and state.response_valid
        values, vectors = np.linalg.eigh(matrix0+point*b)
        slopes = np.einsum("pi,pq,qi->i", vectors, b, vectors)
        pmat = np.asarray(projector(float(point)))
        dp = np.asarray(projector_response(float(point)))
        np.testing.assert_allclose(pmat@pmat, pmat, atol=2e-11)
        np.testing.assert_allclose(pmat@dp@pmat, 0., atol=2e-10)
        np.testing.assert_allclose(dp@pmat+pmat@dp, dp, atol=2e-10)
        ad = np.asarray(responses(float(point)))
        differences = []
        for h in (1e-3, 1e-4, 1e-5):
            ep = np.linalg.eigvalsh(matrix0+(point+h)*b)
            em = np.linalg.eigvalsh(matrix0+(point-h)*b)
            fd = np.array([(oracle(point+h)-oracle(point-h))/(2*h),
                           (ep[:2].sum()-em[:2].sum())/(2*h)])
            differences.append(dict(step=h, derivative=fd.tolist(),
                                    max_error=float(np.max(np.abs(fd-ad)))))
        np.testing.assert_allclose(differences[-1]["derivative"], ad, atol=2e-8, rtol=0)
        scan.append(dict(parameter=float(point), eigenvalues=values.tolist(),
            sorted_root_slopes=None if point == 0 else slopes[:2].tolist(),
            probe_projection=float(oracle(point)), subspace_energy=float(values[:2].sum()),
            ad_derivatives=ad.tolist(), finite_differences=differences,
            boundary_gap=float(state.boundary_gap), internal_gap=float(values[1]-values[0]),
            residual_max=float(np.max(np.asarray(state.residual_norms)))))
    cut_derivative = jax.grad(lambda t: probe@solve(t, replace(config, nroots=1)).projection)(0.)
    assert not np.isfinite(cut_derivative)
    slopes_at_zero = np.linalg.eigvalsh(b[:2, :2])
    one_sided = []
    for h in (1e-3, 1e-4, 1e-5):
        lower_plus = np.linalg.eigvalsh(matrix0+h*b)[0]
        lower_minus = np.linalg.eigvalsh(matrix0-h*b)[0]
        one_sided.append(dict(step=h, left=(1.-lower_minus)/h,
            right=(lower_plus-1.)/h, central=(lower_plus-lower_minus)/(2*h)))
    np.testing.assert_allclose([one_sided[-1]["left"], one_sided[-1]["right"]],
                               slopes_at_zero[::-1], atol=2e-6, rtol=0)
    root = Path(__file__).resolve().parents[2]
    digest = hashlib.sha256()
    for path in sorted((root/"src").rglob("*.py")):
        digest.update(path.relative_to(root).as_posix().encode()); digest.update(path.read_bytes())
    report.update(complete=True, hostname=platform.node(), python=platform.python_version(),
        source_python_sha256=digest.hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        matrix0=matrix0.tolist(), perturbation=b.tolist(), probe=np.asarray(probe).tolist(),
        scan=scan, one_sided_lower_root=one_sided,
        crossing_directional_branch_slopes=slopes_at_zero.tolist(),
        cut_cluster_response_valid=bool(cut.response_valid), cut_derivative_nonfinite=True,
        elapsed_seconds=time.perf_counter()-start,
        scope="Dimensionless first-order isolated rank-2 projector/trace; sorted roots lack a unique derivative at the internal crossing.")
    output = root/"reproducibility/ad_showcase/A03_degenerate_subspace"
    output.mkdir(parents=True, exist_ok=True)
    (output/"results.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({key: value for key, value in report.items() if key != "scan"}, indent=2))
    print("A03 crossing, projector identities, JVP/VJP, multistep dense FD and cut-cluster gates PASS")


if __name__ == "__main__":
    main()

# jy_gpu_1 CPU/x64: 61-point A03 gates PASS in 176.81 s.
# At t=0: projector-observable JVP/VJP=-.115, trace derivative=-.1.
# Smallest-step scan FD error <=1.50e-10; rank-1 cut-cluster response invalid.
