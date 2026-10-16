"""A03: matrix-free VJPs of a complete, internally degenerate eigenspace.

Run numerical work on jy_gpu_1, CPU float64. For example:
  PYTHONPATH=src python examples/solvers/degenerate_vjp.py --dimension 256 --method davidson

The main loss is a weighted three-probe projector tensor. The auxiliary trace
has a stationary first derivative and is not evidence of an adjoint solve.
Dense and Davidson forward methods use exactly the same shared implicit rule.
The model never constructs a full U, P, or parameter Jacobian; only the bounded,
independent NumPy oracle and the explicitly requested dense solver do so.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
import platform
import resource
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np

from gradscf.solvers import (
    EigenResponseConfig, EigenSolverConfig, LinearOperator, LinearSolverConfig,
    solve_hermitian,
)


class DegenerateModel:
    """Known spectrum, thin global rotation, and four full-support perturbations.

    U rotates each cluster coordinate into a different orthonormal bath vector.
    In the known spectral frame, E_a=diag(q_a)+.15*(a_a b_a.T+b_a a_a.T).
    Both vectors have cluster and bath support; this is not an inactive bath
    appended to a small matrix. All quantities and parameters are dimensionless.
    """

    def __init__(self, dimension, *, rank=4, seed=1703):
        if dimension < 2*rank+1 or rank < 2:
            raise ValueError("Use rank >= 2 and dimension >= 2*rank+1")
        if not jax.config.x64_enabled:
            raise ValueError("The demonstration requires JAX float64")
        self.dimension, self.rank, self.seed = dimension, rank, seed
        rng = np.random.default_rng(seed)
        bath, _ = np.linalg.qr(rng.normal(size=(dimension-rank, rank)))
        self.bath = jnp.asarray(bath)
        angles = np.linspace(.45, 1.05, rank)
        self.cosine, self.sine = jnp.asarray(np.cos(angles)), jnp.asarray(np.sin(angles))
        left, right = rng.normal(size=(2, dimension, 4))
        for array, cluster_norm, bath_norm in ((left, 1., .2), (right, .2, 1.)):
            array[:rank] *= cluster_norm/np.linalg.norm(array[:rank], axis=0)
            array[rank:] *= bath_norm/np.linalg.norm(array[rank:], axis=0)
        self.left, self.right = jnp.asarray(left), jnp.asarray(right)
        self.diagonal_directions = jnp.asarray(.03*rng.normal(size=(dimension, 4)))
        self.rotated_left, self.rotated_right = self.rotate(self.left), self.rotate(self.right)
        # Keep cluster/bath probe overlap finite as n grows. The random bath
        # component still has support everywhere, not only in the rotation span.
        probes = rng.normal(size=(dimension, 3))
        probes[:rank] /= np.linalg.norm(probes[:rank], axis=0)
        probes[rank:] /= np.linalg.norm(probes[rank:], axis=0)
        probes[rank:] += right[rank:, :3]
        probes[rank:] /= np.linalg.norm(probes[rank:], axis=0)
        self.probe_frame = probes
        self.probes = self.rotate(jnp.asarray(probes))
        self.weights = jnp.array([[.7, -.2, .1], [-.2, -.4, .15], [.1, .15, .3]])

    def rotate(self, values, *, transpose=False):
        """Apply U or U.T in O(nk), with only thin n-by-block arrays."""
        values = jnp.asarray(values)
        vector = values.ndim == 1
        x = values[:, None] if vector else values
        selected, bath = x[:self.rank], x[self.rank:]
        projected = self.bath.T @ bath
        sine = -self.sine if transpose else self.sine
        top = self.cosine[:, None]*selected-sine[:, None]*projected
        bottom = bath+self.bath @ (sine[:, None]*selected+(self.cosine-1)[:, None]*projected)
        out = jnp.concatenate((top, bottom), axis=0)
        return out[:, 0] if vector else out

    def spectrum(self, internal_gap, external_gap):
        selected = 1.+internal_gap*jnp.linspace(-.5, .5, self.rank)
        bath = 1.+.5*internal_gap+external_gap+jnp.linspace(0., 3., self.dimension-self.rank)
        return jnp.concatenate((selected, bath))

    def rotated_diagonal(self, diagonal):
        """Exact diag(U diag(d) U.T), without forming either square matrix."""
        dc, db = diagonal[:self.rank], diagonal[self.rank:]
        small = self.bath.T @ (db[:, None]*self.bath)
        offset = self.cosine-1
        correction = offset[:, None]*small*offset[None, :]+jnp.diag(self.sine**2*dc)
        top = self.cosine**2*dc+self.sine**2*jnp.diag(small)
        bottom = db+2*db*((self.bath**2)@offset)+jnp.sum((self.bath@correction)*self.bath, axis=1)
        return jnp.concatenate((top, bottom))

    def operator(self, parameters, *, internal_gap, external_gap):
        theta = jnp.asarray(parameters)
        if theta.shape != (4,):
            raise ValueError("Supply four real perturbation parameters")
        diagonal = self.spectrum(internal_gap, external_gap)+self.diagonal_directions@theta

        def action(vectors):
            spectral = self.rotate(vectors, transpose=True)
            out = diagonal[:, None]*spectral
            out += .15*(self.left@(theta[:, None]*(self.right.T@spectral))
                        +self.right@(theta[:, None]*(self.left.T@spectral)))
            return self.rotate(out)

        physical_diagonal = self.rotated_diagonal(diagonal)
        physical_diagonal += .30*((self.rotated_left*self.rotated_right)@theta)
        return LinearOperator((self.dimension, self.dimension), diagonal.dtype,
                              lambda v: action(v[:, None])[:, 0],
                              diagonal=physical_diagonal, matmat=action)

    def analytic_reference(self, internal_gap, external_gap):
        """Independent known-spectrum formula; no eigendecomposition or AD."""
        if internal_gap < 0 or external_gap <= 0:
            raise ValueError("The known-cluster reference requires internal_gap >= 0 and external_gap > 0")
        k = self.rank
        lc = 1.+internal_gap*np.linspace(-.5, .5, k)
        lb = 1.+.5*internal_gap+external_gap+np.linspace(0., 3., self.dimension-k)
        bc, bb = self.probe_frame[:k], self.probe_frame[k:]
        left, right = np.asarray(self.left), np.asarray(self.right)
        weights, diagonal = np.asarray(self.weights), np.asarray(self.diagonal_directions)
        derivatives, trace_derivatives = [], []
        for index in range(4):
            coupling = .15*(np.outer(left[k:, index], right[:k, index])
                            +np.outer(right[k:, index], left[:k, index]))
            # Only cluster-to-bath gaps occur, including at exact internal degeneracy.
            tangent = coupling/(lc[None, :]-lb[:, None])
            half_tensor = bb.T @ tangent @ bc
            derivatives.append(np.sum(weights*(half_tensor+half_tensor.T)))
            trace_derivatives.append(diagonal[:k, index].sum()
                                     +.30*np.dot(left[:k, index], right[:k, index]))
        return dict(loss=float(np.sum(weights*(bc.T@bc))), trace=float(lc.sum()),
                    vjp=np.asarray(derivatives), trace_vjp=np.asarray(trace_derivatives))

    def numpy_matrix(self, parameters, internal_gap, external_gap, *, max_dimension=256):
        """Bounded independent dense construction, used only by the FD oracle."""
        if self.dimension > max_dimension:
            raise ValueError("The independent dense oracle exceeds its explicit bound")
        k, n = self.rank, self.dimension
        v, c, s = np.asarray(self.bath), np.asarray(self.cosine), np.asarray(self.sine)
        rotation = np.block([[np.diag(c), -s[:, None]*v.T],
                             [v*s[None, :], np.eye(n-k)+(v*(c-1)[None, :])@v.T]])
        eigenvalues = np.concatenate((1.+internal_gap*np.linspace(-.5, .5, k),
            1.+.5*internal_gap+external_gap+np.linspace(0., 3., n-k)))
        matrix = np.diag(eigenvalues+np.asarray(self.diagonal_directions)@parameters)
        left, right = np.asarray(self.left), np.asarray(self.right)
        for index, coefficient in enumerate(parameters):
            matrix += .15*coefficient*(np.outer(left[:, index], right[:, index])
                                      +np.outer(right[:, index], left[:, index]))
        return rotation@matrix@rotation.T


def make_solve(model, *, method, max_subspace, eig_tol, response_tol, restart,
               maxiter, nroots=None, initial_vectors=None):
    config = EigenSolverConfig(method=method, nroots=model.rank if nroots is None else nroots,
        atol=eig_tol, maxiter=maxiter, max_subspace=max_subspace,
        max_dense=model.dimension, seed=11)
    response = EigenResponseConfig(target="subspace", linear_config=LinearSolverConfig(
        method="gmres", rtol=response_tol, restart=restart, maxiter=maxiter))

    def solve(parameters, internal_gap, external_gap):
        return solve_hermitian(model.operator(parameters, internal_gap=internal_gap,
            external_gap=external_gap), config=config, response=response,
            probes=model.probes, initial_vectors=initial_vectors)
    return solve


def observables(model, result):
    tensor = model.probes.T@result.projection
    return jnp.stack((jnp.sum(model.weights*tensor), result.eigenvalue_sum))


def timed(function, *arguments):
    started = perf_counter()
    result = function(*arguments)
    jax.block_until_ready(result)
    return result, perf_counter()-started


def peak_rss_mib():
    divisor = 1024**2 if platform.system() == "Darwin" else 1024
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/divisor


def dense_differences(model, internal_gap, external_gap, direction, max_dimension):
    if model.dimension > max_dimension:
        return dict(performed=False, reason="dimension exceeds independent dense-oracle bound")
    probes, weights = np.asarray(model.probes), np.asarray(model.weights)

    def oracle(parameters):
        matrix = model.numpy_matrix(parameters, internal_gap, external_gap,
                                    max_dimension=max_dimension)
        values, vectors = np.linalg.eigh(matrix)
        projected = vectors[:, :model.rank].T@probes
        return np.array([np.sum(weights*(projected.T@projected)), values[:model.rank].sum()])

    rows = []
    # Small external gaps require a correspondingly smaller perturbation.
    scale = min(1., external_gap)
    perturbation_bound = sum(abs(coefficient)*(np.max(np.abs(np.asarray(model.diagonal_directions)[:, index]))
        +.30*np.linalg.norm(np.asarray(model.left)[:, index])*np.linalg.norm(np.asarray(model.right)[:, index]))
        for index, coefficient in enumerate(direction))
    for h in (1e-3*scale, 3e-4*scale, 1e-4*scale):
        if h*perturbation_bound > external_gap/4:
            raise AssertionError("Independent FD step does not preserve the exterior-gap lower bound")
        derivative = (oracle(h*direction)-oracle(-h*direction))/(2*h)
        rows.append(dict(step=float(h), derivative=derivative.tolist()))
    return dict(performed=True, perturbation_norm_upper_bound=float(perturbation_bound), steps=rows)


def run_case(model, solve, forward, end_to_end, internal_gap, external_gap, args):
    zero, cotangent = jnp.zeros(4), jnp.array([1., 0.])
    direction = np.array([.3, -.2, .4, .1])
    direction /= np.linalg.norm(direction)
    state, state_seconds = timed(solve, zero, internal_gap, external_gap)
    before = peak_rss_mib()
    # The pullback closes over the retained forward residuals. Warm applications
    # below reuse this exact callable and do not execute a new forward solve.
    (values, pullback), preparation = timed(jax.vjp,
        lambda p: forward(p, internal_gap, external_gap), zero)
    after_prepare = peak_rss_mib()
    product, first = timed(pullback, cotangent)
    gradient = np.asarray(product[0])
    after_pullback = peak_rss_mib()
    valid = bool(state.response_valid)
    row = dict(internal_gap=float(internal_gap), external_gap=float(external_gap),
        converged=bool(state.converged), response_valid=valid, status=int(state.status),
        boundary_gap=float(state.boundary_gap), values=np.asarray(values).tolist(),
        vjp=gradient.tolist(), raw_eigenvalues=np.asarray(state.raw_values).tolist(),
        ritz_residuals=np.asarray(state.raw_residual_norms).tolist(),
        timing_seconds=dict(forward_state=state_seconds, vjp_preparation=preparation,
                            retained_pullback_first=first),
        process_peak_rss_mib=dict(before_prepare=before, after_prepare=after_prepare,
                                  after_pullback=after_pullback),
        adjoint_residual=None, adjoint_iterations=None, operator_calls=None,
        adjoint_diagnostics="Shared true-residual guard is active; actual residual/counts are not exposed.")
    if not valid:
        row["expected_boundary_rejection"] = external_gap <= 1e-8
        row["derivative_nonfinite"] = bool(not np.isfinite(gradient).all())
        row["passed"] = bool(row["expected_boundary_rejection"] and row["derivative_nonfinite"])
        return row
    reference = model.analytic_reference(internal_gap, external_gap)
    row["reference_vjp"] = reference["vjp"].tolist()
    row["reference_vjp_norm"] = float(np.linalg.norm(reference["vjp"]))
    row["vjp_max_error"] = float(np.max(np.abs(gradient-reference["vjp"])))
    row["vjp_relative_error"] = float(np.linalg.norm(gradient-reference["vjp"])
                                       /np.linalg.norm(reference["vjp"]))
    row["timing_seconds"]["retained_pullback_warm"] = [timed(pullback, cotangent)[1]
                                                          for _ in range(args.warm_repeats)]
    _, row["timing_seconds"]["value_and_grad_first"] = timed(end_to_end, zero, internal_gap, external_gap)
    row["timing_seconds"]["value_and_grad_warm"] = [timed(end_to_end, zero, internal_gap, external_gap)[1]
                                                     for _ in range(args.warm_repeats)]
    trace_product, _ = timed(pullback, jnp.array([0., 1.]))
    row["trace_vjp"] = np.asarray(trace_product[0]).tolist()
    tangent, jvp_seconds = timed(jax.jvp, lambda p: forward(p, internal_gap, external_gap),
                                (zero,), (jnp.asarray(direction),))
    derivative = np.asarray(tangent[1])
    row["jvp"] = derivative.tolist()
    row["jvp_vjp_error"] = float(abs(derivative[0]-gradient@direction))
    row["timing_seconds"]["jvp_first"] = jvp_seconds
    row["direction"] = direction.tolist()
    row["finite_differences"] = dense_differences(model, internal_gap, external_gap,
                                                  direction, args.fd_max_dimension)
    checks = row["finite_differences"]
    fd_error = None
    if checks["performed"]:
        for fd in checks["steps"]:
            fd["max_error"] = float(np.max(np.abs(np.asarray(fd["derivative"])-derivative)))
        fd_error = min(fd["max_error"] for fd in checks["steps"])
    # Relative accuracy is used as the external-gap response grows; no gap
    # regularization or averaged internal splitting is introduced.
    row["passed"] = bool(np.isfinite(gradient).all()
        and np.allclose(values, [reference["loss"], reference["trace"]], atol=2e-7, rtol=2e-7)
        and np.allclose(gradient, reference["vjp"], atol=2e-7, rtol=2e-5)
        and np.allclose(trace_product[0], reference["trace_vjp"], atol=2e-8, rtol=2e-7)
        and np.isclose(derivative[0], gradient@direction, atol=2e-8, rtol=2e-7)
        and (fd_error is None or fd_error <= 3e-6*max(1., np.linalg.norm(derivative))))
    return row


def gauge_and_cut_checks(model, settings):
    rng = np.random.default_rng(29)
    rotation, _ = np.linalg.qr(rng.normal(size=(model.rank, model.rank)))
    frame = np.zeros((model.dimension, model.rank+1))
    frame[:model.rank, :model.rank] = rotation
    frame[model.rank, model.rank] = 1.
    # Exact, rotated initial vectors are ONLY a separate gauge check, never the
    # starting vectors used by the timed main Davidson solve.
    initial = model.rotate(jnp.asarray(frame))
    gauge = make_solve(model, method="davidson", initial_vectors=initial, **settings)
    zero = jnp.zeros(4)
    value, derivative = jax.value_and_grad(lambda p: observables(model, gauge(p, 0., .7))[0])(zero)
    reference = model.analytic_reference(0., .7)
    np.testing.assert_allclose(value, reference["loss"], atol=2e-8, rtol=0)
    np.testing.assert_allclose(derivative, reference["vjp"], atol=2e-7, rtol=2e-5)
    cut = make_solve(model, method="davidson", nroots=model.rank-1, **settings)
    cut_state = cut(zero, 0., .7)
    cut_derivative = jax.grad(lambda p: observables(model, cut(p, 0., .7))[0])(zero)
    if bool(cut_state.response_valid) or np.isfinite(cut_derivative).all():
        raise AssertionError("A rank-three cut through the exact rank-four cluster must reject AD")
    return dict(rotated_frame_vjp=np.asarray(derivative).tolist(),
        rotated_frame_vjp_error=float(np.max(np.abs(np.asarray(derivative)-reference["vjp"]))),
        cut_rank=model.rank-1, cut_response_valid=False, cut_derivative_nonfinite=True)


def run(dimension=256, method="davidson", sweep="smoke", *, max_subspace=64, restart=40,
        maxiter=100, eig_tol=1e-10, response_tol=1e-10, warm_repeats=3,
        fd_max_dimension=256, skip_gauge=False, output=None):
    """Run this example with ordinary Python parameters; no command-line parser."""
    if method not in ("dense", "davidson") or sweep not in ("smoke", "internal", "external", "all"):
        raise ValueError("Use a declared method and sweep")
    args = SimpleNamespace(dimension=dimension, method=method, sweep=sweep, max_subspace=max_subspace,
        restart=restart, maxiter=maxiter, eig_tol=eig_tol, response_tol=response_tol,
        warm_repeats=warm_repeats, fd_max_dimension=fd_max_dimension, skip_gauge=skip_gauge,
        output=None if output is None else Path(output))
    jax.config.update("jax_enable_x64", True)
    model = DegenerateModel(args.dimension)
    settings = dict(max_subspace=min(args.max_subspace, args.dimension), eig_tol=args.eig_tol,
                    response_tol=args.response_tol, restart=args.restart, maxiter=args.maxiter)
    solve = jax.jit(make_solve(model, method=args.method, **settings))
    forward = jax.jit(lambda p, delta, gap: observables(model, solve(p, delta, gap)))
    end_to_end = jax.jit(jax.value_and_grad(lambda p, delta, gap: forward(p, delta, gap)[0]))
    cases = [(0., .7)]
    if args.sweep in ("internal", "all"):
        cases = [(gap, .7) for gap in (.1, 1e-3, 1e-6, 0.)]
    if args.sweep in ("external", "all"):
        cases += [(0., gap) for gap in (.1, .01, 1e-3, 1e-10)]
    output = args.output or Path("reproducibility/ad_showcase/A03_degenerate_vjp")/(
        f"n{args.dimension}_{args.method}_{args.sweep}.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[2]
    digest = hashlib.sha256()
    for path in sorted((root/"src").rglob("*.py")):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(path.read_bytes())
    report = dict(complete=False, passed=False, dimension=model.dimension, rank=model.rank,
        nparameters=4, nprobes=3, seed=model.seed, units="dimensionless", method=args.method,
        response="shared implicit subspace / GMRES", settings=settings,
        source_python_sha256=digest.hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        hostname=platform.node(), python=platform.python_version(), jax=jax.__version__,
        process_id=os.getpid(), started_utc=datetime.now(timezone.utc).isoformat(),
        numpy=np.__version__, devices=[str(d) for d in jax.devices()], dtype="float64",
        persistent_compilation_cache_enabled=bool(jax.config.jax_enable_compilation_cache),
        threads={key: os.environ.get(key) for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS")},
        scope="Complete fixed-rank projector VJP; no individual-state derivative or forward-speed claim.",
        model_structure="Full-support structured diagonal-plus-low-rank operator; not a generic dense or molecular cost model.",
        spectral_probe_norms=dict(cluster=float(np.linalg.norm(model.probe_frame[:model.rank])),
                                  bath=float(np.linalg.norm(model.probe_frame[model.rank:]))),
        cluster_bath_perturbation_norms=[float(np.linalg.norm(.15*(
            np.outer(np.asarray(model.left)[model.rank:, a], np.asarray(model.right)[:model.rank, a])
            +np.outer(np.asarray(model.right)[model.rank:, a], np.asarray(model.left)[:model.rank, a]))))
            for a in range(4)],
        cases=[])
    def save():
        output.write_text(json.dumps(report, indent=2, allow_nan=True)+"\n")
    save()
    started = perf_counter()
    for delta, gap in cases:
        row = run_case(model, solve, forward, end_to_end, delta, gap, args)
        report["cases"].append(row)
        save()
        print(json.dumps({key: row.get(key) for key in (
            "internal_gap", "external_gap", "response_valid", "vjp_max_error", "passed")}), flush=True)
        if not row["passed"]:
            raise RuntimeError("A03 VJP case failed; incomplete report retains the observed diagnostics")
    if not args.skip_gauge:
        report["gauge_and_cut_checks"] = gauge_and_cut_checks(model, settings)
    report.update(complete=True, passed=True, elapsed_seconds=perf_counter()-started,
                  process_peak_rss_mib=peak_rss_mib())
    save()
    print(f"A03 complete-cluster VJP gates PASS: {output}", flush=True)


if __name__ == "__main__":
    run()
