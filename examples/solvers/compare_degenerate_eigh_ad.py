"""Compare ordinary eigh AD and shared implicit AD for the same projector loss.

Both differentiate L=<W,B.T P B>, with P selecting the complete lowest rank-4
cluster and a fixed exterior gap. The ordinary path uses jnp.linalg.eigh with
no stopped gradients or custom derivative. Neither an individual eigenvector
nor a sorted eigenvalue is used as its loss. Success is not predetermined.
Analytic references apply at theta=0 only. Run numerical work on remote CPU.
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

from degenerate_vjp import DegenerateModel, make_solve, observables


def derivative_record(loss, gradient, reference):
    gradient = np.asarray(gradient)
    finite = bool(np.isfinite(gradient).all())
    difference = np.abs(gradient-reference["vjp"])
    return dict(loss=float(loss), loss_absolute_error=float(abs(loss-reference["loss"])),
        vjp=gradient.tolist(), finite=finite, absolute_errors=difference.tolist(),
        max_absolute_error=float(np.max(difference)) if finite else None,
        relative_error=float(np.linalg.norm(difference)/np.linalg.norm(reference["vjp"])) if finite else None,
        within_tolerance=bool(finite and np.allclose(gradient, reference["vjp"], atol=2e-7, rtol=2e-5)
                              and np.isclose(loss, reference["loss"], atol=2e-7, rtol=2e-7)))


def run_dimension(dimension, report, save):
    model = DegenerateModel(dimension, rank=4, seed=1703)
    settings = dict(max_subspace=min(64, dimension), eig_tol=1e-10,
                    response_tol=1e-10, restart=40, maxiter=100)
    solve = jax.jit(make_solve(model, method="dense", **settings))
    shared_ad = jax.jit(jax.value_and_grad(
        lambda theta, delta, gap: observables(model, solve(theta, delta, gap))[0]))

    def ordinary_loss(theta, delta, gap):
        op = model.operator(theta, internal_gap=delta, external_gap=gap)
        matrix = op.matmat(jnp.eye(dimension, dtype=jnp.float64))
        eigenvalues, eigenvectors = jnp.linalg.eigh(matrix)
        selected = eigenvectors[:, :model.rank]
        overlap = selected.T@model.probes
        loss = jnp.sum(model.weights*(overlap.T@overlap))
        return loss, eigenvalues[:model.rank+1]

    ordinary_ad = jax.jit(jax.value_and_grad(ordinary_loss, has_aux=True))
    zero, exterior_gap = jnp.zeros(4), .7
    for delta in (.1, 1e-3, 1e-6, 1e-9, 0.):
        reference = model.analytic_reference(delta, exterior_gap)
        row = dict(dimension=dimension, rank=model.rank, internal_width=delta, exterior_gap=exterior_gap,
            theta=np.zeros(4).tolist(), reference_loss=reference["loss"], reference_vjp=reference["vjp"].tolist(),
            shared_settings=settings, ordinary=None, shared=None, comparison_credible=False)
        report["cases"].append(row)
        save()
        try:
            (ordinary_value, raw_values), ordinary_gradient = ordinary_ad(zero, delta, exterior_gap)
            jax.block_until_ready((ordinary_value, raw_values, ordinary_gradient))
            ordinary = derivative_record(float(ordinary_value), ordinary_gradient, reference)
            raw_values = np.asarray(raw_values)
            ordinary.update(raw_values=raw_values.tolist(),
                raw_internal_gaps=np.diff(raw_values[:model.rank]).tolist(),
                raw_min_internal_gap=float(np.min(np.diff(raw_values[:model.rank]))),
                raw_internal_width=float(np.ptp(raw_values[:model.rank])),
                raw_exterior_gap=float(raw_values[model.rank]-raw_values[model.rank-1]))
            row["ordinary"] = ordinary
        except Exception as error:
            row["ordinary"] = dict(finite=False, within_tolerance=False, vjp=None,
                                   error=f"{type(error).__name__}: {error}")
        save()
        try:
            state = solve(zero, delta, exterior_gap)
            shared_value, shared_gradient = shared_ad(zero, delta, exterior_gap)
            jax.block_until_ready((state, shared_value, shared_gradient))
            shared = derivative_record(float(shared_value), shared_gradient, reference)
            shared.update(converged=bool(state.converged), response_valid=bool(state.response_valid),
                status=int(state.status), raw_values=np.asarray(state.raw_values).tolist(),
                ritz_residuals=np.asarray(state.raw_residual_norms).tolist(),
                boundary_gap=float(state.boundary_gap), adjoint_residual=None)
            row["shared"] = shared
            row["comparison_credible"] = bool(shared["converged"] and shared["response_valid"]
                                               and shared["within_tolerance"])
        except Exception as error:
            row["shared"] = dict(finite=False, within_tolerance=False, vjp=None,
                                 error=f"{type(error).__name__}: {error}")
        save()
        print(json.dumps(dict(dimension=dimension, internal_width=delta,
            ordinary_finite=row["ordinary"]["finite"],
            ordinary_max_error=row["ordinary"].get("max_absolute_error"),
            shared_max_error=row["shared"].get("max_absolute_error"),
            comparison_credible=row["comparison_credible"])), flush=True)


def run(dimensions=(32, 256), *, output=Path("reproducibility/ad_showcase/A03_degenerate_vjp/plain_eigh_comparison.json")):
    """Run this example with ordinary Python parameters; no command-line parser."""
    args = SimpleNamespace(dimensions=list(dimensions), output=Path(output))
    if any(dimension not in (32, 256) for dimension in args.dimensions):
        raise ValueError("This bounded comparison supports dimensions 32 and 256")
    jax.config.update("jax_enable_x64", True)
    root, digest = Path(__file__).resolve().parents[2], hashlib.sha256()
    for path in sorted((root/"src").rglob("*.py")):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(path.read_bytes())
    report = dict(complete=False, comparison_credible=False, passed=False, cases=[], dimensions=args.dimensions,
        rank=4, nparameters=4, seed=1703, units="dimensionless", dtype="float64",
        source_python_sha256=digest.hexdigest(), script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        model_script_sha256=hashlib.sha256(Path(__file__).with_name("degenerate_vjp.py").read_bytes()).hexdigest(),
        hostname=platform.node(), process_id=os.getpid(), started_utc=datetime.now(timezone.utc).isoformat(),
        python=platform.python_version(), jax=jax.__version__, numpy=np.__version__,
        devices=[str(device) for device in jax.devices()],
        persistent_compilation_cache_enabled=bool(jax.config.jax_enable_compilation_cache),
        threads={key: os.environ.get(key) for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
        loss="sum(W * (B.T @ P_lowest_rank4 @ B)); identical complete-cluster loss for both methods",
        ordinary_method="jnp.linalg.eigh with its default AD and symmetrize_input=True; no stop_gradient",
        shared_method="dense forward plus existing shared implicit subspace GMRES response",
        acceptance="Shared finite valid/converged VJP and loss must match the analytic reference; ordinary failures are observations.",
        derivative_atol=2e-7, derivative_rtol=2e-5,
        scope="A fixed exterior gap keeps this projector loss mathematically smooth; no individual-state loss or speed comparison.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    def save():
        divisor = 1024**2 if platform.system() == "Darwin" else 1024
        report.update(elapsed_seconds=perf_counter()-started,
                      process_peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/divisor)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=True)+"\n")
    save()
    try:
        for dimension in args.dimensions:
            run_dimension(dimension, report, save)
        credible = all(row["comparison_credible"] for row in report["cases"])
        report.update(complete=True, comparison_credible=credible, passed=credible,
            ordinary_within_tolerance_count=sum(row["ordinary"]["within_tolerance"] for row in report["cases"]),
            ordinary_finite_count=sum(row["ordinary"]["finite"] for row in report["cases"]))
        save()
    except Exception as error:
        report.update(error=f"{type(error).__name__}: {error}")
        save()
        raise
    if not report["comparison_credible"]:
        raise RuntimeError("Comparison finished, but at least one shared reference gate failed")
    print(f"Comparison complete and credible: {args.output}", flush=True)


if __name__ == "__main__":
    run()
