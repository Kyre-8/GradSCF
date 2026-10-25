"""Measure FCI subspace adaptation at fixed Hamiltonian and seed (CPU/x64).

Run high/low budgets in separate processes. All Hamiltonian and solver math is
provided by src; the optional second execution counts projected eigensolves to
record iterations without including callbacks in the warm timing.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

from gradscf import fci
from gradscf.fci import hamiltonian, solver
from gradscf.solvers import EigenSolverConfig
from gradscf.solvers.eigen import primal


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-memory", type=float, required=True)
    parser.add_argument("--max-space", type=int, default=32)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    jax.config.update("jax_enable_x64", True)
    rng = np.random.default_rng(2902)
    space = fci.make_fci_space(6, (2, 2))
    h = np.diag(np.linspace(-1.5, .9, 6))
    off = rng.normal(scale=.03, size=(6, 6))
    h += (off+off.T)*.5
    factors = rng.normal(scale=.05, size=(7, 6, 6))
    factors = (factors+factors.transpose(0, 2, 1))*.5
    g = np.einsum("Lpq,Lrs->pqrs", factors, factors)
    input_hash = hashlib.sha256(h.tobytes()+g.tobytes()).hexdigest()
    h, g = jnp.asarray(h), jnp.asarray(g)
    config = EigenSolverConfig(max_subspace=args.max_space, maxiter=150, atol=1e-10, seed=2902)
    planned, reserved, batch = solver._plan_fci_memory(
        space, h.dtype, config, args.max_memory, arrays=(h, g, jnp.asarray(0.)))

    def solve(hh, gg):
        return solver.solve_fci(hh, gg, space, config=config, max_memory=args.max_memory)

    start = time.perf_counter()
    compiled = jax.jit(solve).lower(h, g).compile()
    compilation_seconds = time.perf_counter()-start
    start = time.perf_counter()
    result = jax.block_until_ready(compiled(h, g))
    first_seconds = time.perf_counter()-start
    elapsed = []
    for _ in range(7):
        start = time.perf_counter()
        jax.block_until_ready(compiled(h, g))
        elapsed.append(time.perf_counter()-start)
    peak_after_warm = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    iterations = []
    eigh = primal.jnp.linalg.eigh

    def counted_eigh(matrix, *pos, **kw):
        jax.debug.callback(lambda: iterations.append(1), ordered=True)
        return eigh(matrix, *pos, **kw)

    try:
        primal.jnp.linalg.eigh = counted_eigh
        jax.block_until_ready(jax.jit(lambda hh, gg: solve(hh, gg))(h, g))
    finally:
        primal.jnp.linalg.eigh = eigh
    memory = compiled.memory_analysis()
    source_root = Path(solver.__file__).resolve().parents[1]
    sources = ["fci/solver.py", "fci/hamiltonian.py", "solvers/memory.py",
               "solvers/eigen/primal.py", "solvers/eigen/davidson.py"]
    cpu_model = next((line.split(":", 1)[1].strip()
                      for line in Path("/proc/cpuinfo").read_text().splitlines()
                      if line.startswith("model name")), platform.processor())
    tile_plans = {}
    widths = {2, planned.initial_guess_count, planned.collapse_subspace}
    actual_widths = {min(width, batch) for width in widths}
    actual_widths.update(width % batch for width in widths if width % batch)
    for width in sorted(actual_widths):
        effective = jax.ShapeDtypeStruct(g.shape, g.dtype)
        diagonal = jax.ShapeDtypeStruct((space.size,), g.dtype)
        coefficients = jax.ShapeDtypeStruct(space.shape+(width,), g.dtype)
        original_coefficients = jax.ShapeDtypeStruct(space.shape+(width,), g.dtype)
        a, b, r = hamiltonian._action_tiles(space, g.dtype, width,
            hamiltonian.DEFAULT_WORKSPACE, args.max_memory,
            arrays=(effective, h, g, diagonal, coefficients, original_coefficients),
            extra_bytes=reserved+jnp.dtype(g.dtype).itemsize)
        tile_plans[width] = dict(alpha=a, beta=b, rhs=r,
            tiles=((space.shape[0]+a-1)//a)*((space.shape[1]+b-1)//b)*((width+r-1)//r))
    report = dict(complete=True, command=sys.argv, python=sys.version,
        hardware=cpu_model, hostname=platform.node(), affinity=sorted(os.sched_getaffinity(0)),
        backend=jax.default_backend(), jax=jax.__version__, numpy=np.__version__,
        environment={key: os.environ.get(key) for key in (
            "JAX_PLATFORMS", "CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
        dtype=str(h.dtype), seed=2902, space_shape=space.shape, size=space.size,
        max_memory_mb=args.max_memory, input_sha256=input_hash,
        source_sha256={p: hashlib.sha256((source_root/p).read_bytes()).hexdigest() for p in sources},
        requested_config=asdict(config), planned_config=asdict(planned),
        solver_reserved_bytes=reserved, operator_column_batch=batch,
        action_tile_plans=tile_plans,
        compile_seconds=compilation_seconds, first_seconds=first_seconds,
        warm_seconds=elapsed, warm_median_seconds=float(np.median(elapsed)),
        iterations=len(iterations), peak_after_warm_kib=peak_after_warm,
        compiled_memory={k: getattr(memory, k) for k in (
            "argument_size_in_bytes", "output_size_in_bytes", "alias_size_in_bytes", "temp_size_in_bytes")},
        energies=np.asarray(result.total_energies).tolist(),
        residuals=np.asarray(result.residual_norms).tolist(),
        converged=np.asarray(result.converged).tolist())
    args.output.write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2))
    if not np.all(result.converged):
        raise SystemExit("FCI benchmark did not converge")


if __name__ == "__main__":
    main()
