"""G06: one relaxed HVP versus assembling every Cartesian Hessian column.

Each molecule runs in a fresh process to separate JIT caches and peak memory.
The geometries are fixed derivative benchmarks, not optimized minima. NativeRHF
uses implicit orbital response and native second integral derivatives; this
script supplies directions and measures its existing API without a new solver.
"""
import hashlib
import json
import multiprocessing
import platform
import resource
from pathlib import Path
from time import perf_counter

import jax
import numpy as np

jax.config.update("jax_enable_x64", True)

from gradscf_tools.molecular_ep import NativeRHF


SYSTEMS = {
    "H2": "H 0 0 0; H 0 0 1.4",
    "water": "O 0 0 0; H 0 1.43 1.11; H 0 -1.43 1.11",
    "ethylene": "C 0 0 -1.26; C 0 0 1.26; H 0 1.75 -2.32; H 0 -1.75 -2.32; "
                "H 0 1.75 2.32; H 0 -1.75 2.32",
}
OUTPUT = Path("reproducibility/ad_showcase/G06_hessian_vectors")


def peak_memory_mib():
    """Process high-water RSS, including imports/JAX/native allocations."""
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value / (1024**2 if platform.system() == "Darwin" else 1024)


def run(name):
    engine = NativeRHF(SYSTEMS[name], basis="6-31g*")
    coordinates = engine.coordinates
    start = perf_counter()
    state, _ = engine.evaluate(coordinates)
    scf_seconds = perf_counter()-start
    response = engine.response(coordinates, state)
    direction = np.random.default_rng(31).normal(size=coordinates.shape)
    direction /= np.linalg.norm(direction)

    start = perf_counter()
    product, _ = response(direction)  # host arrays synchronize device work
    cold_seconds = perf_counter()-start
    warm_seconds = []
    for _ in range(5):
        start = perf_counter()
        repeated, _ = response(direction)
        warm_seconds.append(perf_counter()-start)
        np.testing.assert_allclose(repeated, product, atol=1e-12, rtol=0)
    hvp_peak = peak_memory_mib()

    # Only the validation phase assembles the 3N by 3N matrix.
    start = perf_counter()
    hessian = np.column_stack([response(v)[0].ravel()
        for v in np.eye(coordinates.size).reshape(-1, *coordinates.shape)])
    dense_seconds = perf_counter()-start
    dense_peak = peak_memory_mib()
    np.testing.assert_allclose(hessian, hessian.T, atol=2e-6, rtol=0)
    np.testing.assert_allclose(product.ravel(), hessian@direction.ravel(), atol=1e-8, rtol=0)
    checks = []
    for step in (1e-3, 3e-4, 1e-4):
        _, plus = engine.evaluate(coordinates+step*direction, init_density=state.density_matrix)
        _, minus = engine.evaluate(coordinates-step*direction, init_density=state.density_matrix)
        error = float(np.max(np.abs((plus-minus)/(2*step)-product)))
        checks.append(dict(step_bohr=step, max_hvp_error=error))
    assert checks[-1]["max_hvp_error"] < 2e-6
    result = dict(name=name, atom_bohr=SYSTEMS[name], basis="6-31g*", cart=True,
                  nao=engine.topology.nao, ncart=coordinates.size,
                  energy_hartree=float(state.total_energy), direction=direction.tolist(),
                  hvp=product.tolist(), hessian=hessian.tolist(), finite_difference_checks=checks,
                  cold_scf_gradient_seconds=scf_seconds, cold_hvp_seconds=cold_seconds,
                  warm_hvp_seconds=warm_seconds, warm_dense_hessian_seconds=dense_seconds,
                  process_peak_after_hvp_mib=hvp_peak, process_peak_after_dense_mib=dense_peak,
                  hostname=platform.node(), python_version=platform.python_version(),
                  jax_version=jax.__version__, dtype="float64", devices=[str(d) for d in jax.devices()],
                  response_source_sha256=hashlib.sha256(Path("src/gradscf_tools/molecular_ep.py").read_bytes()).hexdigest())
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / f"{name}.json").write_text(json.dumps(result, indent=2)+"\n")
    print(f"{name}: nAO={engine.topology.nao}, cold HVP={cold_seconds:.3f}s, "
          f"warm={np.median(warm_seconds):.3f}s, all columns={dense_seconds:.3f}s, "
          f"FD error={checks[-1]['max_hvp_error']:.3e}", flush=True)


if __name__ == "__main__":
    for name in SYSTEMS:
        job = multiprocessing.get_context("spawn").Process(target=run, args=(name,))
        job.start()
        job.join()
        if job.exitcode:
            raise RuntimeError(f"{name} HVP benchmark failed with exit code {job.exitcode}.")

# jy_gpu_1, CPU float64, 2026-10-15; first / median warm / all columns (seconds):
# H2:       5.930 / 0.009 /   0.052; FD error 3.215e-10 Eh/Bohr^2
# water:    8.561 / 0.995 /   7.901; FD error 1.954e-7  Eh/Bohr^2
# ethylene: 20.086 / 10.837 / 198.649; FD error 1.108e-7 Eh/Bohr^2
# First HVP includes compilation; all columns use the already warmed API.
