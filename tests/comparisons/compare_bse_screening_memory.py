"""Synthetic BSE A/B action and factor-gradient memory in fresh CPU processes.

Run with PYTHONPATH=src JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 python this_file.py.
This measures screening plus kernel actions, not SCF/GW or an optical spectrum.
Peak RSS includes Python/JAX/compiler memory; XLA byte counts describe the
compiled value-and-gradient executable. No measured memory is a capacity cap.
"""

from pathlib import Path
import json
import platform
import resource
import subprocess
import sys
import time


def measure(naux, nocc, nvir, method, *, gradient_path=None):
    import jax
    import jax.numpy as jnp
    import numpy as np
    import gradscf
    from gradscf import bse
    from gradscf.gw.screened import build_static_screening
    from gradscf.solvers import LinearSolverConfig

    jax.config.update("jax_enable_x64", True)
    rng = np.random.default_rng(701)
    nmo, block = nocc + nvir, 8
    factors = rng.normal(size=(naux, nmo, nmo)) * .03
    factors = jnp.asarray((factors + factors.transpose(0, 2, 1)) * .5)
    energies = jnp.concatenate((-jnp.linspace(2., .2, nocc), jnp.linspace(.2, 3., nvir)))
    probe = jnp.asarray(rng.normal(size=nocc*nvir))
    space = bse.make_bse_space(nmo, nocc)
    kwargs = {} if method == "direct" else dict(config=LinearSolverConfig(
        method="gmres", rtol=1e-11, atol=1e-13, restart=12, maxiter=100))

    def objective(l):
        screening = build_static_screening(energies, l, occupied=space.occupied,
                                           virtual=space.virtual, **kwargs)
        a, b = bse.build_bse_operators(energies, l, space, screening, block_size=block)
        # Physical factor directions remain symmetric throughout AD.
        return jnp.sum((a.matvec(probe) + b.matvec(probe))**2)

    def physical_objective(l):
        return objective((l + l.swapaxes(1, 2)) * .5)

    started = time.perf_counter()
    compiled = jax.jit(jax.value_and_grad(physical_objective)).lower(factors).compile()
    value, gradient = compiled(factors)
    jax.block_until_ready((value, gradient))
    first = time.perf_counter() - started
    started = time.perf_counter()
    for _ in range(3):
        jax.block_until_ready(compiled(factors))
    steady = (time.perf_counter() - started) / 3
    memory = compiled.memory_analysis()
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform != "darwin":
        rss *= 1024
    assert np.isfinite(value) and np.isfinite(gradient).all()
    if gradient_path is not None:
        np.save(gradient_path, np.asarray(gradient))
    return dict(
        backend=jax.default_backend(), platform=platform.platform(),
        python=platform.python_version(), jax=jax.__version__, dtype="float64",
        source=gradscf.__file__, seed=701, naux=naux, nocc=nocc, nvir=nvir,
        block_size=block, screening_solver=method, rtol=1e-11, atol=1e-13,
        compile_and_first_gradient_seconds=first, steady_seconds=steady,
        peak_rss_bytes=rss, xla_argument_bytes=memory.argument_size_in_bytes,
        xla_output_bytes=memory.output_size_in_bytes,
        xla_temporary_bytes=memory.temp_size_in_bytes,
        objective=float(value), gradient_norm=float(jnp.linalg.norm(gradient)),
        gradient_directional=float(jnp.sum(gradient*factors)),
    )


if __name__ == "__main__":
    # Isolate cumulative process RSS and compilation caches for every case.
    worker = (
        "import json,runpy,sys; "
        "measure=runpy.run_path(sys.argv[1])['measure']; "
        "print(json.dumps(measure(int(sys.argv[2]),6,int(sys.argv[3]),sys.argv[4])))"
    )
    for naux, nvir in ((192, 40), (384, 64)):
        for method in ("direct", "gmres"):
            subprocess.run([sys.executable, "-c", worker, str(Path(__file__).resolve()),
                            str(naux), str(nvir), method], check=True)
