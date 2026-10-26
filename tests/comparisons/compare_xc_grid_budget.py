"""Compare XC grid workspaces at one fixed density, without running SCF.

Both cases use the same DABNA-2 coordinates, level-3 quadrature, Cartesian
def2-SVP basis and supplied AO density. Only the AO block size changes.
Run on an authorized compute node with externally bounded CPU/memory usage.
"""
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import sys
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np

from gradscf import gto
from gradscf.scf.inputs import build_rks_integral_inputs
from gradscf.scf.rks import RKSConfig, _restricted_xc_energy_and_matrix

jax.config.update("jax_enable_x64", True)


def run_benchmark(geometry, density_path, output_dir, max_memory=60000., repeats=2,
                  block_sizes=(256, None)):
    """Measure source XC energy/potential integration with 256 versus auto blocks.

    ``density_matrix`` in the NPZ checkpoint is a fixed trial input; neither HF
    nor PBE SCF convergence is established here. Timed repeats exclude explicit
    compilation and one warmup. Process peak RSS is cumulative across both
    cases, so it cannot isolate the memory cost of either case.
    """
    if repeats < 1:
        raise ValueError("repeats must be positive.")
    geometry, density_path, output_dir = map(Path, (geometry, density_path, output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    source_root = Path(__file__).resolve().parents[2] / "src" / "gradscf"
    sources = ("_memory.py", "integrals/grids/blocks.py", "integrals/grids/ao.py",
               "scf/inputs/grid.py", "scf/inputs/assembly.py", "scf/rks.py")
    report = dict(xc="PBE", basis="Cartesian def2-SVP", grid_level=3,
        max_memory_mb=max_memory, repeats=repeats, dtype="float64",
        geometry_sha256=digest(geometry), density_sha256=digest(density_path),
        density_role="fixed checkpoint trial density; no SCF convergence test",
        benchmark_sha256=digest(Path(__file__)),
        source_sha256={name: digest(source_root / name) for name in sources},
        host=platform.node(), platform=platform.platform(), python=platform.python_version(),
        jax_version=jax.__version__, backend=jax.default_backend(),
        cpu_affinity=sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        thread_settings={name: os.environ.get(name) for name in
            ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "XLA_FLAGS")},
        rss_scope="cumulative process peak; not an independent per-case peak", cases={})

    def save(phase):
        report.update(phase=phase, elapsed_seconds=perf_counter()-started)
        report["process_peak_rss_bytes_cumulative"] = resource.getrusage(
            resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024)
        (output_dir / "status.json").write_text(json.dumps(report, indent=2) + "\n")
        print("XC_GRID_BENCHMARK " + json.dumps(report), flush=True)

    save("setup")
    with np.load(density_path, allow_pickle=False) as archive:
        density = jnp.asarray(archive["density_matrix"], dtype=jnp.float64)
    atom = ";".join(line for line in geometry.read_text().splitlines()[2:] if line.strip())
    mol = gto.M(atom=atom, basis="def2-svp", unit="Angstrom", cart=True)
    cfg = RKSConfig(xc_spec="pbe", jk_backend="direct", max_cycle=100,
                    max_memory=max_memory, grid_block_size=0, grid_max_memory=None)
    inputs = build_rks_integral_inputs(atom=mol.to_spec(), basis=mol.basis, config=cfg,
        grids_level=3, integral_backend="native", include_dipole_integrals=False)
    jax.block_until_ready(inputs.grid_weights)
    if density.shape != inputs.overlap.shape or not np.isfinite(np.asarray(density)).all():
        raise ValueError("The fixed density must be finite and match the AO basis.")
    provider = inputs.grid_ao
    if provider is None:
        raise ValueError("This benchmark requires the automatically streamed large molecular grid.")
    report.update(setup_seconds=perf_counter()-started, nao=density.shape[0],
                  grid_points=provider.ngrids, auto_block_size=provider.block_size)
    values = {}
    for size in block_sizes:
        name = "auto" if size is None else f"forced{size}"
        grid = provider if size is None else replace(provider, block_size=size, adaptive=False)
        assert grid.coords is provider.coords and grid.weights is provider.weights
        case = report["cases"][name] = dict(block_size=grid.block_size)

        def evaluate(dm, ao_grid):
            return _restricted_xc_energy_and_matrix(density=dm, ao=inputs.ao,
                ao_deriv1=inputs.ao_deriv1, weights=ao_grid.weights,
                cfg=cfg, xc_kind="GGA", grid_ao=ao_grid)

        save(name + ":compile")
        phase_started = perf_counter()
        compiled = jax.jit(evaluate).lower(density, grid).compile()
        case["compile_seconds"] = perf_counter()-phase_started
        memory = compiled.memory_analysis()
        case["compiled_memory_bytes"] = {key: getattr(memory, key) for key in
            ("argument_size_in_bytes", "output_size_in_bytes", "temp_size_in_bytes",
             "alias_size_in_bytes", "const_size_in_bytes") if hasattr(memory, key)}
        save(name + ":warmup")
        phase_started = perf_counter()
        result = jax.block_until_ready(compiled(density, grid))
        case["warmup_seconds"] = perf_counter()-phase_started
        case["warm_seconds"] = []
        for _ in range(repeats):
            phase_started = perf_counter()
            result = jax.block_until_ready(compiled(density, grid))
            case["warm_seconds"].append(perf_counter()-phase_started)
            save(name + ":repeat")
        energy, potential = map(np.asarray, result)
        if not np.isfinite(energy).all() or not np.isfinite(potential).all():
            raise FloatingPointError("XC energy and potential must remain finite.")
        values[name] = energy, potential
        case["xc_energy_hartree"] = float(energy)
        case["potential_max_abs_hartree"] = float(np.max(np.abs(potential)))
        np.savez(output_dir / f"{name}.npz", xc_energy_hartree=energy, potential_hartree=potential)
        save(name + ":complete")

    baseline = next(iter(values.values()))
    report["energy_abs_difference_hartree"] = max(
        float(abs(value[0]-baseline[0])) for value in values.values())
    report["potential_max_abs_difference_hartree"] = max(
        float(np.max(np.abs(value[1]-baseline[1]))) for value in values.values())
    save("comparison")
    assert report["energy_abs_difference_hartree"] <= 1e-9
    assert report["potential_max_abs_difference_hartree"] <= 1e-10
    save("complete")
    return report


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[2]
    output = root / "artifacts/dabna2/monomer_pbe_level3"
    run_benchmark(root / "examples/tdscf/dabna2/dabna2_monomer.xyz",
                  output / "scf_density.npz", output / "xc_grid_budget")
