"""Compare determinant and excitation-block MP graphs on identical native inputs.

Run from the checkout root with PYTHONPATH=src JAX_PLATFORMS=cpu. HF and MO
integrals are prepared once per molecule. Warm timings cover energy corrections
only, with explicit synchronization; graph lowering and compilation are separate.
This is an experimental representation comparison, not an end-to-end scaling test.
"""
import hashlib
import json
from pathlib import Path
import platform
import subprocess
from time import perf_counter

import jax
import numpy as np

jax.config.update("jax_enable_x64", True)

from gradscf import gto, scf, mp
from gradscf.mp.series import make_series_space
from gradscf.scf.reference import reference_from_source, unrestricted_reference_from_source

cases = [
    ("H3", "H 0 0 0; H 0 0 .85; H 0 0 1.9", 1),
    ("H4", "H 0 0 0; H 0 0 .8; H 0 0 1.8; H 0 0 2.8", 0),
    ("Li", "Li 0 0 0", 1),
]
records = []

for name, atom, spin in cases:
    mol = gto.M(atom=atom, basis="6-31g*", cart=True, spin=spin, unit="Angstrom")
    mf = (scf.UHF if spin else scf.RHF)(mol, conv_tol=1e-13,
        conv_tol_grad=1e-11, max_cycle=200).run()
    if not mf.converged:
        raise RuntimeError("Converge HF before comparing MP backends")
    ref = (unrestricted_reference_from_source if spin else reference_from_source)(mf)
    h, g = jax.block_until_ready((ref.h1, ref.eri))
    nmo = h[0].shape[0] if spin else h.shape[0]
    for order in (4, 6, 12):
        executables, stats = [], {}
        for algorithm in ("series", "tensor"):
            config = mp.MPConfig(order=order, algorithm=algorithm, with_t2=False)
            space = make_series_space(nmo, ref.nocc, config=config)
            function = jax.jit(lambda a, b: mp.run_mp(
                a, b, nocc=ref.nocc, config=config).corrections)
            start = perf_counter()
            lowered = function.lower(h, g)
            lower_seconds = perf_counter()-start
            start = perf_counter()
            executable = lowered.compile()
            compile_seconds = perf_counter()-start
            for _ in range(3):
                executable(h, g).block_until_ready()
            executables.append(executable)
            stats[algorithm] = dict(lower_seconds=lower_seconds,
                compile_seconds=compile_seconds, cost_analysis=executable.cost_analysis(),
                warm_samples_seconds=[])
        old, new = [np.asarray(exe(h, g).block_until_ready()) for exe in executables]
        np.testing.assert_allclose(new, old, atol=2e-10, rtol=0.)
        for repeat in range(15):
            for index in ((0, 1) if repeat % 2 == 0 else (1, 0)):
                start = perf_counter()
                executables[index](h, g).block_until_ready()
                stats[("series", "tensor")[index]]["warm_samples_seconds"].append(perf_counter()-start)
        for item in stats.values():
            item["warm_median_seconds"] = float(np.median(item["warm_samples_seconds"]))
        record = dict(system=name, atom=atom, spin=spin, nmo=nmo, nocc=ref.nocc,
            order=order, independent_coefficients=space.size, dense_elements=space.dense_elements,
            corrections=new.tolist(), max_difference=float(np.max(np.abs(new-old))), **stats)
        records.append(record)
        print("%s MP%2d: series %.3f ms, tensor %.3f ms, ratio %.2fx, error %.3e Ha" %
            (name, order, stats["series"]["warm_median_seconds"]*1000,
             stats["tensor"]["warm_median_seconds"]*1000,
             stats["series"]["warm_median_seconds"]/stats["tensor"]["warm_median_seconds"],
             record["max_difference"]), flush=True)
        jax.clear_caches()

sources = ["src/gradscf/mp/series.py", "src/gradscf/mp/_tensor.py", "src/gradscf/mp/types.py"]
report = dict(platform=platform.platform(), backend=str(jax.devices()),
    python=platform.python_version(), jax=jax.__version__, dtype="float64",
    basis="6-31g*", cart=True, frozen=None, records=records,
    command="PYTHONPATH=src JAX_PLATFORMS=cpu python examples/mp/benchmark_excitation_blocks.py",
    source_sha256={p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in sources})
if platform.system() == "Darwin":
    item = json.loads(subprocess.check_output(
        ["/usr/sbin/system_profiler", "SPHardwareDataType", "-json"], text=True))["SPHardwareDataType"][0]
    report["hardware"] = dict(cpu=item.get("chip_type"), memory=item.get("physical_memory"))
output = Path("artifacts/mp_blocks/benchmark.json")
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(report, indent=2)+"\n")
print("Detailed measurements:", output)

# Warm medians, Apple M4 Pro CPU / float64, 2026-10-10.
# System  Order   series / ms   tensor / ms   series / tensor
# H3       4        0.124         0.142          0.88
# H3       6        0.145         0.181          0.80
# H3      12        0.256         0.323          0.79
# H4       4        1.245         0.946          1.32
# H4       6        1.555         1.621          0.96
# H4      12        2.661         3.490          0.76
# Li       4        4.911         0.753          6.52
# Li       6        6.295         1.238          5.08
# Li      12       10.604         2.128          4.98
# Largest correction difference: 8.33e-17 Hartree.
# Both paths already share W Psi_j across Taylor degrees. This measures the
# representation change, not the earlier common-subexpression optimization.
# Theory and limitations: src/gradscf/mp/TENSOR_BLOCKS.md.
