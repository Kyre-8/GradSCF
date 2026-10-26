"""Paired CPU timings of MP Taylor graphs, using identical native HF inputs.

Run from this Git checkout with JAX_PLATFORMS=cpu. The historical baseline is
read from Git, avoiding a second maintained MP implementation. HF, integrals
and connection construction are outside warm timings. Every execution is
synchronized. JSON results go to ignored artifacts/mp_graph/benchmark.json.
"""
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
from time import perf_counter
from types import ModuleType

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from gradscf import gto, scf, ci, mp
from gradscf.mp.series import run_series
from gradscf.scf.reference import reference_from_source, unrestricted_reference_from_source
from pyscf import gto as pyscf_gto, scf as pyscf_scf
from _pyscf_reference import mp_coefficients

BASE_COMMIT = "5a0f242ea9573e89d56e5849ec5fc5392f8659fb"
baseline_source = subprocess.check_output(
    ["git", "show", BASE_COMMIT+":src/gradscf/mp/series.py"], text=True)
baseline = ModuleType("gradscf.mp._benchmark_baseline")
baseline.__package__ = "gradscf.mp"
exec(compile(baseline_source, "<MP baseline "+BASE_COMMIT+">", "exec"), baseline.__dict__)

cases = [
    ("H4", "H 0 0 0; H 0 0 .8; H 0 0 1.8; H 0 0 2.8", 0),
    ("H3", "H 0 0 0; H 0 0 .85; H 0 0 1.9", 1),
    ("Li", "Li 0 0 0", 1),
]
records = []

for name, atom, spin in cases:
    mol = gto.M(atom=atom, basis="6-31g*", cart=True, spin=spin)
    mf = (scf.UHF if spin else scf.RHF)(mol, conv_tol=1e-13,
        conv_tol_grad=1e-11, max_cycle=200).run()
    if not mf.converged:
        raise RuntimeError("Converge the native HF reference before timing")
    ref = (unrestricted_reference_from_source if spin else reference_from_source)(mf)
    h, g = jax.block_until_ready((ref.h1, ref.eri))
    nmo = h[0].shape[0] if spin else h.shape[0]
    print("\n%s / Cartesian 6-31G*, %s" % (name, "UHF" if spin else "RHF"), flush=True)
    for order in (4, 6, 8, 10, 12):
        cfg = mp.MPConfig(order=order, algorithm="series", with_t2=False)
        space = (ci.make_uci_space if spin else ci.make_ci_space)(
            nmo, ref.nocc, max_excitation=order)
        ci.hamiltonian_action(h, g, space, jnp.zeros(space.size)).block_until_ready()
        executables, preparations = [], []
        for run in (baseline.run_series, run_series):
            function = jax.jit(lambda a, b: run(a, b, nocc=ref.nocc, config=cfg).corrections)
            start = perf_counter()
            lowered = function.lower(h, g)
            lower_seconds = perf_counter()-start
            start = perf_counter()
            executable = lowered.compile()
            compile_seconds = perf_counter()-start
            result = executable(h, g)
            result.block_until_ready()
            executables.append(executable)
            preparations.append(dict(lower_seconds=lower_seconds, compile_seconds=compile_seconds,
                flops=executable.cost_analysis()["flops"],
                estimated_bytes_accessed=executable.cost_analysis()["bytes accessed"]))
        old, new = [np.asarray(exe(h, g).block_until_ready()) for exe in executables]
        np.testing.assert_allclose(new, old, atol=2e-10, rtol=0.)
        samples = [[], []]
        # Alternate order to reduce bias from CPU scheduling or thermal drift.
        for repeat in range(15):
            for index in ((0, 1) if repeat % 2 == 0 else (1, 0)):
                start = perf_counter()
                executables[index](h, g).block_until_ready()
                samples[index].append(perf_counter()-start)
        for stats, times in zip(preparations, samples):
            stats.update(warm_median_seconds=float(np.median(times)), warm_samples_seconds=times)
        speedup = preparations[0]["warm_median_seconds"]/preparations[1]["warm_median_seconds"]
        record = dict(system=name, order=order, nmo=nmo, nocc=ref.nocc, ndet=space.size,
            baseline=preparations[0], shared=preparations[1], speedup=speedup,
            max_correction_difference=float(np.max(np.abs(new-old))), corrections=new.tolist())
        records.append(record)
        print("MP%2d  ndet=%4d  old=%7.3f ms  shared=%7.3f ms  speedup=%5.2f  difference=%.3e Ha" %
            (order, space.size, 1e3*preparations[0]["warm_median_seconds"],
             1e3*preparations[1]["warm_median_seconds"], speedup, record["max_correction_difference"]), flush=True)
        jax.clear_caches()
    # Independent high-order oracle, outside all timing regions.
    reference_mol = pyscf_gto.M(atom=atom, basis="6-31g*", cart=True, spin=spin, verbose=0)
    reference_mf = (pyscf_scf.UHF if spin else pyscf_scf.RHF)(reference_mol).run(
        conv_tol=1e-13, conv_tol_grad=1e-11, max_cycle=200, init_guess="hcore")
    expected = mp_coefficients(reference_mf, 12)
    np.testing.assert_allclose(records[-1]["corrections"], expected, atol=2e-10, rtol=0.)
    records[-1]["max_independent_reference_difference"] = float(
        np.max(np.abs(np.asarray(records[-1]["corrections"])-expected)))

report = dict(platform=platform.platform(), backend=str(jax.devices()),
    python=platform.python_version(), jax=jax.__version__, dtype="float64",
    baseline_commit=BASE_COMMIT, baseline_source_sha256=hashlib.sha256(baseline_source.encode()).hexdigest(),
    shared_source_sha256=hashlib.sha256(Path("src/gradscf/mp/series.py").read_bytes()).hexdigest(),
    timed_region="compiled MP energy corrections only, 15 paired synchronized executions",
    records=records)
report["hardware"] = dict(machine=platform.machine(), logical_cpus=os.cpu_count())
if platform.system() == "Darwin":
    info = json.loads(subprocess.check_output(
        ["/usr/sbin/system_profiler", "SPHardwareDataType", "-json"], text=True))
    item = info.get("SPHardwareDataType", [{}])[0]
    report["hardware"].update(cpu=item.get("chip_type"), memory=item.get("physical_memory"))
output = Path("artifacts/mp_graph/benchmark.json")
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(report, indent=2)+"\n")
print("\nMeasured results:", output)

# Plotting is also outside the timing regions; requires matplotlib.
import matplotlib.pyplot as plt
plt.rcParams["svg.fonttype"] = "none"

fig, axes = plt.subplots(3, 2, figsize=(10, 8), sharex=True, layout="constrained")
for row, (name, _, spin) in enumerate(cases):
    samples = [r for r in records if r["system"] == name]
    orders = [r["order"] for r in samples]
    for variant, label, color in (("baseline", "Baseline", "#c45145"),
                                   ("shared", "Shared nodes", "#246caa")):
        values = np.array([r[variant]["warm_median_seconds"]*1000 for r in samples])
        quartiles = np.array([np.percentile(r[variant]["warm_samples_seconds"], [25, 75])*1000
                              for r in samples]).T
        axes[row, 0].errorbar(orders, values, yerr=np.stack((values-quartiles[0], quartiles[1]-values)),
            label=label, color=color, marker="o", markersize=4, capsize=3, linewidth=1.7)
    axes[row, 0].set_title("%s / %s; %d determinants" %
        (name, "UHF" if spin else "RHF", samples[0]["ndet"]), loc="left", fontsize=11)
    axes[row, 0].set_ylabel("Warm execution / ms")
    axes[row, 1].plot(orders, [r["speedup"] for r in samples], color="#397357", marker="o")
    axes[row, 1].axhline(1, color="#777777", linewidth=.8, linestyle="--")
    axes[row, 1].set_ylabel("Speedup")
    axes[row, 1].set_title("Baseline / shared-node time", loc="left", fontsize=11)
    for ax in axes[row]:
        ax.grid(alpha=.2)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xticks([4, 6, 8, 10, 12])
axes[0, 0].legend(frameon=False)
for ax in axes[-1]:
    ax.set_xlabel("Requested MP energy order")
fig.suptitle("Shared linear nodes in MP Taylor graphs\n%s CPU · float64 · Cartesian 6-31G* · 15 paired runs (median and IQR)" %
    report["hardware"].get("cpu", platform.machine()), fontsize=12)
fig.savefig(output.with_suffix(".png"), dpi=170)
fig.savefig(output.with_suffix(".svg"))

# Measured warm medians / ms (Apple M4 Pro CPU, float64, 2026-10-10).
# System  MP order  baseline  shared nodes  speedup
# H4       4        4.541    1.304       3.48
# H4       6        5.993    1.678       3.57
# H4       8        7.887    2.063       3.82
# H4      10        9.190    2.344       3.92
# H4      12       11.361    2.659       4.27
# H3       4        0.285    0.118       2.41
# H3       6        0.372    0.158       2.35
# H3       8        0.538    0.232       2.32
# H3      10        0.611    0.218       2.81
# H3      12        0.710    0.236       3.01
# Li       4       18.823    5.010       3.76
# Li       6       27.338    6.538       4.18
# Li       8       35.339    7.799       4.53
# Li      10       43.235    9.230       4.68
# Li      12       51.666   10.623       4.86
# Maximum old/new correction difference: 6.939e-18 Ha.
# Independent PySCF-action reference checks E2--E12; maximum difference 2.297e-13 Ha.
# Detailed theory and scope: src/gradscf/mp/GRAPH_REUSE.md.
