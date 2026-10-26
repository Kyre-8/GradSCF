"""Bounded N2/Cartesian def2-TZVP CI action benchmark, outside normal pytest.

PySCF supplies one common RHF/MO Hamiltonian and an independent MP2 oracle.
Only the CI action/solve timings belong to GradSCF. A second run can request
the complete CISD comparison after the sigma timing and memory are inspected.
"""
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

from gradscf import ci
from gradscf.ci.hamiltonian import build_direct_hamiltonian


OUTPUT = Path("artifacts/ci_mp_validation/ci/n2_def2tzvp_sigma.json")


def run(*, solve=False, cisd_max_cycle=150, cisd_max_space=40):
    import pyscf
    from pyscf import ao2mo, mp

    if not jax.config.x64_enabled:
        raise ValueError("This benchmark requires JAX float64.")
    output = OUTPUT.with_name("n2_def2tzvp_cisd.json") if solve else OUTPUT
    output.parent.mkdir(parents=True, exist_ok=True)
    files = [Path(ci.__file__).with_name(name) for name in
             ("direct.py", "hamiltonian.py", "solver.py", "space.py")]+[Path(__file__)]
    cpu = next((line.partition(":")[2].strip() for line in Path("/proc/cpuinfo").read_text().splitlines()
                if line.startswith("model name")), platform.processor())
    report = dict(complete=False, phase="reference", basis="def2-tzvp", cart=True,
        atom="N 0 0 -1.04; N 0 0 1.04", unit="Bohr", dtype="float64",
        scope="sigma_and_cisd" if solve else "sigma_only",
        reference_source="PySCF RHF; excluded from GradSCF CI timings",
        cpu=cpu, backend=jax.default_backend(), command=sys.argv,
        source_sha256={path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in files},
        jax=jax.__version__, pyscf=pyscf.__version__, python=platform.python_version(),
        affinity=sorted(os.sched_getaffinity(0)),
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        threads={key: os.environ.get(key) for key in
                 ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")})

    def save(phase):
        report.update(phase=phase, peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        output.write_text(json.dumps(report, indent=2)+"\n")
        print(phase, "peak RSS KiB", report["peak_rss_kib"], flush=True)

    save("reference")
    started = perf_counter()
    molecule = pyscf.gto.M(atom=report["atom"], basis=report["basis"], unit="Bohr",
                           cart=True, verbose=0)
    mf = pyscf.scf.RHF(molecule).run(conv_tol=1e-12, conv_tol_grad=1e-10)
    if not mf.converged:
        raise RuntimeError("The external RHF reference did not converge.")
    report.update(reference_seconds=perf_counter()-started, reference_energy_hartree=float(mf.e_tot),
                  reference_mo_sha256=hashlib.sha256(np.asarray(mf.mo_coeff).tobytes()).hexdigest())
    save("mo_integrals")
    started = perf_counter()
    nmo, nocc = mf.mo_coeff.shape[1], molecule.nelectron//2
    h = jnp.asarray(mf.mo_coeff.T@mf.get_hcore()@mf.mo_coeff)
    g = jnp.asarray(ao2mo.restore(1, ao2mo.kernel(molecule, mf.mo_coeff), nmo))
    report.update(integral_seconds=perf_counter()-started, nmo=nmo, nocc=nocc)
    save("topology")
    started = perf_counter()
    space = ci.make_ci_space(nmo, nocc)
    report.update(topology_seconds=perf_counter()-started, determinants=space.size)
    save("operator_build")
    started = perf_counter()
    operator = build_direct_hamiltonian(h, g, space)
    jax.block_until_ready(operator.diagonal)
    report["operator_build_dispatch_seconds"] = perf_counter()-started
    save("first_sigma")
    reference = jnp.zeros(space.size).at[0].set(1.)
    started = perf_counter()
    sigma = np.asarray(jax.block_until_ready(operator.apply(reference)))
    report["first_sigma_including_pending_build_seconds"] = perf_counter()-started
    np.testing.assert_allclose(sigma[0]+molecule.energy_nuc(), mf.e_tot, atol=2e-10, rtol=0)
    save("warm_sigma")
    vector = jnp.asarray(np.random.default_rng(147).normal(size=space.size)/np.sqrt(space.size))
    samples = []
    for _ in range(3):
        started = perf_counter()
        value = jax.block_until_ready(operator.apply(vector))
        samples.append(perf_counter()-started)
    if not np.isfinite(np.asarray(value)).all():
        raise RuntimeError("The direct sigma action produced a nonfinite result.")
    report.update(warm_sigma_seconds=samples, warm_sigma_median_seconds=float(np.median(samples)),
                  sigma_norm=float(jnp.linalg.norm(value)))
    other = jnp.asarray(np.random.default_rng(641).normal(size=space.size)/np.sqrt(space.size))
    applied = jax.block_until_ready(operator.apply(other))
    left, right = float(jnp.dot(vector, applied)), float(jnp.dot(value, other))
    report.update(hermiticity_bilinear_absolute_error=abs(left-right),
                  hermiticity_bilinear_scale=max(1., abs(left), abs(right)))
    np.testing.assert_allclose(left, right, atol=2e-10, rtol=0)
    save("mp2_oracle")
    # For a stationary HF reference, |<D|H|HF>|^2 / (E0-E_D0)
    # summed over normalized spin determinants is exactly the MP2 correction.
    energies = np.tile(mf.mo_energy, 2)
    ref = space.determinants[0]
    def gap(det):
        changed = det ^ ref
        value = 0.
        while changed:
            bit = changed & -changed
            p = bit.bit_length()-1
            value += energies[p]*(1 if det & bit else -1)
            changed ^= bit
        return value
    gaps = np.fromiter((gap(d) for d in space.determinants[1:]), dtype=float)
    correction = -np.dot(sigma[1:]**2, 1/gaps)
    started = perf_counter()
    reference_mp2 = mp.MP2(mf).run(verbose=0)
    report.update(mp2_from_sigma_hartree=float(correction),
                  reference_mp2_hartree=float(reference_mp2.e_corr),
                  reference_mp2_seconds=perf_counter()-started,
                  mp2_absolute_error_hartree=float(abs(correction-reference_mp2.e_corr)))
    np.testing.assert_allclose(correction, reference_mp2.e_corr, atol=2e-9, rtol=0)
    if solve:
        report.update(cisd_max_cycle=cisd_max_cycle, cisd_max_space=cisd_max_space,
                      cisd_conv_tol=1e-10)
        save("cisd")
        started = perf_counter()
        obj = ci.CISD(ci.CIReference(h, g, nocc, molecule.energy_nuc()),
                       conv_tol=1e-10, max_cycle=cisd_max_cycle, max_space=cisd_max_space).run()
        report.update(cisd_seconds=perf_counter()-started, cisd_converged=bool(obj.converged),
                      cisd_energy_hartree=float(obj.e_tot),
                      cisd_residual_norm=float(obj.result.residual_norms[0]),
                      cisd_reference_weight=float(obj.ci[0]**2),
                      cisd_coefficient_norm=float(jnp.linalg.norm(obj.ci)),
                      cisd_true_residual_norm=float(jnp.linalg.norm(operator.apply(obj.ci)
                          -(obj.e_tot-molecule.energy_nuc())*obj.ci)))
        save("cisd_result")
        save("cisd_oracle")
        started = perf_counter()
        external = mf.CISD().run(conv_tol=1e-11, max_cycle=150)
        report.update(reference_cisd_seconds=perf_counter()-started,
                      reference_cisd_converged=bool(external.converged),
                      reference_cisd_energy_hartree=float(external.e_tot),
                      cisd_absolute_error_hartree=float(abs(obj.e_tot-external.e_tot)))
        save("cisd_comparison")
        if not obj.converged:
            raise RuntimeError("The GradSCF CISD solve did not converge.")
        if not external.converged:
            raise RuntimeError("The independent PySCF CISD solve did not converge.")
        np.testing.assert_allclose(obj.e_tot, external.e_tot, atol=2e-9, rtol=0)
    report["complete"] = True
    save("complete")
    return report


if __name__ == "__main__":
    run()
