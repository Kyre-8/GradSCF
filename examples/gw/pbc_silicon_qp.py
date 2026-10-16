"""Four direct Si quasiparticle levels in a small three-dimensional model.

Edit the ordinary run arguments below. This is a finite-model functionality
example, not a converged Si gap or an interpolated GW band structure.
"""
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
from time import perf_counter
import warnings

import jax
import numpy as np

jax.config.update("jax_enable_x64", True)

from examples.pbc.band_structures.systems import structure
from gradscf.pbc import gto, dft
from gradscf.scf import SCFDifferentiationConfig
from gradscf.gw.pbc.ksigma import g0w0_cd_kpoints, _screening_memory_bytes
from gradscf.solvers import LinearSolverConfig
from gradscf.solvers.nonlinear import ScalarRootConfig
from gradscf.tools.spectra import HARTREE_TO_EV


def plot(folder):
    """Plot successful requested levels and label failures without plotting their energies."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    folder = Path(folder)
    report = json.loads((folder/"results.json").read_text())
    if not report.get("complete"):
        raise ValueError("Require a complete direct-QP attempt")
    with np.load(folder/"levels.npz") as arrays:
        ks, qp, mask = (arrays[name] for name in ("ks_hartree", "qp_hartree", "qp_computed_mask"))
        residual = arrays["qp_residual_hartree"]
        accepted = (mask & arrays["converged_mask"] & np.isfinite(qp)
                    & (abs(residual) <= report["qp_residual_gate_hartree"]))
    rows = np.argwhere(mask)
    if not np.any(accepted):
        raise ValueError("No requested QP level passed its convergence/residual gate")
    zero = float(ks[:, report["nocc"]-1].max())
    x = np.arange(len(rows))
    ks_values = np.array([ks[k, p] for k, p in rows])
    qp_values = np.array([qp[k, p] for k, p in rows])
    valid = np.array([accepted[k, p] for k, p in rows])
    labels = [f"k{k}, band {p}" for k, p in rows]
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.5), gridspec_kw={"width_ratios": [1.5, 1]})
    axes[0].scatter(x-.07, (ks_values-zero)*HARTREE_TO_EV, marker="o", s=65,
                    facecolors="none", edgecolors="#555555", label="PBE KS")
    axes[0].scatter(x[valid]+.07, (qp_values[valid]-zero)*HARTREE_TO_EV, marker="D", s=48,
                    color="#1d718e", label="Direct diagonal G₀W₀")
    axes[0].axhline(0., color=".65", lw=.8)
    axes[0].set(ylabel="Energy relative to sampled KS VBM / eV", xticks=x, xticklabels=labels)
    axes[0].legend(frameon=False)
    axes[1].bar(x[valid], (qp_values[valid]-ks_values[valid])*HARTREE_TO_EV, color="#1d718e", width=.55)
    axes[1].axhline(0., color=".65", lw=.8)
    axes[1].set(ylabel="QP − KS / eV", xticks=x, xticklabels=labels)
    for axis in axes:
        axis.grid(axis="y", alpha=.18)
        axis.tick_params(axis="x", labelrotation=25)
        axis.spines[["top", "right"]].set_visible(False)
        for index in x[~valid]:
            axis.text(index, .06, "QP not\nconverged", transform=axis.get_xaxis_transform(),
                      ha="center", va="bottom", fontsize=9, color="#a13931")
    fig.suptitle(f"Si: {int(valid.sum())} converged / {len(rows)} requested direct QP levels", fontsize=14)
    settings = report["settings"]
    text = (f"PBE / GTH-SZV / GTH-PBE · k={settings['kmesh']} · FFT={settings['mesh']} · "
            f"nw={settings['nw']} · finite-size correction={settings['fc']}\n"
            f"Fractional k: {np.round(report['fractional_kpoints'], 4).tolist()}\n"
            "Failed QP candidates are omitted; full masks/residuals remain in the data.\n"
            "Discrete finite-model values; no off-mesh GW interpolation or bulk-convergence claim.")
    fig.text(.07, .025, text, fontsize=9, va="bottom")
    fig.subplots_adjust(top=.84, bottom=.31, left=.09, right=.97, wspace=.38)
    for suffix in ("png", "pdf"):
        fig.savefig(folder/f"silicon_direct_qp.{suffix}", dpi=180, bbox_inches="tight")
    plt.close(fig)
    validation = dict(results_sha256=hashlib.sha256((folder/"results.json").read_bytes()).hexdigest(),
        arrays_sha256=hashlib.sha256((folder/"levels.npz").read_bytes()).hexdigest(),
        plot_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        qp_computed_mask=mask.tolist(), plotted_qp_mask=accepted.tolist(),
        nrequested=int(mask.sum()), nplotted=int(accepted.sum()),
        all_requested_qp_passed=bool(report["passed"]),
        failed_requested=[dict(k_index=int(k), band_index=int(p), residual_hartree=float(residual[k, p]))
                          for k, p in np.argwhere(mask & ~accepted)])
    (folder/"plot_validation.json").write_text(json.dumps(validation, indent=2)+"\n")


def run(*, output="reproducibility/ad_showcase/PBC_silicon_gw", nw=8, fc=True):
    """Use full screening/internal sums and request only bands 3 and 4 at two k points."""
    folder = Path(output)
    folder.mkdir(parents=True, exist_ok=True)
    destination = folder/"results.json"
    if destination.exists():
        raise FileExistsError("Use a fresh output directory; preserve earlier evidence")
    began = perf_counter()
    settings = dict(a_angstrom=5.431, xc="pbe", basis="gth-szv", pseudo="gth-pbe",
        kmesh=[2, 1, 1], mesh=[21, 21, 21], eta_hartree=1e-3, nw=int(nw), fc=bool(fc))
    paths = [Path(__file__), Path("examples/pbc/band_structures/systems.py")]
    paths += [Path("src/gradscf")/name for name in (
        "pbc/scf.py", "pbc/_kpoint.py", "pbc/gto.py", "integrals/periodic/fft.py",
        "gw/pbc/ksigma.py", "gw/pbc/product_basis.py", "gw/pbc/q0.py", "gw/screened.py",
        "solvers/nonlinear/scalar.py", "solvers/linear/complex.py")]
    report = dict(complete=False, passed=False, stage="scf", settings=settings,
        source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        hostname=platform.node(), jax=jax.__version__, dtype="float64/complex128",
        backend=jax.default_backend(), allowed_cpu_ids=sorted(os.sched_getaffinity(0)),
        threads={name:os.environ.get(name) for name in (
            "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")},
        discretization_converged=False, periodic_ad_supported=False,
        scope="Direct diagonal G0W0 in the converged KS basis, two sampled k points. No off-mesh GW interpolation, bulk gap, or degenerate-subspace self-energy diagonalization claim.",
        q0_convention="Existing finite-eta head/wing prescription; compression parity is not an independent physical validation of opposite-wing conjugation.")

    def save():
        report.update(elapsed_seconds=perf_counter()-began,
            peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        destination.write_text(json.dumps(report, indent=2)+"\n")

    save()
    try:
        atoms, _, _ = structure("silicon")
        cell = gto.M(atom=list(zip(atoms.get_chemical_symbols(), atoms.positions)),
            a=atoms.cell.array, unit="Angstrom", basis=settings["basis"], pseudo=settings["pseudo"],
            mesh=settings["mesh"], cart=True, precision=1e-10)
        mf = dft.KRKS(cell, xc="pbe", kpts=cell.make_kpts(settings["kmesh"]),
            temperature=None, exxdiv="ewald", max_cycle=150, damping=.15,
            conv_tol=1e-11, conv_tol_density=1e-10, conv_tol_grad=1e-9,
            differentiation=SCFDifferentiationConfig(mode="implicit", tolerance=1e-10)).run()
        jax.block_until_ready(mf.result)
        reference = mf.result
        ks = np.asarray(reference.mo_energy_spin[0])
        nocc = cell.nelectron//2
        gap = float(ks[:, nocc].min()-ks[:, nocc-1].max())
        fractional = np.asarray(mf.kpts)@np.asarray(cell.lattice).T/(2*np.pi)
        report.update(scf_converged=bool(mf.converged), energy_hartree=float(mf.e_tot),
            scf_seconds=perf_counter()-began, nocc=nocc, nmo=ks.shape[1],
            fractional_kpoints=fractional.tolist(), ks_gap_hartree=gap,
            lattice_bohr=np.asarray(cell.lattice).tolist(),
            coordinates_angstrom=atoms.positions.tolist(), basis_data=cell.basis_spec,
            pseudopotential_data=cell.pseudopotentials)
        save()
        if not mf.converged or gap <= 1e-3:
            raise RuntimeError("Require a converged gapped reference (sampled gap > 1e-3 Eh)")
        orbs, targets = (nocc-1, nocc), tuple(range(len(ks)))
        config = LinearSolverConfig(method="direct", rtol=1e-11, atol=1e-13)
        estimate = _screening_memory_bytes(len(ks), int(np.prod(settings["mesh"])),
            ks.shape[1], nocc, len(targets), len(orbs), nw, "transition")
        report.update(stage="gw", qp_orbitals=list(orbs), qp_kpoints=list(targets),
            complete_screening_transitions=len(ks)*nocc*(ks.shape[1]-nocc),
            estimated_gw_primary_bytes=estimate, primary_array_limit_bytes=1 << 28,
            qp_residual_gate_hartree=ScalarRootConfig().ftol)
        np.savez_compressed(folder/"scf_reference.npz", mo_energy=ks,
            mo_coeff=np.asarray(reference.mo_coeff_spin[0]), density=np.asarray(reference.density_spin))
        save()
        print("SCF complete; direct QP window", targets, orbs, "gap / Eh", gap, flush=True)
        gw_started = perf_counter()
        result = g0w0_cd_kpoints(inputs=mf.inputs, kpts_frac=fractional,
            mo_energy_k=reference.mo_energy_spin[0], mo_coeff_k=reference.mo_coeff_spin[0],
            nocc=nocc, fock_k=reference.fock_spin[0], hcore_k=mf.inputs.hcore,
            density_spin=reference.density_spin, mesh=tuple(settings["mesh"]), nw=nw,
            eta=settings["eta_hartree"], orbs=orbs, qp_kpoints=targets, fc=fc,
            screening_backend="transition", screening_config=config, max_memory_bytes=1 << 28)
        jax.block_until_ready(result)
        qp, mask, residual = map(np.asarray, (result.mo_energy, result.qp_computed_mask, result.qp_residual))
        expected = np.zeros_like(mask)
        expected[:, orbs] = True
        if not np.array_equal(mask, expected):
            raise RuntimeError("Unexpected QP computed mask")
        passed = bool(result.converged) and np.isfinite(qp[mask]).all() and (
            np.max(abs(residual[mask])) <= report["qp_residual_gate_hartree"])
        np.savez_compressed(folder/"levels.npz", ks_hartree=ks, qp_hartree=qp,
            qp_computed_mask=mask, converged_mask=np.asarray(result.converged_mask),
            qp_residual_hartree=residual, fractional_kpoints=fractional)
        report.update(complete=True, passed=bool(passed), stage="complete", gw_seconds=perf_counter()-gw_started,
            qp_computed_mask=mask.tolist(), converged_mask=np.asarray(result.converged_mask).tolist(),
            ks_hartree=ks.tolist(), qp_hartree=qp.tolist(), qp_residual_hartree=residual.tolist(),
            max_qp_residual_hartree=float(np.max(abs(residual[mask]))),
            internal_g_orbitals=list(result.g_orbitals), screening_occupied=list(result.screening_occupied),
            screening_virtual=list(result.screening_virtual))
        save()
        if not passed:
            warnings.warn("Some requested QP levels did not converge; passed=False and all candidates/residuals are retained.",
                          RuntimeWarning)
        if np.any(mask & np.asarray(result.converged_mask)):
            plot(folder)
        print("Si direct QP attempt complete:", destination, "all requested passed:", report["passed"],
              "max residual / Eh", report["max_qp_residual_hartree"], flush=True)
        return report
    except Exception as error:
        report.update(passed=False, error=f"{type(error).__name__}: {error}")
        save()
        raise


if __name__ == "__main__":
    run()

# Measured on 2026-10-16, c20 CPU float64, shared CPU16-31, nw=8/fc=True:
# SCF gap=0.0819884161 Eh; 3/4 requested diagonal QP levels converged.
# (k,band)=(0,3),(0,4),(1,4): residuals <=3.71e-11 Eh.
# (1,3) did not converge: residual=-0.00430449368 Eh; its candidate is not plotted.
# Initial wrapper49.32s, peak RSS2001128KiB. These are finite-model results,
# not a bulk Si gap or frequency/grid convergence study. No rerun was used
# to select a passing window; the full original failed-attempt record is kept.
