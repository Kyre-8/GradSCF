"""Independent PySCF analytic gradients and finite-gradient CIS curvature.

Run the native X03 optimization first. All reference points independently
build HF and the CIS matrix; SciPy dense eigenvectors enter PySCF's public
analytic gradient API. No GradSCF code or electronic arrays are reused.
"""
import hashlib
from functools import partial
import json
from pathlib import Path
import platform
from time import perf_counter

import numpy as np
import pyscf
from pyscf import gto, scf, lib
from pyscf.scf import cphf
from scipy import linalg


output = Path("reproducibility/ad_showcase/X03_formaldehyde_cis_optimization")
native = json.loads((output / "gradscf.json").read_text())
if not native["complete"]:
    raise RuntimeError("Finish the native optimization and curvature gates first.")
started = perf_counter()
root_index = native["selected_root"]
cphf_tolerance = 1e-13
z_residual_tolerance = 1e-10


def evaluate(coordinates):
    mol = gto.M(atom=list(zip(native["symbols"], coordinates)), basis=native["basis"],
                unit="Bohr", cart=True, verbose=0)
    mf = scf.RHF(mol)
    mf.conv_tol, mf.conv_tol_grad, mf.max_cycle = 1e-12, 1e-12, 180
    mf.kernel()
    if not mf.converged:
        raise RuntimeError("An independent displaced RHF calculation failed.")
    td = mf.TDA()
    td.singlet = True
    a, _ = td.get_ab()
    nocc, nvir = a.shape[:2]
    a = a.reshape(nocc*nvir, nocc*nvir)
    np.testing.assert_allclose(a, a.T, atol=1e-10, rtol=0)
    energies, vectors = linalg.eigh(a, subset_by_index=(0, native["nroots"]-1))
    residual = np.linalg.norm(a@vectors-vectors*energies, axis=0)
    if np.max(residual) > 1e-9:
        raise RuntimeError("Independent CIS eigen-residual exceeds its tolerance.")
    gap = float(np.min(np.abs(np.delete(energies, root_index)-energies[root_index])))
    if gap < native["minimum_gap_hartree"]:
        raise RuntimeError("The independent selected state is not isolated.")
    x = vectors[:, root_index].reshape(nocc, nvir)/np.sqrt(2)
    gradient_driver = td.nuc_grad_method()
    # Gradient differences amplify residual CPHF error; keep the physical
    # acceptance thresholds fixed while resolving the analytic gradient more tightly.
    gradient_driver.cphf_conv_tol, gradient_driver.cphf_max_cycle = cphf_tolerance, 100

    original_cphf, original_krylov = cphf.solve, lib.krylov
    z_residuals = []

    def checked_cphf(fvind, orbital_energies, occupations, rhs, *args, **kwargs):
        """Keep PySCF's solver and check its unpreconditioned Z equation."""
        result = original_cphf(fvind, orbital_energies, occupations, rhs, *args, **kwargs)
        z = np.asarray(result[0])
        gaps = orbital_energies[occupations == 0][:, None]-orbital_energies[occupations > 0][None, :]
        residual = gaps*z+fvind(z.ravel()).reshape(z.shape)+rhs
        norm = float(np.linalg.norm(residual))
        z_residuals.append(norm)
        if not np.isfinite(norm) or norm > z_residual_tolerance:
            raise RuntimeError(f"Independent CPHF Z residual {norm:.3e} exceeds {z_residual_tolerance:.1e} Hartree.")
        return result

    # krylov compares squared trial norms with lindep. Match this threshold to
    # tol**2; otherwise the default 1e-13 cutoff can stop before tight tol.
    # Both public callables are restored immediately, including on exceptions.
    with lib.temporary_env(lib, krylov=partial(original_krylov, lindep=cphf_tolerance**2)), \
            lib.temporary_env(cphf, solve=checked_cphf):
        gradient = gradient_driver.kernel(xy=(x, np.zeros_like(x)), state=root_index+1)
    if lib.krylov is not original_krylov or cphf.solve is not original_cphf:
        raise RuntimeError("An independent solver override leaked outside the gradient call.")
    if not z_residuals:
        raise RuntimeError("No independent CPHF residual was checked.")
    orbital_gradient = mf.get_grad(mf.mo_coeff, mf.mo_occ, mf.get_fock())
    return dict(ground_energy=float(mf.e_tot), excitation=float(energies[root_index]),
                total_energy=float(mf.e_tot+energies[root_index]), gradient=gradient,
                gap=gap, max_eigen_residual=float(np.max(residual)),
                scf_orbital_gradient_norm=float(np.linalg.norm(orbital_gradient)),
                z_residual_norm=max(z_residuals))


coordinates = np.asarray(native["coordinates_bohr"])
reference = evaluate(coordinates)
chart_indices = [(1, 2), (2, 1), (2, 2), (3, 0), (3, 1), (3, 2)]
step = native["curvature_steps_bohr"][-1]
columns, displaced_checks = [], []
for atom, axis in chart_indices:
    derivatives = []
    for sign in (1, -1):
        moved = coordinates.copy()
        moved[atom, axis] += sign*step
        point = evaluate(moved)
        # At these small displacements, an isolated root must remain much
        # closer to the reference energy than to a neighboring root.
        if abs(point["excitation"]-reference["excitation"]) > reference["gap"]/4:
            raise RuntimeError("The local independent root identity is not resolved.")
        derivatives.append(np.array([point["gradient"][i, j] for i, j in chart_indices]))
        displaced_checks.append(dict(atom=atom, axis=axis, sign=sign,
                                     excitation_hartree=point["excitation"], gap_hartree=point["gap"],
                                     max_eigen_residual=point["max_eigen_residual"],
                                     scf_orbital_gradient_norm=point["scf_orbital_gradient_norm"],
                                     z_residual_norm=point["z_residual_norm"]))
    columns.append((derivatives[0]-derivatives[1])/(2*step))
hessian = np.array(columns).T
asymmetry = float(np.max(np.abs(hessian-hessian.T)))
curvatures = np.linalg.eigvalsh((hessian+hessian.T)/2)
errors = dict(
    ground_energy_hartree=abs(reference["ground_energy"]-native["ground_energy_hartree"]),
    excitation_hartree=abs(reference["excitation"]-native["excitation_hartree"]),
    total_energy_hartree=abs(reference["total_energy"]-native["total_excited_energy_hartree"]),
    gradient_hartree_per_bohr=float(np.max(np.abs(reference["gradient"]-native["gradient_hartree_per_bohr"]))),
    chart_curvature_hartree_per_bohr2=float(np.max(np.abs(hessian-native["chart_hessians_hartree_per_bohr2"][-1]))),
)
tolerances = dict(ground_energy_hartree=1e-8, excitation_hartree=1e-7,
                  total_energy_hartree=1e-7, gradient_hartree_per_bohr=1e-6,
                  chart_curvature_hartree_per_bohr2=2e-5)
passed = (all(errors[key] <= value for key, value in tolerances.items())
          and np.max(np.abs(reference["gradient"])) <= native["gradient_tolerance"]
          and asymmetry <= 2e-5 and np.min(curvatures) > native["minimum_curvature"])
report = dict(
    passed=bool(passed), reference="Independent PySCF/SciPy dense CIS and PySCF analytic gradients",
    ground_energy_hartree=reference["ground_energy"], excitation_hartree=reference["excitation"],
    total_excited_energy_hartree=reference["total_energy"],
    gradient_hartree_per_bohr=reference["gradient"].tolist(),
    max_cartesian_gradient=float(np.max(np.abs(reference["gradient"]))),
    gap_hartree=reference["gap"], max_eigen_residual=reference["max_eigen_residual"],
    scf_orbital_gradient_norm=reference["scf_orbital_gradient_norm"], scf_gradient_tolerance=1e-12,
    z_residual_norm=reference["z_residual_norm"], z_residual_tolerance=z_residual_tolerance,
    curvature_method="Central differences of independent analytic CIS gradients in the same six-coordinate chart",
    curvature_step_bohr=step, chart_hessian_hartree_per_bohr2=hessian.tolist(),
    raw_curvature_asymmetry=asymmetry, chart_curvature_eigenvalues_hartree_per_bohr2=curvatures.tolist(),
    errors=errors, tolerances=tolerances, displaced_checks=displaced_checks,
    cphf_tolerance=cphf_tolerance, cphf_max_cycle=100,
    krylov_linear_dependence_cutoff=cphf_tolerance**2,
    elapsed_seconds=perf_counter()-started, pyscf=pyscf.__version__, hostname=platform.node(),
    native_sha256=hashlib.sha256((output / "gradscf.json").read_bytes()).hexdigest(),
    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
)
(output / "pyscf.json").write_text(json.dumps(report, indent=2)+"\n")
print("Independent minimum/curvature errors:", errors)
print("Independent max Cartesian gradient:", report["max_cartesian_gradient"])
print("Independent chart curvature eigenvalues:", curvatures)
if not passed:
    raise RuntimeError("The independent minimum/curvature validation failed.")
print(f"Independent minimum and curvature checks passed; elapsed {report['elapsed_seconds']:.2f} s")

# Measured 2026-10-15 on jy_gpu_1, PySCF 2.13.0, CPU float64, two threads:
# max gradient error 3.811e-13 Eh/Bohr; chart curvature error 7.942e-10 Eh/Bohr^2.
# Largest true Z residual norm across 13 points: 2.242e-11 Eh (gate 1e-10).
# Passed in 17.13 s using scoped Krylov lindep=1e-26; see reference_precision.json
# for rejected reference controls. Native tolerances and acceptance gates unchanged.
