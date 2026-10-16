"""Independent analytic PySCF CIS total/excitation gradients for X02.

PySCF builds its own RHF and CIS matrix. SciPy's dense eigensolver supplies
the isolated state; PySCF's public gradient API receives its explicit X/Y.
No GradSCF code, orbitals, amplitudes or derivatives enter the reference.
"""
import hashlib
import json
from pathlib import Path
import platform
from time import perf_counter

import numpy as np
import pyscf
from pyscf import gto, scf
from scipy import linalg


output = Path("reproducibility/ad_showcase/X02_formaldehyde_cis_gradients")
native = json.loads((output / "gradscf.json").read_text())
if not native["complete"]:
    raise RuntimeError("Finish the native X02 calculation first.")
started = perf_counter()
mol = gto.M(atom=list(zip(native["symbols"], native["coordinates_bohr"])),
            basis=native["basis"], unit="Bohr", cart=native["cart"], verbose=0)
mf = scf.RHF(mol)
mf.conv_tol, mf.conv_tol_grad, mf.max_cycle = 1e-12, 1e-10, 180
mf.kernel()
if not mf.converged:
    raise RuntimeError("Independent RHF did not converge.")
td = mf.TDA()
td.singlet = True
a, _ = td.get_ab()
nocc, nvir = a.shape[:2]
a = a.reshape(nocc*nvir, nocc*nvir)
np.testing.assert_allclose(a, a.T, atol=1e-10, rtol=0)
energies, vectors = linalg.eigh(a, subset_by_index=(0, native["nroots"]-1))
residual = np.linalg.norm(a@vectors-vectors*energies, axis=0)
if np.max(residual) > 1e-9:
    raise RuntimeError("Independent CIS eigen-residual exceeds 1e-9 Hartree.")
root = native["selected_root"]
gap = np.min(np.abs(np.delete(energies, root)-energies[root]))
if gap < native["minimum_gap_hartree"]:
    raise RuntimeError("The independent selected state is not isolated.")
x = vectors[:, root].reshape(nocc, nvir)/np.sqrt(2)
gradient_driver = td.nuc_grad_method()
gradient_driver.cphf_conv_tol = 1e-10
gradient_driver.cphf_max_cycle = 100
total_gradient = gradient_driver.kernel(xy=(x, np.zeros_like(x)), state=root+1)
ground_gradient = mf.nuc_grad_method().kernel()
excitation_gradient = total_gradient-ground_gradient
errors = dict(
    ground_energy_hartree=abs(float(mf.e_tot)-native["ground_energy_hartree"]),
    excitation_hartree=abs(float(energies[root])-native["excitation_hartree"]),
    excitation_gradient=float(np.max(np.abs(excitation_gradient-native["excitation_gradient_hartree_per_bohr"]))),
    total_excited_gradient=float(np.max(np.abs(total_gradient-native["total_excited_gradient_hartree_per_bohr"]))),
    ground_gradient=float(np.max(np.abs(ground_gradient-native["ground_gradient_hartree_per_bohr"]))),
)
tolerances = dict(ground_energy_hartree=1e-8, excitation_hartree=1e-7,
                  excitation_gradient=1e-6, total_excited_gradient=1e-6, ground_gradient=1e-6)
passed = all(errors[key] <= value for key, value in tolerances.items())
report = dict(
    reference="Independent PySCF RHF + public get_ab / SciPy dense CIS + PySCF analytic CIS gradient",
    selected_root=root, ground_energy_hartree=float(mf.e_tot), excitation_hartree=float(energies[root]),
    total_excited_energy_hartree=float(mf.e_tot+energies[root]),
    excitation_gradient_hartree_per_bohr=excitation_gradient.tolist(),
    total_excited_gradient_hartree_per_bohr=total_gradient.tolist(),
    ground_gradient_hartree_per_bohr=ground_gradient.tolist(),
    eigen_residuals_hartree=residual.tolist(), gap_hartree=float(gap),
    errors=errors, tolerances=tolerances, passed=passed,
    cphf_tolerance=gradient_driver.cphf_conv_tol, cphf_max_cycle=gradient_driver.cphf_max_cycle,
    elapsed_seconds=perf_counter()-started, pyscf=pyscf.__version__, hostname=platform.node(),
    native_sha256=hashlib.sha256((output / "gradscf.json").read_bytes()).hexdigest(),
    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
)
(output / "pyscf.json").write_text(json.dumps(report, indent=2)+"\n")
print("Independent CIS gradient errors:", errors)
if not passed:
    raise RuntimeError("The independent gradient comparison failed its predefined tolerances.")
print(f"All gradient comparisons passed; elapsed {report['elapsed_seconds']:.2f} s")

# Measured on jy_gpu_1, CPU float64, 2026-10-15:
# Max excitation / total excited gradient errors: 9.5475e-9 / 9.5489e-9 Eh/Bohr.
# Max ground gradient error: 2.8424e-12 Eh/Bohr.
# All gradient comparisons passed; elapsed 1.63 s.
