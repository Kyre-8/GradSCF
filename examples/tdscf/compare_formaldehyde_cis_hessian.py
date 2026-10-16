"""Independent full Cartesian finite-gradient Hessian for the X10 CIS gate."""
import hashlib
import json
from pathlib import Path
import platform
from time import perf_counter

import numpy as np
import pyscf
from pyscf import gto
from pyscf.hessian import thermo

from pyscf_cis_reference import evaluate_cis


output = Path("reproducibility/ad_showcase/X10_formaldehyde_cis_emission")
native = json.loads((output/"excited_ad.json").read_text())
if not native["complete"]:
    raise RuntimeError("Complete the native excited-state Hessian gate first.")
started = perf_counter()
coordinates = np.asarray(native["coordinates_bohr"])
evaluate = lambda r: evaluate_cis(native["symbols"], r, native["basis"],
                                 root=native["selected_root"], nroots=native["nroots"])
reference = evaluate(coordinates)
checks = []
step_results = []
for step in (3e-4, 1e-4):
    columns = []
    for index, direction in enumerate(np.eye(coordinates.size).reshape(-1, *coordinates.shape)):
        gradients = []
        for sign in (1, -1):
            point = evaluate(coordinates+sign*step*direction)
            if abs(point["excitation"]-reference["excitation"]) > reference["gap"]/4:
                raise RuntimeError("The independent local state identity is unresolved.")
            gradients.append(point["gradient"])
            checks.append(dict(step_bohr=step, coordinate=index, sign=sign,
                               excitation_hartree=point["excitation"], gap_hartree=point["gap"],
                               eigen_residual=point["max_eigen_residual"],
                               scf_orbital_gradient_norm=point["scf_orbital_gradient_norm"],
                               z_residual_norm=point["z_residual_norm"]))
        columns.append(((gradients[0]-gradients[1])/(2*step)).ravel())
    hessian = np.column_stack(columns)
    error = float(np.max(np.abs(hessian-native["hessian_hartree_per_bohr2"])))
    asymmetry = float(np.max(np.abs(hessian-hessian.T)))
    step_results.append(dict(step_bohr=step, max_hessian_error=error,
                             raw_hessian_asymmetry=asymmetry, hessian_hartree_per_bohr2=hessian.tolist()))
    print(f"Independent gradient FD {step:.1e} Bohr: AD Hessian error {error:.3e} Eh/Bohr^2", flush=True)

mol = gto.M(atom=list(zip(native["symbols"], coordinates)), basis=native["basis"],
            unit="Bohr", cart=True, verbose=0)
analysis = thermo.harmonic_analysis(
    mol, hessian.reshape(len(coordinates), 3, len(coordinates), 3).transpose(0, 2, 1, 3),
    mass=np.asarray(native["masses_amu"]))
frequencies = np.real_if_close(analysis["freq_wavenumber"])
frequency_valid = not np.iscomplexobj(frequencies) and np.all(frequencies > 0)
errors = dict(ground_energy_hartree=abs(reference["ground_energy"]-native["ground_energy_hartree"]),
              excitation_hartree=abs(reference["excitation"]-native["excitation_hartree"]),
              total_energy_hartree=abs(reference["total_energy"]-native["total_excited_energy_hartree"]),
              gradient_hartree_per_bohr=float(np.max(np.abs(reference["gradient"]-native["gradient_hartree_per_bohr"]))),
              hessian_hartree_per_bohr2=step_results[-1]["max_hessian_error"],
              frequencies_cm1=float(np.max(np.abs(frequencies-native["frequencies_cm1"]))))
tolerances = dict(ground_energy_hartree=1e-8, excitation_hartree=1e-7, total_energy_hartree=1e-7,
                  gradient_hartree_per_bohr=1e-6, hessian_hartree_per_bohr2=2e-6, frequencies_cm1=.02)
passed = (frequency_valid and all(errors[key] <= value for key, value in tolerances.items())
          and step_results[-1]["raw_hessian_asymmetry"] <= 2e-6)
report = dict(
    passed=bool(passed), reference="Independent PySCF/SciPy dense CIS; FD of residual-checked analytic gradients",
    ground_energy_hartree=reference["ground_energy"], excitation_hartree=reference["excitation"],
    total_excited_energy_hartree=reference["total_energy"], gradient_hartree_per_bohr=reference["gradient"].tolist(),
    hessian_hartree_per_bohr2=hessian.tolist(), frequencies_cm1=np.real(frequencies).tolist(),
    frequency_imaginary_parts_cm1=np.imag(frequencies).tolist(), frequency_valid=bool(frequency_valid),
    errors=errors, tolerances=tolerances, finite_difference_checks=step_results, displaced_checks=checks,
    cphf_tolerance=reference["cphf_tolerance"], krylov_linear_dependence_cutoff=reference["krylov_linear_dependence_cutoff"],
    z_residual_tolerance=reference["z_residual_tolerance"],
    maximum_z_residual=max([reference["z_residual_norm"]]+[item["z_residual_norm"] for item in checks]),
    elapsed_seconds=perf_counter()-started, pyscf=pyscf.__version__, hostname=platform.node(),
    native_sha256=hashlib.sha256((output/"excited_ad.json").read_bytes()).hexdigest(),
    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    reference_helper_sha256=hashlib.sha256(Path(__file__).with_name("pyscf_cis_reference.py").read_bytes()).hexdigest(),
)
(output/"excited_pyscf.json").write_text(json.dumps(report, indent=2)+"\n")
if not passed:
    raise RuntimeError(f"Independent full excited-state Hessian gate failed: {errors}")
print(f"Independent full Hessian/frequency gates passed in {report['elapsed_seconds']:.2f} s", flush=True)

# Measured 2026-10-15 on jy_gpu_1, PySCF 2.13.0: 49 points, 58.23 s.
# At 1e-4 Bohr: max Hessian error 1.166e-8 Eh/Bohr^2; frequency error 1.583e-5 cm^-1.
# Max true CPHF Z residual 2.245e-11 Eh; gradient error 4.241e-13 Eh/Bohr.
