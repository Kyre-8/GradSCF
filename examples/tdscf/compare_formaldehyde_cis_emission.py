"""Independent RHF minimum/Hessian and paired CIS energy bookkeeping for X10."""
import hashlib
import json
from pathlib import Path
import platform
from time import perf_counter

import numpy as np
import pyscf
from pyscf import gto, scf
from pyscf.hessian import thermo
from scipy.constants import physical_constants

from pyscf_cis_reference import evaluate_cis


output = Path("reproducibility/ad_showcase/X10_formaldehyde_cis_emission")
native = json.loads((output/"emission.json").read_text())
excited_native = json.loads((output/"excited_ad.json").read_text())
excited_reference = json.loads((output/"excited_pyscf.json").read_text())
excited_hash = hashlib.sha256((output/"excited_ad.json").read_bytes()).hexdigest()
if (not native["complete"] or not excited_reference["passed"]
        or native["excited_native_sha256"] != excited_hash
        or excited_reference["native_sha256"] != excited_hash):
    raise RuntimeError("Complete and independently validate the exact linked native results first.")
started = perf_counter()
coordinates = np.asarray(native["ground_coordinates_bohr"])
mol = gto.M(atom=list(zip(native["symbols"], coordinates)), basis=native["basis"],
            unit="Bohr", cart=True, verbose=0)
mf = scf.RHF(mol)
mf.conv_tol, mf.conv_tol_grad, mf.max_cycle = 1e-12, 1e-12, 180
mf.kernel()
if not mf.converged:
    raise RuntimeError("Independent ground RHF failed to converge.")
gradient = mf.nuc_grad_method().kernel()
raw_hessian = mf.Hessian().kernel()
hessian = raw_hessian.transpose(0, 2, 1, 3).reshape(coordinates.size, coordinates.size)
analysis = thermo.harmonic_analysis(mol, raw_hessian, mass=np.asarray(native["masses_amu"]))
frequencies = np.real_if_close(analysis["freq_wavenumber"])
if np.iscomplexobj(frequencies) or np.any(frequencies <= 0):
    raise RuntimeError("The independent ground Hessian is not a harmonic minimum.")
absorption = evaluate_cis(native["symbols"], coordinates, native["basis"],
                          root=excited_native["selected_root"], nroots=excited_native["nroots"],
                          with_gradient=False)
e0g = float(mf.e_tot)
e1g = e0g+absorption["excitation"]
e0e = excited_reference["ground_energy_hartree"]
e1e = excited_reference["total_excited_energy_hartree"]
hartree_to_wavenumber = physical_constants["hartree-inverse meter relationship"][0]/100
ground_zpe = float(.5*np.sum(frequencies)/hartree_to_wavenumber)
excited_zpe = float(.5*np.sum(excited_reference["frequencies_cm1"])/hartree_to_wavenumber)
energies = dict(ground_at_ground=e0g, excited_at_ground=e1g,
                ground_at_excited=e0e, excited_at_excited=e1e)
gaps = dict(vertical_absorption=e1g-e0g, vertical_emission=e1e-e0e,
            adiabatic_electronic=e1e-e0g, harmonic_zero_zero=e1e-e0g+excited_zpe-ground_zpe,
            excited_relaxation=e1g-e1e, ground_reorganization=e0e-e0g,
            stokes_shift=(e1g-e0g)-(e1e-e0e))
errors = dict(
    ground_energy_hartree=abs(e0g-native["electronic_energies_hartree"]["ground_at_ground"]),
    ground_gradient_hartree_per_bohr=float(np.max(np.abs(gradient-native["ground_gradient_hartree_per_bohr"]))),
    ground_hessian_hartree_per_bohr2=float(np.max(np.abs(hessian-native["ground_hessian_hartree_per_bohr2"]))),
    ground_frequencies_cm1=float(np.max(np.abs(frequencies-native["ground_frequencies_cm1"]))),
    electronic_gaps_hartree=max(abs(gaps[key]-native["gaps_hartree"][key])
                                for key in gaps if key != "harmonic_zero_zero"),
    harmonic_zero_zero_hartree=abs(gaps["harmonic_zero_zero"]-native["gaps_hartree"]["harmonic_zero_zero"]),
)
tolerances = dict(ground_energy_hartree=1e-8, ground_gradient_hartree_per_bohr=1e-7,
                  ground_hessian_hartree_per_bohr2=2e-6, ground_frequencies_cm1=.02,
                  electronic_gaps_hartree=1e-7, harmonic_zero_zero_hartree=1e-6)
passed = (all(errors[key] <= value for key, value in tolerances.items())
          and np.max(np.abs(gradient)) <= 1e-7
          and np.max(np.abs(hessian-hessian.T)) <= 2e-6)
report = dict(
    passed=bool(passed), reference="Independent PySCF analytic RHF Hessian and dense CIS vertical roots",
    electronic_energies_hartree=energies, gaps_hartree=gaps,
    ground_gradient_hartree_per_bohr=gradient.tolist(), ground_hessian_hartree_per_bohr2=hessian.tolist(),
    ground_frequencies_cm1=frequencies.tolist(), ground_harmonic_zpe_hartree=ground_zpe,
    excited_harmonic_zpe_hartree=excited_zpe, ground_max_cartesian_gradient=float(np.max(np.abs(gradient))),
    ground_raw_hessian_asymmetry=float(np.max(np.abs(hessian-hessian.T))),
    errors=errors, tolerances=tolerances, elapsed_seconds=perf_counter()-started,
    pyscf=pyscf.__version__, hostname=platform.node(),
    native_sha256=hashlib.sha256((output/"emission.json").read_bytes()).hexdigest(),
    excited_reference_sha256=hashlib.sha256((output/"excited_pyscf.json").read_bytes()).hexdigest(),
    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    reference_helper_sha256=hashlib.sha256(Path(__file__).with_name("pyscf_cis_reference.py").read_bytes()).hexdigest(),
)
(output/"emission_pyscf.json").write_text(json.dumps(report, indent=2)+"\n")
if not passed:
    raise RuntimeError(f"Independent paired-minimum gate failed: {errors}")
print("Independent paired-minimum errors:", errors, flush=True)
print(f"Independent paired-minimum gates passed in {report['elapsed_seconds']:.2f} s", flush=True)

# Measured 2026-10-15 on jy_gpu_1, PySCF 2.13.0: all gates passed in 5.52 s.
# Ground Hessian max error 1.786e-8 Eh/Bohr^2; frequency error 2.405e-5 cm^-1.
# Electronic-gap max error 1.847e-13 Eh; harmonic 0-0 error 1.801e-11 Eh.
