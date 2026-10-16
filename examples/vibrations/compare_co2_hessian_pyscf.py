"""Independent analytic RHF Hessian and isotope-specific CO2 frequencies."""
import json
from pathlib import Path

import numpy as np
import pyscf
from pyscf import gto, scf
from pyscf.hessian import thermo

output = Path("reproducibility/ad_showcase/V03_co2_selection_rules")
data = json.loads((output / "gradscf.json").read_text())
mol = gto.M(atom=list(zip(data["symbols"], data["coordinates_bohr"])),
            basis=data["basis"], cart=True, unit="Bohr", verbose=0)
mf = scf.RHF(mol)
mf.conv_tol, mf.conv_tol_grad, mf.max_cycle = 1e-12, 1e-10, 150
mf.kernel()
assert mf.converged
gradient = mf.nuc_grad_method().kernel()
hessian = mf.Hessian().kernel()
matrix = hessian.transpose(0, 2, 1, 3).reshape(9, 9)
frequencies = thermo.harmonic_analysis(mol, hessian,
    mass=np.array([12., 15.99491461957, 15.99491461957]))["freq_wavenumber"]
assert np.max(np.abs(np.imag(frequencies))) < 1e-9
np.testing.assert_allclose(matrix, data["hessian_hartree_per_bohr2"], atol=2e-6, rtol=0)
np.testing.assert_allclose(frequencies.real, data["frequencies_cm1"], atol=1e-3, rtol=0)
assert np.max(np.abs(gradient)) < 1e-7
reference = dict(pyscf_version=pyscf.__version__, gradient=gradient.tolist(),
                 hessian_hartree_per_bohr2=matrix.tolist(), frequencies_cm1=frequencies.real.tolist(),
                 max_hessian_error=float(np.max(np.abs(matrix-data["hessian_hartree_per_bohr2"]))),
                 max_frequency_error=float(np.max(np.abs(frequencies.real-data["frequencies_cm1"]))))
(output / "hessian_reference.json").write_text(json.dumps(reference, indent=2)+"\n")
print("Max Hessian error / (Eh/Bohr^2):", reference["max_hessian_error"])
print("Max frequency error / cm^-1:", reference["max_frequency_error"])
