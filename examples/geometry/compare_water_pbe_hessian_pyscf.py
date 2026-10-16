"""Independent finite differences of full-grid PySCF PBE analytic gradients.

The reference Hessian here is a central difference, not a PySCF analytic
Hessian. Every displaced gradient includes grid motion and Becke response,
so it differentiates the same finite quadrature as the complete GradSCF AD.
"""
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pyscf
from pyscf import dft, gto
from pyscf.dft import gen_grid, radi
from pyscf.hessian import thermo

output = Path("reproducibility/ad_showcase/G05_water_pbe_hessian")
data = json.loads((output / "gradscf.json").read_text())
coordinates = np.asarray(data["coordinates_bohr"])


def calculate(coords):
    mol = gto.M(atom=list(zip(data["symbols"], coords)), basis=data["basis"],
                unit="Bohr", cart=data["cart"], verbose=0)
    mf = dft.RKS(mol)
    mf.xc = "pbe"
    mf.conv_tol, mf.conv_tol_grad, mf.max_cycle = 1e-13, 1e-11, 150
    mf.small_rho_cutoff = 0
    mf.grids.level, mf.grids.alignment = data["grid_level"], 1
    mf.grids.radi_method, mf.grids.prune = radi.treutler_ahlrichs, gen_grid.nwchem_prune
    mf.grids.radii_adjust, mf.grids.atomic_radii = radi.treutler_atomic_radii_adjust, radi.BRAGG_RADII
    mf.grids.becke_scheme = gen_grid.original_becke
    mf.kernel()
    if not mf.converged:
        raise RuntimeError("Independent PySCF PBE SCF did not converge.")
    derivative = mf.nuc_grad_method()
    derivative.grid_response = True
    gradient = derivative.kernel()
    if not np.isfinite(gradient).all():
        raise RuntimeError("Independent PySCF gradient is nonfinite.")
    return mol, mf.e_tot, gradient


start = perf_counter()
mol, value, gradient = calculate(coordinates)
np.testing.assert_allclose(value, data["energy_hartree"], atol=1e-8, rtol=0)
assert np.max(np.abs(gradient)) < 1e-7
directions = np.eye(coordinates.size).reshape(-1, *coordinates.shape)
checks = []
for step in (3e-4, 1e-4):
    columns = []
    for direction in directions:
        _, _, plus = calculate(coordinates + step*direction)
        _, _, minus = calculate(coordinates - step*direction)
        columns.append(((plus-minus)/(2*step)).ravel())
    hessian = np.column_stack(columns)
    error = float(np.max(np.abs(hessian - np.asarray(data["hessian_hartree_per_bohr2"]))))
    checks.append(dict(step_bohr=step, max_hessian_error=error,
                       hessian_hartree_per_bohr2=hessian.tolist()))
    print(f"PySCF full-grid gradient FD step={step:.1e}: "
          f"max AD-Hessian error={error:.3e} Eh/Bohr^2", flush=True)
np.testing.assert_allclose(hessian, data["hessian_hartree_per_bohr2"], atol=2e-6, rtol=0)
isotopes = {}
for label, isotope in data["isotopes"].items():
    analysis = thermo.harmonic_analysis(
        mol, hessian.reshape(len(coordinates), 3, len(coordinates), 3).transpose(0, 2, 1, 3),
        mass=np.asarray(isotope["masses_amu"]))
    frequencies = np.real_if_close(analysis["freq_wavenumber"])
    assert not np.iscomplexobj(frequencies)
    np.testing.assert_allclose(frequencies, isotope["frequencies_cm1"], atol=.02, rtol=0)
    isotopes[label] = dict(frequencies_cm1=frequencies.tolist(),
                          max_frequency_error_cm1=float(np.max(np.abs(
                              frequencies - isotope["frequencies_cm1"]))))
    print(label, "PySCF FD-Hessian frequencies / cm^-1:", frequencies, flush=True)
translations = np.tile(np.eye(3), (len(coordinates), 1))
centered = coordinates - np.mean(coordinates, axis=0)
rotations = np.column_stack([np.cross(centered, axis).ravel() for axis in np.eye(3)])
reference = dict(
    pyscf_version=pyscf.__version__, energy_hartree=float(value),
    gradient_hartree_per_bohr=gradient.tolist(), hessian_hartree_per_bohr2=hessian.tolist(),
    max_hessian_error=checks[-1]["max_hessian_error"], finite_difference_checks=checks,
    reference_method="central differences of full-grid PySCF analytic gradients",
    isotopes=isotopes, grid_level=data["grid_level"], grid_response=True,
    conv_tol=1e-13, conv_tol_grad=1e-11, elapsed_seconds=perf_counter()-start,
    raw_hessian_asymmetry=float(np.max(np.abs(hessian-hessian.T))),
    raw_translation_residual_hartree_per_bohr2=float(np.max(np.abs(hessian @ translations))),
    raw_rotation_residual_hartree_per_bohr=float(np.max(np.abs(hessian @ rotations))),
)
(output / "pyscf.json").write_text(json.dumps(reference, indent=2) + "\n")

# Measured on jy_gpu_1 (CPU, float64, 2026-10-15):
# PySCF full-grid gradient FD step=3.0e-04: max AD-Hessian error=3.290e-08 Eh/Bohr^2
# PySCF full-grid gradient FD step=1.0e-04: max AD-Hessian error=1.009e-08 Eh/Bohr^2
# H2O PySCF FD-Hessian frequencies / cm^-1: [1683.33616733 3625.11503050 3751.50855891]
# D2O PySCF FD-Hessian frequencies / cm^-1: [1231.62799030 2614.08086135 2746.75573849]
# This reference is a finite difference of analytic gradients, not an analytic Hessian.
