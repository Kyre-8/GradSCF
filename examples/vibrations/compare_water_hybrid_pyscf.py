"""Independent hybrid water geometry/Hessian and finite-field spectra checks.

PySCF's raw wb97xd spelling supplies the electronic functional. Original CHG
is added independently below using radial pair derivatives, not GradSCF/JAX.
The SCF Hessian, including nuclear repulsion, is a difference of full-grid
analytic gradients. The original CHG correction is added exactly once.
"""
import json
from pathlib import Path
import runpy
from time import perf_counter

import numpy as np
import pyscf
from pyscf import dft, gto
from pyscf.dft import gen_grid, radi
from pyscf.hessian import thermo
from scipy.constants import Avogadro, physical_constants


XC = "pbe0"
METHODS = ("pbe0", "b3lyp", "wb97xd")


def chg_correction(symbols, coordinates):
    """Original CHG E, Cartesian gradient and Hessian; water H/O parameters.

    Table C6 is J nm^6/mol and R0 is Angstrom. For a pair, phi(r) equals
    -C*r^6/(r^12+a), a=6*(R0_i+R0_j)^12. Differentiate this radial function
    analytically, then use K=(phi''-phi'/r) uu.T+(phi'/r) I.
    """
    coordinates = np.asarray(coordinates, dtype=float)
    table = {"H": (.14, 1.001), "O": (.70, 1.342)}
    bohr_m = physical_constants["Bohr radius"][0]
    hartree_j = physical_constants["Hartree energy"][0]
    c6_factor = (1e-9/bohr_m)**6/(Avogadro*hartree_j)
    energy, gradient = 0., np.zeros_like(coordinates)
    hessian = np.zeros((coordinates.size, coordinates.size))
    for i in range(len(symbols)):
        for j in range(i):
            delta = coordinates[i]-coordinates[j]
            r = np.linalg.norm(delta)
            if r == 0:
                continue  # phi is O(r^6), so its first two derivatives vanish.
            ci, ri = table[symbols[i]]
            cj, rj = table[symbols[j]]
            c = np.sqrt(ci*cj)*c6_factor
            a = 6*((ri+rj)*1e-10/bohr_m)**12
            t, denominator = r**12, r**12+a
            energy -= c*r**6/denominator
            first = 6*c*r**5*(t-a)/denominator**2
            second = 6*c*r**4*(-7*t*t+36*a*t-5*a*a)/denominator**3
            unit = delta/r
            gradient[i] += first*unit
            gradient[j] -= first*unit
            block = (second-first/r)*np.outer(unit, unit)+(first/r)*np.eye(3)
            si, sj = slice(3*i, 3*i+3), slice(3*j, 3*j+3)
            hessian[si, si] += block
            hessian[sj, sj] += block
            hessian[si, sj] -= block
            hessian[sj, si] -= block
    return float(energy), gradient, hessian


def geometry_reference(xc):
    if xc not in METHODS:
        raise ValueError(f"Choose one of {METHODS}.")
    output = Path("reproducibility/ad_showcase")/f"M2_water_{xc}_spectra"
    data = json.loads((output/"hessian.json").read_text())
    coordinates = np.asarray(data["coordinates_bohr"])
    gradient_norms = []

    def calculate(coords):
        mol = gto.M(atom=list(zip(data["symbols"], coords)), basis=data["basis"],
                    unit="Bohr", cart=True, verbose=0)
        mf = dft.RKS(mol)
        mf.xc = "b3lypg" if xc == "b3lyp" else xc
        mf.disp = None
        mf.conv_tol, mf.conv_tol_grad, mf.max_cycle = 1e-13, 1e-11, 200
        mf.small_rho_cutoff = 0
        mf.grids.level, mf.grids.alignment = data["grid_level"], 1
        mf.grids.radi_method, mf.grids.prune = radi.treutler_ahlrichs, gen_grid.nwchem_prune
        mf.grids.radii_adjust, mf.grids.atomic_radii = radi.treutler_atomic_radii_adjust, radi.BRAGG_RADII
        mf.grids.becke_scheme = gen_grid.original_becke
        mf.kernel()
        if not mf.converged:
            raise RuntimeError(f"Independent {xc} SCF failed.")
        dm = mf.make_rdm1()
        fock = mf.get_hcore()+mf.get_veff(dm=dm)
        norm = float(np.linalg.norm(mf.get_grad(mf.mo_coeff, mf.mo_occ, fock)))
        if not np.isfinite(norm) or norm > 1e-10:
            raise RuntimeError(f"Independent SCF orbital residual {norm:.3e} is too large.")
        gradient_norms.append(norm)
        derivative = mf.nuc_grad_method()
        derivative.grid_response = True
        gradient = derivative.kernel()
        if not np.isfinite(gradient).all():
            raise RuntimeError("Independent analytic gradient is nonfinite.")
        return mol, float(mf.e_tot), gradient

    start = perf_counter()
    mol, value, scf_gradient = calculate(coordinates)
    correction = (chg_correction(data["symbols"], coordinates) if xc == "wb97xd" else
                  (0., np.zeros_like(coordinates), np.zeros((coordinates.size, coordinates.size))))
    chg_energy, chg_gradient, chg_hessian = correction
    value, gradient = value+chg_energy, scf_gradient+chg_gradient
    np.testing.assert_allclose(value, data["energy_hartree"], atol=1e-8, rtol=0)
    np.testing.assert_allclose(gradient, data["gradient_hartree_per_bohr"], atol=1e-6, rtol=0)
    assert np.max(np.abs(gradient)) < 2e-7
    checks = []
    for step in (3e-4, 1e-4):
        columns = []
        for direction in np.eye(coordinates.size).reshape(-1, *coordinates.shape):
            plus = calculate(coordinates+step*direction)[2]
            minus = calculate(coordinates-step*direction)[2]
            columns.append(((plus-minus)/(2*step)).ravel())
        hessian = np.column_stack(columns)+chg_hessian
        error = float(np.max(np.abs(hessian-np.asarray(data["hessian_hartree_per_bohr2"]))))
        checks.append(dict(step_bohr=step, max_hessian_error=error,
                           hessian_hartree_per_bohr2=hessian.tolist()))
        print(xc, "PySCF full-grid Hessian FD", step, error, flush=True)
    np.testing.assert_allclose(hessian, data["hessian_hartree_per_bohr2"], atol=2e-6, rtol=0)
    isotopes = {}
    for label, isotope in data["isotopes"].items():
        analysis = thermo.harmonic_analysis(mol, hessian.reshape(3, 3, 3, 3).transpose(0, 2, 1, 3),
                                            mass=np.asarray(isotope["masses_amu"]))
        frequencies = np.real_if_close(analysis["freq_wavenumber"])
        assert not np.iscomplexobj(frequencies)
        np.testing.assert_allclose(frequencies, isotope["frequencies_cm1"], atol=.02, rtol=0)
        isotopes[label] = dict(frequencies_cm1=frequencies.tolist(), max_frequency_error_cm1=float(
            np.max(np.abs(frequencies-isotope["frequencies_cm1"]))))
    centered = coordinates-coordinates.mean(axis=0)
    translations = np.tile(np.eye(3), (3, 1))
    rotations = np.column_stack([np.cross(centered, axis).ravel() for axis in np.eye(3)])
    reference = dict(xc=xc, pyscf_xc="b3lypg" if xc == "b3lyp" else xc,
                     pyscf_version=pyscf.__version__, energy_hartree=value,
                     gradient_hartree_per_bohr=gradient.tolist(),
                     max_gradient_error=float(np.max(np.abs(gradient-data["gradient_hartree_per_bohr"]))),
                     hessian_hartree_per_bohr2=hessian.tolist(), finite_difference_checks=checks,
                     isotopes=isotopes, grid_level=data["grid_level"], grid_response=True,
                     chg_energy_hartree=chg_energy, chg_gradient_hartree_per_bohr=chg_gradient.tolist(),
                     chg_hessian_hartree_per_bohr2=chg_hessian.tolist(),
                     reference_method="FD of PySCF full-grid gradients; independent analytic CHG added for WB97XD",
                     dispersion="original CHG" if xc == "wb97xd" else "none",
                     scf_evaluations=len(gradient_norms),
                     max_scf_gradient_norm=max(gradient_norms), conv_tol=1e-13, conv_tol_grad=1e-11,
                     raw_hessian_asymmetry=float(np.max(np.abs(hessian-hessian.T))),
                     raw_translation_residual_hartree_per_bohr2=float(np.max(np.abs(hessian@translations))),
                     raw_rotation_residual_hartree_per_bohr=float(np.max(np.abs(hessian@rotations))),
                     elapsed_seconds=perf_counter()-start)
    (output/"pyscf_geometry.json").write_text(json.dumps(reference, indent=2)+"\n")
    return reference


def run(xc):
    geometry_reference(xc)
    compare = runpy.run_path(str(Path(__file__).with_name("compare_water_spectra_pyscf.py")))["run"]
    return compare(Path("reproducibility/ad_showcase")/f"M2_water_{xc}_spectra")


if __name__ == "__main__":
    run(XC)

# Measured on jy_gpu_1/c20, PySCF 2.13.0, CPU float64, 2026-10-15.
# PBE0 / B3LYP(VWN-RPA) / original WB97XD+CHG, respectively:
# Gradient max errors / Eh Bohr^-1: 6.050e-13 / 2.010e-13 / 1.905e-13.
# Hessian max errors at 1e-4 Bohr / Eh Bohr^-2: 5.429e-9 / 1.120e-8 / 5.298e-9.
# Geometry references: 37 SCFs each; 56.085 / 66.021 / 76.349 seconds.
# Raw WB97XD rotation residual: 5.262999e-4 Eh/Bohr (AD 5.262983e-4).
# The reference adds independent analytic CHG exactly once; E_CHG=-2.743819e-5 Eh.
# Spectral dmu max errors: 3.580e-10 / 3.986e-10 / 3.087e-10.
# Spectral dalpha max errors: 7.411e-7 / 1.199e-6 / 2.379e-7.
# Spectral references used 84 SCFs each, no Newton retries, and full reconvergence.
