"""Independent PySCF finite-field/mode differences for the cases below.

The saved normal directions come from independently checked Hessians. Every
electronic observable is recomputed with PySCF; no GradSCF solver is imported.
"""
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pyscf
from pyscf import dft, gto, scf
from pyscf.dft import gen_grid, radi
from scipy.constants import Avogadro, elementary_charge, epsilon_0, speed_of_light, electron_mass, physical_constants

RESULTS = Path("reproducibility/ad_showcase")
CASES = (RESULTS / "V01_V02_water_rhf_spectra", RESULTS / "V01_V02_water_pbe_spectra")
# run(RESULTS / "V03_co2_selection_rules") also uses this same reference code.
# The CO2 calculation also records the zero-field density initial guess;
# a default-guess near-stationary DIIS failure is retained in the run notes.


def run(case_path, output=None):
    case_path = Path(case_path)
    data = json.loads((case_path / "gradscf.json").read_text())
    coordinates = np.asarray(data["coordinates_bohr"])
    xc = data.get("xc", "HF").upper()
    if xc not in ("HF", "PBE", "PBE0", "B3LYP", "WB97XD"):
        raise ValueError("Choose HF, PBE, PBE0, B3LYP or original WB97XD.")
    scf_retries = []
    gradient_norms = []

    def electric_properties(coords):
        mol = gto.M(atom=list(zip(data["symbols"], coords)), basis=data["basis"],
                    unit="Bohr", cart=data["cart"], verbose=0)
        operators = mol.intor_symmetric("int1e_r", comp=3)
        hcore = scf.RHF(mol).get_hcore()
        nuclear = mol.atom_charges() @ mol.atom_coords()

        def dipole(field, density=None):
            mf = scf.RHF(mol) if xc == "HF" else dft.RKS(mol)
            mf.conv_tol, mf.conv_tol_grad, mf.max_cycle = 1e-13, 1e-11, 150
            if xc != "HF":
                mf.xc, mf.small_rho_cutoff = ("b3lypg" if xc == "B3LYP" else xc.lower()), 0
                # Original CHG is geometry-only: it changes the optimized
                # geometry/modes, but contributes no fixed-R field derivative.
                mf.grids.level, mf.grids.alignment = data["grid_level"], 1
                mf.grids.radi_method, mf.grids.prune = radi.treutler_ahlrichs, gen_grid.nwchem_prune
                mf.grids.radii_adjust, mf.grids.atomic_radii = radi.treutler_atomic_radii_adjust, radi.BRAGG_RADII
                mf.grids.becke_scheme = gen_grid.original_becke
            mf.get_hcore = lambda *args: hcore + np.einsum("x,xpq->pq", field, operators)
            try:
                mf.kernel(dm0=density)
            except np.linalg.LinAlgError as error:
                # A near-stationary CO2 DIIS eigensolve can fail. Retry once
                # with PySCF's Newton solver, retaining Hamiltonian and tolerances.
                density = None
                if mf.mo_coeff is not None and mf.mo_occ is not None:
                    candidate = mf.make_rdm1()
                    if np.isfinite(candidate).all():
                        density = candidate
                retry = dict(case=case_path.name, mode=index+1, field_au=field.tolist(),
                             coordinates_bohr=coords.tolist(), reason=str(error),
                             initial_guess="finite density" if density is not None else "PySCF default",
                             converged=False)
                scf_retries.append(retry)
                print("PySCF DIIS failure; one Newton retry:", retry, flush=True)
                mf = mf.newton()
                mf.kernel(dm0=density)
                retry["converged"] = bool(mf.converged)
                print("PySCF Newton converged:", retry["converged"], flush=True)
            if not mf.converged:
                raise RuntimeError("An independent displaced/finite-field SCF failed.")
            density = mf.make_rdm1()
            fock = mf.get_hcore() + mf.get_veff(dm=density)
            gradient_norm = float(np.linalg.norm(mf.get_grad(mf.mo_coeff, mf.mo_occ, fock)))
            if not np.isfinite(gradient_norm) or gradient_norm > 1e-10:
                raise RuntimeError(f"Independent SCF orbital residual {gradient_norm:.3e} is too large.")
            gradient_norms.append(gradient_norm)
            return nuclear - np.einsum("pq,xqp->x", density, operators), density

        field_step = 1e-4
        columns = []
        for axis in np.eye(3):
            mu_plus, dm_plus = dipole(field_step*axis)
            mu_minus, dm_minus = dipole(-field_step*axis)
            columns.append((mu_plus-mu_minus)/(2*field_step))
        # The mean of an already converged +/- field pair is an O(F^2)
        # zero-field initial guess. Reoptimize it under the same strict
        # thresholds; neither its density nor its energy is accepted directly.
        mu_zero, _ = dipole(np.zeros(3), density=.5*(dm_plus+dm_minus))
        return mu_zero, np.column_stack(columns)

    start = perf_counter()
    checks = []
    for index, direction in enumerate(np.asarray(data["displacements"])):
        for maximum_displacement in (1e-3, 3e-4):
            step = maximum_displacement / np.max(np.abs(direction))
            mu_plus, alpha_plus = electric_properties(coordinates + step*direction)
            mu_minus, alpha_minus = electric_properties(coordinates - step*direction)
            dmu, dalpha = (mu_plus-mu_minus)/(2*step), (alpha_plus-alpha_minus)/(2*step)
            error_mu = float(np.max(np.abs(dmu-data["dipole_derivatives"][index])))
            error_alpha = float(np.max(np.abs(dalpha-data["alpha_derivatives"][index])))
            checks.append(dict(mode=index+1, max_displacement_bohr=maximum_displacement,
                               dipole_derivative=dmu.tolist(), alpha_derivative=dalpha.tolist(),
                               max_dipole_error=error_mu, max_alpha_error=error_alpha))
        np.testing.assert_allclose(dmu, data["dipole_derivatives"][index], atol=1e-6, rtol=1e-4)
        np.testing.assert_allclose(dalpha, data["alpha_derivatives"][index], atol=3e-5, rtol=3e-4)
        # Independent SI IR conversion for Q in Bohr*sqrt(electron mass).
        ir_si = (Avogadro * elementary_charge**2 / (12*epsilon_0*electron_mass*speed_of_light**2)
                 * 1e-3 * np.dot(dmu, dmu))
        np.testing.assert_allclose(ir_si, data["ir_intensities_km_mol"][index], atol=1e-3, rtol=1e-4)
        average = np.trace(dalpha) / 3
        gamma2 = .5*((dalpha[0,0]-dalpha[1,1])**2 + (dalpha[1,1]-dalpha[2,2])**2
                     + (dalpha[2,2]-dalpha[0,0])**2
                     + 6*(dalpha[0,1]**2+dalpha[0,2]**2+dalpha[1,2]**2))
        conversion = (physical_constants["Bohr radius"][0]*1e10)**4 * physical_constants["atomic mass constant"][0]/electron_mass
        raman = (45*average**2 + 7*gamma2) * conversion
        np.testing.assert_allclose(raman, data["raman_activities_angstrom4_amu"][index], atol=1e-3, rtol=3e-4)
        checks[-1].update(ir_intensity_km_mol=float(ir_si), raman_activity_angstrom4_amu=float(raman))
        print(f"{case_path.name}, mode {index+1}: PySCF dmu error={error_mu:.3e}, dalpha error={error_alpha:.3e}", flush=True)
    output = case_path if output is None else Path(output)
    output.mkdir(parents=True, exist_ok=True)
    reference = dict(pyscf_version=pyscf.__version__, xc=xc, grid_level=data.get("grid_level"),
                     field_step_au=1e-4, mode_checks=checks, scf_retries=scf_retries,
                     zero_field_initial_guess="mean of converged +/- z-field densities; reconverged",
                     max_scf_gradient_norm=max(gradient_norms),
                     elapsed_seconds=perf_counter()-start)
    (output / "pyscf.json").write_text(json.dumps(reference, indent=2) + "\n")
    return reference


if __name__ == "__main__":
    for case_path in CASES:
        run(case_path)

# RHF water original reference run, jy_gpu_1, CPU float64, 2026-10-15:
# Mode 1: PySCF dmu error=1.141e-10, dalpha error=6.229e-08
# Mode 2: PySCF dmu error=7.853e-11, dalpha error=6.827e-08
# Mode 3: PySCF dmu error=2.649e-10, dalpha error=4.235e-08

# PBE water measured on jy_gpu_1, CPU float64, 2026-10-15:
# At max Cartesian displacement 3e-4 Bohr and inner field step 1e-4 au:
# Mode 1: PySCF dmu error=1.646e-10, dalpha error=1.826e-6
# Mode 2: PySCF dmu error=5.036e-10, dalpha error=8.153e-7
# Mode 3: PySCF dmu error=4.117e-10, dalpha error=6.409e-7
# Maximum actual SCF orbital-gradient norm 1.760e-11; no Newton retries.
# Reference runtime 111.337 s. Smaller mode steps amplify finite-field noise;
# these nested finite differences are a numerical oracle, not analytic third AD.

# M2 references on jy_gpu_1, 2026-10-15: PBE0/B3LYP(VWN-RPA)/original WB97XD.
# Max dalpha errors: 7.411e-7 / 1.199e-6 / 2.379e-7 (same steps as above).
# Each used 84 SCFs with no Newton retries; elapsed 116.948 / 126.340 / 144.166 s.
# Largest actual orbital norms: 1.283e-11 / 1.420e-11 / 9.864e-12.
