"""Independent PySCF reference for the saved formaldehyde absorption example.

PySCF rebuilds integrals, grids, SCF and A/B matrices from saved physical
inputs. SciPy's existing dense eigensolvers provide a bounded independent
root-coverage oracle, not a claim of PySCF iterative TD convergence. No
GradSCF arrays are reused. Dipole outer products remove state-sign choices.
"""
import csv
import hashlib
import json
from pathlib import Path
import platform
from time import perf_counter

import numpy as np
import pyscf
from pyscf import dft, gto, scf
from pyscf.dft import gen_grid, radi
import scipy
from scipy import linalg
from scipy.constants import Avogadro, physical_constants


def original_chg_dispersion(molecule):
    """Independent original CHG pair formula in tabulated SI/mol units.

    C6 (J nm^6/mol) and R0 (Angstrom): Grimme, JCC 27, 1787 (2006).
    Damping 6 and exponent 12: Chai--Head-Gordon, PCCP 10, 6615 (2008).
    """
    parameters = {"H": (.14, 1.001), "C": (1.75, 1.452), "O": (.70, 1.342)}
    coordinates = molecule.atom_coords()*physical_constants["Bohr radius"][0]*1e9
    energy_per_mole = 0.
    for i in range(molecule.natm):
        c_i, r_i = parameters[molecule.atom_symbol(i)]
        for j in range(i):
            c_j, r_j = parameters[molecule.atom_symbol(j)]
            distance_nm = np.linalg.norm(coordinates[i]-coordinates[j])
            damping = 1+6*(distance_nm/(.1*(r_i+r_j)))**-12
            energy_per_mole -= np.sqrt(c_i*c_j)/(distance_nm**6*damping)
    return energy_per_mole/Avogadro/physical_constants["Hartree energy"][0]

output = Path("reproducibility/ad_showcase/X01_formaldehyde_absorption")
native = json.loads((output / "gradscf.json").read_text())
if not native["complete"]:
    raise RuntimeError("Finish every native case before accepting the comparison.")
tolerances = dict(ground_energy_hartree=1e-8, excitation_hartree=1e-5,
                  electronic_energy_hartree=1e-8, dispersion_energy_hartree=1e-10,
                  oscillator_strength=1e-5, dipole_outer_bohr2=1e-5)
report = dict(pyscf_version=pyscf.__version__, scipy_version=scipy.__version__, python=platform.python_version(),
              hostname=platform.node(), tolerances=tolerances, complete=False, cases=[],
              reference_solver="PySCF public get_ab + scipy.linalg.eigh/eig (complete dense matrices)",
              residual_tolerance_hartree=1e-9,
              native_sha256=hashlib.sha256((output / "gradscf.json").read_bytes()).hexdigest(),
              script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              comparison="Increasing-energy roots at the same fixed geometry; nondegenerate roots checked. Dipole outer products are phase invariant.")
started = perf_counter()
(output / "pyscf.json").write_text(json.dumps(report, indent=2)+"\n")
for functional in native["functional_names"]:
    mol = gto.M(atom=native["atom_angstrom"], basis=native["basis"], unit="Angstrom",
                cart=native["cart"], spin=0, charge=0, verbose=0)
    mf = scf.RHF(mol) if functional == "hf" else dft.RKS(mol)
    mf.conv_tol = native["scf_energy_tolerance"]
    mf.conv_tol_grad = native["scf_gradient_tolerance"]
    mf.max_cycle = native["scf_max_cycle"]
    if functional != "hf":
        mf.xc = functional
        mf.small_rho_cutoff = 0
        mf.grids.level = native["grid_level"]
        mf.grids.alignment = 1
        mf.grids.radi_method = radi.treutler_ahlrichs
        mf.grids.prune = gen_grid.nwchem_prune
        mf.grids.radii_adjust = radi.treutler_atomic_radii_adjust
        mf.grids.atomic_radii = radi.BRAGG_RADII
        mf.grids.becke_scheme = gen_grid.original_becke
    start = perf_counter()
    mf.kernel()
    scf_seconds = perf_counter()-start
    if not mf.converged:
        raise RuntimeError(f"Independent {functional} SCF did not converge.")
    # PySCF's 'wb97xd' supplies the electronic RSH, without original CHG.
    dispersion = original_chg_dispersion(mol) if functional == "wb97xd" else 0.
    nuclear = mol.energy_nuc()
    electronic = float(mf.e_tot)-nuclear
    total_energy = float(mf.e_tot)+dispersion
    start = perf_counter()
    matrix_driver = mf.TDHF() if functional == "hf" else mf.TDDFT()
    a, b = matrix_driver.get_ab()
    nocc, nvir = a.shape[:2]
    dimension = nocc*nvir
    a, b = a.reshape(dimension, dimension), b.reshape(dimension, dimension)
    matrix_file = output/f"pyscf_{functional}_ab.npz"
    np.savez_compressed(matrix_file, a=a, b=b, mo_coeff=mf.mo_coeff,
                        mo_energy=mf.mo_energy, overlap=mf.get_ovlp())
    matrix_sha256 = hashlib.sha256(matrix_file.read_bytes()).hexdigest()
    np.testing.assert_allclose(a, a.T, atol=1e-10, rtol=0)
    np.testing.assert_allclose(b, b.T, atol=1e-10, rtol=0)
    stability = [float(linalg.eigvalsh(a-b, subset_by_index=(0, 0))[0]),
                 float(linalg.eigvalsh(a+b, subset_by_index=(0, 0))[0])]
    if min(stability) <= 0:
        raise RuntimeError("The independent singlet RPA matrix is not stable.")
    matrix_seconds = perf_counter()-start
    for label in ("TDA", "TDDFT"):
        expected = next(case for case in native["cases"] if case["functional"] == functional and case["response"] == label)
        if functional != "hf" and len(mf.grids.weights) != expected["grid_points"]:
            raise RuntimeError("The independent and native DFT quadrature point counts differ.")
        td = mf.TDA() if label == "TDA" else (mf.TDHF() if functional == "hf" else mf.TDDFT())
        td.singlet = True
        start = perf_counter()
        if label == "TDA":
            energies, vectors = linalg.eigh(a, subset_by_index=(0, native["nstates"]-1))
            residuals = np.linalg.norm(a@vectors-vectors*energies, axis=0)
            x, y = vectors.T/np.sqrt(2), np.zeros((len(energies), dimension))
        else:
            matrix = np.block([[a, b], [-b, -a]])
            all_energies, all_vectors = linalg.eig(matrix)
            if np.max(np.abs(all_energies.imag)) > 1e-9:
                raise RuntimeError("The independent stable RPA spectrum is not real.")
            positive = np.flatnonzero(all_energies.real > 1e-8)
            if len(positive) != dimension:
                raise RuntimeError("The independent RPA spectrum has an unexpected positive-root count.")
            selected = positive[np.argsort(all_energies[positive].real)[:native["nstates"]]]
            energies, vectors = all_energies[selected].real, all_vectors[:, selected].real
            residuals = np.linalg.norm(matrix@vectors-vectors*energies, axis=0)
            x, y = vectors[:dimension].T, vectors[dimension:].T
            metric = np.sum(x*x-y*y, axis=1)
            if np.min(metric) <= 0:
                raise RuntimeError("A positive RPA root has nonpositive metric norm.")
            scale = np.sqrt(.5/metric)[:, None]
            x, y = x*scale, y*scale
        if np.max(residuals) > report["residual_tolerance_hartree"]:
            raise RuntimeError("The independent dense eigen-residual exceeds its tolerance.")
        np.testing.assert_allclose(2*(x@x.T-y@y.T), np.eye(len(energies)), atol=1e-9, rtol=0)
        xy = list(zip(x.reshape(-1, nocc, nvir), y.reshape(-1, nocc, nvir)))
        td_seconds = perf_counter()-start
        if np.min(np.diff(energies)) < 1e-5:
            raise RuntimeError("Near-degenerate roots require subspace matching before individual-state comparison.")
        dipoles = np.asarray(td.transition_dipole(xy=xy))
        strengths = np.asarray(td.oscillator_strength(e=energies, xy=xy, gauge="length"))
        outer = np.einsum("si,sj->sij", dipoles, dipoles)
        errors = dict(
            ground_energy_hartree=abs(total_energy-expected["ground_energy_hartree"]),
            electronic_energy_hartree=abs(electronic-expected["electronic_energy_hartree"]),
            dispersion_energy_hartree=abs(dispersion-expected["dispersion_energy_hartree"]),
            excitation_hartree=float(np.max(np.abs(energies-[state["excitation_hartree"] for state in expected["states"]]))),
            oscillator_strength=float(np.max(np.abs(strengths-[state["oscillator_strength"] for state in expected["states"]]))),
            dipole_outer_bohr2=float(np.max(np.abs(outer-[state["dipole_outer_bohr2"] for state in expected["states"]]))),
        )
        states = [dict(root=i+1, excitation_hartree=float(e), excitation_ev=float(e*native["hartree_to_ev"]),
                       transition_dipole_bohr=dipoles[i].tolist(), dipole_outer_bohr2=outer[i].tolist(),
                       oscillator_strength=float(strengths[i])) for i, e in enumerate(energies)]
        passed = all(errors[key] <= value for key, value in tolerances.items())
        report["cases"].append(dict(functional=functional, response=label, scf_converged=True, dense_eigen_check_passed=True,
                                    ground_energy_hartree=total_energy, scf_seconds=scf_seconds,
                                    electronic_energy_hartree=electronic, nuclear_repulsion_hartree=nuclear,
                                    dispersion_energy_hartree=dispersion,
                                    matrix_file=matrix_file.name, matrix_sha256=matrix_sha256,
                                    td_seconds=td_seconds, errors=errors, passed=passed, states=states,
                                    matrix_seconds=matrix_seconds, excitation_dimension=dimension,
                                    dense_matrix_dimension=dimension if label == "TDA" else 2*dimension,
                                    eigen_residuals_hartree=residuals.tolist(),
                                    minimum_a_minus_b_and_a_plus_b_eigenvalues=stability,
                                    grid_points=0 if functional == "hf" else len(mf.grids.weights)))
        print(functional, label, "PASS" if passed else "FAIL", errors, flush=True)
        report["elapsed_seconds"] = perf_counter()-started
        report["complete"] = len(report["cases"]) == len(native["cases"])
        report["passed"] = report["complete"] and all(case["passed"] for case in report["cases"])
        (output / "pyscf.json").write_text(json.dumps(report, indent=2)+"\n")

with (output / "comparison.csv").open("w", newline="") as handle:
    writer = csv.writer(handle)
    writer.writerow(["functional", "response", "root", "native_excitation_ev", "pyscf_excitation_ev",
                     "energy_error_hartree", "native_strength", "pyscf_strength", "strength_error", "max_dipole_outer_error_bohr2"])
    for case, expected in zip(report["cases"], native["cases"]):
        for state, reference in zip(expected["states"], case["states"]):
            writer.writerow([case["functional"], case["response"], state["root"], state["excitation_ev"],
                             reference["excitation_ev"], abs(state["excitation_hartree"]-reference["excitation_hartree"]),
                             state["oscillator_strength"], reference["oscillator_strength"],
                             abs(state["oscillator_strength"]-reference["oscillator_strength"]),
                             np.max(np.abs(np.asarray(state["dipole_outer_bohr2"])-reference["dipole_outer_bohr2"]))])
if not report["passed"]:
    raise RuntimeError("The saved comparison did not meet its predefined tolerances.")
print(f"All {len(report['cases'])} comparisons passed; elapsed {report['elapsed_seconds']:.2f} s", flush=True)

# Measured on jy_gpu_1, CPU float64, 2026-10-15:
# All 10 comparisons passed; elapsed 28.38 s.
# Maximum excitation error: 1.3922803576704723e-08 Hartree.
# Maximum oscillator-strength error: 4.125388447517864e-09.
# Maximum dipole-outer-product error: 1.6397093544062713e-08 Bohr^2.
# Maximum independent dense eigen-residual: 2.3958760799428204e-14 Hartree.
# Original omegaB97X-D CHG independent SI error: 7.13584356534791e-13 Hartree.
