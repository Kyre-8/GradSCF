"""Independent PySCF full-FCI energies, state-to-state RDMs and bond differences.

No GradSCF solver or integral routine is imported. PySCF supplies fresh AO
integrals and all 16 CI states in its own moving Cholesky frame, without HF.
CI phases are aligned only when comparing signed transition RDMs/moments.
"""
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pyscf
from pyscf import ao2mo, gto
from pyscf.fci import direct_spin1, spin_op
from scipy.linalg import solve_triangular


CASE = Path("reproducibility/ad_showcase/X09_h2_fci_esa")


def run():
    data = json.loads((CASE/"gradscf.json").read_text())
    initial = np.asarray(data["initial_roots"])
    norb, nelec, nstates = len(data["active_orbitals"]), tuple(data["electrons"]), data["determinant_count"]
    residuals = []

    def calculate(length, phase_anchor=None):
        coords = np.array([[0., 0., -.5*length], [0., 0., .5*length]])
        mol = gto.M(atom=list(zip(data["symbols"], coords)), basis=data["basis"],
                    unit="Bohr", cart=data["cart"], verbose=0)
        overlap = mol.intor_symmetric("int1e_ovlp")
        frame = solve_triangular(np.linalg.cholesky(overlap).T, np.eye(norb), lower=False)
        hcore = mol.intor_symmetric("int1e_kin") + mol.intor_symmetric("int1e_nuc")
        h1 = frame.T @ hcore @ frame
        eri = ao2mo.restore(1, ao2mo.kernel(mol, frame), norb)
        energies, ci = direct_spin1.kernel(h1, eri, norb, nelec, nroots=nstates,
                                           ecore=mol.energy_nuc(), tol=1e-13, max_cycle=200)
        energies, ci = np.asarray(energies), np.asarray(ci)
        effective = direct_spin1.absorb_h1e(h1, eri, norb, nelec, fac=.5)
        norms = [np.linalg.norm(direct_spin1.contract_2e(effective, c, norb, nelec)
                               + (mol.energy_nuc()-energy)*c) for energy, c in zip(energies, ci)]
        assert np.isfinite(norms).all() and max(norms) < 1e-10
        residuals.extend(norms)
        minimum_gap = float(np.min(np.diff(energies)))
        assert minimum_gap > 1e-4
        spin = np.array([spin_op.spin_square(c, norb, nelec)[0] for c in ci])
        alignment = None
        if phase_anchor is not None:
            overlaps = np.einsum("iab,jab->ij", phase_anchor, ci)
            np.testing.assert_array_equal(np.argmax(np.abs(overlaps), axis=1), np.arange(nstates))
            alignment = np.diag(overlaps)
            assert np.min(np.abs(alignment)) > .999999
            ci *= np.where(alignment < 0, -1., 1.)[:, None, None]
        position = np.einsum("xpq,pi,qj->xij", mol.intor_symmetric("int1e_r", comp=3), frame, frame)
        densities = np.array([[direct_spin1.trans_rdm1(ci[left], right, norb, nelec)
                               for right in ci] for left in initial])
        moments = -np.einsum("xpq,ijqp->ijx", position, densities)
        moments += np.einsum("x,ij->ijx", mol.atom_charges() @ coords,
                             np.einsum("iab,jab->ij", ci[initial], ci))
        strengths = (2/3)*(energies[None, :]-energies[initial, None])*np.sum(moments**2, axis=-1)
        return dict(mol=mol, frame=frame, energies=energies, ci=ci, spin=spin,
                    minimum_gap=minimum_gap, alignment=alignment, transition_rdm1=densities,
                    moments=moments, signed_f=strengths)

    start = perf_counter()
    baseline = calculate(data["bond_bohr"], np.asarray(data["ci_coefficients"]))
    comparisons = {
        "energy_hartree": (baseline["energies"], data["energies_hartree"]),
        "orbital_frame": (baseline["frame"], data["orbital_frame"]),
        "spin_square": (baseline["spin"], data["spin_square"]),
        "transition_rdm1": (baseline["transition_rdm1"], data["transition_rdm1"]),
        "transition_dipole_au": (baseline["moments"], data["transition_dipoles_au"]),
        "oscillator_strength": (baseline["signed_f"], data["signed_oscillator_strengths"]),
    }
    errors = {}
    for label, (reference, native) in comparisons.items():
        np.testing.assert_allclose(reference, native, atol=1e-10, rtol=0)
        errors[label] = float(np.max(np.abs(reference-native)))
        print(label, "max difference", errors[label], flush=True)

    checks, tracking = [], []
    for step in (1e-3, 3e-4, 1e-4):
        displaced = []
        for sign in (1, -1):
            length = data["bond_bohr"] + sign*step
            current = calculate(length)
            cross = gto.intor_cross("int1e_ovlp", baseline["mol"], current["mol"])
            frame_overlap = baseline["frame"].T @ cross @ current["frame"]
            state_overlap = np.einsum("iab,ac,bd,jcd->ij", baseline["ci"],
                                     frame_overlap, frame_overlap, current["ci"])
            assignment = np.argmax(np.abs(state_overlap), axis=1)
            np.testing.assert_array_equal(assignment, np.arange(nstates))
            minimum_overlap = float(np.min(np.abs(np.diag(state_overlap))))
            assert minimum_overlap > .999
            np.testing.assert_allclose(current["spin"], baseline["spin"], atol=1e-9)
            tracking.append(dict(bond_bohr=length, root_assignment=assignment.tolist(),
                                 minimum_wavefunction_overlap=minimum_overlap,
                                 minimum_gap_hartree=current["minimum_gap"]))
            displaced.append(current["signed_f"])
        numerical = (displaced[0]-displaced[1])/(2*step)
        error = float(np.max(np.abs(numerical-data["signed_f_derivative_per_bohr"])))
        checks.append(dict(step_bohr=step, signed_f_derivative_per_bohr=numerical.tolist(),
                           max_derivative_error=error))
        print("PySCF bond step", step, "max df/dR error", error, flush=True)
    np.testing.assert_allclose(numerical, data["signed_f_derivative_per_bohr"], atol=3e-6, rtol=1e-5)
    reference = dict(pyscf_version=pyscf.__version__, reference="Independent full FCI, moving AO Cholesky frame; no HF",
        energies_hartree=baseline["energies"].tolist(), spin_square=baseline["spin"].tolist(),
        transition_rdm1=baseline["transition_rdm1"].tolist(), transition_dipoles_au=baseline["moments"].tolist(),
        signed_oscillator_strengths=baseline["signed_f"].tolist(), errors=errors,
        phase_alignment_overlaps=baseline["alignment"].tolist(), finite_difference_checks=checks, state_tracking=tracking,
        minimum_root_gap_hartree=baseline["minimum_gap"], max_ci_residual_norm=float(max(residuals)),
        fci_evaluations=7, elapsed_seconds=perf_counter()-start)
    (CASE/"pyscf.json").write_text(json.dumps(reference, indent=2)+"\n")
    return reference


if __name__ == "__main__":
    run()

# Measured on jy_gpu_1/c20, PySCF 2.13.0, CPU float64, 2026-10-16:
# Maximum E/RDM/mu/f differences: 4.219e-15 / 8.061e-15 / 1.454e-14 / 7.216e-15.
# df/dR differences at h=1e-3/3e-4/1e-4: 3.830e-6 / 3.446e-7 / 3.826e-8.
# Largest independently recomputed CI residual: 6.386e-15.
# Seven full-FCI calculations, no HF: 0.155 seconds (distinct scope from native AD).
