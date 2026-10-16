"""Independent dense CIS/analytic-gradient reference for X10 and its small gate.

This module imports no GradSCF code. It reuses PySCF's public matrix/gradient
APIs and SciPy diagonalization; the scoped CPHF precision check follows the
validated X03 reference. It implements no electronic or response solver.
"""
from functools import partial

import numpy as np
from pyscf import gto, scf, lib
from pyscf.scf import cphf
from scipy import linalg


def evaluate_cis(symbols, coordinates, basis, *, root=0, nroots=6,
                 minimum_gap=1e-4, with_gradient=True):
    mol = gto.M(atom=list(zip(symbols, coordinates)), basis=basis,
                unit="Bohr", cart=True, verbose=0)
    mf = scf.RHF(mol)
    mf.conv_tol, mf.conv_tol_grad, mf.max_cycle = 1e-12, 1e-12, 180
    mf.kernel()
    if not mf.converged:
        raise RuntimeError("Independent RHF failed to converge.")
    td = mf.TDA()
    td.singlet = True
    matrix, _ = td.get_ab()
    nocc, nvir = matrix.shape[:2]
    matrix = matrix.reshape(nocc*nvir, nocc*nvir)
    np.testing.assert_allclose(matrix, matrix.T, atol=1e-10, rtol=0)
    energies, vectors = linalg.eigh(matrix, subset_by_index=(0, nroots-1))
    residuals = np.linalg.norm(matrix@vectors-vectors*energies, axis=0)
    if np.max(residuals) > 1e-9:
        raise RuntimeError("Independent CIS residual exceeds 1e-9 Hartree.")
    gap = float(np.min(np.abs(np.delete(energies, root)-energies[root])))
    if gap < minimum_gap:
        raise RuntimeError("Independent CIS root is not isolated.")
    report = dict(ground_energy=float(mf.e_tot), excitation=float(energies[root]),
                  total_energy=float(mf.e_tot+energies[root]), gap=gap,
                  max_eigen_residual=float(np.max(residuals)),
                  scf_orbital_gradient_norm=float(np.linalg.norm(
                      mf.get_grad(mf.mo_coeff, mf.mo_occ, mf.get_fock()))),
                  coefficients=np.asarray(mf.mo_coeff), overlap=mf.get_ovlp(),
                  cis_matrix=matrix, excitation_energies=energies,
                  amplitudes=vectors.T.reshape(nroots, nocc, nvir))
    if not with_gradient:
        return report

    cphf_tolerance, z_tolerance = 1e-13, 1e-10
    driver = td.nuc_grad_method()
    driver.cphf_conv_tol, driver.cphf_max_cycle = cphf_tolerance, 100
    original_cphf, original_krylov = cphf.solve, lib.krylov
    residual_norms = []

    def checked_cphf(fvind, orbital_energies, occupations, rhs, *args, **kwargs):
        result = original_cphf(fvind, orbital_energies, occupations, rhs, *args, **kwargs)
        z = np.asarray(result[0])
        gaps = orbital_energies[occupations == 0][:, None]-orbital_energies[occupations > 0][None, :]
        residual = gaps*z+fvind(z.ravel()).reshape(z.shape)+rhs
        norm = float(np.linalg.norm(residual))
        if not np.isfinite(norm) or norm > z_tolerance:
            raise RuntimeError(f"Independent CPHF Z residual {norm:.3e} exceeds {z_tolerance:.1e} Hartree.")
        residual_norms.append(norm)
        return result

    x = vectors[:, root].reshape(nocc, nvir)/np.sqrt(2)
    with lib.temporary_env(lib, krylov=partial(original_krylov, lindep=cphf_tolerance**2)), \
            lib.temporary_env(cphf, solve=checked_cphf):
        gradient = driver.kernel(xy=(x, np.zeros_like(x)), state=root+1)
    if lib.krylov is not original_krylov or cphf.solve is not original_cphf:
        raise RuntimeError("A reference override escaped the gradient call.")
    if not residual_norms:
        raise RuntimeError("The reference did not check a CPHF residual.")
    report.update(gradient=np.asarray(gradient), z_residual_norm=max(residual_norms),
                  z_residual_tolerance=z_tolerance, cphf_tolerance=cphf_tolerance,
                  krylov_linear_dependence_cutoff=cphf_tolerance**2)
    return report
