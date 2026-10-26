"""Independent open-shell analytic gradients on the recorded real pi branch.

The native density is an initial guess only. PySCF recomputes integrals, XC,
SCF and gradients; a fresh orbital residual and density comparison check that
the same orientation of a degenerate real pi determinant was retained.
Post-HF gradients use a separately verified SciPy augmented physical-pi-gauge
UCPHF oracle; stock iterative PySCF response is not claimed to pass that gate.
"""
import json
import hashlib
import platform
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import numpy as np
import pyscf
import scipy
from pyscf import cc, dft, gto, lib, mp, scf
from pyscf.dft import gen_grid, radi
from pyscf.scf import ucphf
from scipy.linalg import solve


OUTPUT = Path("reproducibility/ad_showcase/G03_open_shell_forces")
CASES = ("OH_hf", "OH_pbe", "OH_mp2", "OH_ccsd",
         "NO_hf", "NO_pbe", "NO_mp2", "NO_ccsd")


def correlated_gradient(electronic):
    """PySCF gradient assembly with an independent physical-pi-gauge Z oracle.

    Only these small, real, collinear OH/NO UHF references are supported. The
    original UCPHF operator is retained, including its measured near-zero root.
    A standard SciPy augmented solve fixes the verified redundant pi rotation;
    this is not a claim that stock PySCF Krylov converged to the same residual.
    """
    mf = electronic._scf
    coordinates = mf.mol.atom_coords()
    if np.max(np.abs(coordinates[:, :2])) > 1e-12:
        raise ValueError("This reference oracle requires the recorded z-axis geometry.")
    angular = mf.mol.intor("int1e_cg_irxp", comp=3)[2]
    pi = np.hstack([(coeff[:, occupation == 0].T@angular@coeff[:, occupation > 0]).ravel()
                    for coeff, occupation in zip(mf.mo_coeff, mf.mo_occ)])
    norm_pi = np.linalg.norm(pi)
    if not np.isfinite(norm_pi) or norm_pi < 1e-8:
        raise RuntimeError("The physical pi-rotation generator is unresolved.")
    pi /= norm_pi
    original_solve = ucphf.solve
    diagnostics = []

    def gauge_fixed_response(fvind, eps, occ, rhs, *args, **kwargs):
        if args or kwargs.get("s1") is not None or kwargs.get("level_shift", 0) != 0:
            raise ValueError("Use the field-independent, unshifted Z-vector equation.")
        shapes = [np.asarray(value).shape for value in rhs]
        if any(len(shape) != 2 for shape in shapes):
            raise ValueError("This small reference expects one alpha/beta Z-vector pair.")
        right = np.hstack([np.asarray(value).ravel() for value in rhs])
        gaps = np.hstack([(energy[occupation == 0][:, None]-energy[occupation > 0][None, :]).ravel()
                          for energy, occupation in zip(eps, occ)])
        dimension = right.size
        if dimension > 512 or pi.shape != right.shape:
            raise ValueError("The independent augmented oracle is bounded to 512 response coordinates, with no frozen orbitals.")
        matrix = np.diag(gaps)+np.column_stack([
            np.asarray(fvind(axis[None, :])).ravel() for axis in np.eye(dimension)])
        asymmetry = float(np.max(np.abs(matrix-matrix.T)))
        if asymmetry > 1e-10:
            raise RuntimeError("The independent UCPHF operator is not symmetric.")
        # Symmetric eigenanalysis diagnoses the nullspace. The solve below
        # uses the original matrix, without modifying any eigenvalue.
        spectrum, vectors = np.linalg.eigh(.5*(matrix+matrix.T))
        null = np.abs(spectrum) < 1e-8
        if np.count_nonzero(null) != 1:
            raise RuntimeError("The reference has more or fewer than the expected pi null direction.")
        overlap = float(abs(vectors[:, null].T@pi).item())
        if overlap < 1-1e-8:
            raise RuntimeError("The near-zero direction is not the physical pi generator.")
        rhs_norm = float(np.linalg.norm(right))
        compatibility = float(abs(pi@right))
        if compatibility > 1e-10*rhs_norm+32*np.finfo(float).eps*rhs_norm:
            raise RuntimeError("The response RHS is incompatible with the physical pi gauge.")
        augmented = np.zeros((dimension+1, dimension+1))
        augmented[:dimension, :dimension] = matrix
        augmented[:dimension, dimension], augmented[dimension, :dimension] = pi, pi
        solution = solve(augmented, np.r_[-right, 0.], assume_a="gen")
        z, multiplier = solution[:-1], float(solution[-1])
        applied = gaps*z+np.asarray(fvind(z[None, :])).ravel()
        residual = applied+right  # Full physical residual, with no projection.
        residual_norm = float(np.linalg.norm(residual))
        roundoff = 32*np.finfo(float).eps*(rhs_norm+float(np.linalg.norm(applied)))
        gauge_residual = float(abs(pi@z))
        if (not np.isfinite(residual_norm) or residual_norm > 1e-10
                or residual_norm > 1e-10*rhs_norm+roundoff or gauge_residual > 1e-12):
            raise RuntimeError("The independent augmented UCPHF residual/gauge gate failed.")
        diagnostics.append(dict(
            oracle="SciPy general solve of [[A,pi],[pi.T,0]]; original PySCF fvind and unchanged A spectrum",
            dimension=dimension, matrix_asymmetry=asymmetry, response_eigenvalues_hartree=spectrum.tolist(),
            null_threshold_hartree=1e-8, pi_null_overlap=overlap,
            near_zero_eigenvalue_hartree=float(spectrum[null][0]),
            minimum_resolved_abs_eigenvalue_hartree=float(np.min(np.abs(spectrum[~null]))),
            pi_action_norm=float(np.linalg.norm(matrix@pi)), rhs_norm=rhs_norm,
            rhs_pi_component=compatibility, rhs_numerical_null_component=float(np.linalg.norm(vectors[:, null].T@right)),
            true_residual_l2=residual_norm, relative_true_residual=residual_norm/max(rhs_norm, np.finfo(float).tiny),
            true_residual_max=float(np.max(np.abs(residual))), roundoff_allowance=roundoff,
            pi_dot_z=float(pi@z), lagrange_multiplier=multiplier,
            matrix_action_error=float(np.linalg.norm(matrix@z-applied)),
            rtol=1e-10, absolute_residual_limit=1e-10, passed=True))
        split = int(np.prod(shapes[0]))
        return (z[:split].reshape(shapes[0]), z[split:].reshape(shapes[1])), None

    try:
        with lib.temporary_env(ucphf, solve=gauge_fixed_response):
            gradient = electronic.nuc_grad_method().kernel()
    finally:
        if ucphf.solve is not original_solve:
            raise RuntimeError("The independent response override escaped its scope.")
    if not diagnostics:
        raise RuntimeError("No independent UCPHF residual was checked.")
    return gradient, diagnostics


def run(case, *, check_grid=False):
    data = json.loads((OUTPUT / f"{case}.json").read_text())
    mol = gto.M(atom=data["atom_bohr"], basis=data["basis"], spin=data["spin"],
                unit="Bohr", cart=True, verbose=0)
    mf = dft.UKS(mol) if data["method"] == "PBE" else scf.UHF(mol)
    mf.conv_tol, mf.conv_tol_grad, mf.max_cycle = 1e-13, 1e-11, 250
    if data["method"] == "PBE":
        mf.xc, mf.small_rho_cutoff = "pbe", 0
        mf.grids.level, mf.grids.alignment = data["grid_level"], 1
        mf.grids.radi_method, mf.grids.prune = radi.treutler_ahlrichs, gen_grid.nwchem_prune
        mf.grids.radii_adjust, mf.grids.atomic_radii = radi.treutler_atomic_radii_adjust, radi.BRAGG_RADII
        mf.grids.becke_scheme = gen_grid.original_becke
    start = perf_counter()
    mf.kernel(dm0=np.asarray(data["spin_density"]))
    if not mf.converged:
        raise RuntimeError(f"{case}: independent SCF did not converge.")
    density = mf.make_rdm1()
    fock = mf.get_hcore()+mf.get_veff(dm=density)
    residual = float(np.linalg.norm(mf.get_grad(mf.mo_coeff, mf.mo_occ, fock)))
    assert np.isfinite(residual) and residual < 1e-10
    np.testing.assert_allclose(density, data["spin_density"], atol=2e-6, rtol=0)
    electronic = mf
    if data["method"] in {"MP2", "CCSD"}:
        electronic = (mp.MP2 if data["method"] == "MP2" else cc.CCSD)(mf)
        if data["method"] == "CCSD":
            electronic.conv_tol, electronic.conv_tol_normt, electronic.max_cycle = 1e-12, 1e-11, 200
        electronic.kernel()
        if data["method"] == "CCSD":
            assert electronic.converged
    derivative = electronic.nuc_grad_method()
    if data["method"] == "PBE":
        derivative.grid_response = True
    response_diagnostics = []
    if data["method"] in {"MP2", "CCSD"}:
        gradient, response_diagnostics = correlated_gradient(electronic)
    else:
        gradient = derivative.kernel()
    if data["method"] == "CCSD":
        assert electronic.converged_lambda
    np.testing.assert_allclose(electronic.e_tot, data["energy_hartree"], atol=2e-8, rtol=0)
    np.testing.assert_allclose(gradient, data["gradient_hartree_per_bohr"], atol=2e-6, rtol=0)
    np.testing.assert_allclose(mf.spin_square()[0], data["spin_square"], atol=2e-6, rtol=0)
    result = dict(passed=True, pyscf_version=pyscf.__version__, energy_hartree=float(electronic.e_tot),
                  gradient_hartree_per_bohr=gradient.tolist(),
                  max_gradient_error=float(np.max(np.abs(gradient-data["gradient_hartree_per_bohr"]))),
                  orbital_gradient_norm=residual, spin_square=float(mf.spin_square()[0]),
                  spin_diagnostic=data.get("spin_diagnostic", "SCF determinant"),
                  max_density_difference=float(np.max(np.abs(density-data["spin_density"]))),
                  initial_guess="native density for pi orientation only; independently reconverged",
                  grid_response=data["method"] == "PBE", elapsed_seconds=perf_counter()-start,
                  response_oracle="physical-pi-gauge augmented UCPHF using SciPy" if response_diagnostics else "PySCF analytic SCF gradient",
                  ucphf_responses=response_diagnostics,
                  correlated_converged=None if data["method"] != "CCSD" else bool(electronic.converged),
                  lambda_converged=None if data["method"] != "CCSD" else bool(electronic.converged_lambda),
                  native_sha256=hashlib.sha256((OUTPUT/f"{case}.json").read_bytes()).hexdigest(),
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  hostname=platform.node(), python_version=platform.python_version(),
                  numpy_version=np.__version__, scipy_version=scipy.__version__,
                  recorded_utc=datetime.now(timezone.utc).isoformat())
    if data["method"] == "PBE" and check_grid:
        # This changes quadrature, not the AD formula. Keep its error separate
        # from the matched-grid derivative comparison above.
        refinement_start = perf_counter()
        mf.grids.level = 5
        mf.grids.build()
        mf.kernel(dm0=density)
        attempts = [dict(solver="DIIS", converged=bool(mf.converged), energy_hartree=float(mf.e_tot))]
        if not mf.converged:
            # A nearly flat pi orientation can stall DIIS. One bounded Newton
            # restart retains both convergence tolerances and is recorded.
            mf = mf.newton()
            mf.max_cycle = 80
            mf.kernel(mf.mo_coeff, mf.mo_occ)
            attempts.append(dict(solver="Newton", converged=bool(mf.converged),
                                 energy_hartree=float(mf.e_tot)))
        print(case, "level-5 SCF attempts", attempts, flush=True)
        if not mf.converged:
            rejected_density = mf.make_rdm1()
            rejected_fock = mf.get_hcore()+mf.get_veff(dm=rejected_density)
            rejected = dict(converged=False, grid_level=5, scf_attempts=attempts,
                            orbital_gradient_norm=float(np.linalg.norm(
                                mf.get_grad(mf.mo_coeff, mf.mo_occ, rejected_fock))),
                            energy_tolerance=mf.conv_tol, orbital_tolerance=mf.conv_tol_grad,
                            status="rejected; no level-5 force accepted")
            (OUTPUT / f"{case}_grid_check.json").write_text(json.dumps(rejected, indent=2)+"\n")
            raise RuntimeError(f"{case}: level-5 grid comparison did not converge.")
        refined_density = mf.make_rdm1()
        refined_fock = mf.get_hcore()+mf.get_veff(dm=refined_density)
        refined_residual = float(np.linalg.norm(mf.get_grad(mf.mo_coeff, mf.mo_occ, refined_fock)))
        assert refined_residual < 1e-10
        refined_gradient = mf.nuc_grad_method()
        refined_gradient.grid_response = True
        refined_gradient = refined_gradient.kernel()
        result["level5_grid_check"] = dict(
            energy_hartree=float(mf.e_tot), gradient_hartree_per_bohr=refined_gradient.tolist(),
            orbital_gradient_norm=refined_residual,
            max_gradient_change=float(np.max(np.abs(refined_gradient-gradient))),
            max_density_change=float(np.max(np.abs(refined_density-density))),
            elapsed_seconds=perf_counter()-refinement_start, scf_attempts=attempts)
    (OUTPUT / f"{case}_pyscf.json").write_text(json.dumps(result, indent=2)+"\n")
    print(case, "independent maximum gradient error", result["max_gradient_error"], flush=True)
    return result


if __name__ == "__main__":
    for case in CASES:
        run(case)

# Measured on jy_gpu_1, CPU float64, 2026-10-16 Asia/Shanghai:
# OH/MP2, OH/CCSD, NO/MP2, NO/CCSD gradient errors respectively
# [5.79e-14, 1.40e-12, 9.33e-11, 5.26e-11] Eh Bohr^-1.
# Full unprojected relative UCPHF residuals:
# [2.46e-15, 4.63e-15, 3.14e-14, 1.70e-15]; dimensions [112,112,337,337].
# Actual reference elapsed times: [0.194, 2.429, 0.585, 12.707] s.
# This is the SciPy augmented physical-pi-gauge oracle, not a claim that
# stock PySCF iterative UCPHF meets the same true-residual requirement.
# The executed pre-comment source is archived with its recorded SHA256.
