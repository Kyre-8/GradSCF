"""Independent MP2/CCSD gradients, Hessian differences and finite-field spectra.

PySCF recomputes every displaced SCF and correlated state. No GradSCF import
supplies electronic quantities. Finite fields include the nuclear field energy;
orbital-unrelaxed correlated densities are not substituted for total derivatives.
"""
import json
from functools import partial
from pathlib import Path
from time import perf_counter

import numpy as np
import pyscf
from pyscf import cc, gto, lib, mp, scf
from pyscf.scf import cphf


METHOD = "ccsd"


def folder(method):
    if method not in ("mp2", "ccsd"):
        raise ValueError("Choose mp2 or ccsd.")
    return Path("reproducibility/ad_showcase")/f"V05_water_{method}_spectra"


def evaluate(coordinates, method, *, field=None, gradient=False):
    mol = gto.M(atom=list(zip(("O", "H", "H"), coordinates)), basis="6-31g*",
                unit="Bohr", cart=True, verbose=0)
    mf = scf.RHF(mol)
    mf.conv_tol, mf.conv_tol_grad, mf.max_cycle = 1e-13, 1e-11, 200
    nuclear_field = 0.
    if field is not None:
        if gradient:
            raise ValueError("The analytic gradient oracle is zero-field only.")
        position = mol.intor_symmetric("int1e_r", comp=3)
        hcore = mf.get_hcore()+np.einsum("x,xpq->pq", field, position)
        mf.get_hcore = lambda *args: hcore
        nuclear_field = -np.dot(field, mol.atom_charges() @ mol.atom_coords())
    mf.kernel()
    assert mf.converged
    density = mf.make_rdm1()
    fock = mf.get_hcore()+mf.get_veff(dm=density)
    residual = float(np.linalg.norm(mf.get_grad(mf.mo_coeff, mf.mo_occ, fock)))
    assert residual < 1e-10
    correlated = (mp.MP2 if method == "mp2" else cc.CCSD)(mf)
    if method == "ccsd":
        correlated.conv_tol, correlated.conv_tol_normt, correlated.max_cycle = 1e-12, 1e-11, 200
    correlated.kernel()
    if method == "ccsd":
        assert correlated.converged
    derivative, response_residuals = None, []
    if gradient:
        original_solve, original_krylov = cphf.solve, lib.krylov

        def checked_response(fvind, eps, occ, rhs, *args, **kwargs):
            # CC/MP gradient drivers otherwise use a looser default CPHF solve.
            kwargs.update(tol=1e-13, max_cycle=100)
            answer = original_solve(fvind, eps, occ, rhs, *args, **kwargs)
            z = np.asarray(answer[0])
            gaps = eps[occ == 0][:, None]-eps[occ > 0][None, :]
            norm = float(np.linalg.norm(gaps*z+fvind(z.ravel()).reshape(z.shape)+rhs))
            response_residuals.append(norm)
            if not np.isfinite(norm) or norm > 1e-10:
                raise RuntimeError(f"Independent CPHF residual {norm:.3e} is unresolved.")
            return answer

        # The Krylov dependence check compares squared norms. Both public
        # callables are restored even if the reference calculation fails.
        with lib.temporary_env(cphf, solve=checked_response), \
                lib.temporary_env(lib, krylov=partial(original_krylov, lindep=1e-26)):
            derivative = correlated.nuc_grad_method().kernel()
        assert cphf.solve is original_solve and lib.krylov is original_krylov
        assert response_residuals
        if method == "ccsd":
            assert correlated.converged_lambda
    diagnostics = dict(orbital_gradient_norm=residual,
                       cphf_residual_norm=max(response_residuals, default=0.))
    return float(correlated.e_tot+nuclear_field), derivative, diagnostics


def geometry(method):
    output = folder(method)
    data = json.loads((output/"hessian.json").read_text())
    coordinates = np.array(data["coordinates_bohr"])
    start = perf_counter()
    value, derivative, diagnostics = evaluate(coordinates, method, gradient=True)
    np.testing.assert_allclose(value, data["energy_hartree"], atol=2e-8, rtol=0)
    np.testing.assert_allclose(derivative, data["gradient_hartree_per_bohr"], atol=2e-6, rtol=0)
    checks, maximum_residual = [], diagnostics["orbital_gradient_norm"]
    maximum_response = diagnostics["cphf_residual_norm"]
    for step in (3e-4, 1e-4):
        columns = []
        for direction in np.eye(9).reshape(9, 3, 3):
            _, plus, rp = evaluate(coordinates+step*direction, method, gradient=True)
            _, minus, rm = evaluate(coordinates-step*direction, method, gradient=True)
            maximum_residual = max(maximum_residual, rp["orbital_gradient_norm"], rm["orbital_gradient_norm"])
            maximum_response = max(maximum_response, rp["cphf_residual_norm"], rm["cphf_residual_norm"])
            columns.append(((plus-minus)/(2*step)).ravel())
        hessian = np.column_stack(columns)
        error = float(np.max(np.abs(hessian-data["hessian_hartree_per_bohr2"])))
        checks.append(dict(step_bohr=step, hessian_hartree_per_bohr2=hessian.tolist(),
                           max_hessian_error=error))
        print(method, "independent Hessian", step, error, flush=True)
    assert checks[-1]["max_hessian_error"] < 3e-6
    result = dict(method=method.upper(), pyscf_version=pyscf.__version__, energy_hartree=value,
                  gradient_hartree_per_bohr=derivative.tolist(),
                  max_gradient_error=float(np.max(np.abs(derivative-data["gradient_hartree_per_bohr"]))),
                  hessian_checks=checks, max_orbital_gradient=maximum_residual,
                  max_cphf_residual=maximum_response, cphf_tolerance=1e-13,
                  krylov_dependence_cutoff=1e-26,
                  elapsed_seconds=perf_counter()-start, scf_evaluations=37)
    (output/"pyscf_geometry.json").write_text(json.dumps(result, indent=2)+"\n")
    return result


def field_response(coordinates, method, step):
    center, _, diagnostics = evaluate(coordinates, method, field=np.zeros(3))
    residual = diagnostics["orbital_gradient_norm"]
    dipole, alpha = np.zeros(3), np.zeros((3, 3))
    pairs = []
    for index, axis in enumerate(np.eye(3)):
        plus, _, rp = evaluate(coordinates, method, field=step*axis)
        minus, _, rm = evaluate(coordinates, method, field=-step*axis)
        residual = max(residual, rp["orbital_gradient_norm"], rm["orbital_gradient_norm"])
        pairs.append(plus+minus)
        dipole[index] = -(plus-minus)/(2*step)
        alpha[index, index] = -(plus+minus-2*center)/step**2
    for i, j in ((0, 1), (0, 2), (1, 2)):
        direction = np.eye(3)[i]+np.eye(3)[j]
        plus, _, rp = evaluate(coordinates, method, field=step*direction)
        minus, _, rm = evaluate(coordinates, method, field=-step*direction)
        residual = max(residual, rp["orbital_gradient_norm"], rm["orbital_gradient_norm"])
        alpha[i, j] = alpha[j, i] = -(plus+minus-pairs[i]-pairs[j]+2*center)/(2*step**2)
    return dipole, alpha, residual


def spectra(method):
    output = folder(method)
    data = json.loads((output/"gradscf.json").read_text())
    coordinates = np.array(data["coordinates_bohr"])
    directions = np.array(data["displacements"])
    start = perf_counter()
    checks, maximum_residual = [], 0.
    # Two independent displacement/field windows expose both truncation and
    # cancellation. Neither window generates the native AD spectrum.
    for displacement, field_step in ((1e-3, 4e-3), (3e-4, 2e-3)):
        mu_derivatives, alpha_derivatives = [], []
        for index, direction in enumerate(directions):
            step = displacement/np.max(np.abs(direction))
            mup, ap, rp = field_response(coordinates+step*direction, method, field_step)
            mum, am, rm = field_response(coordinates-step*direction, method, field_step)
            maximum_residual = max(maximum_residual, rp, rm)
            mu_derivatives.append((mup-mum)/(2*step))
            alpha_derivatives.append((ap-am)/(2*step))
            print(method, "field window", field_step, "mode", index+1, flush=True)
        dmu, dalpha = np.array(mu_derivatives), np.array(alpha_derivatives)
        checks.append(dict(max_cartesian_step_bohr=displacement, field_step_au=field_step,
                           dipole_derivatives=dmu.tolist(), alpha_derivatives=dalpha.tolist(),
                           max_dipole_error=float(np.max(np.abs(dmu-data["dipole_derivatives"]))),
                           max_alpha_error=float(np.max(np.abs(dalpha-data["alpha_derivatives"])))))
    np.testing.assert_allclose(dmu, data["dipole_derivatives"], atol=3e-6, rtol=1e-3)
    np.testing.assert_allclose(dalpha, data["alpha_derivatives"], atol=5e-5, rtol=1e-3)
    result = dict(method=method.upper(), pyscf_version=pyscf.__version__,
                  checks=checks, max_orbital_gradient=maximum_residual,
                  elapsed_seconds=perf_counter()-start, scf_evaluations=156,
                  approximation="central total-energy field differences and displaced geometries",
                  nuclear_field_energy="-F dot sum(Z_A R_A)")
    (output/"pyscf.json").write_text(json.dumps(result, indent=2)+"\n")
    return result


if __name__ == "__main__":
    geometry(METHOD)
    spectra(METHOD)
