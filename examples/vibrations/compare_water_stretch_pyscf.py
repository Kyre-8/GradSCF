"""Independent PySCF derivative checks and two-coordinate SciPy sinc DVR.

The DVR uses the identical saved polynomial potential and dipole. Electronic
checks separately compare their coefficients and sample values with PySCF.
No GradSCF solver or vibrational helper is imported by this reference.
"""
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pyscf
from pyscf import gto, scf
from scipy.sparse.linalg import LinearOperator, eigsh


output = Path("reproducibility/ad_showcase/V08_water_two_stretch")


def electronic_reference():
    model = json.loads((output/"model.json").read_text())
    dipole = json.loads((output/"dipole.json").read_text())
    coordinates, directions = np.array(model["coordinates_bohr"]), np.array(model["directions"])
    frame = directions.reshape(9, 2)
    derivative = {tuple(row["powers"]): row["derivative"] for row in model["coefficients"]}
    points = {}

    def evaluate(q, hessian=True):
        q = np.asarray(q, dtype=float)
        key = tuple(q.tolist())
        if key in points and (not hessian or "hessian" in points[key]):
            return points[key]
        geometry = coordinates+np.einsum("axi,i->ax", directions, q)
        mol = gto.M(atom=list(zip(model["symbols"], geometry)), basis=model["basis"],
                    cart=True, unit="Bohr", verbose=0)
        mf = scf.RHF(mol)
        mf.conv_tol, mf.conv_tol_grad, mf.conv_tol_cpscf = 1e-13, 1e-11, 1e-12
        mf.max_cycle = 200
        mf.kernel()
        residual = float(np.linalg.norm(mf.get_grad(mf.mo_coeff, mf.mo_occ)))
        if not mf.converged or residual > 1e-10:
            raise RuntimeError(f"Independent SCF failed at q={q}: orbital gradient={residual}")
        row = dict(q=q.tolist(), energy=float(mf.e_tot),
                   dipole=np.asarray(mf.dip_moment(unit="AU", verbose=0)).tolist(),
                   orbital_gradient_norm=residual)
        if hessian:
            cartesian = mf.Hessian().kernel().transpose(0, 2, 1, 3).reshape(9, 9)
            row["hessian"] = (frame.T@cartesian@frame).tolist()
        points[key] = row
        return row

    start = perf_counter()
    center = evaluate(np.zeros(2))
    h0, mu0 = np.array(center["hessian"]), np.array(center["dipole"])
    np.testing.assert_allclose(center["energy"], model["energy_hartree"], atol=1e-9, rtol=0)
    np.testing.assert_allclose(h0, model["hessian"], atol=2e-10, rtol=0)
    np.testing.assert_allclose(mu0, dipole["constant"], atol=1e-8, rtol=0)
    checks = []
    for step in (.3, .1, .03):
        plus = [evaluate(step*axis) for axis in np.eye(2)]
        minus = [evaluate(-step*axis) for axis in np.eye(2)]
        h_first = [(np.array(p["hessian"])-m["hessian"])/(2*step) for p, m in zip(plus, minus)]
        h_second = [(np.array(p["hessian"])-2*h0+m["hessian"])/step**2 for p, m in zip(plus, minus)]
        mixed_points = [(a*b, evaluate(step*np.array([a, b]))) for a in (-1, 1) for b in (-1, 1)]
        h_mixed = sum(sign*np.array(row["hessian"]) for sign, row in mixed_points)/(4*step**2)
        cubic = np.array([h_first[0][0, 0], h_first[1][0, 0], h_first[0][1, 1], h_first[1][1, 1]])
        quartic = np.array([h_second[0][0, 0], h_mixed[0, 0], h_second[1][0, 0], h_mixed[1, 1], h_second[1][1, 1]])
        mu_first = np.column_stack([(np.array(p["dipole"])-m["dipole"])/(2*step) for p, m in zip(plus, minus)])
        mu_second = np.zeros((3, 2, 2))
        for axis in range(2):
            mu_second[:, axis, axis] = (np.array(plus[axis]["dipole"])-2*mu0+minus[axis]["dipole"])/step**2
        mu_mixed = sum(sign*np.array(row["dipole"]) for sign, row in mixed_points)/(4*step**2)
        mu_second[:, 0, 1] = mu_second[:, 1, 0] = mu_mixed
        checks.append(dict(step=step, cubic=cubic.tolist(), quartic=quartic.tolist(),
            max_cubic_error=float(np.max(np.abs(cubic-[derivative[(3-i, i)] for i in range(4)]))),
            max_quartic_error=float(np.max(np.abs(quartic-[derivative[(4-i, i)] for i in range(5)]))),
            dipole_linear_error=float(np.max(np.abs(mu_first-dipole["linear"]))),
            dipole_quadratic_error=float(np.max(np.abs(mu_second-dipole["quadratic"])))))
        print("PySCF derivative check", checks[-1], flush=True)
    assert checks[-1]["max_cubic_error"] < 1e-9 and checks[-1]["max_quartic_error"] < 5e-9
    assert min(row["dipole_linear_error"] for row in checks) < 1e-7
    assert min(row["dipole_quadratic_error"] for row in checks) < 1e-7
    samples = []
    for native_energy, native_mu in zip(model["samples"], dipole["samples"], strict=True):
        np.testing.assert_array_equal(native_energy["q"], native_mu["q"])
        row = evaluate(native_energy["q"], False)
        samples.append(dict(q=row["q"], energy_error=row["energy"]-center["energy"]-native_energy["energy_relative"],
                            dipole_error=float(np.linalg.norm(np.array(row["dipole"])-native_mu["actual"]))))
    assert max(abs(row["energy_error"]) for row in samples) < 2e-8
    assert max(row["dipole_error"] for row in samples) < 1e-7
    result = dict(pyscf_version=pyscf.__version__, checks=checks, sampled_surface_checks=samples,
                  electronic_points=list(points.values()), seconds=perf_counter()-start,
                  scf_energy_tolerance=1e-13, scf_gradient_tolerance=1e-11,
                  cpscf_tolerance=1e-12, scf_max_cycle=200, actual_orbital_gradient_gate=1e-10,
                  method="PySCF analytic Hessians and relaxed dipoles, multi-step geometry differences")
    (output/"electronic_reference.json").write_text(json.dumps(result, indent=2)+"\n")
    return result


def sinc_kinetic(size, step):
    difference = np.arange(size)[:, None]-np.arange(size)[None, :]
    denominator = np.where(difference == 0, 1, difference)**2
    matrix = np.where(difference % 2 == 0, 1., -1.)/(step**2*denominator)
    np.fill_diagonal(matrix, np.pi**2/(6*step**2))
    return matrix


def dvr(half_width, size, potential, dipole):
    grid = np.linspace(-half_width, half_width, size)
    first, second = grid[:, None], grid[None, :]
    values = sum(c*first**p[0]*second**p[1] for p, c in potential.items())
    kinetic = sinc_kinetic(size, grid[1]-grid[0])

    def product(vector):
        wavefunction = np.asarray(vector).reshape(size, size)
        return (kinetic@wavefunction+wavefunction@kinetic.T+values*wavefunction).ravel()

    operator = LinearOperator((size**2, size**2), matvec=product, rmatvec=product, dtype=np.float64)
    initial = np.random.default_rng(67).normal(size=size**2)
    levels, vectors = eigsh(operator, k=13, which="SA", tol=1e-11, maxiter=12000, ncv=52, v0=initial)
    order = np.argsort(levels)
    levels, vectors = levels[order], vectors[:, order]
    residual = np.linalg.norm(np.column_stack([product(vectors[:, i]) for i in range(13)])-vectors*levels, axis=0)
    if np.max(residual) > 1e-9:
        raise RuntimeError(f"Unresolved independent DVR residual {np.max(residual)}")
    moments = []
    for terms in dipole:
        surface = sum(c*first**p[0]*second**p[1] for p, c in terms.items())
        moments.append(vectors[:, 0]@(surface.ravel()[:, None]*vectors[:, 1:12]))
    moments = np.array(moments).T
    return dict(half_width=half_width, size_per_coordinate=size, dimension=size**2,
                spacing=float(grid[1]-grid[0]), levels_hartree=levels[:12].tolist(),
                dipole_strengths_au=np.sum(moments**2, axis=1).tolist(),
                residual_norms=residual.tolist(), guard_gap=float(levels[12]-levels[11]))


def dvr_reference():
    model = json.loads((output/"model.json").read_text())
    dipole = json.loads((output/"dipole.json").read_text())
    spectrum = json.loads((output/"spectrum.json").read_text())
    potential = {tuple(row["powers"]): row["coefficient"] for row in model["coefficients"]}
    constant, linear, quadratic = (np.array(dipole[key]) for key in ("constant", "linear", "quadratic"))
    surfaces = [{(0, 0): constant[a], (1, 0): linear[a, 0], (0, 1): linear[a, 1],
                 (2, 0): quadratic[a, 0, 0]/2, (1, 1): quadratic[a, 0, 1],
                 (0, 2): quadratic[a, 1, 1]/2} for a in range(3)]
    start = perf_counter()
    grid_checks = []
    for size in (51, 71, 91):
        row = dvr(45., size, potential, surfaces)
        grid_checks.append(row)
        print("DVR grid", size, "level error", np.max(np.abs(np.array(row["levels_hartree"])-spectrum["levels_hartree"])), flush=True)
    domain_checks = [dvr(width, 2*width+1, potential, surfaces) for width in (30, 37, 43)]
    domain_checks.append(grid_checks[-1])
    for row in (grid_checks[-1], *domain_checks[-2:]):
        np.testing.assert_allclose(row["levels_hartree"], spectrum["levels_hartree"], atol=2e-7, rtol=0)
        np.testing.assert_allclose(row["dipole_strengths_au"], spectrum["dipole_strengths_au"], atol=2e-7, rtol=5e-4)
    harmonic_potential = {p: c for p, c in potential.items() if sum(p) == 2}
    harmonic = dvr(45., 91, harmonic_potential, surfaces)
    frequencies = np.sqrt(np.linalg.eigvalsh(np.array(model["hessian"])))
    occupations = np.array(list(np.ndindex(12, 12)))
    exact = np.sort(((occupations+.5)*frequencies).sum(axis=1))[:12]
    np.testing.assert_allclose(harmonic["levels_hartree"], exact, atol=2e-7, rtol=0)
    result = dict(grid_convergence=grid_checks, domain_convergence=domain_checks, harmonic_limit=harmonic,
                  max_level_error=float(np.max(np.abs(np.array(grid_checks[-1]["levels_hartree"])-spectrum["levels_hartree"]))),
                  max_strength_error=float(np.max(np.abs(np.array(grid_checks[-1]["dipole_strengths_au"])-spectrum["dipole_strengths_au"]))),
                  seconds=perf_counter()-start, method="Independent SciPy sinc-DVR for identical two-coordinate polynomials")
    (output/"dvr_reference.json").write_text(json.dumps(result, indent=2)+"\n")
    return result


if __name__ == "__main__":
    electronic_reference()
    dvr_reference()

# jy_gpu_1 CPU float64, PySCF2.13.0, 2026-10-16:
# Qstep=.03 cubic/quartic errors1.114e-11/2.675e-13; dipole errors4.355e-10/4.540e-10.
# All105 independent points passed (max actual orbital gradient5.604e-12).
# One far corner required max_cycle200; tolerances unchanged, no Newton needed.
# Two-mode DVR-HO max level/strength errors5.502e-9Eh/1.108e-14au².
# Domain half-widths30/37/43/45 errors8.883e-4/1.244e-5/4.994e-8/5.502e-9Eh;
# both final domains pass the original tolerance; all earlier results retained.
