"""Independent H2 equilibrium/force constants and finite-domain sinc DVR.

PySCF analytic gradients/Hessians validate the native electronic calculation.
SciPy diagonalizes an independent real-space representation of the SAME saved
quartic model; this separates force-constant and vibrational-basis errors.
"""
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pyscf
from pyscf import gto, scf
from scipy.constants import physical_constants
from scipy.linalg import eigh
from scipy.optimize import root_scalar


output = Path("reproducibility/ad_showcase/V08_h2_quartic")
direction = np.array([[0., 0., -.5], [0., 0., .5]])


def electronic_reference(data):
    points = {}

    def evaluate(length, hessian=False):
        mol = gto.M(atom=list(zip(["H", "H"], length*direction)),
                    basis=data["basis"], cart=True, unit="Bohr", verbose=0)
        mf = scf.RHF(mol)
        mf.conv_tol, mf.conv_tol_grad, mf.conv_tol_cpscf = 1e-13, 1e-11, 1e-12
        mf.kernel()
        residual = float(np.linalg.norm(mf.get_grad(mf.mo_coeff, mf.mo_occ)))
        if not mf.converged or residual > 1e-10:
            raise RuntimeError(f"Independent SCF failed at R={length}: residual={residual}")
        gradient = float(np.sum(mf.nuc_grad_method().kernel()*direction))
        row = dict(length_bohr=length, energy=float(mf.e_tot), gradient=gradient,
                   orbital_gradient_norm=residual)
        if hessian:
            matrix = mf.Hessian().kernel().transpose(0, 2, 1, 3).reshape(6, 6)
            row["curvature"] = float(direction.ravel()@matrix@direction.ravel())
        points[length] = row
        return row

    optimized = root_scalar(lambda r: evaluate(r)["gradient"], bracket=(1.1, 1.7),
                            method="brentq", xtol=1e-13, rtol=1e-14)
    assert optimized.converged
    length = data["equilibrium_length_bohr"]
    np.testing.assert_allclose(optimized.root, length, atol=1e-8, rtol=0)
    center = evaluate(length, True)
    np.testing.assert_allclose([center["energy"], center["gradient"], center["curvature"]],
                               data["derivatives"][:3], atol=2e-8, rtol=0)
    checks = []
    for step in (3e-3, 1e-3, 3e-4):
        plus, minus = evaluate(length+step, True), evaluate(length-step, True)
        cubic = (plus["curvature"]-minus["curvature"])/(2*step)
        quartic = (plus["curvature"]-2*center["curvature"]+minus["curvature"])/step**2
        checks.append(dict(step_bohr=step, cubic=cubic, quartic=quartic,
                           cubic_error=abs(cubic-data["derivatives"][3]),
                           quartic_error=abs(quartic-data["derivatives"][4])))
    assert checks[-1]["cubic_error"] < 3e-5 and checks[-1]["quartic_error"] < 3e-4
    sampled = []
    for native in data["sampled_true_rhf_potential"]:
        reference = evaluate(length+native["displacement_bohr"])
        relative = reference["energy"]-center["energy"]
        error = relative-native["energy_relative"]
        sampled.append(dict(displacement_bohr=native["displacement_bohr"],
                            energy_relative=relative, native_error=error))
    assert max(abs(row["native_error"]) for row in sampled) < 1e-9
    return dict(equilibrium_length_bohr=float(optimized.root),
                equilibrium_error_bohr=abs(float(optimized.root)-length),
                derivative_checks=checks, sampled_potential=sampled,
                electronic_points=list(points.values()), pyscf_version=pyscf.__version__,
                scf_energy_tolerance=1e-13, scf_gradient_tolerance=1e-11,
                cpscf_tolerance=1e-12, actual_orbital_gradient_gate=1e-10)


def sinc_dvr(bounds, size, mass, coefficients):
    """Independent finite sinc lattice; interval and spacing must both converge."""
    grid = np.linspace(*bounds, size)
    step = grid[1]-grid[0]
    difference = np.arange(size)[:, None]-np.arange(size)[None, :]
    denominator = np.where(difference == 0, 1, difference)**2
    kinetic = np.where(difference % 2 == 0, 1., -1.)/(mass*step**2*denominator)
    np.fill_diagonal(kinetic, np.pi**2/(6*mass*step**2))
    potential = sum(value*grid**order for order, value in coefficients.items())
    matrix = kinetic+np.diag(potential)
    values, vectors = eigh(matrix, subset_by_index=(0, 5), driver="evr")
    residuals = np.linalg.norm(matrix@vectors-vectors*values, axis=0)
    assert np.max(residuals) < 1e-9
    return dict(bounds_bohr=list(bounds), size=size, spacing_bohr=float(step),
                levels_hartree=values.tolist(), residual_norms=residuals.tolist())


def main():
    data = json.loads((output/"model.json").read_text())
    spectrum = json.loads((output/"spectrum.json").read_text())
    start = perf_counter()
    electronic = electronic_reference(data)
    mass = data["hydrogen_mass_amu"]*physical_constants["atomic mass constant"][0]/physical_constants["electron mass"][0]/2
    np.testing.assert_allclose(mass, data["reduced_mass_electron_masses"], atol=1e-10, rtol=0)
    coefficients = {row["powers"][0]: row["coefficient"] for row in data["coefficients"]}
    grid_checks = [sinc_dvr((-1.2, 1.8), size, mass, coefficients) for size in (201, 301, 401, 601)]
    domain_checks = []
    for bounds in ((-.65, .8), (-.85, 1.1), (-1.05, 1.5), (-1.2, 1.8)):
        size = round((bounds[1]-bounds[0])/.01)+1
        domain_checks.append(sinc_dvr(bounds, size, mass, coefficients))
    expected = spectrum["levels_hartree"]
    np.testing.assert_allclose(grid_checks[-1]["levels_hartree"], expected, atol=2e-8, rtol=0)
    np.testing.assert_allclose(domain_checks[-1]["levels_hartree"], expected, atol=2e-8, rtol=0)
    # At zero anharmonicity the sixth state has a wider negative-q tail.
    # Hold dx=.005 fixed while expanding the interval. The last mathematical
    # whole-line harmonic check is distinct from the R>0 quartic comparisons.
    harmonic_exact = data["harmonic_frequency_hartree"]*(np.arange(6)+.5)
    harmonic_checks = []
    for left in (-1.2, -1.32, -1.8):
        size = round((1.8-left)/.005)+1
        row = sinc_dvr((left, 1.8), size, mass, {2: .5*data["derivatives"][2]})
        row["max_exact_error"] = float(np.max(np.abs(np.array(row["levels_hartree"])-harmonic_exact)))
        harmonic_checks.append(row)
    for row in harmonic_checks[1:]:
        np.testing.assert_allclose(row["levels_hartree"], harmonic_exact, atol=2e-8, rtol=0)
    result = dict(electronic=electronic, dvr_grid_convergence=grid_checks,
                  dvr_domain_convergence=domain_checks, dvr_harmonic_limit=harmonic_checks[-1],
                  harmonic_limit_domain_convergence=harmonic_checks,
                  max_dvr_ho_level_error=float(np.max(np.abs(np.array(grid_checks[-1]["levels_hartree"])-expected))),
                  seconds=perf_counter()-start,
                  scope="Independent PySCF electronic reference and SciPy sinc-DVR for the identical supplied quartic")
    (output/"reference.json").write_text(json.dumps(result, indent=2)+"\n")
    print("PySCF equilibrium error / bohr", electronic["equilibrium_error_bohr"])
    print("Cubic/quartic smallest-step errors", electronic["derivative_checks"][-1])
    print("Maximum DVR-HO level error / Eh", result["max_dvr_ho_level_error"])
    return result


if __name__ == "__main__":
    main()

# jy_gpu_1 CPU float64, PySCF2.13.0, 2026-10-16:
# Independent Re error7.803e-13 bohr; h=.0003 cubic/quartic errors2.721e-7/1.286e-6.
# Identical quartic model: max six-level sinc-DVR/HO error3.610e-14 Eh.
# Fixed dx=.005 harmonic left boundaries -1.2/-1.32/-1.8 give errors
# 9.049e-8 / 7.964e-10 / 3.108e-14 Eh. Original narrow-domain failure retained.
