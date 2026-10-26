"""V08: a bounded one-dimensional H2/RHF quartic vibrational model.

Optimize the actual electronic energy first. Native AD supplies the local
force constants; padded oscillator matrices and the shared Hermitian solver
supply model levels. H2 electric-dipole IR is forbidden: no IR sticks are made.
"""
import json
from math import factorial
from pathlib import Path
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import root_scalar

jax.config.update("jax_enable_x64", True)

from gradscf import gto, scf
from gradscf.solvers import EigenSolverConfig, solve_hermitian
from gradscf_tools.anharmonic import polynomial_hamiltonian, quartic_certificate
from gradscf_tools.molecular_response import MolecularEnergy
from gradscf_tools.vibrations import AMU_TO_ELECTRON_MASS, HARTREE_TO_WAVENUMBER


output = Path("reproducibility/ad_showcase/V08_h2_quartic")
hydrogen_mass_amu = 1.00782503223
reduced_mass = hydrogen_mass_amu*AMU_TO_ELECTRON_MASS/2
direction = jnp.array([[0., 0., -.5], [0., 0., .5]])


def build_model():
    mol = gto.M(atom="H 0 0 -.7; H 0 0 .7", basis="6-31g", unit="Bohr", cart=True)
    controls = dict(integral_backend="native", conv_tol=1e-13,
                    conv_tol_density=1e-12, conv_tol_grad=1e-11)
    mf = scf.RHF(mol, **controls).run()
    energy = MolecularEnergy(mf, gradient_tolerance=1e-11)
    scalar = lambda length: energy(length*direction)
    value_gradient = jax.jit(jax.value_and_grad(scalar))
    history = []

    def derivative(length):
        value, gradient = map(float, value_gradient(jnp.asarray(length)))
        if not np.isfinite([value, gradient]).all():
            raise RuntimeError("Nonfinite relaxed energy/gradient during nuclear optimization.")
        history.append(dict(length_bohr=length, energy=value, gradient=gradient))
        return gradient

    start = perf_counter()
    solution = root_scalar(derivative, bracket=(1.1, 1.7), method="brentq", xtol=1e-13, rtol=1e-14)
    length = float(solution.root)
    assert solution.converged and abs(derivative(length)) < 1e-10
    functions = [scalar, jax.grad(scalar)]
    for _ in range(2, 5):
        previous = functions[-1]
        functions.append(lambda r, previous=previous: jax.jvp(previous, (r,), (jnp.ones_like(r),))[1])
    compiled = [jax.jit(function) for function in functions]
    derivatives = []
    for order, function in enumerate(compiled):
        value = float(function(jnp.asarray(length)))
        if not np.isfinite(value):
            raise RuntimeError(f"Nonfinite derivative order {order}.")
        derivatives.append(value)
        print(f"equilibrium R={length:.12f}, derivative{order}={value:.13g}", flush=True)
    assert derivatives[2] > 0
    state = energy.solve(length*direction)
    assert bool(state.stationary) and float(state.gradient_norm) < 1e-10
    equilibrium_mol = gto.M(atom=list(zip(["H", "H"], np.asarray(length*direction))),
                            basis="6-31g", unit="Bohr", cart=True)
    equilibrium_mf = scf.RHF(equilibrium_mol, **controls).run()
    stability = equilibrium_mf.stability(solver="dense", eigensolver_tol=1e-10)
    assert stability.stable and stability.spectrum_certified
    coefficients = {(order,): derivatives[order]/factorial(order) for order in range(1, 5)}
    certificate = quartic_certificate(coefficients)
    if certificate["quartic_status"] != "positive" or certificate["stationary_status"] != "unique":
        raise RuntimeError(f"The global one-well quartic model was not certified: {certificate}")
    checks = []
    for step in (3e-3, 1e-3, 3e-4):
        plus, minus = float(compiled[2](length+step)), float(compiled[2](length-step))
        checks.append(dict(step_bohr=step,
            cubic_error=abs((plus-minus)/(2*step)-derivatives[3]),
            quartic_error=abs((plus-2*derivatives[2]+minus)/step**2-derivatives[4])))
    assert checks[-1]["cubic_error"] < 3e-5 and checks[-1]["quartic_error"] < 3e-4
    sampled = []
    for displacement in np.linspace(-.6, .8, 29):
        actual = float(compiled[0](length+displacement))-derivatives[0]
        polynomial = sum(value*displacement**powers[0] for powers, value in coefficients.items())
        sampled.append(dict(displacement_bohr=float(displacement), energy_relative=actual,
                            quartic_energy=polynomial, error=polynomial-actual))
    data = dict(method="RHF", symbols=["H", "H"], basis="6-31g", cart=True,
                equilibrium_length_bohr=length, coordinates_bohr=np.asarray(length*direction).tolist(),
                coordinate="q=R-Re; centers move by (-q/2,+q/2) along z",
                hydrogen_mass_amu=hydrogen_mass_amu, reduced_mass_electron_masses=reduced_mass,
                mass_metric="sum_A m_A |dR_A/dq|^2=m_H/2", derivatives=derivatives,
                coefficients=[dict(powers=list(p), coefficient=c) for p, c in coefficients.items()],
                harmonic_frequency_hartree=float(np.sqrt(derivatives[2]/reduced_mass)),
                certificate=certificate, nuclear_optimization=history,
                orbital_gradient_norm=float(state.gradient_norm),
                minimum_internal_orbital_curvature=float(stability.minimum_curvature),
                curvature_finite_differences=checks, sampled_true_rhf_potential=sampled,
                seconds_including_compilation=perf_counter()-start,
                scf_energy_tolerance=1e-13, scf_gradient_tolerance=1e-11,
                implicit_response_tolerance=1e-10,
                scope="Local RHF-derived one-coordinate quartic; fixed rotor, no dipole IR, no experimental accuracy claim")
    output.mkdir(parents=True, exist_ok=True)
    (output/"model.json").write_text(json.dumps(data, indent=2)+"\n")
    return data


def spectrum():
    data = json.loads((output/"model.json").read_text())
    coefficients = {tuple(row["powers"]): row["coefficient"] for row in data["coefficients"]}
    frequency, mass = data["harmonic_frequency_hartree"], data["reduced_mass_electron_masses"]
    convergence = []
    start = perf_counter()
    for size in (16, 24, 32, 48):
        matrix = polynomial_hamiltonian((frequency,), (mass,), (size,), coefficients)
        solved = solve_hermitian(matrix, config=EigenSolverConfig(method="dense", nroots=6, atol=1e-11))
        if not np.all(solved.converged) or not np.all(solved.response_valid):
            raise RuntimeError("The bounded oscillator eigenproblem is unresolved.")
        levels = np.asarray(solved.values)
        convergence.append(dict(size=size, levels_hartree=levels.tolist(),
            transitions_cm1=((levels-levels[0])*HARTREE_TO_WAVENUMBER).tolist(),
            residual_norms=np.asarray(solved.residual_norms).tolist()))
        print("HO basis", size, "transitions / cm^-1", convergence[-1]["transitions_cm1"], flush=True)
    np.testing.assert_allclose(convergence[-1]["levels_hartree"], convergence[-2]["levels_hartree"],
                               atol=1e-8, rtol=0)
    harmonic = polynomial_hamiltonian((frequency,), (mass,), (48,), {(2,): .5*data["derivatives"][2]})
    harmonic_result = solve_hermitian(harmonic, config=EigenSolverConfig(method="dense", nroots=6, atol=1e-12))
    np.testing.assert_allclose(harmonic_result.values, frequency*(np.arange(6)+.5), atol=1e-12, rtol=0)
    # Normalized Hermite functions reconstruct probability density. This is an
    # analytic basis evaluation, not a second vibrational solver.
    grid = np.linspace(-3., 3., 6001)
    x = np.sqrt(mass*frequency)*grid
    basis = [(mass*frequency/np.pi)**.25*np.exp(-x*x/2)]
    basis.append(np.sqrt(2)*x*basis[0])
    for n in range(2, 48):
        basis.append(np.sqrt(2/n)*x*basis[-1]-np.sqrt((n-1)/n)*basis[-2])
    wavefunctions = np.array(basis).T@np.asarray(solved.vectors)
    density = wavefunctions**2
    normalization = np.trapezoid(density, grid, axis=0)
    np.testing.assert_allclose(normalization, 1., atol=1e-10, rtol=0)
    windows = []
    for left, right in ((-.3, .3), (-.4, .4), (-.5, .5), (-.6, .6), (-.6, .8)):
        selected = (grid >= left) & (grid <= right)
        inside = np.trapezoid(density[selected], grid[selected], axis=0)
        points = [row for row in data["sampled_true_rhf_potential"]
                  if left-1e-12 <= row["displacement_bohr"] <= right+1e-12]
        windows.append(dict(interval_bohr=[left, right], probability_outside=(normalization-inside).tolist(),
                            max_sampled_taylor_error_hartree=max(abs(row["error"]) for row in points)))
    nonphysical = grid <= -data["equilibrium_length_bohr"]
    nonphysical_probability = np.trapezoid(density[nonphysical], grid[nonphysical], axis=0)
    result = dict(convergence=convergence, harmonic_levels_hartree=np.asarray(harmonic_result.values).tolist(),
                  nuclear_wavefunction_norms=normalization.tolist(), sampling_windows=windows,
                  probability_R_nonpositive=nonphysical_probability.tolist(),
                  levels_hartree=levels.tolist(), transitions_cm1=convergence[-1]["transitions_cm1"],
                  coefficients_in_oscillator_basis=np.asarray(solved.vectors).tolist(),
                  seconds=perf_counter()-start, hartree_to_wavenumber=HARTREE_TO_WAVENUMBER,
                  model_note="Whole-line quartic oscillator; physical R>0 and local Taylor validity audited separately")
    (output/"spectrum.json").write_text(json.dumps(result, indent=2)+"\n")
    return result


if __name__ == "__main__":
    build_model()
    spectrum()

# jy_gpu_1 CPU float64, OMP/BLAS=2, 2026-10-16:
# Re=1.3794239869227614 bohr; E'=-2.082e-16 Eh/bohr; reduced mass=918.57632372 me.
# E''=.4116596065025, E'''=-1.414209272718, E''''=4.785027113105 in Eh/bohr^n.
# N=48 model gaps1/2: 4547.6669388 / 9160.4095261 cm^-1; no H2 dipole IR.
# Full sampled [-.6,.8] window: max Taylor error .0309917 Eh; v5 tail13.52%.
# These are converged polynomial-model levels, not accurate global H2 overtones.
