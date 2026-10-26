"""V08: two fixed OH-stretch coordinates, relaxed RHF quartic and dipole surfaces.

The bend normal coordinate and rotations are frozen. This finite Cartesian
subspace model omits Watson/Coriolis terms and is not multidimensional VPT2.
Run each function separately to bound production stages; no CLI is needed.
"""
import json
from math import factorial
from pathlib import Path
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from gradscf import gto, scf
from gradscf.solvers import EigenSolverConfig, solve_hermitian
from gradscf_tools.anharmonic import polynomial_hamiltonian, polynomial_matrix, quartic_certificate
from gradscf_tools.molecular_response import MolecularEnergy
from gradscf_tools.vibrations import AMU_TO_ELECTRON_MASS, HARTREE_TO_WAVENUMBER


output = Path("reproducibility/ad_showcase/V08_water_two_stretch")
harmonic_path = Path("reproducibility/ad_showcase/G05_water_rhf_hessian/gradscf.json")


def setup():
    source = json.loads(harmonic_path.read_text())
    coordinates = jnp.array(source["coordinates_bohr"])
    directions = jnp.array(source["isotopes"]["H2O"]["displacements"])[..., 1:3]
    masses = np.array(source["isotopes"]["H2O"]["masses_amu"])*AMU_TO_ELECTRON_MASS
    metric = np.einsum("axi,a,axj->ij", directions, masses, directions)
    np.testing.assert_allclose(metric, np.eye(2), atol=1e-12, rtol=0)
    mol = gto.M(atom=list(zip(source["symbols"], np.array(coordinates))),
                basis=source["basis"], cart=True, unit="Bohr")
    mf = scf.RHF(mol, integral_backend="native", conv_tol=1e-13,
                 conv_tol_density=1e-12, conv_tol_grad=1e-11).run()
    energy = MolecularEnergy(mf, gradient_tolerance=1e-11)
    geometry = lambda q: coordinates+jnp.einsum("axi,i->ax", directions, q)
    return source, energy, geometry, directions, metric, mf


def curvature_differences(curvature, step):
    origin = np.zeros(2)
    center = np.asarray(curvature(origin))
    plus = [np.asarray(curvature(origin+step*axis)) for axis in np.eye(2)]
    minus = [np.asarray(curvature(origin-step*axis)) for axis in np.eye(2)]
    first = [(p-m)/(2*step) for p, m in zip(plus, minus)]
    second = [(p-2*center+m)/step**2 for p, m in zip(plus, minus)]
    mixed = sum(a*b*np.asarray(curvature(step*np.array([a, b])))
                for a in (-1, 1) for b in (-1, 1))/(4*step**2)
    cubic = np.array([first[0][0, 0], first[1][0, 0], first[0][1, 1], first[1][1, 1]])
    quartic = np.array([second[0][0, 0], mixed[0, 0], second[1][0, 0], mixed[1, 1], second[1][1, 1]])
    return cubic, quartic, abs(second[1][0, 0]-second[0][1, 1])


def build_model():
    source, energy, geometry, directions, metric, mf = setup()
    origin = jnp.zeros(2)
    scalar = lambda q: energy(geometry(q))
    value = jax.jit(scalar)
    start = perf_counter()
    reference_energy = float(value(origin))
    state = energy.solve(geometry(origin))
    stability = mf.stability(solver="dense", eigensolver_tol=1e-10)
    assert bool(state.stationary) and float(state.gradient_norm) < 1e-10
    assert stability.stable and stability.spectrum_certified
    derivatives, coefficients, functions = {}, {}, {}
    for order in range(1, 5):
        def product(q, vectors, order=order):
            function = lambda x: jnp.vdot(jax.grad(scalar)(x), vectors[0])
            for index in range(1, order):
                previous = function
                function = lambda x, previous=previous, index=index: jax.jvp(
                    previous, (x,), (vectors[index],))[1]
            return function(q)
        functions[order] = jax.jit(product)
        for second in range(order+1):
            powers = (order-second, second)
            vectors = jnp.eye(2)[jnp.array([0]*powers[0]+[1]*powers[1])]
            derivative = float(functions[order](origin, vectors))
            if not np.isfinite(derivative):
                raise RuntimeError(f"Nonfinite relaxed energy derivative {powers}.")
            derivatives[powers] = derivative
            coefficients[powers] = derivative/(factorial(powers[0])*factorial(powers[1]))
            print("energy derivative", powers, derivative, flush=True)
    permutation_errors = []
    for indices, powers in (([1, 0, 1], (1, 2)), ([1, 0, 1, 0], (2, 2))):
        error = abs(float(functions[len(indices)](origin, jnp.eye(2)[jnp.array(indices)]))-derivatives[powers])
        permutation_errors.append(error)
    assert max(permutation_errors) < 1e-12
    curvature = jax.jit(jax.hessian(scalar))
    matrix = np.array(curvature(origin))
    if not np.isfinite(matrix).all() or np.any(np.diag(matrix) <= 0):
        raise RuntimeError("The selected stretch quadratic curvatures are not positive and finite.")
    checks = []
    for step in (.3, .1, .03):
        cubic, quartic, mixed_error = curvature_differences(curvature, step)
        checks.append(dict(step_mass_weighted_bohr=step, cubic=cubic.tolist(), quartic=quartic.tolist(),
            max_cubic_error=float(np.max(np.abs(cubic-[derivatives[(3-i, i)] for i in range(4)]))),
            max_quartic_error=float(np.max(np.abs(quartic-[derivatives[(4-i, i)] for i in range(5)]))),
            mixed_hessian_permutation_error=float(mixed_error)))
        print("curvature finite differences", checks[-1], flush=True)
    assert checks[-1]["max_cubic_error"] < 1e-9 and checks[-1]["max_quartic_error"] < 5e-9
    certificate = quartic_certificate(coefficients)
    sampled = []
    for first in np.arange(-20., 21., 5.):
        for second in np.arange(-20., 21., 5.):
            q = np.array([first, second])
            actual = float(value(q))-reference_energy
            polynomial = sum(c*first**p[0]*second**p[1] for p, c in coefficients.items())
            if not np.isfinite(actual):
                raise RuntimeError(f"Sampled electronic state failed at {q}.")
            sampled.append(dict(q=q.tolist(), energy_relative=actual, quartic_energy=polynomial,
                                energy_error=polynomial-actual))
    data = dict(method="RHF", basis=source["basis"], cart=True, symbols=source["symbols"],
                coordinates_bohr=source["coordinates_bohr"], source_harmonic=str(harmonic_path),
                selected_source_modes=[1, 2], directions=directions.tolist(), mass_metric=metric.tolist(),
                mode_coordinate_units="bohr*sqrt(electron mass)", coordinate_masses=[1., 1.],
                energy_hartree=reference_energy, hessian=matrix.tolist(),
                harmonic_basis_frequencies=np.sqrt(np.diag(matrix)).tolist(),
                coefficients=[dict(powers=list(p), derivative=derivatives[p], coefficient=c)
                              for p, c in coefficients.items()], certificate=certificate,
                curvature_checks=checks, AD_permutation_errors=permutation_errors,
                orbital_gradient_norm=float(state.gradient_norm),
                minimum_internal_orbital_curvature=float(stability.minimum_curvature),
                samples=sampled, seconds_including_compilation=perf_counter()-start,
                scope="Two fixed Cartesian OH-stretch normal coordinates; bend/rotor frozen; no Watson/Coriolis or full VPT2")
    output.mkdir(parents=True, exist_ok=True)
    (output/"model.json").write_text(json.dumps(data, indent=2)+"\n")
    if certificate["quartic_status"] != "positive" or certificate["stationary_status"] != "unique":
        raise RuntimeError(f"Full-domain two-mode polynomial not certified: {certificate}")
    return data


def dipole_surface():
    model = json.loads((output/"model.json").read_text())
    _, energy, geometry, *_ = setup()
    mu = jax.jit(lambda q: energy.dipole(geometry(q)))
    origin = jnp.zeros(2)
    start = perf_counter()
    constant = np.asarray(mu(origin))
    linear = np.asarray(jax.jit(jax.jacfwd(mu))(origin))
    quadratic = np.asarray(jax.jit(jax.jacfwd(jax.jacfwd(mu)))(origin))
    assert all(np.isfinite(value).all() for value in (constant, linear, quadratic))
    np.testing.assert_allclose(quadratic, quadratic.swapaxes(-1, -2), atol=1e-11, rtol=0)
    checks = []
    for step in (.3, .1, .03):
        plus, minus = [np.asarray(mu(step*axis)) for axis in np.eye(2)], [np.asarray(mu(-step*axis)) for axis in np.eye(2)]
        first = np.column_stack([(p-m)/(2*step) for p, m in zip(plus, minus)])
        second = np.zeros((3, 2, 2))
        for axis in range(2):
            second[:, axis, axis] = (plus[axis]-2*constant+minus[axis])/step**2
        mixed = sum(a*b*np.asarray(mu(step*np.array([a, b]))) for a in (-1, 1) for b in (-1, 1))/(4*step**2)
        second[:, 0, 1] = second[:, 1, 0] = mixed
        checks.append(dict(step=step, linear_error=float(np.max(np.abs(first-linear))),
                           quadratic_error=float(np.max(np.abs(second-quadratic)))))
    assert min(row["linear_error"] for row in checks) < 1e-7
    assert min(row["quadratic_error"] for row in checks) < 1e-7
    samples = []
    for row in model["samples"]:
        q = np.array(row["q"])
        actual = np.asarray(mu(q))
        if not np.isfinite(actual).all():
            raise RuntimeError(f"Sampled dipole state failed at {q}.")
        approximation = constant+linear@q+.5*np.einsum("aij,i,j->a", quadratic, q, q)
        samples.append(dict(q=q.tolist(), actual=actual.tolist(), approximation=approximation.tolist(),
                            error_norm=float(np.linalg.norm(approximation-actual))))
    result = dict(constant=constant.tolist(), linear=linear.tolist(), quadratic=quadratic.tolist(),
                  finite_differences=checks, samples=samples, seconds=perf_counter()-start,
                  units="dipole e*bohr; derivatives with respect to the stored mass-weighted coordinates",
                  expression="stationary total dipole identity with full implicit density and moving-position response")
    (output/"dipole.json").write_text(json.dumps(result, indent=2)+"\n")
    return result


def spectrum():
    model = json.loads((output/"model.json").read_text())
    dipole = json.loads((output/"dipole.json").read_text())
    if model["certificate"]["stationary_status"] != "unique":
        raise RuntimeError("The full-domain polynomial is not certified.")
    potential = {tuple(row["powers"]): row["coefficient"] for row in model["coefficients"]}
    frequency = model["harmonic_basis_frequencies"]
    constant, linear, quadratic = (np.array(dipole[key]) for key in ("constant", "linear", "quadratic"))
    mu_coefficients = []
    for axis in range(3):
        mu_coefficients.append({(0, 0): constant[axis], (1, 0): linear[axis, 0], (0, 1): linear[axis, 1],
            (2, 0): quadratic[axis, 0, 0]/2, (1, 1): quadratic[axis, 0, 1], (0, 2): quadratic[axis, 1, 1]/2})
    convergence = []
    start = perf_counter()
    for size in (10, 14, 18, 22):
        matrix = polynomial_hamiltonian(frequency, (1., 1.), (size, size), potential)
        solved = solve_hermitian(matrix, config=EigenSolverConfig(method="dense", nroots=12, atol=1e-10))
        if not np.all(solved.converged) or not np.all(solved.response_valid):
            raise RuntimeError("Unresolved vibrational eigenproblem; do not assign isolated states.")
        levels, vectors = np.array(solved.values), np.array(solved.vectors)
        mu_matrices = [np.asarray(polynomial_matrix(frequency, (1., 1.), (size, size), terms)) for terms in mu_coefficients]
        moments = np.array([vectors[:, 0]@matrix@vectors for matrix in mu_matrices]).T
        strengths = np.sum(moments[1:]**2, axis=1)
        convergence.append(dict(size_per_mode=size, dimension=size**2, levels_hartree=levels.tolist(),
            frequencies_cm1=((levels[1:]-levels[0])*HARTREE_TO_WAVENUMBER).tolist(),
            dipole_strengths_au=strengths.tolist(), residual_norms=np.asarray(solved.residual_norms).tolist()))
        print("water oscillator size", size, "frequencies", convergence[-1]["frequencies_cm1"], flush=True)
    np.testing.assert_allclose(convergence[-1]["levels_hartree"], convergence[-2]["levels_hartree"], atol=2e-7, rtol=0)
    np.testing.assert_allclose(convergence[-1]["dipole_strengths_au"], convergence[-2]["dipole_strengths_au"], atol=1e-7, rtol=2e-4)
    occupations = np.array(list(np.ndindex(22, 22)))
    assignments = []
    for column in vectors.T:
        leading = np.argsort(column**2)[-3:][::-1]
        assignments.append([dict(occupation=occupations[index].tolist(), weight=float(column[index]**2)) for index in leading])
    quadratic_potential = {p: c for p, c in potential.items() if sum(p) == 2}
    harmonic = polynomial_hamiltonian(frequency, (1., 1.), (22, 22), quadratic_potential)
    harmonic_result = solve_hermitian(harmonic, config=EigenSolverConfig(method="dense", nroots=12, atol=1e-10))
    force_roots = solve_hermitian(jnp.array(model["hessian"]), config=EigenSolverConfig(method="dense", nroots=2, atol=1e-12))
    harmonic_frequencies = np.sqrt(np.asarray(force_roots.raw_values))
    expected = np.sort(((occupations+.5)*harmonic_frequencies).sum(axis=1))[:12]
    np.testing.assert_allclose(harmonic_result.values, expected, atol=1e-10, rtol=0)
    # Reconstruct actual normalized model wavefunctions to audit the probability
    # outside the domain where the electronic potential/dipole were sampled.
    grid = np.linspace(-60., 60., 601)
    basis_functions = []
    for omega in frequency:
        x = np.sqrt(omega)*grid
        values = [(omega/np.pi)**.25*np.exp(-x*x/2)]
        values.append(np.sqrt(2)*x*values[0])
        for n in range(2, 22):
            values.append(np.sqrt(2/n)*x*values[-1]-np.sqrt((n-1)/n)*values[-2])
        basis_functions.append(np.array(values))
    widths = (5., 10., 15., 20.)
    normalizations, tails = [], [[] for _ in widths]
    for column in vectors.T:
        wavefunction = basis_functions[0].T@column.reshape(22, 22)@basis_functions[1]
        density = wavefunction**2
        norm = float(np.trapezoid(np.trapezoid(density, grid, axis=1), grid))
        normalizations.append(norm)
        for index, width in enumerate(widths):
            selected = np.abs(grid) <= width+1e-12
            local = density[np.ix_(selected, selected)]
            inside = np.trapezoid(np.trapezoid(local, grid[selected], axis=1), grid[selected])
            tails[index].append(float(norm-inside))
    np.testing.assert_allclose(normalizations, 1., atol=1e-9, rtol=0)
    windows = []
    for width, probabilities in zip(widths, tails):
        energy_points = [row for row in model["samples"] if max(map(abs, row["q"])) <= width]
        dipole_points = [row for row in dipole["samples"] if max(map(abs, row["q"])) <= width]
        windows.append(dict(half_width_mass_weighted_bohr=width, probability_outside=probabilities,
            max_sampled_taylor_energy_error=max(abs(row["energy_error"]) for row in energy_points),
            max_sampled_dipole_error=max(row["error_norm"] for row in dipole_points)))
    result = dict(convergence=convergence, levels_hartree=levels.tolist(),
                  frequencies_cm1=convergence[-1]["frequencies_cm1"], transition_dipoles_au=moments[1:].tolist(),
                  dipole_strengths_au=strengths.tolist(), oscillator_strengths=(2/3*(levels[1:]-levels[0])*strengths).tolist(),
                  dominant_harmonic_components=assignments, coefficients_in_oscillator_basis=vectors.tolist(),
                  harmonic_levels_hartree=np.asarray(harmonic_result.values).tolist(),
                  harmonic_frequencies_hartree=harmonic_frequencies.tolist(), seconds=perf_counter()-start,
                  sampling_windows=windows, wavefunction_norms=normalizations,
                  wavefunction_grid=dict(bounds=[-60., 60.], points_per_axis=601),
                  state_labels="Largest harmonic components are assignments, not exact quantum numbers; near-resonant mixing retained",
                  scope="Zero-temperature two-stretch quartic/quadratic-dipole model; no bend, rotor, lifetime or experimental-spectrum claim")
    (output/"spectrum.json").write_text(json.dumps(result, indent=2)+"\n")
    return result


if __name__ == "__main__":
    build_model()
    dipole_surface()
    spectrum()

# jy_gpu_1 CPU float64, OMP/BLAS=2, 2026-10-16:
# Two-stretch quartic: exact-rational all-angle positive/unique certificate passed.
# Qstep=.03 native cubic/quartic FD errors1.110e-11 / 3.460e-13.
# Dipole linear/quadratic FD errors4.529e-10 / 3.395e-11 at the same step.
# Model fundamentals3969.63813/4082.94957cm^-1; combination-like gap8027.20840cm^-1.
# At sampled half-width20, max energy/dipole Taylor errors .05208Eh/.01534au;
# combination-state probability outside is1.744%. These constrain physical use.
