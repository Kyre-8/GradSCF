"""V06: full-space TDHF polarizability and its harmonic mode derivatives.

The model is frequency-dependent Placzek activity, not a complete resonant
Kramers-Heisenberg vibronic spectrum. Run after G05 water RHF and V01/V02.
"""
import hashlib
import json
import platform
import resource
from pathlib import Path
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np
from scipy.constants import physical_constants
from scipy.integrate import simpson

jax.config.update('jax_enable_x64', True)

from gradscf import gto, scf
from gradscf.tddft import gen_tdhf_vind
from gradscf_tools.dynamic_response import MolecularPolarizability, dynamic_polarizability
from gradscf_tools.vibrations import AMU_TO_ELECTRON_MASS


BASE = Path('reproducibility/ad_showcase')
HARMONIC = BASE/'G05_water_rhf_hessian/gradscf.json'
STATIC = BASE/'V01_V02_water_rhf_spectra/gradscf.json'
OUTPUT = BASE/'V06_water_dynamic_raman'
FREQUENCIES = np.linspace(0., .55, 881)
DAMPINGS = np.array([.02, .01, .005])
HARTREE_TO_EV = physical_constants['Hartree energy in eV'][0]
RAMAN_CONVERSION = physical_constants['Bohr radius'][0]**4*1e40*AMU_TO_ELECTRON_MASS


def activity(tensor):
    """Rotational invariant of a complex symmetric polarizability derivative."""
    isotropic = np.trace(tensor, axis1=-2, axis2=-1)/3
    traceless = tensor-isotropic[..., None, None]*np.eye(3)
    anisotropy = 1.5*np.sum(abs(traceless)**2, axis=(-2, -1))
    return (45*abs(isotropic)**2+7*anisotropy)*RAMAN_CONVERSION


def main():
    started = perf_counter()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT/'gradscf.json').write_text('{"complete": false}\n')
    harmonic, previous = json.loads(HARMONIC.read_text()), json.loads(STATIC.read_text())
    coordinates = jnp.asarray(harmonic['coordinates_bohr'])
    modes = np.moveaxis(np.asarray(harmonic['isotopes']['H2O']['displacements']), -1, 0)
    masses = np.repeat(np.asarray(harmonic['isotopes']['H2O']['masses_amu'])*AMU_TO_ELECTRON_MASS, 3)
    flat = modes.reshape(3, -1)
    np.testing.assert_allclose((flat*masses)@flat.T, np.eye(3), atol=2e-12, rtol=0)
    np.testing.assert_array_equal(previous['coordinates_bohr'], harmonic['coordinates_bohr'])
    np.testing.assert_array_equal(previous['displacements'], modes)
    molecule = gto.M(atom=list(zip(harmonic['symbols'], harmonic['coordinates_bohr'])),
                     basis=harmonic['basis'], unit='Bohr', cart=True)
    mean_field = scf.RHF(molecule, conv_tol=1e-12, conv_tol_density=1e-11,
                        conv_tol_grad=1e-10, integral_backend='native').run()
    assert mean_field.converged
    response = MolecularPolarizability(mean_field, gradient_tolerance=1e-10)
    make_reference = jax.jit(response.reference)
    reference, stationary = make_reference(coordinates)
    assert stationary
    dimension = response.nocc*(reference.mo_coeff.shape[1]-response.nocc)
    matrix = np.asarray(gen_tdhf_vind(reference, response.functional)(jnp.eye(2*dimension))).T
    a, b = matrix[:dimension, :dimension], matrix[:dimension, dimension:]
    np.testing.assert_allclose(a, a.T, atol=2e-11, rtol=0)
    np.testing.assert_allclose(b, b.T, atol=2e-11, rtol=0)
    stability = [float(np.linalg.eigvalsh(a-b)[0]), float(np.linalg.eigvalsh(a+b)[0])]
    assert min(stability) > 1e-5
    roots = np.linalg.eigvals(matrix)
    assert np.max(abs(roots.imag)) < 1e-10
    first_root = float(np.min(roots.real[roots.real > 1e-8]))

    # Apply the chain rule in two reusable pieces. Each tangent contains the
    # relaxed SCF orbitals/Fock, moving ERIs, dipoles, and (for DFT) grid/AOs.
    # Nothing is frozen when taking the mode derivative; the reference JVP is
    # reused over laser frequencies instead of rerunning SCF for every point.
    reference_jvp = jax.jit(lambda r, v: jax.jvp(lambda q: response.reference(q)[0], (r,), (v,)))
    tangents = [reference_jvp(coordinates, jnp.asarray(mode))[1] for mode in modes]
    evaluate = jax.jit(lambda ref, omega, eta: dynamic_polarizability(ref, response.functional, omega, damping=eta))
    mode_jvp = jax.jit(lambda ref, tangent, omega, eta: jax.jvp(
        lambda record: dynamic_polarizability(record, response.functional, omega, damping=eta).polarizability,
        (ref,), (tangent,))[1])
    static = np.asarray(evaluate(reference, 0., 0.).polarizability)
    static_modes = np.array([mode_jvp(reference, tangent, 0., 0.) for tangent in tangents])
    static_alpha_from_energy = -jax.jit(jax.hessian(lambda field: response.energy(coordinates, field)))(jnp.zeros(3))
    np.testing.assert_allclose(static.real, static_alpha_from_energy, atol=2e-8, rtol=2e-8)
    np.testing.assert_allclose(static_modes.real, previous['alpha_derivatives'], atol=3e-6, rtol=3e-5)
    np.testing.assert_allclose(activity(static_modes), previous['raman_activities_angstrom4_amu'], atol=2e-3, rtol=3e-5)

    polarizabilities, derivatives, residuals = [], [], []
    for damping in DAMPINGS:
        print('water RHF full-space scan, damping / Eh:', damping, flush=True)
        values, changes, norms = [], [], []
        for omega in FREQUENCIES:
            solved = evaluate(reference, omega, damping)
            assert solved.converged
            values.append(np.asarray(solved.polarizability))
            changes.append(np.asarray([mode_jvp(reference, tangent, omega, damping) for tangent in tangents]))
            norms.append(float(solved.residual_norm))
        polarizabilities.append(values); derivatives.append(changes); residuals.append(norms)
    polarizabilities, derivatives = np.array(polarizabilities), np.array(derivatives)
    np.testing.assert_allclose(polarizabilities, polarizabilities.swapaxes(-1, -2), atol=2e-8, rtol=2e-9)
    np.testing.assert_allclose(derivatives, derivatives.swapaxes(-1, -2), atol=2e-7, rtol=2e-8)
    activities = activity(derivatives)
    frequency_checks = []
    for eta, curve in zip(DAMPINGS, polarizabilities):
        absorption = np.trace(curve.imag, axis1=-2, axis2=-1)/3
        areas = [float(simpson(absorption[::stride], x=FREQUENCIES[::stride])) for stride in [4, 2, 1]]
        relative = abs(areas[-1]-areas[-2])/abs(areas[-1])
        assert relative < 2e-4
        frequency_checks.append(dict(damping_hartree=float(eta), samples=[len(FREQUENCIES[::stride]) for stride in [4, 2, 1]],
                                     integrated_imaginary_isotropic_alpha=areas, last_relative_change=relative))

    checks = []
    probes = [(.11, .01), (.85*first_root, .005)]
    for mode, (direction, tangent) in enumerate(zip(modes, tangents), 1):
        for omega, eta in probes:
            actual = np.asarray(mode_jvp(reference, tangent, omega, eta))
            for maximum_step in [3e-4, 1e-4]:
                step = maximum_step/np.max(abs(direction))
                plus, plus_valid = make_reference(coordinates+step*direction)
                minus, minus_valid = make_reference(coordinates-step*direction)
                assert plus_valid & minus_valid
                finite = (evaluate(plus, omega, eta).polarizability-evaluate(minus, omega, eta).polarizability)/(2*step)
                error = float(np.max(abs(actual-finite)))
                checks.append(dict(mode=mode, frequency_hartree=omega, damping_hartree=eta,
                    max_cartesian_step_bohr=maximum_step, normal_coordinate_step=float(step), max_error=error,
                    derivative_real=actual.real.tolist(), derivative_imag=actual.imag.tolist()))
            np.testing.assert_allclose(actual, finite, atol=3e-5, rtol=3e-4)
    np.savez_compressed(OUTPUT/'response.npz', frequencies_hartree=FREQUENCIES, dampings_hartree=DAMPINGS,
        polarizability=polarizabilities, mode_derivatives=derivatives, activities_angstrom4_amu=activities,
        static_polarizability=static, static_mode_derivatives=static_modes)
    source = hashlib.sha256()
    for path in sorted(Path('src').rglob('*.py')):
        source.update(path.as_posix().encode()); source.update(path.read_bytes())
    report = dict(complete=True, method='RHF/TDHF', basis=harmonic['basis'], cart=True,
        symbols=harmonic['symbols'], coordinates_bohr=harmonic['coordinates_bohr'],
        mode_frequencies_cm1=harmonic['isotopes']['H2O']['frequencies_cm1'], mode_displacements=modes.tolist(),
        mode_units='bohr sqrt(electron mass); V.T M_electron V=I; fixed harmonic directions',
        polarizability_units='bohr^3', derivative_units='bohr^3/(bohr sqrt(electron mass))',
        activity_units='angstrom^4/amu', raman_conversion=RAMAN_CONVERSION,
        frequency_range_hartree=[0., .55], dampings_hartree=DAMPINGS.tolist(), frequency_samples=len(FREQUENCIES),
        occupied_virtual_dimension=dimension, real_linear_dimension=4*dimension, truncated_roots=False,
        stability_min_eigenvalues_hartree=stability, first_electronic_root_hartree=first_root,
        maximum_linear_residual=float(np.max(residuals)), frequency_grid_checks=frequency_checks,
        geometry_fd_checks=checks, static_energy_ad_error=float(np.max(abs(static.real-static_alpha_from_energy))),
        static_previous_mode_error=float(np.max(abs(static_modes.real-previous['alpha_derivatives']))),
        static_activities_angstrom4_amu=activity(static_modes).tolist(),
        input_sha256={str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in [HARMONIC, STATIC]},
        source_python_sha256=source.hexdigest(), elapsed_seconds=perf_counter()-started,
        peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, hostname=platform.node(),
        python=platform.python_version(), jax=jax.__version__, dtype='float64',
        approximation='Harmonic frequency-dependent Placzek activity. Phenomenological damping; no full resonant vibronic KHD, thermal, scattered-frequency or absolute cross-section factors.')
    (OUTPUT/'gradscf.json').write_text(json.dumps(report, indent=2)+'\n')
    print('V06 native full-space/static/mode/frequency gates PASS', flush=True)


if __name__ == '__main__':
    main()
