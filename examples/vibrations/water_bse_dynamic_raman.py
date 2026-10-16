"""Full-space G0W0/BSE polarizability and harmonic mode derivatives of water.

Run after G05 water RHF. The orbitals, QP roots, screening, BSE amplitudes
and position integrals all move. These are frequency-dependent Placzek
activities at an RHF minimum, not a vibronic KHD spectrum or GW forces.
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
from scipy.integrate import simpson

jax.config.update("jax_enable_x64", True)

from gradscf import bse, gto, scf
from gradscf.gw.screened import build_static_screening
from gradscf.solvers import LinearSolverConfig
from gradscf_tools.gw_response import MolecularGW
from gradscf_tools.vibrations import AMU_TO_ELECTRON_MASS
from water_dynamic_raman import activity


BASE = Path("reproducibility/ad_showcase")
HARMONIC = BASE/"G05_water_rhf_hessian/gradscf.json"
OUTPUT = BASE/"V06_water_bse_dynamic_raman"
FREQUENCIES = np.linspace(0., .55, 1761)
DAMPINGS = np.array([.02, .01, .005])


def main():
    started = perf_counter()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT/"gradscf.json").write_text('{"complete": false}\n')
    harmonic = json.loads(HARMONIC.read_text())
    coordinates = jnp.asarray(harmonic["coordinates_bohr"])
    modes = np.moveaxis(np.asarray(harmonic["isotopes"]["H2O"]["displacements"]), -1, 0)
    masses = np.repeat(np.asarray(harmonic["isotopes"]["H2O"]["masses_amu"])*AMU_TO_ELECTRON_MASS, 3)
    flat = modes.reshape(3, -1)
    np.testing.assert_allclose((flat*masses)@flat.T, np.eye(3), atol=2e-12, rtol=0)
    molecule = gto.M(atom=list(zip(harmonic["symbols"], harmonic["coordinates_bohr"])),
                     basis=harmonic["basis"], unit="Bohr", cart=True)
    mean_field = scf.RHF(molecule, conv_tol=1e-12, conv_tol_grad=1e-10,
                        integral_backend="native").run()
    assert mean_field.converged

    # Select numerical seeds once outside AD, then continue the converged
    # local finite-eta roots. The selector has no derivative assigned to it.
    settings = dict(nw=100, eta=1e-5, qp_solver="newton")
    initial = MolecularGW(mean_field, **settings)
    gradient = np.asarray(jax.jit(jax.grad(initial.reference))(coordinates))
    assert np.max(abs(gradient)) < 1e-7
    seeds = initial.pole_seeds()
    assert np.asarray(seeds.valid).all()
    initial = MolecularGW(mean_field, qp_initial=seeds.seeds, **settings)
    anchor = jax.jit(initial)(coordinates)
    assert anchor.converged
    model = MolecularGW(mean_field, qp_initial=anchor.mo_energy, **settings)
    space = bse.make_bse_space(model.nmo, model.nocc)
    config = bse.BSEConfig(nroots=space.size, tda=False, solver="dense", conv_tol=1e-10,
        response_config=LinearSolverConfig(method="direct", rtol=1e-10, max_dense=256))

    def reference(r):
        qp, state, position, _ = model.optical(r, config=config)
        return (state, position), (qp.mo_energy, qp.qp_residual, qp.converged_mask)

    def check_reference(record, diagnostics):
        state, position = record
        assert all(np.asarray(x).all() for x in (state.converged, state.stable,
                   state.response_valid, state.screening_valid, diagnostics[2]))
        assert np.isfinite(np.asarray(position)).all()
        assert np.max(abs(np.asarray(diagnostics[1]))) < 1e-10

    forward = jax.jit(reference)
    start = perf_counter()
    point, diagnostics = forward(coordinates)
    jax.block_until_ready(point)
    cold_forward = perf_counter()-start
    check_reference(point, diagnostics)
    start = perf_counter()
    jax.block_until_ready(forward(coordinates))
    warm_forward = perf_counter()-start
    # Cache three geometry tangents, not three frequency-dependent SCF runs.
    reference_jvp = jax.jit(lambda r, v: jax.jvp(reference, (r,), (v,), has_aux=True))
    tangents, response_times = [], []
    for mode, direction in enumerate(modes, 1):
        start = perf_counter()
        _, tangent, diagnostic = reference_jvp(coordinates, jnp.asarray(direction))
        jax.block_until_ready(tangent)
        response_times.append(perf_counter()-start)
        tangents.append(tangent)
        print("BSE mode", mode, "geometry JVP / s", response_times[-1], flush=True)
    start = perf_counter()
    jax.block_until_ready(reference_jvp(coordinates, jnp.asarray(modes[0])))
    warm_derivative = perf_counter()-start

    evaluate = jax.jit(lambda record, omega, eta: bse.polarizability(*record, space, omega, eta=eta))
    mode_jvp = jax.jit(lambda record, tangent, omega, eta: jax.jvp(
        lambda value: bse.polarizability(*value, space, omega, eta=eta), (record,), (tangent,))[1])
    alpha = np.array([evaluate(point, FREQUENCIES, eta) for eta in DAMPINGS])
    changes = np.array([[mode_jvp(point, tangent, FREQUENCIES, eta)
                        for tangent in tangents] for eta in DAMPINGS])
    assert np.isfinite(alpha).all() and np.isfinite(changes).all()
    np.testing.assert_allclose(alpha, alpha.swapaxes(-1, -2), atol=2e-9, rtol=2e-10)
    np.testing.assert_allclose(changes, changes.swapaxes(-1, -2), atol=2e-8, rtol=2e-9)
    static = np.asarray(evaluate(point, 0., 0.))
    static_changes = np.array([mode_jvp(point, tangent, 0., 0.) for tangent in tangents])
    frequency_checks = []
    for eta, curve in zip(DAMPINGS, alpha):
        imaginary = np.trace(curve.imag, axis1=-2, axis2=-1)/3
        areas = [float(simpson(imaginary[::stride], x=FREQUENCIES[::stride])) for stride in (4, 2, 1)]
        error = abs(areas[-1]-areas[-2])/abs(areas[-1])
        frequency_checks.append(dict(eta_hartree=float(eta), samples=[441, 881, 1761],
            integrated_imaginary_alpha=areas, last_relative_change=error))
        assert error < 2e-4

    checks = []
    for mode, direction in enumerate(modes):
        for step in (.01, .003, .001):
            plus, plus_diagnostic = forward(coordinates+step*direction)
            minus, minus_diagnostic = forward(coordinates-step*direction)
            check_reference(plus, plus_diagnostic); check_reference(minus, minus_diagnostic)
            finite = np.array([(evaluate(plus, FREQUENCIES, eta)-evaluate(minus, FREQUENCIES, eta))/(2*step)
                               for eta in DAMPINGS])
            difference = abs(changes[:, mode]-finite)
            scaled = float(np.max(difference/(3e-5+3e-4*abs(changes[:, mode]))))
            checks.append(dict(mode=mode+1, step_bohr_sqrt_me=step,
                maximum_cartesian_displacement_bohr=float(step*np.max(abs(direction))),
                max_absolute_error=float(np.max(difference)), max_scaled_error=scaled))
            print("mode/step/FD scaled error", mode+1, step, scaled, flush=True)
        assert scaled < 1., "Complete spectral derivative failed its fixed AD-FD gate."

    # Export the existing kernel for a separate SciPy resolvent oracle. This
    # validation matrix is not used to produce the spectral AD result above.
    qp, factors = jax.jit(model._solve)(coordinates)
    factors = .5*(factors+factors.swapaxes(-1, -2))
    screen = build_static_screening(qp.screening_energy, factors,
                                   occupied=space.occupied, virtual=space.virtual)
    assert screen.valid
    a, b = bse.build_bse_operators(qp.mo_energy, factors, space, screen)
    identity = jnp.eye(space.size)
    a, b = np.asarray(a.apply(identity)), np.asarray(b.apply(identity))
    position = np.asarray(point[1])[:, :model.nocc, model.nocc:].reshape(3, -1).T
    np.savez_compressed(OUTPUT/"response.npz", frequencies_hartree=FREQUENCIES,
        dampings_hartree=DAMPINGS, polarizability=alpha, mode_derivatives=changes,
        activities_angstrom4_amu=activity(changes), static_polarizability=static,
        static_mode_derivatives=static_changes, a=a, b=b, position=position,
        excitation_energies=np.asarray(point[0].excitation_energies),
        x=np.asarray(point[0].x_amplitudes), y=np.asarray(point[0].y_amplitudes))
    source = hashlib.sha256()
    for path in sorted(Path("src").rglob("*.py")):
        source.update(path.as_posix().encode()); source.update(path.read_bytes())
    report = dict(complete=True, method="RHF/G0W0/full static BSE", basis=harmonic["basis"], cart=True,
        symbols=harmonic["symbols"], coordinates_bohr=harmonic["coordinates_bohr"],
        rhf_gradient_hartree_bohr=gradient.tolist(),
        mode_frequencies_cm1=harmonic["isotopes"]["H2O"]["frequencies_cm1"], displacements=modes.tolist(),
        normal_coordinate_units="bohr sqrt(electron mass); V.T M_electron V=I",
        polarizability_units="bohr^3", derivative_units="bohr^3/(bohr sqrt(electron mass))",
        activity_units="angstrom^4/amu", nmo=model.nmo, nocc=model.nocc, nroots=space.size,
        truncated_roots=False, nw=100, qp_eta_hartree=1e-5, qp_tolerance_hartree=1e-11,
        qp_anchor_hartree=np.asarray(anchor.mo_energy).tolist(), qp_weight=np.asarray(anchor.qp_weight).tolist(),
        qp_residual_hartree=np.asarray(diagnostics[1]).tolist(), bse_residual_hartree=np.asarray(point[0].residual_norms).tolist(),
        factor_tolerance=model.factor_tolerance, dampings_hartree=DAMPINGS.tolist(),
        frequency_grid_checks=frequency_checks, geometry_fd_checks=checks,
        static_polarizability=static.real.tolist(), static_activities_angstrom4_amu=activity(static_changes).tolist(),
        input_sha256={str(HARMONIC): hashlib.sha256(HARMONIC.read_bytes()).hexdigest()},
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), source_python_sha256=source.hexdigest(),
        geometry_jvp_seconds=response_times, elapsed_seconds=perf_counter()-started,
        timings_seconds=dict(cold_forward=cold_forward, warm_forward=warm_forward,
                             cold_derivative=response_times[0], warm_derivative=warm_derivative),
        peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, hostname=platform.node(),
        python=platform.python_version(), jax=jax.__version__, backend=jax.default_backend(), dtype="float64",
        approximation="Harmonic frequency-dependent Placzek activity at the RHF minimum. Phenomenological optical width; no GW total-energy Hessian, vibronic KHD, thermal or absolute Raman cross-section factors.")
    (OUTPUT/"gradscf.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print("Full-space BSE dynamic polarizability/mode/frequency gates PASS", flush=True)


if __name__ == "__main__":
    main()

# Recorded on jy_gpu_1, CPU float64, OMP=2 / BLAS=1 (2026-10-16):
# 70 full-BSE states; 3 modes x 3 optical widths x 1761 frequencies: PASS.
# Cold/warm optical forward: 12.1461 / 0.2351 s (after the QP anchor).
# Cold/warm complete geometry JVP: 19.3802 / 0.7610 s.
# Final scaled spectral FD errors: [5.4824e-4, 2.4082e-3, 5.3353e-3] < 1.
# Static activities / angstrom^4 amu^-1: [5.11025004, 59.61947644, 34.73846244].
# Calculation body: 125.114 s; process peak RSS: 4,401,268 KiB.
