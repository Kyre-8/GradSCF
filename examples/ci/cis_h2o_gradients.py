"""X06: three distinct sensitivities of a gauge-invariant water CIS probe.

The target is the squared length-gauge transition dipole |mu_01|^2.
Geometry derivatives include relaxed HF and CIS response. Amplitude derivatives
vary a normalized CIS state at fixed orbitals. Reference derivatives rotate
occupied/virtual orbitals at fixed amplitudes, without an off-HF CIS solve.
The three derivatives are separate partial derivatives, not additive parts of
the geometry derivative. No PySCF or CLI is used in this native calculation.
"""
import hashlib
import json
from pathlib import Path
import platform
import resource
from time import perf_counter

import jax
import jax.numpy as jnp
from jax.scipy.linalg import expm
import numpy as np

jax.config.update("jax_enable_x64", True)

from gradscf import gto, scf
from gradscf.solvers import EigenResponseConfig, LinearSolverConfig
from gradscf_tools.excited import CISGeometry, singlet_cis_densities


BASIS = "6-31g*"
ATOM = "O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692"
OUTPUT = Path("reproducibility/ad_showcase/X06_water_cis_probes")


def _givens_rotation(theta, nmo, pairs):
    """Retain the original example's explicit selected-pair rotation chart."""
    rotation = jnp.eye(nmo)
    for angle, (occupied, virtual) in zip(theta, pairs):
        cosine, sine = jnp.cos(angle), jnp.sin(angle)
        givens = jnp.eye(nmo).at[occupied, occupied].set(cosine).at[virtual, virtual].set(cosine)
        givens = givens.at[occupied, virtual].set(sine).at[virtual, occupied].set(-sine)
        rotation = rotation @ givens
    return rotation


def wavefunction_probe(coefficients, amplitudes, position, nocc):
    """Normalized singlet transition strength; no SVD frame is differentiated."""
    transition = singlet_cis_densities(coefficients, amplitudes, nocc=nocc).transition
    moment = -jnp.einsum("pq,xqp->x", transition, position)
    return jnp.sum(moment**2)


def main():
    started = perf_counter()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT/"gradscf.json").write_text('{"complete": false}\n')
    molecule = gto.M(atom=ATOM, basis=BASIS, unit="Angstrom", cart=True)
    hf = scf.RHF(molecule, integral_backend="native", conv_tol=1e-12,
                 conv_tol_density=1e-11, conv_tol_grad=1e-11, max_cycle=180).run()
    if not hf.converged:
        raise RuntimeError("RHF did not converge.")
    policy = EigenResponseConfig(target="eigenpairs",
        linear_config=LinearSolverConfig(method="direct", max_dense=128, rtol=1e-11))
    model = CISGeometry(hf, nroots=6, gradient_tolerance=1e-11, response=policy)
    coordinates = np.asarray(model.energy.coordinates)
    values, derivatives, state = model.transition_derivatives(coordinates, 0)
    coefficients, amplitude = jnp.asarray(state.coefficients), jnp.asarray(state.amplitudes[0])
    nocc, nmo = model.nocc, coefficients.shape[1]
    position = model.energy.plan.evaluate("dipole", model.energy.parameters, origin=jnp.zeros(3))
    overlap = np.asarray(model.energy.plan.evaluate("overlap", model.energy.parameters))
    np.testing.assert_allclose(np.sum(np.asarray(amplitude)**2), 1., atol=1e-12, rtol=0)
    geometry_value = float(values[:3]@values[:3])
    geometry_gradient = 2*np.einsum("x,xai->ai", values[:3], derivatives[:3])
    amplitude_value, amplitude_gradient = jax.value_and_grad(
        lambda x: wavefunction_probe(coefficients, x, position, nocc))(amplitude)

    def reference_rotation_probe(theta):
        # A norm-preserving reference chart at fixed X. It changes the
        # electronic states' embedding, not a reconverged HF/CIS solution.
        k = jnp.zeros((nmo, nmo)).at[:nocc, nocc:].set(theta)
        k = k.at[nocc:, :nocc].set(-theta.T)
        return wavefunction_probe(coefficients@expm(k), amplitude, position, nocc)

    theta0 = jnp.zeros((nocc, nmo-nocc))
    reference_value, reference_gradient = jax.jit(jax.value_and_grad(reference_rotation_probe))(theta0)
    np.testing.assert_allclose([amplitude_value, reference_value], geometry_value, atol=1e-10, rtol=0)
    np.testing.assert_allclose(jnp.sum(amplitude_gradient*amplitude), 0., atol=1e-10, rtol=0)
    # Preserve the old selected-pair chart and check its infinitesimal map
    # against the full occupied/virtual sensitivity heatmap.
    pairs = tuple((i, nocc+i) for i in range(min(nocc, nmo-nocc, 3)))
    selected_gradient = jax.grad(lambda theta: wavefunction_probe(
        coefficients@_givens_rotation(theta, nmo, pairs), amplitude, position, nocc))(jnp.zeros(len(pairs)))
    np.testing.assert_allclose(selected_gradient, np.asarray(reference_gradient)[np.arange(len(pairs)), np.arange(len(pairs))],
                               atol=1e-10, rtol=0)

    rng = np.random.default_rng(106)
    u, _ = np.linalg.qr(rng.normal(size=(nocc, nocc)))
    v, _ = np.linalg.qr(rng.normal(size=(nmo-nocc, nmo-nocc)))
    rotated_c = jnp.column_stack([coefficients[:, :nocc]@u, coefficients[:, nocc:]@v])
    rotated_x = u.T@amplitude@v
    original_fields = singlet_cis_densities(coefficients, amplitude, nocc=nocc)
    rotated_fields = singlet_cis_densities(rotated_c, rotated_x, nocc=nocc)
    gauge_errors = dict(
        transition=float(np.max(np.abs(original_fields.transition-rotated_fields.transition))),
        attachment=float(np.max(np.abs(original_fields.attachment-rotated_fields.attachment))),
        scalar=abs(float(wavefunction_probe(rotated_c, rotated_x, position, nocc))-geometry_value),
        phase=abs(float(wavefunction_probe(coefficients, -amplitude, position, nocc))-geometry_value),
    )
    rotated_gradient = jax.grad(lambda x: wavefunction_probe(rotated_c, x, position, nocc))(rotated_x)
    gauge_errors["amplitude_gradient_covariance"] = float(np.max(np.abs(rotated_gradient-u.T@amplitude_gradient@v)))
    if max(gauge_errors.values()) > 1e-10:
        raise RuntimeError(f"Orbital/state gauge check failed: {gauge_errors}")

    anchor = model.anchor(coordinates, state, 0)
    tracking = []
    def geometry_probe(r):
        value, current = model.transition_properties(r, 0)
        match = model.match(r, current, anchor)
        if match["root"] != 0:
            value, current = model.transition_properties(r, match["root"])
        tracking.append(match)
        return float(value[:3]@value[:3])

    checks = []
    for kind, shape, gradient, function in (
        ("geometry", coordinates.shape, geometry_gradient, geometry_probe),
        ("amplitude", amplitude.shape, np.asarray(amplitude_gradient),
         lambda x: float(wavefunction_probe(coefficients, x, position, nocc))),
        ("reference_rotation", theta0.shape, np.asarray(reference_gradient),
         lambda theta: float(reference_rotation_probe(jnp.asarray(theta))))):
        point = coordinates if kind == "geometry" else np.asarray(amplitude) if kind == "amplitude" else np.asarray(theta0)
        for index in range(3):
            direction = rng.normal(size=shape)
            if kind == "geometry":
                direction -= direction.mean(axis=0)
            elif kind == "amplitude":
                direction -= np.sum(direction*np.asarray(amplitude))*np.asarray(amplitude)
            direction /= np.linalg.norm(direction)
            expected = float(np.sum(gradient*direction))
            finite = []
            for step in (1e-3, 3e-4, 1e-4):
                derivative = (function(point+step*direction)-function(point-step*direction))/(2*step)
                finite.append(dict(step=step, derivative=derivative, absolute_error=abs(derivative-expected)))
            np.testing.assert_allclose(finite[-1]["derivative"], expected, atol=2e-6, rtol=0)
            checks.append(dict(kind=kind, index=index, direction=direction.tolist(),
                               derivative=expected, finite_differences=finite))
            print(kind, index, "last FD error", finite[-1]["absolute_error"], flush=True)

    normalized_x = np.asarray(amplitude)/np.linalg.norm(amplitude)
    purity = float(np.sum((normalized_x@normalized_x.T)**2))
    repository = Path(__file__).resolve().parents[2]
    digest = hashlib.sha256()
    for path in sorted((repository/"src").rglob("*.py")):
        digest.update(path.relative_to(repository).as_posix().encode())
        digest.update(path.read_bytes())
    report = dict(
        complete=True, molecule="H2O", symbols=list(model.energy.spec.symbols), basis=BASIS, cart=True,
        coordinates_bohr=coordinates.tolist(), nroots=6, selected_root=0, nocc=nocc, nmo=nmo,
        ground_energy_hartree=float(state.ground_energy),
        excitation_energies_hartree=np.asarray(state.excitation_energies).tolist(),
        cis_residual_norms=np.asarray(state.ci_residual_norms).tolist(),
        coefficients_ao=np.asarray(coefficients).tolist(), amplitudes=np.asarray(amplitude).tolist(),
        position_integrals_bohr=np.asarray(position).tolist(), overlap=overlap.tolist(),
        transition_dipole_au=values[:3].tolist(), oscillator_strength=float(values[3]),
        probe_definition="Squared normalized singlet transition dipole |mu_01|^2, in (e Bohr)^2",
        probe=geometry_value, nuclear_gradient=geometry_gradient.tolist(),
        amplitude_gradient=np.asarray(amplitude_gradient).tolist(),
        reference_rotation_gradient=np.asarray(reference_gradient).tolist(),
        reference_rotation_pairs=[list(pair) for pair in pairs],
        selected_reference_gradient=np.asarray(selected_gradient).tolist(),
        reference_scope="Fixed X, orthogonal OV rotation of reference/excitation embedding; no off-HF CIS solve",
        nto_purity=purity, gauge_errors=gauge_errors, finite_difference_checks=checks, tracking=tracking,
        source_python_sha256=digest.hexdigest(), script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        elapsed_seconds=perf_counter()-started, peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        hostname=platform.node(), python=platform.python_version(), jax=jax.__version__, dtype="float64",
    )
    (OUTPUT/"gradscf.json").write_text(json.dumps(report, indent=2)+"\n")
    print("X06 native probe gates passed", report["elapsed_seconds"], "s", flush=True)
    return report


if __name__ == "__main__":
    main()

# Measured jy_gpu_1, CPU float64, two threads: native X06 gates passed in 65.59 s.
# q=.0727781650412459 (e Bohr)^2; geometry/amplitude/reference last-step FD
# errors <=2.43e-9 / 9.18e-10 / 3.09e-10, respectively. They are distinct partials.
# State-sign / same-block gauge scalar errors <=1.12e-16; no off-HF CIS solve.
