"""A02: JVP, VJP and energy HVP of real water RHF observables.

Twelve dimensionless parameters scale all nine Cartesian coordinates and
three electric-field components. The dense matrices below are validation
references; the product calls themselves use JAX contractions. This is a
fixed-geometry response benchmark, not a vibrational-frequency calculation.
Run on jy_gpu_1 with CPU float64. No CLI and no replacement SCF solver.
"""
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from gradscf import gto, scf
from gradscf_tools.molecular_response import MolecularEnergy


OUTPUT = Path("reproducibility/ad_showcase/A02_molecular_products")
R_SCALE = .05   # bohr per dimensionless coordinate parameter
F_SCALE = .002  # atomic electric-field unit per dimensionless field parameter
POINT = np.array([.12, -.08, .05, -.18, .09, .11, .06, -.01, -.16, .35, -.20, .25])
FD_STEPS = (1e-2, 3e-3, 1e-3)


def timed(function, *arguments):
    start = perf_counter()
    result = function(*arguments)
    jax.block_until_ready(result)
    return result, perf_counter()-start


def peak_rss_mib():
    divisor = 1024**2 if platform.system() == "Darwin" else 1024
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/divisor


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[2]
    source = hashlib.sha256()
    for path in sorted((root/"src").rglob("*.py")):
        source.update(path.relative_to(root).as_posix().encode())
        source.update(path.read_bytes())
    report = dict(complete=False, source_python_sha256=source.hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        symbols=["O", "H", "H"], basis="6-31g*", cart=True, method="RHF",
        coordinate_scale_bohr=R_SCALE, field_scale_au=F_SCALE, point_theta=POINT.tolist(),
        parameter_labels=[f"{atom}_{axis}" for atom in ("O", "H1", "H2") for axis in "xyz"]
                         + ["F_x", "F_y", "F_z"],
        observable_labels=["E/Eh", "mu_x/(e bohr)", "mu_y/(e bohr)", "mu_z/(e bohr)"],
        hessian_definition="d2(E/Eh)/dtheta_i dtheta_j; not an unscaled Cartesian Hessian",
        origin_bohr=[0., 0., 0.], electronic_response="implicit",
        hostname=platform.node(), python=platform.python_version(), jax=jax.__version__,
        numpy=np.__version__, dtype="float64", devices=[str(device) for device in jax.devices()],
        persistent_compilation_cache_enabled=bool(jax.config.jax_enable_compilation_cache),
        threads={key: os.environ.get(key) for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS")})
    path = OUTPUT/"gradscf.json"
    path.write_text(json.dumps(report, indent=2)+"\n")
    mol = gto.M(atom="O 0 0 0; H 0 1.43 1.11; H 0 -1.43 1.11",
                basis=report["basis"], unit="Bohr", cart=True)
    mf = scf.RHF(mol, conv_tol=1e-12, conv_tol_density=1e-11, conv_tol_grad=1e-10,
                 max_cycle=150, integral_backend="native").run()
    assert mf.converged
    model = MolecularEnergy(mf, gradient_tolerance=1e-10)
    base = model.coordinates
    point = jnp.asarray(POINT)
    rng = np.random.default_rng(641)
    direction = rng.normal(size=12)
    direction /= np.linalg.norm(direction)
    cotangent = np.array([.7, -.2, .3, .4])
    cotangent /= np.linalg.norm(cotangent)
    direction, cotangent = jnp.asarray(direction), jnp.asarray(cotangent)

    def unpack(theta):
        return base+R_SCALE*theta[:9].reshape(3, 3), F_SCALE*theta[9:]

    def energy(theta):
        return model(*unpack(theta))  # E/Eh, numerical normalization one

    def observables(theta):
        r, field = unpack(theta)
        return jnp.concatenate([model(r, field)[None], model.dipole(r, field)])

    solve = jax.jit(lambda theta: model.solve(*unpack(theta)))
    state, forward_seconds = timed(solve, point)
    assert bool(state.stationary) and int(state.iterations) > 0
    forward = jax.jit(observables)
    values, _ = timed(forward, point)
    assert np.isfinite(values).all()
    print("A02 forward stationary; iterations", int(state.iterations), flush=True)

    jvp_function = jax.jit(lambda theta, v: jax.jvp(observables, (theta,), (v,))[1])
    jvp, jvp_cold = timed(jvp_function, point, direction)
    jvp_warm = [timed(jvp_function, point, direction)[1] for _ in range(3)]
    (_, pullback), preparation_seconds = timed(jax.vjp, forward, point)
    product, pullback_cold = timed(pullback, cotangent)
    vjp = product[0]
    pullback_warm = [timed(pullback, cotangent)[1] for _ in range(3)]
    gradient_function = jax.jit(jax.grad(energy))
    hvp_function = jax.jit(lambda theta, v: jax.jvp(jax.grad(energy), (theta,), (v,))[1])
    hvp, hvp_cold = timed(hvp_function, point, direction)
    hvp_warm = [timed(hvp_function, point, direction)[1] for _ in range(3)]
    product_rss = peak_rss_mib()
    print("A02 JVP/VJP/HVP computed; assembling small dense validation references", flush=True)

    jacobian, dense_jacobian_seconds = timed(jax.jit(jax.jacrev(observables)), point)
    hessian, dense_hessian_seconds = timed(jax.jit(jax.jacfwd(jax.grad(energy))), point)
    gradient = gradient_function(point)
    np.testing.assert_allclose(jvp, jacobian@direction, atol=2e-8, rtol=2e-7)
    np.testing.assert_allclose(vjp, jacobian.T@cotangent, atol=2e-8, rtol=2e-7)
    np.testing.assert_allclose(hvp, hessian@direction, atol=2e-7, rtol=2e-6)
    np.testing.assert_allclose(hessian, hessian.T, atol=2e-7, rtol=2e-6)
    np.testing.assert_allclose(cotangent@jvp, direction@vjp, atol=2e-9, rtol=2e-7)
    np.testing.assert_allclose(-gradient[9:]/F_SCALE, values[1:], atol=1e-7, rtol=1e-7)
    np.testing.assert_allclose(gradient[:9].reshape(3, 3).sum(axis=0)/R_SCALE,
                               0., atol=2e-8, rtol=0)
    checks = []
    for step in FD_STEPS:
        plus, minus = point+step*direction, point-step*direction
        for displaced in (plus, minus):
            endpoint = solve(displaced)
            assert bool(endpoint.stationary), float(endpoint.gradient_norm)
        fd_jvp = (forward(plus)-forward(minus))/(2*step)
        fd_hvp = (gradient_function(plus)-gradient_function(minus))/(2*step)
        scalar_fd = cotangent@fd_jvp
        check = dict(step_theta=step, jvp=np.asarray(fd_jvp).tolist(), hvp=np.asarray(fd_hvp).tolist(),
            max_jvp_error=float(jnp.max(jnp.abs(fd_jvp-jvp))),
            max_hvp_error=float(jnp.max(jnp.abs(fd_hvp-hvp))),
            scalar_vjp_directional_error=float(jnp.abs(scalar_fd-direction@vjp)))
        checks.append(check)
        print("A02 FD", step, check["max_jvp_error"], check["max_hvp_error"], flush=True)
    np.testing.assert_allclose(checks[-1]["jvp"], jvp, atol=2e-7, rtol=2e-5)
    np.testing.assert_allclose(checks[-1]["hvp"], hvp, atol=2e-6, rtol=2e-5)
    errors = dict(jvp_dense=float(jnp.max(jnp.abs(jvp-jacobian@direction))),
        vjp_dense=float(jnp.max(jnp.abs(vjp-jacobian.T@cotangent))),
        hvp_dense=float(jnp.max(jnp.abs(hvp-hessian@direction))),
        adjoint=float(jnp.abs(cotangent@jvp-direction@vjp)),
        hessian_symmetry=float(jnp.max(jnp.abs(hessian-hessian.T))))
    report.update(complete=True, base_coordinates_bohr=np.asarray(base).tolist(),
        coordinates_bohr=np.asarray(unpack(point)[0]).tolist(), field_au=np.asarray(unpack(point)[1]).tolist(),
        direction_theta=np.asarray(direction).tolist(), cotangent=np.asarray(cotangent).tolist(),
        values=np.asarray(values).tolist(), gradient_theta=np.asarray(gradient).tolist(),
        jacobian_theta=np.asarray(jacobian).tolist(), hessian_theta=np.asarray(hessian).tolist(),
        jvp=np.asarray(jvp).tolist(), vjp=np.asarray(vjp).tolist(), hvp=np.asarray(hvp).tolist(),
        contraction_errors=errors, finite_difference_checks=checks,
        forward_stationary=bool(state.stationary), orbital_iterations=int(state.iterations),
        orbital_gradient_norm=float(state.gradient_norm), orbital_gradient_tolerance=1e-10,
        implicit_tolerance=1e-10, implicit_max_iter=100, implicit_restart=40,
        seed_coefficients_sha256=hashlib.sha256(np.asarray(mf.mo_coeff).tobytes()).hexdigest(),
        timing_seconds=dict(forward_state=forward_seconds, jvp_first=jvp_cold, jvp_warm=jvp_warm,
            vjp_preparation=preparation_seconds, pullback_first=pullback_cold, pullback_warm=pullback_warm,
            hvp_first=hvp_cold, hvp_warm=hvp_warm, dense_jacobian_validation=dense_jacobian_seconds,
            dense_hessian_validation=dense_hessian_seconds),
        cumulative_peak_after_products_mib=product_rss, peak_after_validation_mib=peak_rss_mib(),
        scope="Scaled 12-parameter coordinate/field response at one nonzero point. Dense matrices are small validation oracles; no scaling or vibrational-frequency claim.")
    path.write_text(json.dumps(report, indent=2)+"\n")
    print("A02 native contraction, adjoint, dense-reference and multistep FD gates PASS", errors, flush=True)


if __name__ == "__main__":
    main()

# Measured on jy_gpu_1, frozen v7 CPU/x64 source: native gates PASS,
# dense/adjoint maximum error <=1.22e-14; independent four-energy HVP <=5.77e-9.
# Matrices differentiate normalized observables with respect to scaled theta.
