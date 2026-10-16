"""A05: bounded H2 DF-HF exponent optimization at fixed basis topology.

The auxiliary basis and geometry are fixed. Native normalized 1e/3c integrals
are rebuilt inside the objective; only the unchanged auxiliary metric is
cached. For this four-AO example, DF factors form a small approximate ERI
tensor for the existing orbital solver. This is not exact four-center HF.
No CLI; run on jy_gpu_1 with CPU float64.
"""
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import platform
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize

jax.config.update("jax_enable_x64", True)

from gradscf import gto, integrals, scf
from gradscf.integrals.molecular.density_fitting import unpack_factors


OUTPUT = Path("reproducibility/ad_showcase/A05_basis_exponents")
BASIS = {"H": [[0, [3.42525091, .15432897], [.62391373, .53532814],
                 [.16885540, .44463454]], [0, [.08, 1.]]]}
AUXILIARY_BASIS = "def2-universal-jkfit"
LOWER, UPPER = -.7, .7  # log-scale bounds; alpha = alpha0 * exp(theta)


def make_problem():
    mol = gto.M(atom="H 0 0 -.7; H 0 0 .7", basis=BASIS, unit="Bohr", cart=True)
    seed = scf.RHF(mol, conv_tol=1e-12, conv_tol_grad=1e-10,
                   integral_backend="native").run()
    assert seed.converged
    spec = mol.to_spec()
    topology, parameters = integrals.prepare_basis(spec, BASIS, cart=True)
    auxiliary, auxiliary_parameters = integrals.prepare_basis(spec, AUXILIARY_BASIS, cart=True)
    plan = integrals.make_plan(topology)
    ri = integrals.make_auxiliary_plan(topology, auxiliary)
    metric = ri.metric_factor(parameters, auxiliary_parameters)
    occupations = jnp.stack([seed.mo_occ/2]*2)
    assert topology.nao == 4 and topology.primitive_counts == (3, 1, 3, 1)

    def inputs(theta):
        # Equal H atoms share two scale parameters, with all raw coefficients
        # fixed. The integral plan differentiates primitive/shell normalization.
        bound = replace(parameters, exponents=tuple(
            alpha*jnp.exp(theta[shell % 2]) for shell, alpha in enumerate(parameters.exponents)))
        overlap = plan.evaluate("overlap", bound)
        hcore = plan.evaluate("kinetic", bound)+plan.evaluate("nuclear", bound)
        factors = unpack_factors(ri.factors(bound, auxiliary_parameters, metric_factor=metric), topology.nao)
        eri_df = jnp.einsum("Qpq,Qrs->pqrs", factors, factors)
        return overlap, hcore, eri_df

    def objective(theta):
        overlap, hcore, eri_df = inputs(theta)
        n = topology.nao
        result = scf.minimize_roks_from_integrals(
            overlap=overlap, hcore=hcore, eri=eri_df, nuclear_repulsion=spec.nuclear_repulsion,
            ao=jnp.zeros((0, n)), ao_deriv1=jnp.zeros((4, 0, n)), grid_weights=jnp.zeros(0),
            mo_coeff=seed.mo_coeff, mo_occ=occupations, xc_spec="hf",
            orthonormalize_initial=True, gradient_tolerance=1e-10, max_iterations=300,
            differentiation=scf.SCFDifferentiationConfig(max_iter=100, restart=40))
        aux = dict(stationary=result.stationary, orbital_gradient_norm=result.gradient_norm,
            iterations=result.iterations, normalization_error=jnp.max(jnp.abs(jnp.diag(overlap)-1)),
            minimum_overlap_eigenvalue=jnp.linalg.eigvalsh(overlap)[0])
        return jnp.where(result.stationary, result.total_energy, jnp.nan), aux

    metadata = dict(basis=BASIS, auxiliary_basis=AUXILIARY_BASIS, cart=True,
        symbols=["H", "H"], coordinates_bohr=np.asarray(parameters.nuclear_coords).tolist(),
        topology=asdict(topology), fixed_auxiliary_metric_signature=metric.signature,
        metric_method=metric.method, metric_lindep=metric.lindep,
        theta_definition="Two dimensionless log scales, shared by both H atoms: contracted s then diffuse s.",
        exponent_units="bohr^-2", energy_units="Hartree", bounds_theta=[LOWER, UPPER],
        seed_coefficients_sha256=hashlib.sha256(np.asarray(seed.mo_coeff).tobytes()).hexdigest())
    return objective, inputs, metadata


def main():
    start = perf_counter()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    objective, inputs, metadata = make_problem()
    value_grad = jax.jit(jax.value_and_grad(objective, has_aux=True))
    forward = jax.jit(objective)
    evaluations, trajectory = [], []

    def checked(theta):
        (value, info), gradient = value_grad(jnp.asarray(theta))
        assert bool(info["stationary"]) and np.isfinite(gradient).all() and np.isfinite(value)
        assert float(info["normalization_error"]) < 2e-12
        assert float(info["minimum_overlap_eigenvalue"]) > 1e-5
        record = dict(theta=np.asarray(theta).tolist(), energy=float(value),
            gradient_theta=np.asarray(gradient).tolist(),
            orbital_iterations=int(info["iterations"]),
            orbital_gradient_norm=float(info["orbital_gradient_norm"]),
            normalization_error=float(info["normalization_error"]),
            minimum_overlap_eigenvalue=float(info["minimum_overlap_eigenvalue"]))
        evaluations.append(record)
        return float(value), np.asarray(gradient)

    initial = np.zeros(2)
    initial_energy, _ = checked(initial)
    trajectory.append(evaluations[-1])

    def accepted(theta):
        checked(theta)
        trajectory.append(evaluations[-1])
        print("A05 accepted", len(trajectory)-1, theta, trajectory[-1]["energy"], flush=True)

    fit = minimize(checked, initial, jac=True, method="L-BFGS-B",
        bounds=[(LOWER, UPPER)]*2, callback=accepted,
        options=dict(maxiter=60, ftol=1e-13, gtol=1e-8, maxls=30))
    final_energy, final_gradient = checked(fit.x)
    assert fit.success, fit.message
    assert final_energy < initial_energy-1e-5
    assert np.all(fit.x >= LOWER-1e-12) and np.all(fit.x <= UPPER+1e-12)
    projected = fit.x-np.clip(fit.x-final_gradient, LOWER, UPPER)
    assert np.max(np.abs(projected)) < 2e-7
    checks = []
    for label, point in (("initial", initial), ("optimized", fit.x)):
        _, gradient = checked(point)
        derivatives = []
        for step in (5e-3, 1e-3, 2e-4):
            fd = []
            for axis in np.eye(2):
                endpoints = [forward(jnp.asarray(point+sign*step*axis)) for sign in (1, -1)]
                assert all(bool(info["stationary"]) for _, info in endpoints)
                fd.append(float((endpoints[0][0]-endpoints[1][0])/(2*step)))
            derivatives.append(dict(step=step, gradient_theta=fd,
                max_error=float(np.max(np.abs(np.asarray(fd)-gradient)))))
        np.testing.assert_allclose(derivatives[-1]["gradient_theta"], gradient, atol=2e-7, rtol=2e-5)
        # Shell normalization remains one and has zero parameter derivative.
        diagonal_jacobian = jax.jacfwd(lambda x: jnp.diag(inputs(x)[0]))(jnp.asarray(point))
        np.testing.assert_allclose(diagonal_jacobian, 0., atol=2e-12, rtol=0)
        checks.append(dict(point=label, theta=np.asarray(point).tolist(), gradient_theta=gradient.tolist(),
            finite_differences=derivatives, normalization_jacobian_max=float(jnp.max(jnp.abs(diagonal_jacobian)))))
    root = Path(__file__).resolve().parents[2]
    digest = hashlib.sha256()
    for path in sorted((root/"src").rglob("*.py")):
        digest.update(path.relative_to(root).as_posix().encode()); digest.update(path.read_bytes())
    report = dict(complete=True, **metadata, initial_energy=initial_energy, final_energy=final_energy,
        optimized_theta=fit.x.tolist(), optimized_scale=np.exp(fit.x).tolist(),
        projected_gradient_max=float(np.max(np.abs(projected))), optimizer_iterations=int(fit.nit),
        optimizer_message=str(fit.message), trajectory=trajectory, evaluations=evaluations,
        derivative_checks=checks, source_python_sha256=digest.hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        hostname=platform.node(), python=platform.python_version(), jax=jax.__version__,
        dtype="float64", devices=[str(d) for d in jax.devices()], elapsed_seconds=perf_counter()-start,
        scope="Fixed geometry/topology/auxiliary metric. First exponent derivatives include normalized 1e and 3c DF integrals and relaxed HF. No exact 4c, auxiliary, geometry or higher exponent derivatives.")
    (OUTPUT/"gradscf.json").write_text(json.dumps(report, indent=2)+"\n")
    print("A05 bounded DF-HF optimization and first derivative gates PASS", initial_energy, final_energy, flush=True)


if __name__ == "__main__":
    main()

# jy_gpu_1 CPU/x64: 15 accepted updates, 34.21 s; DF-HF energy
# -1.1199118212206445 -> -1.1200445639724483 Eh; scales [1.08814213,1.78332298].
# Independent PySCF energy error <=4.45e-16 Eh; first exponent FD <=2.58e-9 Eh.
