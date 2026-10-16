"""A01: matched explicit/implicit derivatives of H2 energy and dipole.

Run on jy_gpu_1 with CPU float64. There is no CLI. Each method/mode/observable
runs in a fresh spawned process. Forward, VJP preparation, and pullback are
timed directly and synchronized; their wall times are never subtracted.
"""
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import platform
import resource
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from gradscf import dft, gto, scf
from gradscf_tools.molecular_response import MolecularEnergy


METHODS = ("HF", "PBE")
MODES = ("explicit", "implicit")
OBSERVABLES = ("energy", "dipole_z")
OUTPUT = Path("reproducibility/ad_showcase/A01_explicit_implicit")
BASIS = "6-31g"
POINT = np.array([.25, .4])
FD_STEPS = (2e-2, 7e-3, 2e-3)
REPEATS = 5


def provenance():
    root = Path(__file__).resolve().parents[2]
    digest = hashlib.sha256()
    for path in sorted((root/"src").rglob("*.py")):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(path.read_bytes())
    return dict(source_python_sha256=digest.hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        hostname=platform.node(), python=platform.python_version(), jax=jax.__version__,
        numpy=np.__version__, dtype="float64", devices=[str(device) for device in jax.devices()],
        persistent_compilation_cache_enabled=bool(jax.config.jax_enable_compilation_cache),
        threads={key: os.environ.get(key) for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS")})


def peak_rss_mib():
    divisor = 1024**2 if platform.system() == "Darwin" else 1024
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/divisor


def synchronized_call(function, *arguments):
    start = perf_counter()
    result = function(*arguments)
    jax.block_until_ready(result)
    return result, perf_counter()-start


def run_case(method, mode, observable, output=OUTPUT):
    """One fresh-process measurement; optional output isolates a smoke check."""
    if method not in METHODS or mode not in MODES or observable not in OBSERVABLES:
        raise ValueError("Choose one of the declared method/mode/observable cases.")
    if jax.config.jax_enable_compilation_cache:
        raise RuntimeError("Set JAX_ENABLE_COMPILATION_CACHE=false for fresh-process cold measurements.")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    record = dict(method=method, mode=mode, observable=observable, complete=False,
                  fresh_process_pid=os.getpid(), **provenance())
    path = output/f"{method.lower()}_{mode}_{observable}.json"
    path.write_text(json.dumps(record, indent=2)+"\n")

    start = perf_counter()
    mol = gto.M(atom="H 0 0 -.7; H 0 0 .7", basis=BASIS, unit="Bohr", cart=True)
    settings = dict(conv_tol=1e-12, conv_tol_density=1e-11, conv_tol_grad=1e-10,
                    max_cycle=150, integral_backend="native")
    mf = (scf.RHF(mol, **settings) if method == "HF" else
          dft.RKS(mol, xc="pbe", grids_level=0, **settings)).run()
    if not mf.converged:
        raise RuntimeError("The common zero-field seed did not converge.")
    model = MolecularEnergy(mf, mode=mode, gradient_tolerance=1e-10)
    assert not jax.config.jax_enable_compilation_cache
    seed_seconds = perf_counter()-start
    coordinates = model.coordinates
    stretch = jnp.array([[0., 0., -.05], [0., 0., .05]])  # bohr per theta[0]
    field_direction = jnp.array([0., 0., .005])          # field au per theta[1]

    def unpack(theta):
        return coordinates+theta[0]*stretch, theta[1]*field_direction

    def observable_value(theta):
        r, field = unpack(theta)
        # E/Eh and mu_z/(e*bohr) have numerical scale one in atomic units.
        return model(r, field) if observable == "energy" else model.dipole(r, field)[2]

    point = jnp.asarray(POINT)
    forward = jax.jit(observable_value)
    value, forward_cold = synchronized_call(forward, point)
    if not np.isfinite(float(value)):
        raise RuntimeError("The measured forward solve is not stationary/finite.")
    forward_warm = [synchronized_call(forward, point)[1] for _ in range(REPEATS)]
    forward_rss = peak_rss_mib()
    print(method, mode, observable, "forward complete", flush=True)

    # Preparation includes the primal computation and retained residuals.
    # Blocking the Partial pytree also synchronizes its array residuals.
    (prepared_value, pullback), preparation_cold = synchronized_call(jax.vjp, forward, point)
    preparation_warm = []
    for _ in range(REPEATS):
        (prepared_value, pullback), seconds = synchronized_call(jax.vjp, forward, point)
        preparation_warm.append(seconds)
    preparation_rss = peak_rss_mib()
    print(method, mode, observable, "VJP preparation complete", flush=True)
    cotangent = jnp.asarray(1., dtype=point.dtype)
    product, pullback_cold = synchronized_call(pullback, cotangent)
    gradient = np.asarray(product[0])
    pullback_warm = []
    for _ in range(REPEATS):
        repeated, seconds = synchronized_call(pullback, cotangent)
        pullback_warm.append(seconds)
        np.testing.assert_allclose(repeated[0], gradient, atol=2e-12, rtol=2e-12)
    timed_rss = peak_rss_mib()
    np.testing.assert_allclose(prepared_value, value, atol=2e-12, rtol=0)
    assert np.isfinite(gradient).all()

    # Validation follows the timing window; its allocations are reported
    # separately and are not attributed to the timed pullback.
    solve = jax.jit(lambda theta: model.solve(*unpack(theta)))
    state = solve(point)
    jax.block_until_ready(state)
    assert bool(state.stationary) and bool(state.optimizer_success)
    assert state.gradient_mode == mode
    assert int(state.iterations) > 0
    decrease = float(state.initial_energy-state.total_energy)
    assert decrease > 1e-10, "A zero-iteration stationary seed is not this benchmark."
    checks = []
    for step in FD_STEPS:
        finite = []
        for axis in jnp.eye(2):
            plus, minus = point+step*axis, point-step*axis
            for displaced in (plus, minus):
                endpoint = solve(displaced)
                assert bool(endpoint.stationary), float(endpoint.gradient_norm)
            finite.append(float((forward(plus)-forward(minus))/(2*step)))
        error = float(np.max(np.abs(np.asarray(finite)-gradient)))
        checks.append(dict(step_theta=step, gradient_theta=finite, max_error=error))
    np.testing.assert_allclose(checks[-1]["gradient_theta"], gradient, atol=2e-6, rtol=2e-5)
    record.update(complete=True, basis=BASIS, cart=True, symbols=["H", "H"],
        seed_coordinates_bohr=np.asarray(coordinates).tolist(),
        coordinate_direction_bohr=np.asarray(stretch).tolist(),
        field_direction_au=np.asarray(field_direction).tolist(), point_theta=POINT.tolist(),
        grid_level=0 if method == "PBE" else None, seed_scf_seconds=seed_seconds,
        seed_coefficients_sha256=hashlib.sha256(np.asarray(mf.mo_coeff).tobytes()).hexdigest(),
        seed_occupations=np.asarray(mf.mo_occ).tolist(), value=float(value),
        observable_normalization="E/Eh" if observable == "energy" else "mu_z/(e bohr)",
        gradient_theta=gradient.tolist(), finite_difference_checks=checks,
        stationary=bool(state.stationary), gradient_norm=float(state.gradient_norm),
        orbital_iterations=int(state.iterations), initial_energy_hartree=float(state.initial_energy),
        final_energy_hartree=float(state.total_energy), relaxation_energy_hartree=decrease,
        final_density=np.asarray(state.density_matrix).tolist(),
        orbital_gradient_tolerance=1e-10, orbital_max_iterations=300,
        implicit_tolerance=1e-10, implicit_max_iter=100, implicit_restart=40,
        forward_cold_seconds=forward_cold, forward_warm_seconds=forward_warm,
        vjp_preparation_cold_seconds=preparation_cold, vjp_preparation_warm_seconds=preparation_warm,
        pullback_cold_seconds=pullback_cold, pullback_warm_seconds=pullback_warm,
        cumulative_peak_after_forward_mib=forward_rss,
        cumulative_peak_after_preparation_mib=preparation_rss,
        timed_fresh_process_peak_rss_mib=timed_rss,
        peak_after_validation_mib=peak_rss_mib(),
        timing_definition="First transformed call includes tracing/compilation. VJP preparation includes a primal. Pullback timings directly execute the retained pullback, with no wall-time subtraction.",
        memory_definition="Fresh-process high-water RSS includes imports, seed SCF and all timed phases. Phase readings are cumulative; no difference is interpreted as backward-only memory.")
    path.write_text(json.dumps(record, indent=2)+"\n")
    print(method, mode, observable, "PASS; iterations", int(state.iterations),
          "FD error", checks[-1]["max_error"], "timed RSS MiB", timed_rss, flush=True)
    return record


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    report = dict(complete=False, cases=[], parity=[], methods=list(METHODS),
                  point_theta=POINT.tolist(), theta_definition="R=Rseed+theta0*stretch; F=theta1*(0,0,.005au)")
    (OUTPUT/"gradscf.json").write_text(json.dumps(report, indent=2)+"\n")
    for method in METHODS:
        for mode in MODES:
            for observable in OBSERVABLES:
                job = multiprocessing.get_context("spawn").Process(target=run_case, args=(method, mode, observable))
                job.start()
                job.join(1800)
                if job.is_alive():
                    job.terminate()
                    job.join()
                    raise RuntimeError("A bounded A01 worker exceeded 30 minutes.")
                if job.exitcode:
                    raise RuntimeError(f"A01 worker failed: {method}/{mode}/{observable}, exit {job.exitcode}.")
                data = json.loads((OUTPUT/f"{method.lower()}_{mode}_{observable}.json").read_text())
                assert data["complete"]
                report["cases"].append(data)
                (OUTPUT/"gradscf.json").write_text(json.dumps(report, indent=2)+"\n")
        cases = [case for case in report["cases"] if case["method"] == method]
        assert len({case["seed_coefficients_sha256"] for case in cases}) == 1
        assert len({case["source_python_sha256"] for case in cases}) == 1
        for observable in OBSERVABLES:
            explicit, implicit = [case for case in cases if case["observable"] == observable]
            np.testing.assert_allclose(explicit["value"], implicit["value"], atol=2e-10, rtol=0)
            np.testing.assert_allclose(explicit["final_density"], implicit["final_density"], atol=2e-9, rtol=0)
            np.testing.assert_allclose(explicit["gradient_theta"], implicit["gradient_theta"], atol=1e-7, rtol=1e-6)
            report["parity"].append(dict(method=method, observable=observable,
                value_error=abs(explicit["value"]-implicit["value"]),
                explicit_iterations=explicit["orbital_iterations"],
                implicit_iterations=implicit["orbital_iterations"],
                same_iteration_count=explicit["orbital_iterations"] == implicit["orbital_iterations"],
                max_gradient_error=float(np.max(np.abs(np.asarray(explicit["gradient_theta"])-implicit["gradient_theta"])))))
        for mode in MODES:
            energy, dipole = [case for case in cases if case["mode"] == mode]
            np.testing.assert_allclose(-energy["gradient_theta"][1]/.005, dipole["value"], atol=1e-7, rtol=1e-7)
    assert len({case["source_python_sha256"] for case in report["cases"]}) == 1
    assert len({case["script_sha256"] for case in report["cases"]}) == 1
    report["complete"] = True
    (OUTPUT/"gradscf.json").write_text(json.dumps(report, indent=2)+"\n")
    print("All eight fresh-process cases, forward/derivative parity and field identity passed.", flush=True)


if __name__ == "__main__":
    main()

# Measured on jy_gpu_1, frozen v7 CPU/x64 source: all eight cases passed,
# each with 9 orbital iterations; explicit/implicit derivative error <=1.26e-16.
# Independent PySCF FD at the smallest theta step: maximum error <=8.10e-10.
# Direct phase times and process RSS: A01_explicit_implicit/README.md and JSON.
