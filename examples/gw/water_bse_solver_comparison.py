"""Fixed water GW result: dense/Davidson BSE and direct/GMRES screening.

Use run(tda=..., solver=..., screening=...) in a fresh bounded process for
each setting. This reuses the accepted 19-orbital reference and full-window
G0W0 result; it never reruns SCF or GW. Six isolated optical states are a
representation check, not an all-state polarizability or complete spectrum.
"""
import hashlib
import json
from pathlib import Path
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from gradscf import bse
from gradscf.gw.screened import build_static_screening, solve_static_screening
from gradscf.solvers import LinearSolverConfig
from water_forward_convergence import NMO, NOCC, OCCUPIED, VIRTUAL, OUTPUT, read_reference


def run(*, tda=True, solver="dense", screening="direct"):
    data = read_reference()
    with np.load(OUTPUT / "nw200/qp.npz") as stored:
        qp = {key: jnp.asarray(stored[key]) for key in stored.files}
    space = bse.make_bse_space(NMO, NOCC)
    linear = LinearSolverConfig(method=screening, rtol=1e-11, atol=1e-13,
                                 restart=40, maxiter=100)
    config = bse.BSEConfig(tda=tda, nroots=6, solver=solver, conv_tol=1e-10,
        max_space=80, screening_config=linear,
        response_config=LinearSolverConfig(method="direct", rtol=1e-10, max_dense=256))

    def calculate(arrays, poles):
        return bse.run_bse(poles["mo_energy"], poles["screening_energy"],
            arrays["mo_factors"], space, config=config,
            qp_computed_mask=poles["qp_computed_mask"],
            qp_converged_mask=poles["converged_mask"])

    # Arrays remain runtime arguments, so compilation cannot precompute the
    # fixed-reference result as a constant. Three warm calls follow one cold call.
    started = perf_counter()
    compiled = jax.jit(calculate).lower(data, qp).compile()
    result = compiled(data, qp)
    np.asarray(result.excitation_energies)
    cold = perf_counter() - started
    warm = []
    for _ in range(3):
        started = perf_counter()
        np.asarray(compiled(data, qp).excitation_energies)
        warm.append(perf_counter() - started)
    memory = compiled.memory_analysis()
    strengths = bse.oscillator_strengths(result, data["dipole_mo"], space)

    # Recompute residuals in the original equations, outside the eigensolver.
    # The symmetric projection is the same accepted roundoff convention as BSE.
    factors = data["mo_factors"]
    np.testing.assert_allclose(factors, factors.swapaxes(1, 2), atol=1e-13, rtol=0)
    factors = (factors + factors.swapaxes(1, 2)) * .5
    state = build_static_screening(qp["screening_energy"], factors,
        occupied=OCCUPIED, virtual=VIRTUAL, config=linear)
    if tda:
        a = bse.build_tda_operator(qp["mo_energy"], factors, space, state)
        operators = (a, None)
    else:
        operators = bse.build_bse_operators(qp["mo_energy"], factors, space, state)
    a, b = operators
    x = result.x_amplitudes.reshape(6, -1).T
    y = result.y_amplitudes.reshape(6, -1).T
    rx = a.apply(x) - x * result.excitation_energies
    if tda:
        residual = np.linalg.norm(np.asarray(rx), axis=0)
    else:
        rx = rx + b.apply(y)
        ry = b.apply(x) + a.apply(y) + y * result.excitation_energies
        residual = np.sqrt(np.sum(np.asarray(rx)**2 + np.asarray(ry)**2, axis=0))

    # Diagonal and transition-pair RHS probe the same physical dielectric.
    p = np.array([0, 1, 0, 0, 5, 5])
    q = np.array([0, 1, 5, 10, 5, 10])
    rhs = factors[:, p, q]
    screened = solve_static_screening(state, rhs)
    screening_residual = np.linalg.norm(
        np.asarray(state.operator().apply(screened.solution) - rhs), axis=0)
    rhs_norm = np.linalg.norm(np.asarray(rhs), axis=0)
    screening_relative = screening_residual / np.where(rhs_norm > 0, rhs_norm, 1.)
    valid = bool(np.asarray(result.converged & result.stable & result.response_valid).all())
    valid &= bool(np.asarray(screened.converged).all()) and bool(state.valid)
    valid &= bool(np.isfinite(strengths).all() and np.max(residual) <= 1e-9)
    valid &= bool(np.all(screening_residual <= 1e-13 + 1e-11 * rhs_norm))
    name = f"{'tda' if tda else 'full'}_{solver}_{screening}"
    folder = OUTPUT / "representations"
    folder.mkdir(parents=True, exist_ok=True)
    report = dict(tda=tda, solver=solver, screening=screening, nroots=6,
        energy_hartree=result.excitation_energies.tolist(), strength=strengths.tolist(),
        reported_residual_hartree=result.residual_norms.tolist(),
        recomputed_residual_hartree=residual.tolist(),
        screening_rhs_pairs=np.column_stack((p, q)).tolist(),
        screening_absolute_residual=screening_residual.tolist(),
        screening_relative_residual=screening_relative.tolist(),
        converged=np.asarray(result.converged).tolist(),
        stable=np.asarray(result.stable).tolist(),
        response_valid=np.asarray(result.response_valid).tolist(),
        stability_certified=bool(result.stability_certified),
        cold_seconds=cold, warm_seconds=warm, passed=valid,
        xla_argument_bytes=memory.argument_size_in_bytes,
        xla_output_bytes=memory.output_size_in_bytes,
        xla_temporary_bytes=memory.temp_size_in_bytes,
        memory_scope="Compiled solve buffers; process RSS separately includes residual probes.",
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (folder / f"{name}.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    np.savez_compressed(folder / f"{name}.npz", x=result.x_amplitudes,
                        y=result.y_amplitudes, energy=result.excitation_energies)
    print(name, "cold / warm median seconds:", cold, np.median(warm),
          "true BSE residual / Eh:", np.max(residual), flush=True)
    assert valid, "The actual physical residual or convergence gate failed."
    return report


if __name__ == "__main__":
    run()

# Measured on jy_gpu_1, CPU float64, six states, max_space=80:
# Default TDA/dense/direct: cold 1.084 s; three-call warm median 0.00812 s.
# Recomputed original-equation residual: 1.18e-14 Eh.
# Full-BSE/dense/direct warm median: 0.00957 s.
# Historical eager-response baseline: full-BSE/Davidson/GMRES 55.0554 s warm.
# With lazy structured-RPA response, a separate whole-result-sync acceptance
# measured 4.5633 s warm, with identical energies and strengths. See
# reproducibility/ad_showcase/GW_BSE_lazy_response/README.md for exact source.
# All eight representations pass their numerical gates. Shared-node process
# times/RSS do not establish an exclusive or asymptotic performance ranking.
