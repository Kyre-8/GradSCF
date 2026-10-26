"""Water/6-31G*: nuclear response through HF, GW and static singlet BSE.

The QP, screening, AO metric, HF orbitals and transition operator all move.
This gives QP and excitation-energy derivatives, not a variational GW+BSE
total excited-state force. Run one ordinary run() call per bounded job.
"""
import hashlib
import json
import platform
from pathlib import Path
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from gradscf import bse, gto, scf
from gradscf.solvers import LinearSolverConfig
from gradscf_tools.gw_response import MolecularGW
from gradscf_tools.excited import transition_density_overlaps
from gradscf_tools.molecular_ep import NativeRHF


COORDINATES = np.array([[0., 0., 0.], [0., 1.43, 1.11], [0., -1.43, 1.11]])
SYMBOLS = ("O", "H", "H")
OUTPUT = Path("reproducibility/ad_showcase/X02_water_gw_bse_response")


def water_reference():
    """The same native RHF reference for the forward and response examples."""
    mol = gto.M(atom=list(zip(SYMBOLS, COORDINATES)), basis="6-31g*", unit="Bohr", cart=True)
    return scf.RHF(mol, conv_tol=1e-12, conv_tol_grad=1e-10,
                   integral_backend="native").run()


def run(method="g0w0", *, tda=True, nw=100, track_qp=True, qp_anchor=None, output=OUTPUT,
        max_cycle=100, damp=.3, diis_space=0):
    mf = water_reference()
    qp_solver = "newton" if method == "g0w0" else "secant"
    if qp_anchor is not None and not track_qp:
        raise ValueError("A supplied QP anchor requires tracked roots.")
    model = MolecularGW(mf, method=method, nw=nw, eta=1e-5, qp_solver=qp_solver,
                        qp_initial=qp_anchor, max_cycle=max_cycle, damp=damp, diis_space=diis_space)
    anchor = None
    start = perf_counter()
    if track_qp:
        # This defines a local root branch. It does not select the largest-Z
        # solution or turn a satellite into a dominant quasiparticle peak.
        anchor = jax.jit(model)(model.coordinates)
        assert bool(anchor.converged)
        model = MolecularGW(mf, method=method, nw=nw, eta=1e-5, qp_solver=qp_solver,
                            qp_initial=anchor.mo_energy, max_cycle=max_cycle, damp=damp,
                            diis_space=diis_space)
    anchor_seconds = perf_counter()-start
    configuration = bse.BSEConfig(nroots=6, tda=tda, solver="dense", conv_tol=1e-10,
        response_config=LinearSolverConfig(method="direct", rtol=1e-10, max_dense=256))
    coordinates = model.coordinates
    # A generic normalized Cartesian direction avoids symmetry-forced zeros.
    direction = np.random.default_rng(20261016).normal(size=COORDINATES.shape)
    direction -= np.mean(direction, axis=0)
    direction /= np.linalg.norm(direction)
    direction = jnp.asarray(direction)

    def observable(r):
        qp, excited, position, space = model.optical(r, config=configuration)
        strength = bse.oscillator_strengths(excited, position, space)
        values = jnp.concatenate([qp.mo_energy, qp.qp_weight, qp.screening_energy,
                                  excited.excitation_energies, strength])
        diagnostics = (qp.qp_residual, qp.converged_mask, excited.residual_norms,
                       excited.converged, excited.stable, excited.response_valid,
                       excited.x_amplitudes, excited.y_amplitudes, qp.mo_coeff)
        return values, diagnostics

    forward = jax.jit(observable)
    start = perf_counter()
    central, diagnostics = forward(coordinates)
    central = np.asarray(central)
    cold_forward = perf_counter()-start
    qp_residual, qp_converged, bse_residual, converged, stable, response_valid, x, y, coefficients = map(np.asarray, diagnostics)
    assert np.isfinite(central).all() and qp_converged.all()
    assert converged.all() and stable.all() and response_valid.all()
    derivative = jax.jit(lambda r: jax.jvp(lambda point: observable(point)[0], (r,), (direction,))[1])
    start = perf_counter()
    tangent = np.asarray(derivative(coordinates))
    cold_derivative = perf_counter()-start
    assert np.isfinite(tangent).all()
    start = perf_counter()
    np.asarray(forward(coordinates)[0])
    warm_forward = perf_counter()-start
    start = perf_counter()
    np.asarray(derivative(coordinates))
    warm_derivative = perf_counter()-start

    nmo = model.nmo
    blocks = dict(qp_energy=(0, nmo), qp_weight=(nmo, 2*nmo), screening_energy=(2*nmo, 3*nmo),
                  excitation_energy=(3*nmo, 3*nmo+6), oscillator_strength=(3*nmo+6, 3*nmo+12))
    overlap = NativeRHF(model.reference.spec, basis=model.reference.basis)
    # These are transition-density matches, not many-electron overlaps.
    def state_match(r, data):
        cross = overlap.cross_overlap(coordinates, r)
        c = np.asarray(data[8])
        mo_overlap = coefficients.T@np.asarray(cross)@c
        scores = np.stack([transition_density_overlaps(coefficients, amplitude,
            c, np.asarray(data[6]+data[7]), cross, nocc=model.nocc) for amplitude in x+y])
        assert np.array_equal(np.argmax(scores, axis=1), np.arange(6))
        assert np.min(np.diag(scores)) > .99 and np.min(np.abs(np.diag(mo_overlap))) > .99
        return dict(min_transition_density_overlap=float(np.min(np.diag(scores))),
                    min_orbital_overlap=float(np.min(np.abs(np.diag(mo_overlap)))))

    checks = []
    for step in (1e-3, 3e-4, 1e-4, 3e-5, 1e-5):
        plus, plus_diagnostics = forward(coordinates+step*direction)
        minus, minus_diagnostics = forward(coordinates-step*direction)
        for data in (plus_diagnostics, minus_diagnostics):
            assert all(np.asarray(data[index]).all() for index in (1, 3, 4, 5))
        finite = np.asarray((plus-minus)/(2*step))
        errors = {name: float(np.max(np.abs(tangent[a:z]-finite[a:z]))) for name, (a, z) in blocks.items()}
        checks.append(dict(step_bohr=step, errors=errors, finite_difference=finite.tolist(),
            plus_values=np.asarray(plus).tolist(), minus_values=np.asarray(minus).tolist(),
            plus_tracking=state_match(coordinates+step*direction, plus_diagnostics),
            minus_tracking=state_match(coordinates-step*direction, minus_diagnostics)))
        print(method, "TDA" if tda else "full BSE", "step", step, errors, flush=True)
    accepted = max(checks[-1]["errors"].values()) < 2e-5

    source_paths = [Path(__file__), Path("src/gradscf_tools/gw_response.py"), Path("src/gradscf_tools/molecular_response.py"),
                    Path("src/gradscf/integrals/molecular/factorization.py")]
    for module in ("gw", "bse", "solvers/eigen", "solvers/nonlinear", "solvers/linear"):
        source_paths.extend(Path("src/gradscf", module).glob("*.py"))
    record = dict(validation_passed=accepted, method=method, tda=tda, basis="6-31g*", cart=True, symbols=SYMBOLS,
        coordinates_bohr=COORDINATES.tolist(), direction=np.asarray(direction).tolist(),
        qp_solver=qp_solver, qp_branch_continuation=anchor is not None,
        qp_tolerance_hartree=1e-11 if method == "g0w0" else None,
        outer_gw_tolerance_hartree=None if method == "g0w0" else 1e-10,
        outer_max_cycle=None if method == "g0w0" else max_cycle,
        outer_damping=None if method == "g0w0" else damp,
        outer_diis_space=None if method == "g0w0" else diis_space,
        anchor_qp_hartree=None if anchor is None else np.asarray(anchor.mo_energy).tolist(),
        anchor_qp_weight=None if anchor is None else np.asarray(anchor.qp_weight).tolist(),
        initial_qp_seed=None if qp_anchor is None else np.asarray(qp_anchor).tolist(),
        direction_norm=1., hf_energy_hartree=float(mf.e_tot), nw=nw, eta_hartree=1e-5,
        nmo=nmo, nocc=model.nocc, qp_orbitals=list(range(nmo)), g_orbitals=list(range(nmo)),
        screening_occupied=list(range(model.nocc)), screening_virtual=list(range(model.nocc, nmo)),
        factor_tolerance=1e-12, factorization="symmetric AO-pair square root, no auxiliary basis",
        bse_nroots=6, bse_electronic_space=model.nocc*(nmo-model.nocc),
        bse_response_solver="shared direct, rtol=1e-10, max_dense=256",
        values=central.tolist(), derivative=tangent.tolist(), blocks=blocks,
        qp_residual_hartree=qp_residual.tolist(), bse_residual_hartree=bse_residual.tolist(),
        bse_response_valid=response_valid.tolist(), finite_difference_checks=checks,
        timings_seconds=dict(anchor=anchor_seconds, cold_forward=cold_forward, warm_forward=warm_forward,
                             cold_derivative=cold_derivative, warm_derivative=warm_derivative),
        hostname=platform.node(), python=platform.python_version(), jax=jax.__version__,
        backend=jax.default_backend(), dtype="float64",
        source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(set(source_paths))},
        scope="First nuclear directional response of isolated QP/optical states; six BSE roots; not a variational excited-state total-energy force.")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    label = "continued" if anchor is not None else "independent_seeds"
    path = output/f"{method}_{'tda' if tda else 'full'}_nw{nw}_{label}.json"
    path.write_text(json.dumps(record, indent=2, allow_nan=False)+"\n")
    print("Saved", path, "timings / s", record["timings_seconds"], flush=True)
    assert accepted, "The saved complete geometry-response comparison did not pass."
    return record


if __name__ == "__main__":
    run("g0w0", tda=True)
