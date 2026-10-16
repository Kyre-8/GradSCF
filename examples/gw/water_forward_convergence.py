"""Water: separate quadrature, orbital-window and BSE solver convergence.

Prepare one native RHF/Cartesian 6-31G* reference, then reuse its arrays.
The geometry and local QP branch are those of water_primary_qp.py. Every
comparison stays in the same 19-orbital Gaussian basis. Run one ordinary
function call per bounded process; there is no command-line parser.
"""
import hashlib
import json
from pathlib import Path
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from gradscf import bse, gw
from gradscf.gw.types import GWResult
from gradscf.integrals.molecular.factorization import eri_pair_matrix_to_df_factors_traceable
from gradscf.solvers import EigenSolverConfig, LinearSolverConfig, solve_hermitian
from gradscf.tools.spectra import HARTREE_TO_EV
from water_geometry_response import COORDINATES, water_reference


OUTPUT = Path("reproducibility/ad_showcase/GW_BSE_forward_gates")
ACCEPTED = Path("reproducibility/ad_showcase/X02_water_gw_bse_response/primary/nw200/native.json")
NOCC, NMO = 5, 19
OCCUPIED = tuple(range(NOCC))
VIRTUAL = tuple(range(NOCC, NMO))
GRID_EV = np.linspace(0., 50., 1000)
ETA = 1e-5


def save_record(path, record):
    record["script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    path.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")


def prepare():
    """Save one physical reference; no later sweep reruns or differentiates HF."""
    OUTPUT.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    mf = water_reference()
    reference, state = mf.to_reference(), mf.scf_result
    assert mf.converged and mf.mo_coeff.shape == (NMO, NMO)
    coefficients = jnp.asarray(mf.mo_coeff)
    fock = jnp.asarray(state.fock_matrix)
    canonical = solve_hermitian(coefficients.T @ fock @ coefficients,
        config=EigenSolverConfig(method="dense", nroots=NMO, atol=1e-11))
    assert np.asarray(canonical.converged).all()
    coefficients = coefficients @ canonical.vectors

    # Use the same symmetric AO-pair square root as the geometry-response
    # adapter, including its fixed truncation threshold and full factor shape.
    pair = jnp.asarray(reference.eri_pair_matrix)
    assert pair.shape == (190, 190)
    factors = eri_pair_matrix_to_df_factors_traceable(pair, nao=NMO, tol=1e-12)
    rank = int(np.count_nonzero(np.linalg.eigvalsh(np.asarray(pair)) > 1e-12))
    assert rank == 165
    mo_factors = jnp.einsum("Ppq,pi,qj->Pij", factors, coefficients, coefficients)
    dipole = jnp.einsum("pi,xpq,qj->xij", coefficients,
                        reference.dipole_integrals, coefficients)
    accepted = json.loads(ACCEPTED.read_text())
    np.testing.assert_allclose(mf.e_tot, accepted["hf_energy_hartree"], atol=1e-10, rtol=0)
    np.testing.assert_allclose(canonical.values, accepted["hf_orbital_hartree"],
                               atol=1e-9, rtol=0)
    np.savez_compressed(OUTPUT / "reference.npz", coordinates=COORDINATES,
        mo_energy=canonical.values, mo_coeff=coefficients, df_factors=factors,
        fock_matrix=fock, hcore_matrix=state.hcore_matrix,
        density_matrix=state.density_matrix, mo_factors=mo_factors,
        dipole_mo=dipole, qp_anchor=accepted["qp_energy_hartree"])
    record = dict(basis="6-31g*", cart=True, method="RHF", nmo=NMO, nocc=NOCC,
        coordinates_bohr=COORDINATES.tolist(), hf_energy_hartree=float(mf.e_tot),
        factor_shape=list(factors.shape), factor_rank=rank, factor_cutoff=1e-12,
        seconds=perf_counter() - started, accepted_native_record=str(ACCEPTED),
        accepted_sha256=hashlib.sha256(ACCEPTED.read_bytes()).hexdigest(),
        jax=jax.__version__, backend=jax.default_backend(), dtype="float64")
    save_record(OUTPUT / "reference.json", record)
    print("Frozen RHF reference / Eh:", record["hf_energy_hartree"],
          "retained pair rank:", rank, flush=True)


def read_reference():
    with np.load(OUTPUT / "reference.npz") as data:
        return {key: jnp.asarray(data[key]) for key in data.files}


def read_qp(data, folder):
    """Reuse an actual converged fixed-reference QP result, including its masks."""
    with np.load(folder / "qp.npz") as stored:
        arrays = {key: jnp.asarray(stored[key]) for key in stored.files}
    assert np.asarray(arrays["converged_mask"] & arrays["qp_computed_mask"]).all()
    return GWResult(**arrays, mo_coeff=data["mo_coeff"], converged=jnp.array(True),
        nw=200, g_orbitals=tuple(range(NMO)), screening_occupied=OCCUPIED,
        screening_virtual=VIRTUAL, method="g0w0")


def qp_calculation(data, *, nw=200, g_virtual=VIRTUAL, w_virtual=VIRTUAL):
    """Continue recorded native QP roots in the unchanged finite-eta equation."""
    common = {key: data[key] for key in (
        "mo_energy", "mo_coeff", "df_factors", "fock_matrix",
        "hcore_matrix", "density_matrix")}
    return gw.g0w0_cd_restricted(**common, nocc=NOCC, nw=nw, eta=ETA,
        orbs=tuple(range(NMO)), g_orbitals=OCCUPIED + tuple(g_virtual),
        screening_occupied=OCCUPIED, screening_virtual=tuple(w_virtual),
        resolvent_expansion=True, qp_solver="newton", qp_tol=1e-11,
        qp_initial=data["qp_anchor"])


def optical_calculation(data, qp, *, tda=True, virtual=VIRTUAL,
                        w_virtual=VIRTUAL, solver="dense", screening="direct", nroots=None):
    space = bse.make_bse_space(NMO, NOCC, virtual=tuple(virtual))
    screen = bse.make_bse_space(NMO, NOCC, virtual=tuple(w_virtual))
    config = bse.BSEConfig(tda=tda, nroots=space.size if nroots is None else nroots,
        solver=solver, conv_tol=1e-10, max_space=80, block_size=16,
        screening_config=LinearSolverConfig(method=screening, rtol=1e-11,
                                            atol=1e-13, restart=40, maxiter=100),
        response_config=LinearSolverConfig(method="direct", rtol=1e-10, max_dense=256))
    result = bse.run_bse(qp.mo_energy, qp.screening_energy, data["mo_factors"],
        space, screening_space=screen, qp_computed_mask=qp.qp_computed_mask,
        qp_converged_mask=qp.converged_mask, config=config)
    return result, space


def run(nw=200, *, window=None, count=14):
    """One full TDA/full-BSE forward point; a single window changes at a time."""
    if window not in (None, "g", "w", "optical") or count not in (4, 8, 12, 14):
        raise ValueError("Choose g, w or optical with 4, 8, 12 or 14 virtual orbitals.")
    if window is not None and nw != 200:
        raise ValueError("Window sensitivity fixes the quadrature at 200 nodes.")
    if window is not None and count == 14:
        print("Reuse nw200 as the identical full-window endpoint.", flush=True)
        return json.loads((OUTPUT / "nw200/result.json").read_text())
    name = f"nw{nw}" if window is None else f"{window}_virtual{count}"
    folder = OUTPUT / name
    folder.mkdir(parents=True, exist_ok=True)
    data = read_reference()
    selected = VIRTUAL[:count]
    g_virtual = selected if window == "g" else VIRTUAL
    w_virtual = selected if window == "w" else VIRTUAL
    optical_virtual = selected if window == "optical" else VIRTUAL
    seed_source = str(ACCEPTED)
    if window in ("g", "w"):
        previous = {12: "nw200", 8: f"{window}_virtual12", 4: f"{window}_virtual8"}
        seed_folder = OUTPUT / previous[count]
        with np.load(seed_folder / "qp.npz") as stored:
            assert np.asarray(stored["converged_mask"] & stored["qp_computed_mask"]).all()
            data["qp_anchor"] = jnp.asarray(stored["mo_energy"])
        seed_source = str(seed_folder / "qp.npz")
    if window == "optical":
        # Only the optical transition space changes: reuse the identical GW
        # solution instead of solving its unchanged equations for every window.
        qp = read_qp(data, OUTPUT / "nw200")
        cold, warm = None, None
    else:
        started = perf_counter()
        solve = jax.jit(lambda arrays: qp_calculation(
            arrays, nw=nw, g_virtual=g_virtual, w_virtual=w_virtual))
        qp = solve(data)
        np.asarray(qp.mo_energy)
        cold = perf_counter() - started
        started = perf_counter()
        np.asarray(solve(data).mo_energy)
        warm = perf_counter() - started
    np.savez_compressed(folder / "qp.npz", mo_energy=qp.mo_energy,
        screening_energy=qp.screening_energy, qp_weight=qp.qp_weight,
        qp_residual=qp.qp_residual, qp_computed_mask=qp.qp_computed_mask,
        converged_mask=qp.converged_mask)
    # Retain the actual diagnostic arrays even if a truncated model fails.
    assert bool(qp.converged) and np.asarray(qp.converged_mask).all()
    assert np.isfinite(qp.qp_weight).all() and np.min(qp.qp_weight) > 0
    assert np.max(np.abs(qp.qp_residual)) <= 1e-10
    report = dict(method="g0w0", nw=nw, eta_hartree=ETA, qp_tolerance=1e-11,
        qp_seed_source=seed_source, qp_seed_energy_hartree=data["qp_anchor"].tolist(),
        qp_reused=window == "optical",
        g_virtual=list(g_virtual), w_virtual=list(w_virtual),
        optical_virtual=list(optical_virtual), qp_energy_hartree=qp.mo_energy.tolist(),
        qp_weight=qp.qp_weight.tolist(), qp_residual_max_hartree=float(np.max(np.abs(qp.qp_residual))),
        qp_cold_seconds=cold, qp_warm_seconds=warm, optical=[])
    for tda in (True, False):
        label = "tda" if tda else "full"
        optical = jax.jit(lambda arrays, q: optical_calculation(
            arrays, q, tda=tda, virtual=optical_virtual, w_virtual=w_virtual)[0])
        started = perf_counter()
        result = optical(data, qp)
        np.asarray(result.excitation_energies)
        cold = perf_counter() - started
        started = perf_counter()
        np.asarray(optical(data, qp).excitation_energies)
        warm = perf_counter() - started
        assert np.asarray(result.converged & result.stable).all()
        assert np.max(result.residual_norms) <= 1e-9
        space = bse.make_bse_space(NMO, NOCC, virtual=optical_virtual)
        strengths = bse.oscillator_strengths(result, data["dipole_mo"], space)
        alpha = bse.polarizability(result, data["dipole_mo"], space).real
        cross_section = bse.absorption_cross_section(result, data["dipole_mo"],
            space, GRID_EV / HARTREE_TO_EV, eta=.01, unit="au")
        np.savez_compressed(folder / f"{label}.npz", energy=result.excitation_energies,
            strengths=strengths, static_alpha=alpha, grid_ev=GRID_EV,
            cross_section=cross_section, x=result.x_amplitudes, y=result.y_amplitudes)
        report["optical"].append(dict(tda=tda, nroots=space.size,
            energy_hartree=result.excitation_energies.tolist(), strengths=strengths.tolist(),
            static_alpha=alpha.tolist(), residual_max_hartree=float(np.max(result.residual_norms)),
            cold_seconds=cold, warm_seconds=warm))
        print(name, label, "first excitation / eV:",
              float(result.excitation_energies[0]) * HARTREE_TO_EV, flush=True)
    save_record(folder / "result.json", report)
    print(name, "maximum QP residual / Eh:", report["qp_residual_max_hartree"], flush=True)
    return report


if __name__ == "__main__":
    prepare()
    run(200)

# Measured on jy_gpu_1, CPU float64, fixed X02 water reference:
# Frozen RHF energy / Eh: -76.01049281824507; retained AO-pair rank: 165.
# nw200 first TDA / full-BSE excitation / eV: 8.90723778268 / 8.86961245790.
# Maximum QP equation residual / Eh: 1.22e-15.
# Separate 200-to-400-node comparison: max d(QP energy)=3.55e-15 Eh,
# max dZ=9.95e-13. These are fixed-basis quadrature checks, not basis accuracy.
