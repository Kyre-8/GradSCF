"""X15: matched frozen and relaxed diagonal exciton-mode vertices for water.

Use the G05 RHF/Cartesian 6-31G* minimum and its three H2O normal modes.
Both models share one native main-QP branch, three TDA states, base dipoles,
and frequencies. Only the diagonal local vertices differ. Run the separate
finite-difference comparison before constructing their 0 K vibronic spectra.
"""
import json
from pathlib import Path
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from gradscf import bse, gto, scf
from gradscf.scf.rks import RKSConfig
from gradscf.solvers import LinearSolverConfig
from gradscf_tools.gw_response import MolecularGW
from gradscf_tools.molecular_ep import NativeRHF


HARMONIC = Path("reproducibility/ad_showcase/G05_water_rhf_hessian/gradscf.json")
OUTPUT = Path("reproducibility/ad_showcase/X15_water_matched_vertices")
NSTATES = 3
NW = 200
ETA = 1e-5
QP_TOL = 1e-11


def load_modes():
    data = json.loads(HARMONIC.read_text())
    assert data["method"] == "RHF" and data["basis"].lower() == "6-31g*" and data["cart"]
    coordinates = np.asarray(data["coordinates_bohr"])
    isotope = data["isotopes"]["H2O"]
    modes = np.asarray(isotope["displacements"])
    omega = np.asarray(isotope["frequencies_cm1"])/data["hartree_to_wavenumber"]
    mass = np.repeat(np.asarray(isotope["masses_amu"])*data["amu_to_electron_mass"], 3)
    vectors = modes.reshape(9, 3)
    np.testing.assert_allclose(vectors.T @ (mass[:, None]*vectors), np.eye(3), atol=1e-10)
    assert np.all(omega > 0) and modes.shape == (3, 3, 3)
    return data, coordinates, modes, omega


def reference(data, coordinates):
    mol = gto.M(atom=list(zip(data["symbols"], coordinates)), basis=data["basis"],
                unit="Bohr", cart=True)
    mf = scf.RHF(mol, integral_backend="native", conv_tol=1e-13,
                 conv_tol_density=1e-12, conv_tol_grad=1e-10, max_cycle=150).run()
    engine = NativeRHF(mol.to_spec(), basis=data["basis"], config=RKSConfig(
        xc_spec="hf", max_cycle=150, conv_tol=1e-13, conv_tol_density=1e-12,
        conv_tol_grad=1e-10))
    hf, gradient = engine.evaluate(coordinates, init_density=mf.make_rdm1())
    assert engine.topology.cart and engine.topology.nao == 19
    assert np.max(np.abs(gradient)) < 1e-7
    return mf, engine, hf, gradient


def configuration():
    return bse.BSEConfig(nroots=NSTATES, tda=True, solver="dense", conv_tol=1e-10,
        response_config=LinearSolverConfig(method="direct", rtol=1e-10, max_dense=256))


def gw_model(mf, anchor=None):
    return MolecularGW(mf, method="g0w0", nw=NW, eta=ETA, qp_solver="newton",
                       qp_tol=QP_TOL, qp_initial=anchor)


def run():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    data, coordinates, modes, omega = load_modes()
    start = perf_counter()
    mf, engine, hf, gradient = reference(data, coordinates)
    hf_seconds = perf_counter()-start
    print("HF reference / Eh:", float(mf.e_tot), "gradient maximum:", np.max(np.abs(gradient)), flush=True)

    # Select native eta=0 main-weight seeds at THIS optimized geometry. The
    # finite-eta roots then define one local branch for all following changes.
    selector = gw_model(mf)
    start = perf_counter()
    seeds = selector.pole_seeds(max_dense=2048)
    assert np.asarray(seeds.valid).all()
    preliminary = gw_model(mf, seeds.seeds)
    anchor = jax.jit(preliminary)(jnp.asarray(coordinates))
    assert bool(anchor.converged) and np.asarray(anchor.converged_mask).all()
    print("Native main-QP anchor maximum residual / Eh:", np.max(np.abs(anchor.qp_residual)), flush=True)
    model = gw_model(mf, anchor.mo_energy)
    seed_seconds = perf_counter()-start
    cfg = configuration()
    optical = lambda r: model.optical(r, config=cfg)[:3]
    forward = jax.jit(optical)
    start = perf_counter()
    qp, excited, position = forward(jnp.asarray(coordinates))
    c0 = np.asarray(qp.mo_coeff)
    base_seconds = perf_counter()-start
    assert np.asarray(qp.converged_mask).all() and np.asarray(qp.qp_computed_mask).all()
    assert np.asarray(excited.converged & excited.stable & excited.response_valid).all()
    assert bool(excited.screening_valid)
    space = bse.make_bse_space(19, 5)
    x0 = np.asarray(excited.x_amplitudes)
    energies = np.asarray(excited.excitation_energies)
    dipoles = np.asarray(bse.transition_dipoles(excited, position, space))
    print("Common TDA excitation energies / Eh:", energies, flush=True)

    # Convert the screened HF derivative to the actual QP MO frame. The
    # transported reference operator is HF Fock, never the QP energy diagonal.
    overlap = np.asarray(hf.overlap_matrix)
    rotation = np.asarray(hf.mo_coeff).T @ overlap @ c0
    np.testing.assert_allclose(rotation.T @ rotation, np.eye(19), atol=2e-9)
    assert np.max(np.abs(rotation[:5, 5:])) < 1e-7
    np.testing.assert_allclose(2*c0[:, :5] @ c0[:, :5].T, hf.density_matrix, atol=2e-8)
    f0, minimum_overlap = engine.transport(hf, c0, coordinates, coordinates)
    np.testing.assert_allclose(f0, c0.T @ np.asarray(hf.fock_matrix) @ c0, atol=1e-8)
    np.testing.assert_allclose(f0, np.diag(np.asarray(qp.screening_energy)), atol=1e-8)
    hf_response = engine.response(coordinates, hf)

    def observable(r):
        charged, neutral, _ = optical(r)
        return jnp.concatenate((charged.mo_energy, neutral.excitation_energies))

    directional = jax.jit(lambda r, v: jax.jvp(observable, (r,), (v,))[1])
    relaxed, frozen, qp_slopes, fock_slopes, timings = [], [], [], [], []
    hessian = np.asarray(data["hessian_hartree_per_bohr2"])
    for mode in range(3):
        vector = modes[:, :, mode]
        start = perf_counter()
        slope = np.asarray(directional(jnp.asarray(coordinates), jnp.asarray(vector)))
        relaxed_seconds = perf_counter()-start
        assert np.isfinite(slope).all()
        start = perf_counter()
        hvp, dfock = hf_response(vector)
        transported = rotation.T @ dfock @ rotation
        operator = bse.ep_coupling.project_couplings(x0, space, transported[None])[0]
        frozen_slope = np.diag(np.asarray(operator))
        frozen_seconds = perf_counter()-start
        hessian_error = float(np.max(np.abs(hvp.ravel()-hessian @ vector.ravel())))
        assert hessian_error < 2e-8, hessian_error
        qp_slopes.append(slope[:19])
        relaxed.append(slope[19:])
        frozen.append(frozen_slope)
        fock_slopes.append(transported)
        timings.append(dict(mode=mode, relaxed_seconds=relaxed_seconds,
            frozen_seconds=frozen_seconds, hessian_action_error=hessian_error,
            discarded_frozen_offdiagonal_norm=float(np.linalg.norm(operator-np.diag(frozen_slope)))))
        print(f"mode {mode}: relaxed {slope[19:]} / frozen {frozen_slope} Eh/Q", flush=True)

    relaxed, frozen = np.asarray(relaxed), np.asarray(frozen)
    conversion = 1/np.sqrt(2*omega[:, None])
    frozen_g, relaxed_g = frozen*conversion, relaxed*conversion
    # Reflection y -> -y exchanges the two identical H nuclei. The odd mode
    # has no diagonal first-energy derivative for these isolated states.
    mass = np.repeat(np.asarray(data["isotopes"]["H2O"]["masses_amu"])
                     * data["amu_to_electron_mass"], 3)
    reflected = modes[[0, 2, 1]] * np.array([1., -1., 1.])[None, :, None]
    parity = np.sum(modes.reshape(9, 3)*mass[:, None]*reflected.reshape(9, 3), axis=0)
    odd = parity < -.99
    assert np.count_nonzero(odd) == 1
    assert np.max(np.abs(relaxed_g[odd])) < 2e-8
    assert np.max(np.abs(frozen_g[odd])) < 2e-8

    np.savez_compressed(OUTPUT/"vertices.npz", coordinates=coordinates, modes=modes, omega=omega,
        qp_energy=qp.mo_energy, qp_weight=qp.qp_weight, qp_coeff=c0, hf_fock=f0, hf_overlap=overlap,
        excitation_energy=energies, dipoles=dipoles, amplitudes=x0,
        qp_slopes=np.asarray(qp_slopes), relaxed_slopes=relaxed, frozen_slopes=frozen,
        relaxed_g=relaxed_g, frozen_g=frozen_g, frozen_fock_slopes=np.asarray(fock_slopes))
    record = dict(basis=data["basis"], cart=True, method="G0W0/TDA singlet", nstates=NSTATES,
        symbols=data["symbols"], coordinates_bohr=coordinates.tolist(), nw=NW, eta_hartree=ETA,
        qp_tolerance_hartree=QP_TOL, qp_orbitals=list(range(19)), g_orbitals=list(range(19)),
        screening_occupied=list(range(5)), screening_virtual=list(range(5, 19)),
        harmonic_source=str(HARMONIC), normal_coordinate="bohr*sqrt(electron mass); V.T M V=I",
        hf_gradient_hartree_per_bohr=np.asarray(gradient).tolist(), frame_min_overlap=minimum_overlap,
        frame_orthogonality_error=float(np.max(np.abs(rotation.T @ rotation-np.eye(19)))),
        frequencies_cm1=data["isotopes"]["H2O"]["frequencies_cm1"], mode_reflection_parity=parity.tolist(),
        native_seed_hartree=np.asarray(seeds.seeds).tolist(), seed_weight_gap=np.asarray(seeds.weight_gap).tolist(),
        qp_energy_hartree=np.asarray(qp.mo_energy).tolist(), qp_weight=np.asarray(qp.qp_weight).tolist(),
        qp_residual_hartree=np.asarray(qp.qp_residual).tolist(),
        excitation_hartree=energies.tolist(), dipoles_bohr=dipoles.tolist(),
        bse_residual_hartree=np.asarray(excited.residual_norms).tolist(),
        relaxed_slope_hartree_per_Q=relaxed.tolist(), frozen_slope_hartree_per_Q=frozen.tolist(),
        relaxed_g_hartree=relaxed_g.tolist(), frozen_g_hartree=frozen_g.tolist(),
        timing=dict(hf_seconds=hf_seconds, native_seed_anchor_seconds=seed_seconds,
                    base_gw_bse_seconds=base_seconds, modes=timings),
        scope="Matched local diagonal models; common base energies/dipoles/modes; no nonadiabatic offdiagonal, quadratic DW or thermal multiphonon claim.")
    (OUTPUT/"gradscf.json").write_text(json.dumps(record, indent=2)+"\n")


if __name__ == "__main__":
    run()

# Measured on jy_gpu_1, CPU float64, 2026-10-16:
# RHF energy -76.01074650846044 Eh; max nuclear gradient 4.288e-8 Eh/bohr.
# Common TDA excitation energies: [0.33144466, 0.41165548, 0.43231143] Eh.
# Symmetric-stretch g, frozen / relaxed, in eV:
# S1 0.38347600 / 0.49418136; S2 0.47570864 / 0.61186359;
# S3 0.49679502 / 0.41781136. Odd-mode diagonal g is below 1.2e-12 eV.
# Program wall time 99.68 s; first/warm relaxed JVP 21.675 / 0.667-0.684 s.
