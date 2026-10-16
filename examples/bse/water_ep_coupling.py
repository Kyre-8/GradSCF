"""Water: native RHF/3-21G -> G0W0 -> TDA-BSE -> vibronic absorption.

Three molecular modes; screened RHF vertices, frozen GW correction and BSE
kernel. Compare one-phonon Fan/DW with a converged linear vibronic Hamiltonian
and the diagonal displaced-oscillator reference. Zero-temperature Condon
absorption, not an experimental or dissociative water spectrum. No CLI/PySCF.
"""

from dataclasses import replace
from itertools import product
from pathlib import Path
import json
import platform
import time

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize
from scipy.special import gammaln
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from gradscf import bse
from gradscf.bse import ep_coupling as ep
from gradscf.integrals.molecular.factorization import eri_to_df_factors
from gradscf.gw import g0w0_cd_restricted
from gradscf.gw.ep_coupling import PhononModel
from gradscf.solvers import (
    EigenSolverConfig,
    LinearSolverConfig,
    solve_hermitian,
    solve_complex,
)
from gradscf_tools.molecular_ep import NativeRHF
from gradscf.tools.spectra import HARTREE_TO_EV

jax.config.update("jax_enable_x64", True)
started = time.perf_counter()
engine = NativeRHF("O 0 0 0; H 1.4 0 1.1; H -1.4 0 1.1", basis="3-21g")
folder = (
    Path(__file__).resolve().parents[2] / "reproducibility/gw_bse/water_bse_phonons"
)
folder.mkdir(parents=True, exist_ok=True)


def coordinates(parameters):
    length, angle = parameters
    x, z = length * np.sin(angle / 2), length * np.cos(angle / 2)
    return np.array([[0.0, 0.0, 0.0], [x, 0.0, z], [-x, 0.0, z]])


def objective(parameters):
    r = coordinates(parameters)
    result, gradient = engine.evaluate(r)
    length, angle = parameters
    dr = np.array(
        [
            [0.0, 0.0, 0.0],
            [np.sin(angle / 2), 0.0, np.cos(angle / 2)],
            [-np.sin(angle / 2), 0.0, np.cos(angle / 2)],
        ]
    )
    da = np.array(
        [
            [0.0, 0.0, 0.0],
            [length * np.cos(angle / 2) / 2, 0.0, -length * np.sin(angle / 2) / 2],
            [-length * np.cos(angle / 2) / 2, 0.0, -length * np.sin(angle / 2) / 2],
        ]
    )
    return float(result.total_energy), np.array(
        [np.sum(gradient * dr), np.sum(gradient * da)]
    )


minimum = minimize(
    objective,
    [1.8, np.deg2rad(105.0)],
    jac=True,
    method="BFGS",
    options={"gtol": 1e-9, "maxiter": 60},
)
r = coordinates(minimum.x)
reference, gradient = engine.evaluate(r)
assert np.max(abs(gradient)) < 2e-6
print("Optimized water", minimum.x, "gradient", np.max(abs(gradient)), flush=True)

# Full stationary RHF Hessian from native second integrals + implicit response.
response = engine.response(r, reference)
columns, derivatives = [], []
for direction in np.eye(r.size).reshape(-1, *r.shape):
    hvp, df = response(direction)
    columns.append(hvp.ravel())
    derivatives.append(df)
hessian = np.column_stack(columns)
asymmetry = float(np.max(abs(hessian - hessian.T)))
assert asymmetry < 2e-6
mass = np.repeat(
    np.array([15.99491461957, 1.00782503223, 1.00782503223]) * 1822.888486209, 3
)
center = np.average(r, axis=0, weights=mass.reshape(-1, 3)[:, 0])
rigid = []
for axis in np.eye(3):
    rigid.append((np.tile(axis, (3, 1)) * np.sqrt(mass).reshape(-1, 3)).ravel())
    rigid.append((np.cross(r - center, axis) * np.sqrt(mass).reshape(-1, 3)).ravel())
u, singular, _ = np.linalg.svd(np.array(rigid).T, full_matrices=True)
assert np.count_nonzero(singular > 1e-8) == 6
internal = u[:, 6:]
dynamic = (hessian + hessian.T) / 2 / np.sqrt(mass[:, None] * mass[None, :])
normal = solve_hermitian(
    jnp.asarray(internal.T @ dynamic @ internal),
    config=EigenSolverConfig(method="dense", nroots=3, atol=1e-10),
)
assert bool(jnp.all(normal.converged)) and bool(jnp.all(normal.values > 0))
omega = np.sqrt(np.asarray(normal.values))
modes = internal @ np.asarray(normal.vectors)
directions = (modes / np.sqrt(mass)[:, None]).T.reshape(3, 3, 3)
g = (
    np.einsum("aij,la->lij", derivatives, directions.reshape(3, -1))
    / np.sqrt(2 * omega)[:, None, None]
)
quadratic = []
for direction, frequency in zip(directions, omega):
    dq = 0.002 / np.max(abs(direction))
    matrices = []
    for sign in (-1.0, 1.0):
        moving = r + sign * dq * direction
        result, _ = engine.evaluate(
            moving, init_density=reference.density_matrix, gradient=False
        )
        fock, _ = engine.transport(result, reference.mo_coeff, r, moving)
        matrices.append(fock)
    quadratic.append(
        (sum(matrices) - 2 * np.diag(reference.mo_energy)) / dq**2 / (2 * frequency)
    )
phonons = PhononModel(
    jnp.asarray(omega),
    jnp.asarray(g),
    jnp.asarray(quadratic),
    reference="native water RHF/3-21G; frozen GW correction/kernel",
)
print("Modes / cm^-1", omega * 219474.6313632, flush=True)

_, h, eri, _ = engine.integrals(r)
factors = eri_to_df_factors(eri, tol=1e-12)
qp = g0w0_cd_restricted(
    mo_energy=reference.mo_energy,
    mo_coeff=reference.mo_coeff,
    nocc=5,
    df_factors=factors,
    fock_matrix=reference.fock_matrix,
    hcore_matrix=h,
    density_matrix=reference.density_matrix,
    nw=64,
    eta=1e-5,
    resolvent_expansion=True,
)
assert bool(qp.converged) and bool(jnp.all(qp.qp_computed_mask))
c = reference.mo_coeff
mo_factors = jnp.einsum("Qmn,mi,nj->Qij", factors, c, c)
parameters = replace(
    engine.parameters,
    nuclear_coords=jnp.asarray(r),
    centers=jnp.asarray(r)[engine.owners],
)
dipole_ao = engine.plan.evaluate("dipole", parameters)
dipole_mo = jnp.einsum("xmn,mi,nj->xij", dipole_ao, c, c)
source = bse.BSEReference.from_gw_result(
    qp, mo_factors=mo_factors, nocc=5, dipole_mo=dipole_mo
)
electronic = bse.BSE(source, nroots=3, tda=True, solver="dense").run()
model = ep.project_phonons(electronic.result, electronic.space, phonons)
linear = replace(model, quadratic=None)
dipoles = electronic.transition_dipole()
np.savez(
    folder / "inputs.npz",
    excitation_energies=electronic.e,
    mode_energies=model.energies,
    couplings=model.couplings,
    quadratic=model.quadratic,
    dipoles=dipoles,
    coordinates=r,
    hessian=hessian,
    directions=directions,
)
print(
    "BSE / eV",
    np.asarray(electronic.e) * HARTREE_TO_EV,
    "Huang-Rhys",
    (np.diagonal(np.asarray(model.couplings), axis1=1, axis2=2) / omega[:, None]) ** 2,
    flush=True,
)

# Spectra from complete finite-space lines. This is a forward comparison;
# AD below uses the shared resolvent, not individual eigenvector derivatives.
axis_ev = np.arange(4.0, 18.0001, 0.002)
axis = axis_ev / HARTREE_TO_EV
eta_ev = 0.01


def cross_section(energies, moments, eta_ev=0.01):
    energies, moments = np.asarray(energies), np.asarray(moments)
    z = axis[:, None] + 1j * eta_ev / HARTREE_TO_EV
    result = np.zeros(len(axis))
    for start in range(0, len(energies), 128):
        e = energies[start : start + 128]
        strength = np.sum(abs(moments[start : start + 128]) ** 2, axis=1) / 3
        result += np.imag((1 / (e - z) + 1 / (e + z)) @ strength)
    return 4 * np.pi * axis * result / 137.035999084 * (0.529177210903**2 * 100)


curves = {"bse": cross_section(electronic.e, dipoles)}
beta = 1e8  # Fixed vibrational vacuum, consistently in all comparisons.
for name, value in [("fan", linear), ("fan_dw", model)]:
    curves[name] = np.asarray(
        jax.jit(
            lambda: ep.absorption_cross_section(
                electronic.e,
                value,
                dipoles,
                jnp.asarray(axis),
                beta=beta,
                eta=eta_ev / HARTREE_TO_EV,
                unit="Mb",
            )
        )()
    )

# C2v selection rules: modes 0/1 are diagonal in these three states; mode 2
# couples states 0/1. State 2 forms an independent two-mode block, with the
# antisymmetric mode remaining in its vacuum. Check the numerical symmetry
# residual before using it; do not silently drop a physical coupling.
clean_g = np.zeros_like(np.asarray(linear.couplings))
for mode in (0, 1):
    clean_g[mode] = np.diag(np.diag(np.asarray(linear.couplings[mode])))
clean_g[2, 0, 1] = clean_g[2, 1, 0] = float(linear.couplings[2, 0, 1])
symmetry_residual = float(np.max(abs(clean_g - np.asarray(linear.couplings))))
assert symmetry_residual < 1e-10
paired_model = ep.PhononModel(model.energies, jnp.asarray(clean_g[:, :2, :2]))
single_model = ep.PhononModel(model.energies[:2], jnp.asarray(clean_g[:2, 2:3, 2:3]))


def finite_lines(h, phonons, d, cutoff):
    matrix, moments = ep.vibronic_hamiltonian(h, phonons, d, max_quanta=cutoff)
    roots = solve_hermitian(
        matrix,
        config=EigenSolverConfig(
            method="dense", nroots=matrix.shape[0], max_dense=matrix.shape[0], atol=1e-9
        ),
    )
    assert bool(jnp.all(roots.converged)) and float(roots.values.min()) > 0
    line_dipoles = np.asarray(roots.vectors.T @ moments)
    np.testing.assert_allclose(
        np.sum(abs(line_dipoles) ** 2), np.sum(abs(d) ** 2), rtol=1e-10
    )
    return np.asarray(roots.values), line_dipoles, float(roots.residual_norms.max())


convergence = []
previous = previous_pair = previous_fine = previous_pair_fine = None
paired_converged = False
for cutoff in (4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44):
    begin = time.perf_counter()
    if not paired_converged:
        pair_e, pair_d, pair_residual = finite_lines(
            electronic.e[:2], paired_model, dipoles[:2], cutoff
        )
        pair_curve = cross_section(pair_e, pair_d)
        pair_fine = cross_section(pair_e, pair_d, 0.003)
        pair_change = (
            None
            if previous_pair is None
            else float(
                np.sum(abs(pair_curve - previous_pair)) / np.sum(abs(pair_curve))
            )
        )
        pair_fine_change = (
            None
            if previous_pair_fine is None
            else float(
                np.sum(abs(pair_fine - previous_pair_fine)) / np.sum(abs(pair_fine))
            )
        )
        paired_converged = cutoff >= 12 and max(pair_change, pair_fine_change) < 0.001
        pair_cutoff = cutoff
        previous_pair = pair_curve
        previous_pair_fine = pair_fine
    single_e, single_d, single_residual = finite_lines(
        electronic.e[2:], single_model, dipoles[2:], cutoff
    )
    line_energies = np.concatenate((pair_e, single_e))
    line_dipoles = np.concatenate((pair_d, single_d))
    current = pair_curve + cross_section(single_e, single_d)
    current_fine = pair_fine + cross_section(single_e, single_d, 0.003)
    record = dict(
        max_quanta=cutoff,
        paired_max_quanta=pair_cutoff,
        sector_dimensions=[len(pair_e), len(single_e)],
        seconds=time.perf_counter() - begin,
        relative_l1_change=(
            None
            if previous is None
            else float(np.sum(abs(current - previous)) / np.sum(abs(current)))
        ),
        paired_relative_l1_change=pair_change,
        fine_eta_relative_l1_change=(
            None
            if previous_fine is None
            else float(
                np.sum(abs(current_fine - previous_fine)) / np.sum(abs(current_fine))
            )
        ),
        paired_fine_eta_relative_l1_change=pair_fine_change,
        lowest_transition_ev=float(line_energies.min() * HARTREE_TO_EV),
        max_residual_ha=max(pair_residual, single_residual),
    )
    convergence.append(record)
    curves[f"vibronic_{cutoff}"] = current
    previous = current
    previous_fine = current_fine
    print("Vibrational convergence", record, flush=True)
    if (
        paired_converged
        and cutoff >= 16
        and max(record["relative_l1_change"], record["fine_eta_relative_l1_change"])
        < 0.005
    ):
        break
assert convergence[-1]["relative_l1_change"] < 0.005, "Increase the vibrational cutoff"
assert convergence[-1]["fine_eta_relative_l1_change"] < 0.005, "Resolve the smaller eta"
assert paired_converged, "Resolve the coupled pair to its independent tolerance"
final_key = f"vibronic_{cutoff}"
curves["vibronic_fine_eta"] = current_fine
np.savez(
    folder / "vibronic_lines.npz", energies_ha=line_energies, dipoles_bohr=line_dipoles
)

# Independent-mode Franck-Condon oracle: deliberately removes state mixing.
# Keep raw Poisson weights; never renormalize away a missing vibrational tail.
diagonal_g = np.diagonal(np.asarray(model.couplings), axis1=1, axis2=2)
huang_rhys = (diagonal_g / omega[:, None]) ** 2
occupations = np.array(list(product(range(25), repeat=3)))
fc_e, fc_d, retained = [], [], []
for state in range(3):
    s = huang_rhys[:, state]
    log_weight = np.sum(
        -s + occupations * np.log(np.maximum(s, 1e-300)) - gammaln(occupations + 1),
        axis=1,
    )
    weight = np.exp(log_weight)
    retained.append(float(weight.sum()))
    fc_e.append(
        float(electronic.e[state])
        - np.sum(diagonal_g[:, state] ** 2 / omega)
        + occupations @ omega
    )
    fc_d.append(np.sqrt(weight[:, None]) * np.asarray(dipoles[state]))
curves["diagonal_fc"] = cross_section(np.concatenate(fc_e), np.concatenate(fc_d))
diagonal_model = replace(linear, couplings=jax.vmap(jnp.diag)(jnp.asarray(diagonal_g)))
curves["diagonal_fan"] = np.asarray(
    ep.absorption_cross_section(
        electronic.e,
        diagonal_model,
        dipoles,
        jnp.asarray(axis),
        beta=beta,
        eta=eta_ev / HARTREE_TO_EV,
        unit="Mb",
    )
)

# Gauge-invariant differentiable absorption at fixed finite vibrational cutoff.
probe = float(electronic.e[0]) + 0.3 * omega[0]
config = LinearSolverConfig(method="direct", rtol=1e-10, max_dense=1024)


def loss(scale):
    matrix, moments = ep.vibronic_hamiltonian(
        electronic.e,
        replace(linear, couplings=linear.couplings * scale),
        dipoles,
        max_quanta=4,
    )

    def moment(w):
        resolvent = solve_complex(
            (w + 1j * eta_ev / HARTREE_TO_EV) * jnp.eye(matrix.shape[0]) - matrix,
            moments,
            config=config,
        )
        return moments.conj().T @ resolvent.solution

    alpha = -moment(probe) - moment(-probe).conj()
    return (
        4
        * jnp.pi
        * probe
        * jnp.trace(alpha).imag
        / 3
        / 137.035999084
        * (0.529177210903**2 * 100)
    )


value, derivative = jax.jit(jax.value_and_grad(loss))(1.0)
fd = (loss(1.0 + 1e-5) - loss(1.0 - 1e-5)) / (2e-5)
np.testing.assert_allclose(derivative, fd, rtol=2e-5, atol=2e-7)
summary = dict(
    molecule="H2O",
    basis="3-21g",
    method="native RHF -> G0W0 -> three-root singlet TDA-BSE",
    approximation="RHF-screened vertices, frozen GW correction/kernel; linear vibronic Hamiltonian; Condon vacuum",
    hardware=platform.machine(),
    backend=jax.default_backend(),
    jax_version=jax.__version__,
    dtype="float64",
    temperature_k=0.0,
    coordinates_bohr=r.tolist(),
    hf_energy_ha=float(reference.total_energy),
    max_gradient_ha_bohr=float(np.max(abs(gradient))),
    hessian_asymmetry_ha_bohr2=asymmetry,
    mode_frequencies_cm1=(omega * 219474.6313632).tolist(),
    excitation_energies_ev=np.asarray(electronic.e * HARTREE_TO_EV).tolist(),
    oscillator_strengths=np.asarray(electronic.oscillator_strength()).tolist(),
    qp_max_residual_ha=float(abs(qp.qp_residual).max()),
    symmetry_residual_ha=symmetry_residual,
    huang_rhys=huang_rhys.tolist(),
    diagonal_fc_envelope_std_ev=(
        np.sqrt(np.sum(diagonal_g**2, axis=0)) * HARTREE_TO_EV
    ).tolist(),
    fc_retained_weight=retained,
    numerical_eta_ev=eta_ev,
    smaller_eta_ev=0.003,
    vibrational_convergence=convergence,
    probe_energy_ev=probe * HARTREE_TO_EV,
    probe_absorption_mb=float(value),
    probe_max_quanta=4,
    probe_differentiation="linear vertex scale at fixed H_X, phonon energies, dipoles and finite basis",
    coupling_scale_derivative_mb=float(derivative),
    coupling_scale_finite_difference_mb=float(fd),
    wall_seconds=time.perf_counter() - started,
)
(folder / "results.json").write_text(json.dumps(summary, indent=2) + "\n")
np.savetxt(
    folder / "absorption.csv",
    np.column_stack((axis_ev, *curves.values())),
    delimiter=",",
    header="photon_energy_ev," + ",".join(curves),
    comments="",
)

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)
fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
for key, label, color in [
    ("bse", "Fixed-nuclei GW-BSE", "#2166ac"),
    ("fan", "One-phonon Fan (linear model)", "#5a8c48"),
    ("fan_dw", "One-phonon Fan + DW", "#ce8426"),
    (final_key, "Multiphonon, linear Hamiltonian", "#a63d40"),
]:
    axes[0, 0].plot(axis_ev, curves[key], label=label, color=color, lw=1.2)
axes[0, 0].set(
    xlim=(6, 15),
    ylabel="Absorption cross section (Mb)",
    title="Three electronic states, three vibrational modes",
)
axes[0, 0].legend(frameon=False, fontsize=9)
for key, label in [
    ("bse", "Fixed nuclei"),
    ("diagonal_fan", "Diagonal one-phonon Fan"),
    ("diagonal_fc", "Diagonal multiphonon Franck-Condon"),
]:
    axes[0, 1].plot(axis_ev, curves[key], label=label, lw=1.2)
axes[0, 1].set(
    xlim=(6, 15),
    ylabel="Absorption cross section (Mb)",
    title="Same diagonal coupling: approximation comparison",
)
axes[0, 1].legend(frameon=False, fontsize=9)
for record in convergence[-3:]:
    cutoff = record["max_quanta"]
    label = f"Cutoffs {record['paired_max_quanta']} / {cutoff}"
    axes[1, 0].plot(axis_ev, curves[f"vibronic_{cutoff}"], label=label, lw=1.2)
axes[1, 0].set(
    xlim=(6, 15),
    xlabel="Photon energy (eV)",
    ylabel="Absorption cross section (Mb)",
    title="Convergence: coupled pair / separate bright state",
)
axes[1, 0].legend(frameon=False, fontsize=9)
axes[1, 1].plot(axis_ev, curves[final_key], label="eta = 10 meV", lw=1.2)
axes[1, 1].plot(axis_ev, curves["vibronic_fine_eta"], label="eta = 3 meV", lw=1.0)
first = float(electronic.e[0] * HARTREE_TO_EV)
axes[1, 1].set(
    xlim=(first - 1.5, first + 1.5),
    xlabel="Photon energy (eV)",
    ylabel="Absorption cross section (Mb)",
    title="Resolved lines versus numerical smoothing",
)
axes[1, 1].legend(frameon=False, fontsize=9)
fig.suptitle("Water / 3-21G: vibronic absorption at 0 K, frozen electronic kernel")
fig.savefig(folder / "water_bse_phonons.png", dpi=180)
fig.savefig(folder / "water_bse_phonons.pdf")
plt.close(fig)
print(json.dumps(summary, indent=2))
print(folder / "water_bse_phonons.png")

# CPU arm64, float64, JAX 0.8.1 output (2026-10-05):
# RHF/3-21G modes / cm^-1: [1799.2880168, 3812.37623376, 3945.83205393]
# BSE excitations / eV: [8.9263328303, 11.1390119063, 11.5445332999]
# Final coupled-pair / separate-state cutoffs: 28 / 40.
# Last spectral relative L1 changes (eta=10 / 3 meV): 8.68918e-5 / 5.54227e-4.
# Diagonal FC envelope standard deviations, bright states / eV: 0.5170, 0.7275.
# Finite-cutoff AD probe (total quanta <= 4): 1.1853226125 Mb.
# Linear-coupling-scale derivative / Mb: 152.6769102802.
# Recomputed central finite difference / Mb: 152.6767882203.
# Frozen GW correction/kernel, harmonic vacuum: no experimental spectrum claim.

# Refreshed on jy_gpu_1, CPU float64, JAX 0.10.0 (2026-10-16):
# Modes / cm^-1: [1799.2881713, 3812.3761608, 3945.8318650].
# BSE / eV: [8.9263327801, 11.1390116905, 11.5445338880].
# Cutoffs 28 / 40; final eta10/eta3 spectral L1 changes: 8.6891e-5 / 5.5421e-4.
# Coupling-scale derivative / Mb: AD152.67699523, FD152.67687317 (original gate PASS).
# Calculation body347.889s; process peak9,263,848KiB. The frozen model is unchanged.
