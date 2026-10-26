"""Native H2: RHF/3-21G -> G0W0 -> TDA-BSE -> exciton–vibration absorption.

HF-screened linear vertices, finite-displacement quadratic vertices, frozen
GW correction and electronic BSE kernel, fixed harmonic bath at 300 K.
This is a finite-basis Hamiltonian demonstration, without experimental fitting,
Herzberg–Teller dipoles or multiphonon Franck–Condon factors. No PySCF or CLI.
"""

from dataclasses import replace
from pathlib import Path
import json
import time

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize_scalar
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from gradscf import bse
from gradscf.bse import ep_coupling
from gradscf.integrals.molecular.factorization import eri_to_df_factors
from gradscf.gw import g0w0_cd_restricted
from gradscf.gw.ep_coupling import PhononModel
from gradscf_tools.molecular_ep import NativeRHF
from gradscf.tools.spectra import HARTREE_TO_EV

jax.config.update("jax_enable_x64", True)
started = time.perf_counter()
engine = NativeRHF("H 0 0 0; H 0 0 1.4", basis="3-21g")


def coordinates(distance):
    return np.array([[0.0, 0.0, 0.0], [0.0, 0.0, distance]])


def energy(distance):
    result, _ = engine.evaluate(coordinates(distance), gradient=False)
    return float(result.total_energy)


minimum = minimize_scalar(
    energy, bounds=(1.1, 1.7), method="bounded", options={"xatol": 1e-9}
)
r = coordinates(minimum.x)
reference, gradient = engine.evaluate(r)
assert np.max(abs(gradient)) < 2e-6
mass = 1.00782503223 * 1822.888486209
# The diatomic internal space is exactly its mass-weighted bond stretch.
direction = np.array([[0.0, 0.0, -1.0], [0.0, 0.0, 1.0]]) / np.sqrt(2 * mass)
hvp, derivative = engine.response(r, reference)(direction)
omega = np.sqrt(np.sum(direction * hvp))
g = derivative / np.sqrt(2 * omega)

# Lambda is a second derivative of the transported Hamiltonian operator.
dq = 0.002 / np.max(abs(direction))
matrices = []
for sign in (1.0, -1.0):
    displaced = r + sign * dq * direction
    result, _ = engine.evaluate(
        displaced, init_density=reference.density_matrix, gradient=False
    )
    fock, _ = engine.transport(result, reference.mo_coeff, r, displaced)
    matrices.append(fock)
quadratic = (
    (matrices[0] + matrices[1] - 2 * np.diag(reference.mo_energy)) / dq**2 / (2 * omega)
)
phonons = PhononModel(
    jnp.array([omega]),
    jnp.asarray(g[None]),
    jnp.asarray(quadratic[None]),
    reference="native RHF/3-21G bond stretch; frozen GW correction",
)

s, h, eri, enuc = engine.integrals(r)
factors = eri_to_df_factors(eri, tol=1e-12)
qp = g0w0_cd_restricted(
    mo_energy=reference.mo_energy,
    mo_coeff=reference.mo_coeff,
    nocc=1,
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
    qp, mo_factors=mo_factors, nocc=1, dipole_mo=dipole_mo
)
response = bse.BSE(source, nroots=3, tda=True, solver="dense").run()
model = ep_coupling.project_phonons(response.result, response.space, phonons)
dipoles = response.transition_dipole()

beta = 1 / (3.166811563e-6 * 300.0)
eta = 0.01 / HARTREE_TO_EV
dw = ep_coupling.debye_waller(model, beta)
margin = (
    float(
        (
            omega
            + jnp.linalg.norm(model.couplings[0], ord=2)
            + jnp.linalg.norm(dw, ord=2)
        )
        * HARTREE_TO_EV
    )
    + 0.5
)
low = float(response.e.min() * HARTREE_TO_EV) - margin
high = float(response.e.max() * HARTREE_TO_EV) + margin
axis_ev = np.linspace(low, high, int(np.ceil((high - low) / 0.0025)) + 1)
axis = jnp.asarray(axis_ev / HARTREE_TO_EV)
spectrum = jax.jit(
    lambda value: ep_coupling.absorption_cross_section(
        response.e, value, dipoles, axis, beta=beta, eta=eta, unit="Mb"
    )
)
zero = replace(model, couplings=jnp.zeros_like(model.couplings), quadratic=None)
curves = {
    "bse": np.asarray(spectrum(zero)),
    "fan": np.asarray(spectrum(replace(model, quadratic=None))),
    "fan_dw": np.asarray(spectrum(model)),
}
probe = response.e[0] + 0.3 * omega
loss = lambda scale: ep_coupling.absorption_cross_section(
    response.e,
    replace(model, couplings=model.couplings * scale),
    dipoles,
    probe,
    beta=beta,
    eta=eta,
    unit="Mb",
)
value, derivative = jax.jit(jax.value_and_grad(loss))(1.0)
finite_difference = (loss(1.0 + 1e-5) - loss(1.0 - 1e-5)) / (2e-5)
np.testing.assert_allclose(derivative, finite_difference, rtol=2e-5, atol=2e-7)

folder = Path(__file__).resolve().parents[2] / "reproducibility/gw_bse/h2_bse_phonons"
folder.mkdir(parents=True, exist_ok=True)
np.savetxt(
    folder / "absorption.csv",
    np.column_stack((axis_ev, *curves.values())),
    delimiter=",",
    header="photon_energy_ev,bse_mb,fan_mb,fan_dw_mb",
    comments="",
)
np.savez(
    folder / "inputs.npz",
    excitation_energies=response.e,
    mode_energies=model.energies,
    couplings=model.couplings,
    quadratic=model.quadratic,
    dipoles=dipoles,
    coordinates=r,
)
summary = dict(
    basis="3-21g",
    method="native RHF -> G0W0 -> TDA-BSE; frozen electronic kernel",
    backend=jax.default_backend(),
    jax_version=jax.__version__,
    dtype="float64",
    temperature_k=300.0,
    bond_length_bohr=float(minimum.x),
    hf_energy_ha=float(reference.total_energy),
    maximum_gradient_ha_bohr=float(np.max(abs(gradient))),
    mode_frequency_cm1=float(omega * 219474.6313632),
    qp_max_residual_ha=float(abs(qp.qp_residual).max()),
    excitation_energies_ev=np.asarray(response.e * HARTREE_TO_EV).tolist(),
    oscillator_strengths=np.asarray(response.oscillator_strength()).tolist(),
    debye_waller_ev=np.asarray(
        ep_coupling.debye_waller(model, beta) * HARTREE_TO_EV
    ).tolist(),
    numerical_eta_ev=0.01,
    spectral_range_ev=[low, high],
    maximum_coupling_over_mode_energy=float(abs(model.couplings).max() / omega),
    probe_energy_ev=float(probe * HARTREE_TO_EV),
    probe_absorption_mb=float(value),
    coupling_scale_derivative_mb=float(derivative),
    coupling_scale_finite_difference_mb=float(finite_difference),
    wall_seconds=time.perf_counter() - started,
)
(folder / "results.json").write_text(json.dumps(summary, indent=2) + "\n")

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)
fig, axes = plt.subplots(1, 3, figsize=(13.5, 4), constrained_layout=True)
for key, label, color in [
    ("bse", "GW–BSE", "#2166ac"),
    ("fan", "+ exciton Fan", "#ce8426"),
    ("fan_dw", "+ Fan + DW", "#a63d40"),
]:
    axes[0].plot(axis_ev, curves[key], label=label, color=color, lw=1.5)
    axes[1].plot(axis_ev, curves[key], color=color, lw=1.5)
axes[0].set(
    xlabel="Photon energy (eV)", ylabel="Absorption cross section (Mb)", ylim=(0, None)
)
axes[0].legend(frameon=False)
peak = float(response.e[0] * HARTREE_TO_EV)
local = (
    float((omega + 2 * abs(model.couplings[0, 0, 0]) + abs(dw[0, 0])) * HARTREE_TO_EV)
    + 0.1
)
axes[1].set(
    xlim=(peak - local, peak + local),
    ylim=(0, None),
    xlabel="Photon energy (eV)",
    ylabel="Absorption cross section (Mb)",
    title="First excitation and satellite",
)
image = axes[2].imshow(
    abs(np.asarray(model.couplings[0])) * HARTREE_TO_EV * 1000, cmap="viridis"
)
axes[2].set(
    xlabel="Exciton state", ylabel="Exciton state", xticks=range(3), yticks=range(3)
)
fig.colorbar(image, ax=axes[2], label="|Coupling| (meV)")
fig.suptitle("Native H2/3-21G, 300 K; frozen GW correction and BSE kernel")
fig.savefig(folder / "h2_bse_phonons.png", dpi=180)
fig.savefig(folder / "h2_bse_phonons.pdf")
plt.close(fig)
print(json.dumps(summary, indent=2))
print(folder / "h2_bse_phonons.png")

# CPU float64 output (2026-10-05; native integrals, JAX 0.8.1):
# Optimized R / Bohr: 1.3886142448271075
# Stretching mode / cm^-1: 4657.106164344072
# BSE energies / eV: [16.2660620794592, 32.47146941008201, 46.326591735684545]
# Probe absorption / Mb: 0.1557420625840397
# AD coupling-scale derivative / Mb: -0.31212720798362176
# Central finite difference / Mb: -0.31212720805057215
# Maximum |G|/omega: 6.205381784002731 (one-phonon stress model; see results README).
