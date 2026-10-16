"""Plot stored EOM/FCI residues and a broadened one-particle spectral trace.

The imposed Lorentzian HWHM is 0.15 eV. This is a density-of-states model, not
a photoionization cross section: no continuum matrix elements are present.
Negative spectral energies are removal poles (-IP); positive ones are EA
eigenvalues E(N+1)-E(N). All curves include the closed-shell spin factor two.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


output = Path("reproducibility/ad_showcase/X14_h2_eom_dyson")
data = json.loads((output/"gradscf.json").read_text())
reference = json.loads((output/"pyscf_fci.json").read_text())
values = np.asarray(data["values"])
tangent = np.asarray(data["derivative_per_bohr"])
hartree_to_ev = 27.211386245988
sign = np.array([-1., -1., 1., 1.])
poles, dpoles = values[:4]*sign*hartree_to_ev, tangent[:4]*sign*hartree_to_ev
weights, dweights = 2*values[4:], 2*tangent[4:]
grid = np.linspace(-80., 80., 16001)
width = .15
delta = grid[:, None]-poles
lorentzian = width/(np.pi*(delta**2+width**2))
spectrum = lorentzian @ weights
response = lorentzian @ dweights + np.sum(
    2*width*delta/(np.pi*(delta**2+width**2)**2)*weights*dpoles, axis=1)

# Report the actual finite-window area and missing Lorentzian tails.
pole_areas = (np.arctan((grid[-1]-poles)/width)-np.arctan((grid[0]-poles)/width))/np.pi
area = float(pole_areas @ weights)
darea = float(pole_areas @ dweights+np.sum(weights*dpoles*width/np.pi*(
    1/((grid[0]-poles)**2+width**2)-1/((grid[-1]-poles)**2+width**2))))
area_error = abs(np.trapezoid(spectrum, grid)-area)
response_area_error = abs(np.trapezoid(response, grid)-darea)
np.testing.assert_allclose(area_error, 0., atol=2e-9)
np.testing.assert_allclose(response_area_error, 0., atol=2e-9)

def spectral_trace(data):
    data = np.asarray(data)
    separation = grid[:, None]-data[:4]*sign*hartree_to_ev
    return (width/(np.pi*(separation**2+width**2))) @ (2*data[4:])


spectral_checks = []
for check in data["finite_differences"]:
    numerical = (spectral_trace(check["plus_values"])-spectral_trace(check["minus_values"]))/(2*check["step_bohr"])
    spectral_checks.append(dict(step_bohr=check["step_bohr"],
        max_error_per_ev_bohr=float(np.max(np.abs(numerical-response)))))
assert spectral_checks[-1]["max_error_per_ev_bohr"] < 1e-3, spectral_checks

plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
fig, axes = plt.subplots(2, 2, figsize=(10, 6.8), constrained_layout=True)
labels = ["IP 1", "IP 2", "EA 1", "EA 2"]
axes[0, 0].plot(poles, weights, "o", label="GradSCF EOM-CCSD")
ref = np.asarray(reference["values"])
axes[0, 0].plot(ref[:4]*sign*hartree_to_ev, 2*ref[4:], "+", ms=10, label="PySCF FCI")
for energy, weight, label in zip(poles, weights, labels):
    axes[0, 0].vlines(energy, 0, weight, color="C0", alpha=.5)
    axes[0, 0].annotate(label, (energy, weight), xytext=(4, 6), textcoords="offset points")
axes[0, 0].set(xlabel="Spectral energy / eV", ylabel="Spin-summed pole weight",
                title="Full two-orbital removal and addition spaces", ylim=(-.05, 2.18))
axes[0, 0].legend(frameon=False, fontsize=8)
axes[0, 1].bar(labels, dweights, color=["C0", "C0", "C1", "C1"])
axes[0, 1].axhline(0, color=".5", lw=.7)
axes[0, 1].set(ylabel=r"$d(2S_k)/dR_{HH}$ / bohr$^{-1}$", title="Correlation transfers spectral weight")
axes[1, 0].plot(grid, spectrum, color="C0")
axes[1, 0].set(xlabel="Spectral energy / eV", ylabel=r"$A(E)$ / eV$^{-1}$",
               title="Imposed Lorentzian HWHM 0.15 eV", xlim=(-55, 55))
axes[1, 1].plot(grid, response, color="C3")
axes[1, 1].axhline(0, color=".5", lw=.7)
axes[1, 1].set(xlabel="Spectral energy / eV", ylabel=r"$dA/dR_{HH}$ / (eV bohr)$^{-1}$",
               title="Complete SCF + CC + Lambda + EOM response", xlim=(-55, 55))
fig.suptitle("H2 / STO-3G, R = 1.4 bohr: EOM Dyson residues and bond response")
for extension in ("png", "pdf"):
    fig.savefig(output/("dyson_response."+extension), dpi=180)
np.savetxt(output/"spectrum.csv", np.column_stack((grid, spectrum, response)), delimiter=",",
           header="energy_eV,spin_summed_spectrum_per_eV,bond_response_per_eV_bohr", comments="")
(output/"spectrum_checks.json").write_text(json.dumps(dict(
    lorentzian_hwhm_ev=width, energy_window_ev=[float(grid[0]), float(grid[-1])],
    total_weight=float(weights.sum()), finite_window_area=area,
    missing_tail_weight=float(weights.sum()-area), area_integration_error=float(area_error),
    area_derivative_per_bohr=darea, area_derivative_integration_error=float(response_area_error),
    reconverged_spectral_checks=spectral_checks,
    model="One-particle trace; no continuum photoionization matrix elements"), indent=2)+"\n")

# Measured: total spin-summed weight 4.0; finite-window area 3.994966604392.
# Missing Lorentzian tails: 0.005033395608. Area integration error: 1.694e-11.
# Final reconverged spectral-response FD error: 2.792e-4 / (eV bohr).
