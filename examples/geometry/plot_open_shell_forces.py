"""G03: bond derivatives, independent errors and spin diagnostics from JSON."""
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


output = Path("reproducibility/ad_showcase/G03_open_shell_forces")
systems, methods = ("OH", "NO"), ("HF", "PBE", "MP2", "CCSD")
colors = {"HF": "#337a9b", "PBE": "#b16529", "MP2": "#89752e", "CCSD": "#b35f76"}
plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "pdf.fonttype": 42})
fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), layout="constrained")
rows = []
provenance = {}
for j, method in enumerate(methods):
    for i, name in enumerate(systems):
        case = f"{name}_{method.lower()}"
        native = json.loads((output / f"{case}.json").read_text())
        reference = json.loads((output / f"{case}_pyscf.json").read_text())
        native_hash = hashlib.sha256((output / f"{case}.json").read_bytes()).hexdigest()
        correlated = method in ("MP2", "CCSD")
        if correlated:
            assert reference["passed"] and reference["native_sha256"] == native_hash
            assert reference["ucphf_responses"] and all(row["passed"] for row in reference["ucphf_responses"])
        assert np.isfinite(reference["max_gradient_error"]) and reference["max_gradient_error"] < 2e-6
        gradient = float(np.sum(np.array(native["gradient_hartree_per_bohr"])*native["direction"]))
        reference_gradient = float(np.sum(np.array(reference["gradient_hartree_per_bohr"])*native["direction"]))
        x = i+(j-1.5)*.18
        label = method if i == 0 else None
        axes[0].bar(x, gradient, width=.16, color=colors[method], label=label)
        axes[0].plot(x, reference_gradient, "k_", markersize=7)
        if method in ("HF", "PBE"):
            axes[2].bar(i+(j-.5)*.34, native["spin_square"]-.75,
                        width=.3, color=colors[method], label=label)
        checks = native["finite_difference_checks"]
        marker = "o" if name == "OH" else "s"
        axes[1].loglog([c["step_bohr"] for c in checks], [c["error"] for c in checks],
                       marker=marker, color=colors[method], label=f"{name}/{method}")
        rows.append(dict(system=name, method=method, energy_hartree=native["energy_hartree"],
                         bond_gradient_hartree_per_bohr=gradient, spin_square=native["spin_square"],
                         spin_diagnostic=native.get("spin_diagnostic", "SCF determinant"),
                         reference_minimum_orbital_curvature_hartree=native["internal_minimum_curvature"],
                         max_pyscf_gradient_error=reference["max_gradient_error"],
                         smallest_step_fd_error=checks[-1]["error"],
                         cold_energy_gradient_seconds=native["cold_energy_gradient_seconds"],
                         median_warm_energy_gradient_seconds=float(np.median(native["warm_energy_gradient_seconds"])),
                         median_warm_forward_seconds=float(np.median(native["warm_forward_seconds"])),
                         reference_response_oracle=reference.get("response_oracle", "PySCF analytic SCF gradient"),
                         reference_true_response_residual=max((item["true_residual_l2"] for item in reference.get("ucphf_responses", [])), default=None),
                         reference_relative_response_residual=max((item["relative_true_residual"] for item in reference.get("ucphf_responses", [])), default=None)))
        provenance[case] = dict(
            native_snapshot="ad-showcase-integration-20261015" if correlated else "ad-showcase-stationary-20261015",
            native_sha256=native_hash, recorded_runtime_source_hashes=native["runtime_source_hashes"],
            reference_snapshot="ad-showcase-open-shell-reference-20261015" if correlated else "ad-showcase-stationary-20261015",
            reference_sha256=hashlib.sha256((output/f"{case}_pyscf.json").read_bytes()).hexdigest(),
            reference_executed_script_sha256=reference.get("script_sha256"),
            provenance_boundary="Named execution snapshots and selected hashes recorded by each native run; not a complete source-tree hash.")
axes[0].axhline(0., color=".55", linewidth=.7)
axes[0].set(title="Fixed bond lengths", ylabel="dE/dR / Eh Bohr⁻¹")
axes[0].legend(frameon=False)
axes[0].text(.02, .03, "Black ticks: independent reference", transform=axes[0].transAxes, fontsize=8)
axes[1].set(title="Bond-displacement validation", xlabel="Central-difference step / Bohr",
            ylabel="Absolute derivative error / Eh Bohr⁻¹")
axes[1].legend(frameon=False, fontsize=8, ncols=2)
axes[2].set(title="SCF/reference spin diagnostics", ylabel="⟨S²⟩ − 0.75")
axes[2].legend(frameon=False)
for ax in (axes[0], axes[2]):
    ax.set(xticks=[0, 1], xticklabels=["OH · 1.85 Bohr", "NO · 2.18 Bohr"])
fig.suptitle("OH/NO · UHF, UKS-PBE, UMP2 and UCCSD · Cartesian 6-31G*")
for suffix in ("png", "pdf"):
    fig.savefig(output / f"open_shell_forces.{suffix}", dpi=220)
plt.close(fig)
with (output / "summary.csv").open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
(output / "provenance.json").write_text(json.dumps(provenance, indent=2)+"\n")
