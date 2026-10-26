"""Sensitivity heatmaps and independently checked X06 directional errors."""
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


output = Path("reproducibility/ad_showcase/X06_water_cis_probes")
native = json.loads((output/"gradscf.json").read_text())
reference = json.loads((output/"pyscf.json").read_text())
digest = hashlib.sha256((output/"gradscf.json").read_bytes()).hexdigest()
if not native["complete"] or not reference["passed"] or reference["native_sha256"] != digest:
    raise RuntimeError("Only current independently validated X06 records are plotted.")
plt.rcParams.update({"font.size": 10, "pdf.fonttype": 42,
                     "axes.spines.top": False, "axes.spines.right": False})
fig, axes = plt.subplots(2, 3, figsize=(14.4, 8.), layout="constrained")
panels = (
    ("geometry", "nuclear_gradient", "Relaxed nuclear response", "q / Bohr"),
    ("amplitude", "amplitude_gradient", "Normalized amplitude partial", "q / amplitude"),
    ("reference_rotation", "reference_rotation_gradient", "Fixed-X reference rotation partial", "q / radian"),
)
for column, (kind, key, title, unit) in enumerate(panels):
    matrix = np.asarray(native[key])
    axis = axes[0, column]
    limit = max(float(np.abs(matrix).max()), 1e-12)
    artist = axis.imshow(matrix, cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto")
    axis.set_title(title, fontsize=11)
    if kind == "geometry":
        axis.set(xticks=range(3), xticklabels=["x", "y", "z"], yticks=range(3), yticklabels=["O", "H1", "H2"])
    else:
        axis.set(xticks=np.arange(0, matrix.shape[1], 2), xticklabels=np.arange(native["nocc"]+1, native["nmo"]+1, 2),
                 yticks=range(matrix.shape[0]), yticklabels=np.arange(1, native["nocc"]+1),
                 xlabel="Virtual MO (recorded frame)", ylabel="Occupied MO")
    fig.colorbar(artist, ax=axis, orientation="horizontal", shrink=.9, pad=.10, label=unit)
    with (output/f"{key}.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["row_0_based", "column_0_based", "derivative", "units"])
        for index, value in np.ndenumerate(matrix):
            writer.writerow([*index, value, unit])
    axis = axes[1, column]
    for data, label, marker, color in ((native, "Native FD", "o", "#0072B2"),
                                        (reference, "Independent PySCF FD", "s", "#D55E00")):
        checks = [item for item in data["finite_difference_checks"] if item["kind"] == kind]
        steps = [item["step"] for item in checks[0]["finite_differences"]]
        errors = [max(check["finite_differences"][i]["absolute_error"] for check in checks) for i in range(len(steps))]
        axis.loglog(steps, errors, marker+"-", color=color, label=label)
    axis.axhline(2e-6, color=".5", linestyle="--", linewidth=1, label="Acceptance gate")
    axis.set(title="Three controlled directions", xlabel="Central-difference step", ylabel="Max directional derivative error")
    axis.legend(frameon=False, fontsize=9)
fig.suptitle("Water S1 · RHF/CIS · Cartesian 6-31G*\n"
             r"Probe $q=|\mu_{01}|^2$: distinct parameter derivatives; heatmap coordinates use the recorded MO frame", fontsize=13)
for suffix in ("png", "pdf"):
    fig.savefig(output/f"probe_sensitivities.{suffix}", dpi=220)
plt.close(fig)
print("Saved X06 sensitivity heatmaps, directional checks and three CSV tables", flush=True)
