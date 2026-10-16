"""Plot validated X07 linear bright and quadratic dark intensity modulation."""
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


output = Path("reproducibility/ad_showcase/X07_formaldehyde_oscillator_response")
native = json.loads((output/"gradscf.json").read_text())
reference = json.loads((output/"pyscf.json").read_text())
digest = hashlib.sha256((output/"gradscf.json").read_bytes()).hexdigest()
if not native["complete"] or not reference["passed"] or reference["native_sha256"] != digest:
    raise RuntimeError("Only independently validated current X07 records are plotted.")
values, derivatives = np.asarray(native["values"]), np.asarray(native["mode_derivatives"])
jacobian = np.asarray(native["cartesian_jacobians"])
centered = np.asarray(native["coordinates_bohr"])-np.mean(native["coordinates_bohr"], axis=0)
rigid = dict(translation=float(np.max(np.abs(jacobian.sum(axis=2)))),
             moment_rotation=0., strength_rotation=0.)
for axis in np.eye(3):
    direction = np.cross(axis, centered)
    response = np.einsum("sxai,ai->sx", jacobian, direction)
    rigid["moment_rotation"] = max(rigid["moment_rotation"],
        float(np.max(np.abs(response[:, :3]-np.cross(axis, values[:, :3])))))
    rigid["strength_rotation"] = max(rigid["strength_rotation"], float(np.max(np.abs(response[:, 3]))))
if max(rigid.values()) > 2e-7:
    raise RuntimeError(f"Saved Cartesian response violates rigid-motion covariance: {rigid}")
(output/"rigid_response_checks.json").write_text(json.dumps(dict(
    passed=True, errors=rigid, tolerance=2e-7, native_sha256=digest,
    scope="Contractions of saved AD Cartesian Jacobians; independent FD checks span all six internal modes",
    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()), indent=2)+"\n")
gamma = np.asarray(native["dark_quadratic_coefficients"])
bright_mode, dark_mode = int(np.argmax(np.abs(derivatives[1, 3]))), int(np.argmax(gamma))
last = [check for check in reference["finite_difference_checks"] if check["step_bohr_sqrt_amu"] == 2e-4]
plt.rcParams.update({"font.size": 10, "pdf.fonttype": 42,
                     "axes.spines.top": False, "axes.spines.right": False})
fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.4), layout="constrained")
indices = np.arange(1, 7)
axis = axes[0, 0]
axis.bar(indices, derivatives[1, 3], color="#0072B2", width=.65, label="Native AD")
axis.scatter(indices, [check["derivative"][1][3] for check in last], marker="x", color="black", label="Independent FD")
axis.axhline(0, color=".4", linewidth=.8)
axis.set(xticks=indices, xlabel="Ground normal mode", ylabel=r"$df_{bright}/dQ$ / (Bohr $\sqrt{amu}$)$^{-1}$",
         title="Bright-state linear intensity response")
axis.legend(frameon=False)
axis = axes[0, 1]
axis.bar(indices, gamma, color="#D55E00", width=.65, label="From native first moment derivative")
axis.scatter(indices, [check["dark_quadratic_coefficient"] for check in last], marker="x", color="black", label="Independent quadratic FD")
axis.set(xticks=indices, xlabel="Ground normal mode", ylabel=r"$\gamma$ / (Bohr $\sqrt{amu}$)$^{-2}$",
         title=r"Dark state: $f(Q)=\gamma Q^2+O(Q^3)$")
axis.legend(frameon=False, fontsize=9)
for axis, mode, slot, prediction_key, title in (
    (axes[1, 0], dark_mode, 0, "dark_quadratic_prediction", "Dark-state activation"),
    (axes[1, 1], bright_mode, 1, "bright_linear_prediction", "Bright-state modulation")):
    points = [point for point in native["scans"] if point["mode"] == mode]
    refs = [point for point in reference["scans"] if point["mode"] == mode]
    displacement = [point["displacement_bohr_sqrt_amu"] for point in points]
    offset = 0. if slot == 0 else values[slot, 3]
    axis.plot(displacement, [point["values"][slot][3]-offset for point in points], "o-", color="#0072B2", label="Native tracked states")
    axis.scatter(displacement, [point["values"][slot][3]-offset for point in refs], marker="x", color="black", label="Independent PySCF")
    axis.plot(displacement, [point[prediction_key]-offset for point in points], "--", color="#D55E00", label="Local AD prediction")
    axis.set(xlabel=r"Normal coordinate $Q$ / (Bohr $\sqrt{amu}$)",
             ylabel="Oscillator strength" if slot == 0 else "Change in oscillator strength",
             title=f"{title}: mode {mode+1}, {native['frequencies_cm1'][mode]:.1f} cm$^{{-1}}$")
    axis.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
    axis.legend(frameon=False, fontsize=9)
fig.suptitle("Formaldehyde · RHF/CIS · Cartesian 6-31+G*\n"
             "Tracked continuations of X05 S1/S4 at the RHF minimum; fixed harmonic mode directions", fontsize=13)
for suffix in ("png", "pdf"):
    fig.savefig(output/f"intensity_modulation.{suffix}", dpi=220)
plt.close(fig)
with (output/"mode_response.csv").open("w", newline="") as handle:
    writer = csv.writer(handle)
    writer.writerow(["state_label", "root_1_based", "mode_1_based", "frequency_cm1", "dmu_x_dQ", "dmu_y_dQ", "dmu_z_dQ", "df_dQ", "dark_gamma"])
    for slot, label in enumerate(native["original_state_labels"]):
        for mode in range(6):
            writer.writerow([label, native["selected_roots"][slot]+1, mode+1, native["frequencies_cm1"][mode],
                             *derivatives[slot, :, mode], gamma[mode] if slot == 0 else ""])
with (output/"scans.csv").open("w", newline="") as handle:
    writer = csv.writer(handle)
    writer.writerow(["mode_1_based", "Q_bohr_sqrt_amu", "dark_f", "bright_f", "dark_quadratic_prediction", "bright_linear_prediction"])
    for point in native["scans"]:
        writer.writerow([point["mode"]+1, point["displacement_bohr_sqrt_amu"], point["values"][0][3], point["values"][1][3],
                         point["dark_quadratic_prediction"], point["bright_linear_prediction"]])
with (output/"cartesian_response.csv").open("w", newline="") as handle:
    writer = csv.writer(handle)
    writer.writerow(["state_label", "property", "atom_1_based", "Cartesian_axis", "derivative_per_bohr"])
    for (slot, prop, atom, axis), value in np.ndenumerate(np.asarray(native["cartesian_jacobians"])):
        writer.writerow([native["original_state_labels"][slot], native["value_columns"][prop], atom+1, "xyz"[axis], value])
with (output/"continuation.csv").open("w", newline="") as handle:
    writer = csv.writer(handle)
    writer.writerow(["fraction", "original_state", "root_1_based", "transition_overlap", "gap_hartree"])
    for point in native["continuation"]:
        for label, match in zip(native["original_state_labels"], point["matches"]):
            writer.writerow([point["fraction"], label, match["root"]+1, match["overlap"], match["gap_hartree"]])
print("Saved X07 intensity_modulation.png/.pdf and four CSV tables", flush=True)
print("Rigid-response covariance checks:", rigid, flush=True)
