"""Plot saved X03 accepted steps, structural change and local curvature."""
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


output = Path("reproducibility/ad_showcase/X03_formaldehyde_cis_optimization")
native = json.loads((output / "gradscf.json").read_text())
reference = json.loads((output / "pyscf.json").read_text())
if not native["complete"] or not reference["passed"]:
    raise RuntimeError("Only a validated stationary minimum is plotted.")
if reference["native_sha256"] != hashlib.sha256((output / "gradscf.json").read_bytes()).hexdigest():
    raise RuntimeError("The independent comparison is stale.")
history = native["history"]
iteration = np.array([item["iteration"] for item in history])
energy = np.array([item["energy_hartree"] for item in history])
gradient = np.array([item["max_cartesian_gradient"] for item in history])
plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "pdf.fonttype": 42})
fig = plt.figure(figsize=(10.5, 7.7), layout="constrained")
ax = fig.add_subplot(2, 2, 1)
ax.plot(iteration, (energy-energy[0])*27.211386245988, "o-", color="#0072B2", markersize=4)
ax.set(xlabel="Accepted step", ylabel="Energy change from distorted start / eV",
       title="Tracked total excited-state energy")
ax.text(.43, .8, f"Min step overlap: {min(item['overlap'] for item in history):.6f}", transform=ax.transAxes)
ax = fig.add_subplot(2, 2, 2)
ax.semilogy(iteration, gradient, "o-", color="#D55E00", markersize=4)
ax.axhline(native["gradient_tolerance"], color=".5", linestyle="--", linewidth=1)
ax.set(xlabel="Accepted step", ylabel="Max Cartesian |gradient| / (Eh/Bohr)",
       title="Full Cartesian stationarity")

ax = fig.add_subplot(2, 2, 3, projection="3d")
for coordinates, color, alpha, label in (
    (native["initial_coordinates_bohr"], "#999999", .45, "Distorted start"),
    (native["coordinates_bohr"], "#0072B2", 1., "S1 minimum"),
):
    r = np.asarray(coordinates)
    for other in (1, 2, 3):
        ax.plot(r[[0, other], 0], r[[0, other], 1], r[[0, other], 2],
                color=color, alpha=alpha, linewidth=2, label=label if other == 1 else None)
    ax.scatter(r[:, 0], r[:, 1], r[:, 2], color=color, alpha=alpha, s=[70, 85, 35, 35])
combined = np.vstack([native["initial_coordinates_bohr"], native["coordinates_bohr"]])
center = (combined.max(axis=0)+combined.min(axis=0))/2
radius = np.max(np.ptp(combined, axis=0))/2+.2
ax.set(xlim=(center[0]-radius, center[0]+radius),
       ylim=(center[1]-radius, center[1]+radius), zlim=(center[2]-radius, center[2]+radius))
ax.set_box_aspect((1, 1, 1))
ax.view_init(elev=20, azim=55)
ax.set(xlabel="x / Bohr", ylabel="y / Bohr", zlabel="z / Bohr", title="Geometry change in the fixed chart")
ax.legend(frameon=False, loc="upper left", fontsize=9)

ax = fig.add_subplot(2, 2, 4)
positions = np.arange(1, 7)
ax.bar(positions-.16, native["chart_curvature_eigenvalues_hartree_per_bohr2"], width=.30,
       color="#0072B2", label="FD of native AD gradients")
ax.bar(positions+.16, reference["chart_curvature_eigenvalues_hartree_per_bohr2"], width=.30,
       color="#D55E00", label="FD of PySCF gradients")
ax.axhline(0., color=".5", linewidth=1)
ax.set(xlabel="Ordered eigenvalue", ylabel="Chart curvature / (Eh/Bohr²)",
       title="Local curvature: gradient differences", xticks=positions)
ax.legend(frameon=False, fontsize=9)
ax.text(.04, .6, f"Minimum: {min(native['chart_curvature_eigenvalues_hartree_per_bohr2']):.6f} Eh/Bohr²",
        transform=ax.transAxes, fontsize=9)
fig.suptitle("Formaldehyde S1 · RHF/CIS · Cartesian 6-31+G*\n"
             "Accepted-step tracking; finite-difference curvature, not an AD Hessian", fontsize=12)
for suffix in ("png", "pdf"):
    fig.savefig(output / f"optimization_checks.{suffix}", dpi=220)
plt.close(fig)
with (output / "history.csv").open("w", newline="") as handle:
    writer = csv.writer(handle)
    writer.writerow(["accepted_step", "total_excited_energy_hartree", "max_cartesian_gradient_hartree_per_bohr",
                     "selected_root_zero_based", "transition_density_overlap", "gap_hartree"])
    for item in history:
        writer.writerow([item["iteration"], item["energy_hartree"], item["max_cartesian_gradient"],
                         item["root"], item["overlap"], item["gap_hartree"]])


def geometry_metrics(coordinates):
    r = np.asarray(coordinates)
    co, ch1, ch2 = r[1:]-r[0]
    normal = np.cross(ch1, ch2)
    angle = np.degrees(np.arccos(np.clip(ch1@ch2/(np.linalg.norm(ch1)*np.linalg.norm(ch2)), -1., 1.)))
    tilt = np.degrees(np.arcsin(np.clip(abs(co@normal)/(np.linalg.norm(co)*np.linalg.norm(normal)), 0., 1.)))
    return dict(co_angstrom=float(np.linalg.norm(co)*native["angstrom_per_bohr"]),
                ch_angstrom=(np.linalg.norm(np.stack([ch1, ch2]), axis=1)*native["angstrom_per_bohr"]).tolist(),
                hch_angle_degrees=float(angle), co_angle_to_chh_plane_degrees=float(tilt))


metrics = {name: geometry_metrics(native[key]) for name, key in (
    ("planar_reference", "planar_reference_coordinates_bohr"),
    ("distorted_start", "initial_coordinates_bohr"), ("optimized", "coordinates_bohr"))}
(output / "geometry_metrics.json").write_text(json.dumps(metrics, indent=2)+"\n")
print("Saved optimization_checks.png/.pdf, history.csv and geometry_metrics.json")
