"""Plot validated X04 relaxed/unrelaxed difference fields and permanent dipoles."""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
from time import perf_counter

import jax
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

jax.config.update("jax_enable_x64", True)

from volume_views import (sample_ao_grid, density_field, boundary_maximum,
                          extract_surfaces, common_limits, draw_surfaces, POSITIVE, NEGATIVE)


output = Path("reproducibility/ad_showcase/X04_formaldehyde_cis_density")
native = json.loads((output/"gradscf.json").read_text())
reference = json.loads((output/"pyscf.json").read_text())
native_hash = hashlib.sha256((output/"gradscf.json").read_bytes()).hexdigest()
if not native["complete"] or not reference["passed"] or reference["native_sha256"] != native_hash:
    raise RuntimeError("Only current independently validated X04 results are plotted.")
started = perf_counter()
coordinates = np.asarray(native["coordinates_bohr"])
ao, axes, shape = sample_ao_grid(native["symbols"], coordinates, native["basis"])
ground = np.asarray(native["ground_density_ao"])
fields = {name: density_field(ao, matrix, shape) for name, matrix in (
    ("relaxed_difference", np.asarray(native["relaxed_response_density_ao"])-ground),
    ("unrelaxed_difference", np.asarray(native["unrelaxed_cis_density_ao"])-ground),
    ("attachment", native["attachment_density_ao"]),
    ("detachment", native["detachment_density_ao"]))}
del ao
level = .003  # electrons / Bohr^3; identical across the two difference fields.
surfaces = [extract_surfaces(fields[name], axes, level) for name in ("relaxed_difference", "unrelaxed_difference")]
limits = common_limits(coordinates, surfaces)
plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "pdf.fonttype": 42})
fig = plt.figure(figsize=(11.2, 8.5), layout="constrained")
for index, (title, mesh) in enumerate(zip(("Relaxed response difference", "Unrelaxed CIS difference"), surfaces)):
    axis = fig.add_subplot(2, 2, index+1, projection="3d")
    draw_surfaces(axis, mesh, coordinates, native["symbols"], limits)
    axis.set_title(title, fontsize=12)
fig.legend(handles=[Patch(color=POSITIVE, label="Electron accumulation"),
                    Patch(color=NEGATIVE, label="Electron depletion")],
           loc="upper center", bbox_to_anchor=(.5, .905), ncol=2, frameon=False)

axis = fig.add_subplot(2, 2, 3)
labels = ["RHF ground", "CIS unrelaxed", "CIS relaxed"]
dipoles = [native["ground_dipole_au"][2], native["excited_unrelaxed_dipole_au"][2],
           native["excited_relaxed_dipole_au"][2]]
independent = [reference["ground_dipole_au"][2], reference["excited_unrelaxed_dipole_au"][2],
               reference["field_finite_differences"][-1]["dipoles_au"][1][2]]
axis.bar(range(3), dipoles, color=["#777777", "#56B4E9", "#0072B2"], width=.6)
axis.scatter(range(3), independent, marker="x", s=65, color="black", label="Independent PySCF", zorder=3)
for index, value in enumerate(dipoles):
    axis.text(index, value-.035, f"{value:.6f}", ha="center", va="top", fontsize=10)
axis.axhline(0., color=".3", linewidth=1)
axis.set(xticks=range(3), xticklabels=labels, ylabel=r"Permanent $\mu_z$ / (e Bohr)",
         ylim=(min(dipoles)-.18, .05), title="C=O-axis permanent dipole")
axis.legend(frameon=False, fontsize=9, loc="lower right")

axis = fig.add_subplot(2, 2, 4)
for checks, label, color, marker in ((native["field_finite_differences"], "Native energy FD", "#0072B2", "o"),
                                     (reference["field_finite_differences"], "Independent PySCF FD", "#D55E00", "s")):
    axis.loglog([item["step_au"] for item in checks], [item["max_error_au"] for item in checks],
                marker+"-", color=color, label=label)
axis.axhline(2e-6, color=".5", linestyle="--", linewidth=1, label="Acceptance gate")
axis.set(xlabel="Symmetric field step / au", ylabel="Max dipole difference / (e Bohr)",
         title="Field derivatives of ground and excited energies")
axis.legend(frameon=False, fontsize=9)
fig.suptitle("Formaldehyde S1 · RHF/CIS · Cartesian 6-31+G*\n"
             "Fixed geometry; density differences use the same ±0.003 e/Bohr³ surfaces", fontsize=13)
for suffix in ("png", "pdf"):
    fig.savefig(output/f"density_dipole.{suffix}", dpi=220)
plt.close(fig)
cell_volume = float(np.prod([axis[1]-axis[0] for axis in axes]))
diagnostics = dict(
    grid_shape=list(shape), grid_bounds_bohr=[[float(a[0]), float(a[-1])] for a in axes],
    grid_spacing_bohr=[float(a[1]-a[0]) for a in axes], isovalue_electrons_per_bohr3=level,
    boundary_maxima={name: boundary_maximum(value) for name, value in fields.items()},
    finite_box_uniform_grid_integrals={name: float(value.sum()*cell_volume) for name, value in fields.items()},
    integral_scope="Visualization-grid diagnostics only; electron-count gates use exact AO metric traces",
    surface_triangle_counts=[sum(len(faces) for _, faces, _ in mesh) for mesh in surfaces],
    common_display_limits_angstrom=[list(pair) for pair in limits],
    native_sha256=native_hash, reference_sha256=hashlib.sha256((output/"pyscf.json").read_bytes()).hexdigest(),
    elapsed_seconds=perf_counter()-started, hostname=platform.node(),
    package_versions={name: importlib.metadata.version(name) for name in ("numpy", "matplotlib", "scikit-image")},
    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    volume_helper_sha256=hashlib.sha256(Path(__file__).with_name("volume_views.py").read_bytes()).hexdigest(),
)
(output/"field_diagnostics.json").write_text(json.dumps(diagnostics, indent=2)+"\n")
print("Saved density_dipole.png/.pdf and field_diagnostics.json", flush=True)
