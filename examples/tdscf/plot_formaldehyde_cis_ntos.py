"""Plot only independently validated X05 NTOs and transition-density fields."""
import hashlib
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


output = Path("reproducibility/ad_showcase/X05_formaldehyde_cis_ntos")
native = json.loads((output/"gradscf.json").read_text())
reference = json.loads((output/"pyscf.json").read_text())
native_hash = hashlib.sha256((output/"gradscf.json").read_bytes()).hexdigest()
if not native["complete"] or not reference["passed"] or reference["native_sha256"] != native_hash:
    raise RuntimeError("Only current independently validated X05 results are plotted.")
started = perf_counter()
coordinates = np.asarray(native["coordinates_bohr"])
ao, axes, shape = sample_ao_grid(native["symbols"], coordinates, native["basis"])
fields, surfaces = [], []
orbital_level, transition_level = .05, .003
for state in native["states"]:
    local = [(ao@np.asarray(state["hole_ntos_ao"])[:, 0]).reshape(shape),
             (ao@np.asarray(state["particle_ntos_ao"])[:, 0]).reshape(shape),
             density_field(ao, state["transition_density_ao"], shape)]
    fields.append(local)
    surfaces.extend(extract_surfaces(field, axes, level)
                    for field, level in zip(local, (orbital_level, orbital_level, transition_level)))
del ao
limits = common_limits(coordinates, surfaces)
plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "pdf.fonttype": 42})
fig = plt.figure(figsize=(16.6, 8.4), layout="constrained")
for row, state in enumerate(native["states"]):
    for col, label in enumerate(("Leading hole NTO", "Leading particle NTO", "Transition density")):
        axis = fig.add_subplot(2, 4, row*4+col+1, projection="3d")
        draw_surfaces(axis, surfaces[row*3+col], coordinates, native["symbols"], limits)
        axis.set_title(f"{state['state_label']}: {label}", fontsize=11)
    axis = fig.add_subplot(2, 4, row*4+4)
    weights = state["weights"]
    axis.bar(np.arange(1, len(weights)+1), weights, color="#0072B2", width=.7)
    axis.set(xticks=np.arange(1, len(weights)+1), xlabel="NTO pair", ylabel="Normalized pair weight",
             ylim=(0., 1.12), title=f"{state['state_label']}: all pair weights")
    axis.text(.97, .95, f"Leading: {weights[0]:.4%}\nf = {state['oscillator_strength']:.5g}\n"
              f"Centroid separation\n{state['centroid_separation_bohr']:.4f} Bohr",
              transform=axis.transAxes, ha="right", va="top", fontsize=10)
fig.suptitle("Formaldehyde · RHF/CIS · Cartesian 6-31+G*\n"
             "S1 and the brightest of six retained roots; individual orbital/state phases are arbitrary", fontsize=13)
fig.legend(handles=[Patch(color=POSITIVE, label="Positive phase"), Patch(color=NEGATIVE, label="Negative phase")],
           loc="lower center", bbox_to_anchor=(.45, -.035), ncol=2, frameon=False)
fig.text(.45, -.045, "Orbital surfaces: ±0.05 Bohr$^{-3/2}$; transition density: ±0.003 e/Bohr³. "
         "Signs do not denote charge accumulation/depletion.", ha="center", va="top", fontsize=10)
for suffix in ("png", "pdf"):
    fig.savefig(output/f"ntos_transition_density.{suffix}", dpi=200, bbox_inches="tight")
plt.close(fig)
diagnostics = dict(
    grid_shape=list(shape), grid_bounds_bohr=[[float(a[0]), float(a[-1])] for a in axes],
    grid_spacing_bohr=[float(a[1]-a[0]) for a in axes],
    orbital_isovalue_bohr_minus_three_halves=orbital_level,
    transition_isovalue_electrons_per_bohr3=transition_level,
    boundary_maxima=[[boundary_maximum(field) for field in local] for local in fields],
    common_display_limits_angstrom=[list(pair) for pair in limits],
    scope="Leading pair is visualized; all weights and full transition matrices are retained",
    native_sha256=native_hash, reference_sha256=hashlib.sha256((output/"pyscf.json").read_bytes()).hexdigest(),
    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    volume_helper_sha256=hashlib.sha256(Path(__file__).with_name("volume_views.py").read_bytes()).hexdigest(),
    elapsed_seconds=perf_counter()-started, hostname=platform.node(),
)
(output/"field_diagnostics.json").write_text(json.dumps(diagnostics, indent=2)+"\n")
print("Saved ntos_transition_density.png/.pdf and field_diagnostics.json", flush=True)
