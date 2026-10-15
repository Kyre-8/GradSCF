"""Animate every accepted Boys optimization step for all eight occupied MOs.

Run after building GradSCF native integrals:
    PYTHONPATH=src JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 python examples/localization/animate_ethylene_boys.py

Requires matplotlib >= 3.7, pillow and a NumPy-compatible scikit-image.
No PySCF or CLI is used.
Frames are actual L-BFGS-B/Newton-CG iterates from the selected start, without
interpolation, per-frame orbital sorting, or per-frame amplitude rescaling.
"""

from pathlib import Path
import runpy

import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
from PIL import Image
from skimage.measure import marching_cubes

from gradscf import integrals

root = Path(__file__).resolve().parents[2]
example = runpy.run_path(str(root / "examples" / "localization" / "ethylene_boys.py"))
mol = example["mol"]
c_occ = np.asarray(example["c_occ"])
rotations = np.asarray(example["rotation_history"])
stages = example["iteration_stages"]
nframes, norb = len(rotations), c_occ.shape[1]
assert nframes == 1 + example["best"].nit + example["polished"].nit
np.testing.assert_allclose(c_occ @ rotations[-1], example["c_boys"], atol=1e-12)

# Evaluate AO values once, using GradSCF. Each frame only rotates the fixed MOs.
# All grid coordinates are Bohr; only the displayed vertices use Angstrom.
bohr_to_angstrom = 0.529177210903
axes_bohr = [np.linspace(-5.5, 5.5, 68), np.linspace(-4.5, 4.5, 56),
             np.linspace(-3.5, 3.5, 44)]
grid_shape = tuple(len(axis) for axis in axes_bohr)
points = np.stack(np.meshgrid(*axes_bohr, indexing="ij"), axis=-1).reshape(-1, 3)
basis = integrals.basis_from_spec(mol.atom, basis=mol.basis, unit=mol.unit,
                                 precompute_eri_groups=False)
ao = integrals.evaluate_cartesian_ao(basis, jnp.asarray(points), chunk_size=2048)
occupied_grid = np.asarray(ao @ jnp.asarray(c_occ))
del ao
origin = np.array([axis[0] for axis in axes_bohr])
spacing = tuple(axis[1] - axis[0] for axis in axes_bohr)
isovalue = 0.12  # orbital amplitude, Bohr**(-3/2), fixed across all frames
envelope = np.linalg.norm(occupied_grid, axis=1).reshape(grid_shape)
edge_max = max(envelope[0].max(), envelope[-1].max(), envelope[:, 0].max(),
               envelope[:, -1].max(), envelope[:, :, 0].max(), envelope[:, :, -1].max())
assert edge_max < isovalue, "Enlarge the grid to avoid clipping orbital surfaces"

coords = np.asarray(example["parameters"].nuclear_coords) * bohr_to_angstrom
bonds = [(0, 1), (0, 2), (0, 3), (1, 4), (1, 5)]
dipole = np.asarray(example["dipole_mo"])
scores, gradient_norms = [], []
for u in rotations:
    current = np.einsum("pi,xpq,qj->xij", u, dipole, u)
    scores.append(np.sum(np.diagonal(current, axis1=1, axis2=2)**2))
    grad = example["gradient"](jnp.zeros(example["nparam"]), jnp.asarray(current))
    gradient_norms.append(float(jnp.linalg.norm(grad)))
assert np.min(np.diff(scores)) > -1e-10

output = root / "artifacts" / "ethylene_boys_animation"
frames_dir = output / "frames"
frames_dir.mkdir(parents=True, exist_ok=True)
np.savez(output / "trajectory.npz", c_occ=c_occ, rotations=rotations,
         stages=np.asarray(stages), scores=scores, gradient_norms=gradient_norms,
         atom_coords_angstrom=coords, isovalue=isovalue, selected_start=example["best_index"])

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
positive, negative = "#cf7639", "#3677a9"
fig = plt.figure(figsize=(14, 8.5), dpi=110, facecolor="#f8fafc")
grid = fig.add_gridspec(2, 4, left=.015, right=.985, bottom=.20, top=.85,
                       wspace=.01, hspace=.02)
panels = [fig.add_subplot(grid[i // 4, i % 4], projection="3d") for i in range(norb)]
fig.text(.04, .957, "ETHYLENE  /  HF TO BOYS", fontsize=23, weight="bold", color="#192b3a")
fig.text(.04, .92, "8 occupied orbitals  |  Cartesian cc-pVDZ  |  fixed occupied subspace",
         fontsize=11, color="#526576")
fig.legend(handles=[Patch(color=positive, label="positive phase"),
                    Patch(color=negative, label="negative phase")],
           loc="upper right", bbox_to_anchor=(.98, .975), frameon=False, ncol=2)
iteration_text = fig.text(.5, .873, "", ha="center", weight="bold", color="#233d51")
score_axis = fig.add_axes([.075, .072, .66, .082], facecolor="#f8fafc")
score_axis.plot(range(nframes), scores, color="#d0d9e1", lw=2)
score_line, = score_axis.plot([], [], color=negative, lw=2.3)
score_dot, = score_axis.plot([], [], "o", color=positive, markersize=6)
score_axis.set(xlim=(-.5, nframes - .5), ylim=(-2, max(scores) * 1.08),
               xlabel="Accepted optimizer step", ylabel=r"Boys score / bohr$^2$")
score_axis.spines[["top", "right"]].set_visible(False)
score_axis.tick_params(labelsize=9, colors="#526576")
metrics_text = fig.text(.78, .125, "", va="center", fontsize=11, color="#233d51")
fig.text(.5, .18, r"Fixed view and orbital labels. Isosurfaces: $\psi = \pm 0.12$ bohr$^{-3/2}$.",
         ha="center", fontsize=10, color="#526576")
light = LightSource(azdeg=315, altdeg=40)

images = []
for frame, u in enumerate(rotations):
    orbitals = (occupied_grid @ u).reshape(grid_shape + (norb,))
    for orbital, panel in enumerate(panels):
        panel.clear()
        panel.set_facecolor("#f8fafc")
        panel.set_axis_off()
        panel.set_proj_type("ortho")
        panel.view_init(elev=24, azim=-65)
        panel.set_box_aspect((11, 9, 7), zoom=1.6)
        panel.set(xlim=axes_bohr[0][[0, -1]] * bohr_to_angstrom,
                  ylim=axes_bohr[1][[0, -1]] * bohr_to_angstrom,
                  zlim=axes_bohr[2][[0, -1]] * bohr_to_angstrom)
        panel.set_title(f"Orbital {orbital + 1}", y=.92, fontsize=12, color="#233d51")
        for a, b in bonds:
            panel.plot(*coords[[a, b]].T, color="#4b5963", linewidth=2.1)
        panel.scatter(*coords[:2].T, color="#364652", s=55, depthshade=False)
        panel.scatter(*coords[2:].T, color="#e5e9ed", edgecolor="#697984",
                      linewidth=.6, s=25, depthshade=False)
        for sign, color in ((1., positive), (-1., negative)):
            field = sign * orbitals[..., orbital]
            if field.max() <= isovalue:
                continue
            vertices, faces, _, _ = marching_cubes(field, isovalue, spacing=spacing,
                                                   allow_degenerate=False)
            vertices = (vertices + origin) * bohr_to_angstrom
            surface = Poly3DCollection(vertices[faces], facecolors=color, linewidths=0,
                                        alpha=.72, shade=True, lightsource=light)
            panel.add_collection3d(surface)
    stage = "Initial symmetry-breaking rotation" if frame == 0 else stages[frame]
    iteration_text.set_text(f"Step {frame:02d} / {nframes - 1:02d}   |   {stage}")
    score_line.set_data(range(frame + 1), scores[:frame + 1])
    score_dot.set_data([frame], [scores[frame]])
    metrics_text.set_text(f"B = {scores[frame]:.6f}\nLocal |g| = {gradient_norms[frame]:.2e}")
    fig.canvas.draw()
    image = Image.fromarray(np.asarray(fig.canvas.buffer_rgba()).copy()).convert("RGB")
    image.save(frames_dir / f"step_{frame:03d}.png")
    # Reuse one palette so quantization does not change the phase colors.
    if not images:
        images.append(image.quantize(colors=256))
    else:
        images.append(image.quantize(palette=images[0], dither=Image.Dither.NONE))
    print(f"Rendered frame {frame + 1}/{nframes}", flush=True)

durations = [240] * nframes
durations[0], durations[-1] = 800, 1800
gif_path = output / "ethylene_boys.gif"
images[0].save(gif_path, save_all=True, append_images=images[1:], duration=durations,
               loop=0, optimize=False, disposal=2)
plt.close(fig)
with Image.open(gif_path) as gif:
    assert gif.n_frames == nframes
print(f"Saved {gif_path} ({nframes} frames)")
