"""Render every accepted BODIPY Boys step from a saved GradSCF trajectory.

Run examples/localization/bodipy_boys.py first. This script does not rerun SCF or optimize
orbitals. It needs matplotlib >= 3.7, pillow and a NumPy-compatible scikit-image.
All 49 occupied orbitals are shown on separate pages, with fixed phase colors,
view and isovalue. There is no CLI and no PySCF dependency.
"""

from pathlib import Path
from tempfile import TemporaryDirectory

import jax
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

jax.config.update("jax_enable_x64", True)
root = Path(__file__).resolve().parents[2]
output = root / "artifacts" / "bodipy_boys"
with np.load(output / "trajectory.npz") as archive:
    data = dict(archive)
c_occ, rotations = data["c_occ"], data["rotations"]
stages = data["stages"]
nframes, norb = len(rotations), c_occ.shape[1]
assert norb == 49 and len(stages) == nframes
np.testing.assert_allclose(c_occ @ rotations[-1], data["c_boys"], atol=1e-11)

# The cached orbitals and fixed geometry fully specify the rendering inputs.
bohr_to_angstrom = 0.529177210903
coords_bohr = data["coords_bohr"]
coords = coords_bohr * bohr_to_angstrom
basis = integrals.basis_from_spec(str(data["atom"]), basis=str(data["basis"]),
                                 unit="Angstrom", precompute_eri_groups=False)
lower, upper = coords_bohr.min(axis=0) - 3.5, coords_bohr.max(axis=0) + 3.5
axes = [np.linspace(lo, hi, int(np.ceil((hi - lo) / .24)) + 1)
        for lo, hi in zip(lower, upper)]
shape = tuple(len(axis) for axis in axes)
spacing = tuple(axis[1] - axis[0] for axis in axes)
points = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
occupied = np.empty((len(points), norb))
# Keep only occupied MO grid values, not the full AO grid, in memory.
for start in range(0, len(points), 2048):
    block = jnp.asarray(points[start:start + 2048])
    ao = integrals.evaluate_cartesian_ao(basis, block)
    occupied[start:start + len(block)] = np.asarray(ao @ jnp.asarray(c_occ))
isovalue = .10  # Bohr**(-3/2), unchanged between orbitals and frames
envelope = np.linalg.norm(occupied, axis=1).reshape(shape)
edge = max(envelope[0].max(), envelope[-1].max(), envelope[:, 0].max(),
           envelope[:, -1].max(), envelope[:, :, 0].max(), envelope[:, :, -1].max())
assert edge < isovalue, "Enlarge the grid before rendering"

bonds = [(0, 1), (1, 2), (1, 3), (3, 4), (4, 5), (5, 6), (6, 7),
         (7, 8), (8, 9), (9, 10), (10, 11), (11, 12), (12, 13),
         (13, 1), (7, 3), (13, 9), (4, 14), (5, 15), (6, 16),
         (8, 17), (10, 18), (11, 19), (12, 20)]
atom_colors = {"C": "#364652", "H": "#e5e9ed", "N": "#41658b",
               "B": "#b58b72", "F": "#529887"}
colors = [atom_colors[s] for s in data["symbols"]]
p, q = np.triu_indices(norb, 1)
scores, grad_norms = [], []
for u in rotations:
    d = np.einsum("pi,xpq,qj->xij", u, data["dipole_mo"], u, optimize=True)
    centers = np.diagonal(d, axis1=1, axis2=2)
    scores.append(np.sum(centers**2))
    # Local-chart analytic gradient, checked against AD in the calculation.
    g = 4 * np.sum((centers[:, p] - centers[:, q]) * d[:, p, q], axis=0)
    grad_norms.append(np.linalg.norm(g))

positive, negative = "#cf7639", "#3677a9"
light = LightSource(azdeg=315, altdeg=40)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
pages = [list(range(start, min(start + 8, norb))) for start in range(0, norb, 8)]
figures = []
for indices in pages:
    fig = plt.figure(figsize=(14, 8.5), dpi=100, facecolor="#f8fafc")
    rows, columns = (2, 4) if len(indices) > 4 else (1, len(indices))
    grid = fig.add_gridspec(rows, columns, left=.015, right=.985,
                           bottom=.20, top=.84, wspace=.01, hspace=.02)
    panels = [fig.add_subplot(grid[i // columns, i % columns], projection="3d")
              for i in range(len(indices))]
    fig.text(.04, .95, "BODIPY  /  HF TO BOYS", fontsize=23, weight="bold", color="#192b3a")
    fig.text(.04, .91, f"Orbitals {indices[0] + 1}–{indices[-1] + 1} of 49  |  Cartesian cc-pVDZ  |  fixed UFF geometry",
             color="#526576")
    fig.legend(handles=[Patch(color=positive, label="positive phase"),
                        Patch(color=negative, label="negative phase")],
               loc="upper right", bbox_to_anchor=(.98, .975), frameon=False, ncol=2)
    step_text = fig.text(.5, .865, "", ha="center", weight="bold", color="#233d51")
    graph = fig.add_axes([.08, .075, .65, .08], facecolor="#f8fafc")
    graph.plot(range(nframes), scores, color="#d0d9e1", lw=2)
    line, = graph.plot([], [], color=negative, lw=2)
    dot, = graph.plot([], [], "o", color=positive, markersize=5)
    graph.set(xlim=(-1, nframes), ylim=(min(scores) - 20, max(scores) + 20),
              xlabel="Accepted optimizer step", ylabel=r"Boys score / bohr$^2$")
    graph.spines[["top", "right"]].set_visible(False)
    metrics = fig.text(.78, .115, "", va="center", color="#233d51")
    fig.text(.5, .18, r"Fixed view and orbital labels. Isosurfaces: $\psi = \pm 0.10$ bohr$^{-3/2}$.",
             ha="center", color="#526576")
    figures.append((fig, panels, step_text, line, dot, metrics))

# Temporary PNGs bound memory while preserving every accepted iteration.
# Only GIFs, endpoint previews and the numerical trajectory are retained.
with TemporaryDirectory(prefix="bodipy-boys-frames-") as temporary:
    temporary = Path(temporary)
    for frame, u in enumerate(rotations):
        fields = (occupied @ u).reshape(shape + (norb,))
        for page, (indices, artists) in enumerate(zip(pages, figures)):
            fig, panels, step_text, line, dot, metrics = artists
            for orbital, panel in zip(indices, panels):
                panel.clear()
                panel.set_facecolor("#f8fafc")
                panel.set_axis_off()
                panel.set_proj_type("ortho")
                panel.view_init(elev=30, azim=-75)
                panel.set_box_aspect(upper - lower, zoom=1.5)
                panel.set(xlim=np.array([lower[0], upper[0]]) * bohr_to_angstrom,
                          ylim=np.array([lower[1], upper[1]]) * bohr_to_angstrom,
                          zlim=np.array([lower[2], upper[2]]) * bohr_to_angstrom)
                panel.set_title(f"Orbital {orbital + 1}", y=.93, color="#233d51")
                for a, b in bonds:
                    panel.plot(*coords[[a, b]].T, color="#697984", linewidth=1.3)
                panel.scatter(*coords.T, c=colors, s=15, edgecolor="#697984",
                              linewidth=.3, depthshade=False)
                for sign, color in ((1., positive), (-1., negative)):
                    volume = sign * fields[..., orbital]
                    if volume.max() <= isovalue:
                        continue
                    vertices, faces, _, _ = marching_cubes(volume, isovalue,
                        spacing=spacing, allow_degenerate=False)
                    vertices = (vertices + lower) * bohr_to_angstrom
                    surface = Poly3DCollection(vertices[faces], facecolors=color,
                        linewidths=0, alpha=.72, shade=True, lightsource=light)
                    panel.add_collection3d(surface)
            step_text.set_text(f"Step {frame:03d} / {nframes - 1:03d}   |   {stages[frame]}")
            line.set_data(range(frame + 1), scores[:frame + 1])
            dot.set_data([frame], [scores[frame]])
            metrics.set_text(f"B = {scores[frame]:.6f}\nLocal |g| = {grad_norms[frame]:.2e}")
            fig.canvas.draw()
            image = Image.fromarray(np.asarray(fig.canvas.buffer_rgba()).copy()).convert("RGB")
            image.save(temporary / f"page_{page:02d}_step_{frame:04d}.png")
            if frame in (0, nframes - 1):
                endpoint = "initial" if frame == 0 else "final"
                image.save(output / f"page_{page + 1:02d}_{endpoint}.png")
        print(f"Rendered step {frame}/{nframes - 1}, all {len(pages)} pages", flush=True)

    for page, indices in enumerate(pages):
        images = []
        for frame in range(nframes):
            with Image.open(temporary / f"page_{page:02d}_step_{frame:04d}.png") as image:
                rgb = image.convert("RGB")
                images.append(rgb.quantize(colors=256) if not images else
                              rgb.quantize(palette=images[0], dither=Image.Dither.NONE))
        durations = [100] * nframes
        durations[0], durations[-1] = 800, 1600
        path = output / f"bodipy_boys_{indices[0] + 1:02d}-{indices[-1] + 1:02d}.gif"
        images[0].save(path, save_all=True, append_images=images[1:], duration=durations,
                       loop=0, optimize=True, disposal=1)
        with Image.open(path) as gif:
            assert gif.n_frames == nframes
        print(f"Saved {path}", flush=True)
        del images
plt.close("all")
