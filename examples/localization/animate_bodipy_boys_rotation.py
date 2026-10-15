"""Animate the saved occupied-orbital rotation U in C_current = C_HF @ U.

Run bodipy_boys.py and combine_bodipy_boys.py first. Outputs a matrix movie and
a synchronized orbital/matrix overview. Only NumPy, Matplotlib and Pillow are
needed; no SCF, optimization, orbital relabeling or CLI is involved.
"""

from pathlib import Path
from tempfile import TemporaryDirectory

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
from PIL import Image

root = Path(__file__).resolve().parents[2]
output = root / "artifacts" / "bodipy_boys"
with np.load(output / "trajectory.npz") as archive:
    rotations, stages = archive["rotations"], archive["stages"]
    c_occ, c_boys = archive["c_occ"], archive["c_boys"]
nframes, norb, _ = rotations.shape
assert rotations.shape == (len(stages), 49, 49)
np.testing.assert_allclose(c_occ @ rotations[-1], c_boys, atol=1e-11, rtol=0)
orth_errors = np.linalg.norm(
    rotations.transpose(0, 2, 1) @ rotations - np.eye(norb), axis=(1, 2))
assert np.max(orth_errors) < 1e-11
increments = np.linalg.norm(np.diff(rotations, axis=0), axis=(1, 2))

# A fixed diverging scale preserves the meaning of each color across frames.
background, foreground = "#f8fafc", "#233d51"
cmap = LinearSegmentedColormap.from_list("orbital_coefficients",
                                         ["#3677a9", background, "#cf7639"])
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 13,
                     "text.color": foreground, "axes.labelcolor": foreground,
                     "xtick.color": foreground, "ytick.color": foreground})
fig = plt.figure(figsize=(12, 13), dpi=100, facecolor=background)
fig.text(.07, .96, "BODIPY  /  ORBITAL ROTATION", fontsize=24, weight="bold")
fig.text(.07, .923, r"$C_{\mathrm{current}} = C_{\mathrm{HF}}\,U$", fontsize=21)
step_text = fig.text(.07, .885, "", fontsize=15, weight="bold")
ax = fig.add_axes([.10, .27, .76, .58], facecolor=background)
matrix = ax.imshow(rotations[0], cmap=cmap, vmin=-1, vmax=1,
                    interpolation="nearest", origin="upper")
ticks = np.array([1, 7, 14, 21, 28, 35, 42, 49])
ax.set(xticks=ticks - 1, xticklabels=ticks, yticks=ticks - 1, yticklabels=ticks,
       xlabel="Current orbital j (column)", ylabel="Original HF orbital i (row)")
colorbar = fig.colorbar(matrix, cax=fig.add_axes([.89, .265, .022, .545]))
colorbar.set_label(r"Signed coefficient $U_{ij}$")
colorbar.set_ticks([-1, -.5, 0, .5, 1])
fig.text(.10, .195, "Each column expands one current orbital in the fixed HF basis.", fontsize=12)
fig.text(.10, .173, "Fixed labels and signs; blue = negative, orange = positive.", fontsize=12)
metrics = fig.text(.10, .115, "", fontsize=14)
fig.text(.10, .086, "Frame 0 includes the optimizer's small initial random rotation.",
         fontsize=12, color="#526576")
fig.text(.10, .062, "One frame per saved state; no interpolation or per-frame color rescaling.",
         fontsize=12, color="#526576")

# Sample the full color scale explicitly: the almost-identity first matrix
# alone would give a poor GIF palette for subsequent mixed orbitals.
ramp = (255 * cmap(np.linspace(0, 1, 192))[:, :3]).astype(np.uint8)
gray = np.repeat(np.linspace(0, 255, 64).astype(np.uint8)[:, None], 3, axis=1)
matrix_palette = Image.new("P", (1, 1))
matrix_palette.putpalette(np.concatenate([ramp, gray]).ravel().tolist())
durations = []
with TemporaryDirectory(prefix="bodipy-rotation-") as temporary:
    temporary = Path(temporary)
    with Image.open(output / "bodipy_boys_all.gif") as orbitals:
        assert orbitals.n_frames == nframes
        for frame, u in enumerate(rotations):
            orbitals.seek(frame)
            durations.append(orbitals.info["duration"])
            matrix.set_data(u)
            step_text.set_text(f"Step {frame:03d} / {nframes - 1:03d}  |  {stages[frame]}")
            change = "n/a (initial state)" if frame == 0 else f"{increments[frame - 1]:.3e}"
            metrics.set_text(f"Step change ||U[k] - U[k-1]||_F = {change}\n"
                             f"Orthogonality ||U.T U - I||_F = {orth_errors[frame]:.2e}")
            fig.canvas.draw()
            panel = Image.fromarray(np.asarray(fig.canvas.buffer_rgba()).copy()).convert("RGB")
            # Keep all 49 orbitals visible alongside the numerical rotation.
            left_size = (1815, 1504)
            combined = Image.new("RGB", (3055, 1544), background)
            combined.paste(orbitals.convert("RGB").resize(left_size, Image.Resampling.LANCZOS),
                           (20, 20))
            combined.paste(panel, (1855, 122))
            for name, image in (("rotation", panel), ("all_with_rotation", combined)):
                if frame in (0, nframes - 1):
                    endpoint = "initial" if frame == 0 else "final"
                    image.save(output / f"bodipy_boys_{name}_{endpoint}.png")
                if name == "rotation":
                    encoded = image.quantize(palette=matrix_palette, dither=Image.Dither.NONE)
                else:
                    if frame == 0:
                        # The colorbar supplies the entire matrix range here too.
                        combined_palette = image.quantize(colors=256)
                    encoded = image.quantize(palette=combined_palette, dither=Image.Dither.NONE)
                encoded.save(temporary / f"{name}_{frame:04d}.png")
            if frame % 20 == 0:
                print(f"Rendered rotation {frame}/{nframes - 1}", flush=True)

    for name in ("rotation", "all_with_rotation"):
        frames = []
        for frame in range(nframes):
            with Image.open(temporary / f"{name}_{frame:04d}.png") as image:
                frames.append(image.copy())
        path = output / f"bodipy_boys_{name}.gif"
        frames[0].save(path, save_all=True, append_images=frames[1:], duration=durations,
                       loop=0, optimize=True, disposal=1)
        with Image.open(path) as movie:
            assert movie.n_frames == nframes
            for frame, duration in enumerate(durations):
                movie.seek(frame)
                assert movie.info["duration"] == duration
        print(f"Saved {path}: {nframes} frames, {sum(durations) / 1000:.2f} s", flush=True)
        del frames
plt.close(fig)
