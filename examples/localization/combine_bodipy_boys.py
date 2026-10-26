"""Combine the seven saved BODIPY movies into a synchronized 7-by-7 GIF.

Run animate_bodipy_boys.py first. This only crops and assembles existing frames;
it does not rerun SCF, localization or isosurface rendering. Requires Pillow,
NumPy and Matplotlib (for its bundled fonts). There is no CLI.
"""

from contextlib import ExitStack
from pathlib import Path

import matplotlib
import numpy as np
from PIL import Image, ImageDraw, ImageFont

root = Path(__file__).resolve().parents[2]
output = root / "artifacts" / "bodipy_boys"
with np.load(output / "trajectory.npz") as archive:
    stages = archive["stages"]
nframes = len(stages)
font_dir = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
title_font = ImageFont.truetype(str(font_dir / "DejaVuSans-Bold.ttf"), 40)
label_font = ImageFont.truetype(str(font_dir / "DejaVuSans.ttf"), 22)
step_font = ImageFont.truetype(str(font_dir / "DejaVuSans-Bold.ttf"), 23)
background, foreground = "#f8fafc", "#233d51"
margin, cell_width, cell_height, header = 20, 340, 224, 145
width = 2 * margin + 7 * cell_width
footer_size = (2240, 272)
footer_y = header + 7 * cell_height
height = footer_y + footer_size[1] + margin

# These crops follow the fixed 1400-by-850 layout in animate_bodipy_boys.py.
# The 3D axes on page seven are 2.02 times taller than on the other pages.
# Map the crop about the axes center, so the molecular scale stays unchanged.
panel_width = 1400 * (.985 - .015) / (4 + 3 * .01)
panel_height = 850 * (.84 - .20) / (2 + .02)
crops = []
for orbital in range(48):
    row, column = divmod(orbital % 8, 4)
    center_x = 21 + panel_width * (column * 1.01 + .5)
    top = round(180 + row * panel_height * 1.02)
    left = round(center_x - 165)
    crops.append((left, top, left + 330, top + 180))
scale = 2.02
center_y = 136 + panel_height / 2
crops.append(tuple(round(v) for v in (
    700 - 165 * scale, 408 + (180 - center_y) * scale,
    700 + 165 * scale, 408 + (360 - center_y) * scale,
)))

frames, durations = [], []
with ExitStack() as stack:
    pages = [stack.enter_context(Image.open(output / (
        f"bodipy_boys_{start:02d}-{min(start + 7, 49):02d}.gif")))
        for start in range(1, 50, 8)]
    assert all(page.size == (1400, 850) and page.n_frames == nframes
               for page in pages), "Source movies must share the saved layout and timeline"
    for frame in range(nframes):
        for page in pages:
            page.seek(frame)
        duration = pages[0].info["duration"]
        assert all(page.info["duration"] == duration for page in pages)
        durations.append(duration)
        sources = [page.convert("RGB") for page in pages]
        canvas = Image.new("RGB", (width, height), background)
        draw = ImageDraw.Draw(canvas)
        draw.text((40, 18), "BODIPY  /  ALL 49 OCCUPIED ORBITALS", font=title_font,
                  fill="#192b3a")
        draw.text((40, 72), "GradSCF RHF to Boys  |  Cartesian cc-pVDZ  |  Fixed UFF geometry",
                  font=label_font, fill="#526576")
        draw.text((40, 109), f"Step {frame:03d} / {nframes - 1:03d}  |  {stages[frame]}",
                  font=step_font, fill=foreground)
        for x, color, label in ((1810, "#cf7639", "Positive phase"),
                                (2100, "#3677a9", "Negative phase")):
            draw.rectangle((x, 80, x + 25, 96), fill=color)
            draw.text((x + 36, 73), label, font=label_font, fill=foreground)
        for orbital, crop in enumerate(crops):
            row, column = divmod(orbital, 7)
            x, y = margin + column * cell_width, header + row * cell_height
            tile = sources[orbital // 8].crop(crop)
            if orbital == 48:
                tile = tile.resize((330, 180), Image.Resampling.LANCZOS)
            canvas.paste(tile, (x + 5, y + 30))
            draw.text((x + cell_width / 2, y + 3), f"Orbital {orbital + 1}",
                      anchor="mt", font=label_font, fill=foreground)
        footer = sources[0].crop((0, 680, 1400, 850))
        canvas.paste(footer.resize(footer_size, Image.Resampling.LANCZOS),
                     ((width - footer_size[0]) // 2, footer_y))
        if frame in (0, nframes - 1):
            endpoint = "initial" if frame == 0 else "final"
            canvas.save(output / f"bodipy_boys_all_{endpoint}.png")
        frames.append(canvas.quantize(colors=256) if not frames else
                      canvas.quantize(palette=frames[0], dither=Image.Dither.NONE))
        if frame % 20 == 0:
            print(f"Combined frame {frame}/{nframes - 1}", flush=True)

path = output / "bodipy_boys_all.gif"
frames[0].save(path, save_all=True, append_images=frames[1:], duration=durations,
               loop=0, optimize=True, disposal=1)
with Image.open(path) as movie:
    assert movie.n_frames == nframes and movie.size == (width, height)
    for frame, duration in enumerate(durations):
        movie.seek(frame)
        assert movie.info["duration"] == duration
print(f"Saved {path}: {nframes} frames, {width} x {height}, "
      f"{sum(durations) / 1000:.2f} seconds", flush=True)
