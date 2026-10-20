"""Build a rigid DABNA-2 dimer for CPU memory/scaling measurements.

This is a computational benchmark geometry, not an optimized aggregate or
crystal structure. Both monomers retain the public gas-phase PBEh-3c geometry.
The central B/N plane defines the stacking direction; a deterministic slip
search avoids interfragment nuclear distances below 2.5 Angstrom.
Run on the authorized compute node, before the electronic-structure examples.
"""
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time

import numpy as np


root = Path(__file__).resolve().parent
started = time.perf_counter()
source = root / "dabna2_monomer.xyz"
lines = source.read_text().splitlines()
symbols = [line.split()[0] for line in lines[2:] if line.strip()]
coordinates = np.array([[float(x) for x in line.split()[1:4]]
                        for line in lines[2:] if line.strip()], dtype=np.float64)
assert len(symbols) == int(lines[0]) == 96
center = coordinates.mean(axis=0)
core = coordinates[[i for i, symbol in enumerate(symbols) if symbol in ("B", "N")]]
_, _, axes = np.linalg.svd(core - core.mean(axis=0), full_matrices=False)
normal = axes[-1]
normal *= np.sign(normal[np.argmax(np.abs(normal))])
first = axes[0]
first *= np.sign(first[np.argmax(np.abs(first))])
second = np.cross(normal, first)
first = np.cross(second, normal)

# Search modest slips first so that a large lateral translation is not used
# merely to separate the fragments. The second stage widens both the slip
# and normal distance. These are benchmark placements, not energy minima.
searches = (
    dict(name="compact_stack", normal_min=3.5, normal_max=6.0, slip_radius=3.5),
    dict(name="expanded_stack", normal_min=3.5, normal_max=8.0, slip_radius=6.0),
)
angle_degrees = [0., 180.]
for increment in (30., 60., 90., 120., 150., 15., 45., 75., 105., 135., 165.):
    angle_degrees.extend((increment, 360. - increment))
rotations = []
for degrees in angle_degrees:
    angle = np.deg2rad(degrees)
    rotation = (np.outer(normal, normal)
                + np.cos(angle)*(np.eye(3)-np.outer(normal, normal))
                + np.sin(angle)*np.array([[0., -normal[2], normal[1]],
                                         [normal[2], 0., -normal[0]],
                                         [-normal[1], normal[0], 0.]]))
    rotations.append((degrees, rotation, (coordinates-center) @ rotation.T + center))

chosen = None
attempts = 0
minimum_distance = 2.5
seen = set()
for search in searches:
    radius = search["slip_radius"]
    offsets = ((float(x), float(y))
               for x in np.arange(-radius, radius+.01, .5)
               for y in np.arange(-radius, radius+.01, .5)
               if x*x+y*y <= radius*radius)
    offsets = tuple(offsets)
    placements = [(float(separation), x, y)
                  for separation in np.arange(search["normal_min"], search["normal_max"]+.01, .25)
                  for x, y in offsets]
    placements.sort(key=lambda p: ((p[0]-3.5)**2+p[1]**2+p[2]**2, p[0], p[1], p[2]))
    for separation, x, y in placements:
        if (separation, x, y) in seen:
            continue
        seen.add((separation, x, y))
        translation = separation*normal + x*first + y*second
        for degrees, rotation, rotated in rotations:
            other = rotated + translation
            distances = np.linalg.norm(coordinates[:, None, :] - other[None, :, :], axis=-1)
            attempts += 1
            if distances.min() >= minimum_distance+1e-8:
                chosen = other, separation, x, y, degrees, rotation, distances, search["name"]
                break
        if chosen is not None:
            break
    if chosen is not None:
        break
if chosen is None:
    # A finite fallback is guaranteed by disjoint projected coordinate ranges.
    # Label it explicitly: a distant pair is not a compact stacked aggregate.
    thickness = np.ptp(coordinates @ normal)
    separation = float(np.ceil((thickness+minimum_distance+1e-8)*4)/4)
    other = coordinates + separation*normal
    distances = np.linalg.norm(coordinates[:, None, :] - other[None, :, :], axis=-1)
    attempts += 1
    chosen = other, separation, 0., 0., 0., np.eye(3), distances, "separated_fallback"

other, separation, x, y, degrees, rotation, distances, stage = chosen
original_internal = np.linalg.norm(coordinates[:, None, :] - coordinates[None, :, :], axis=-1)
other_internal = np.linalg.norm(other[:, None, :] - other[None, :, :], axis=-1)
rigid_error = float(np.max(np.abs(other_internal-original_internal)))
if distances.min() < minimum_distance or rigid_error > 1e-10:
    raise RuntimeError("The dimer failed its distance or rigid-fragment check.")
output = root / "dabna2_dimer.xyz"
comment = (f"Rigid benchmark dimer; normal separation {separation:.2f} Angstrom; "
           f"slip ({x:.2f}, {y:.2f}); rotation {degrees:.0f} degrees; "
           f"minimum interfragment distance {distances.min():.6f} Angstrom")
all_coordinates = np.concatenate((coordinates, other))
output.write_text("192\n" + comment + "\n" + "".join(
    f"{symbol:2s} {r[0]: .12f} {r[1]: .12f} {r[2]: .12f}\n"
    for symbol, r in zip(symbols*2, all_coordinates)))
closest = np.unravel_index(np.argmin(distances), distances.shape)
record = dict(
    purpose="rigid computational benchmark; no aggregate-physics or optimized-geometry claim",
    source_geometry_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
    generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    dimer_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
    search=dict(stages=searches, normal_step_angstrom=.25, slip_step_angstrom=.5,
                angle_degrees=angle_degrees,
                placement_order="squared normal excess above 3.5 Angstrom plus squared lateral slip",
                selected_stage=stage, evaluated_candidates=attempts),
    normal_separation_angstrom=separation, slip_angstrom=[x, y], rotation_degrees=degrees,
    stacking_normal=normal.tolist(), slip_axes=[first.tolist(), second.tolist()],
    rotation_matrix=rotation.tolist(),
    minimum_interfragment_distance_angstrom=float(distances.min()),
    maximum_internal_distance_change_angstrom=rigid_error,
    closest_atom_indices=[int(closest[0]), int(closest[1])+96],
    atoms=192, electrons=776, charge=0, spin=0,
    basis="Cartesian def2-SVP", expected_nao=2120,
    expected_occupied=388, expected_virtual=1732, expected_transition_dimension=672016,
    execution=dict(host=platform.node(), machine=platform.machine(), python=sys.version.split()[0],
                   executable=sys.executable, numpy=np.__version__,
                   cpu_affinity=sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
                   cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
                   elapsed_seconds=time.perf_counter()-started))
(root / "dimer_provenance.json").write_text(json.dumps(record, indent=2) + "\n")
print(json.dumps(record, indent=2))
