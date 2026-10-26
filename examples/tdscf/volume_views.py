"""Small field views using GradSCF AO sampling and scikit-image surfaces.

The sampling/mesh pattern follows the existing Boys-localization examples.
No SCF or property calculation is run here. Sampling uses Bohr; displayed
coordinates use Angstrom. Callers supply validated matrices/orbital columns.
"""
import jax
import jax.numpy as jnp
import numpy as np
from matplotlib.colors import LightSource
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage.measure import marching_cubes

from gradscf import integrals


BOHR_TO_ANGSTROM = 0.529177210903
POSITIVE, NEGATIVE = "#0072B2", "#D55E00"


def sample_ao_grid(symbols, coordinates, basis_name, *, padding=8., spacing=.25):
    if not jax.config.x64_enabled:
        raise ValueError("Enable float64 before sampling orbital fields.")
    coordinates = np.asarray(coordinates)
    lower, upper = coordinates.min(axis=0)-padding, coordinates.max(axis=0)+padding
    axes = [np.linspace(a, b, int(np.ceil((b-a)/spacing))+1) for a, b in zip(lower, upper)]
    shape = tuple(map(len, axes))
    if np.prod(shape) > 1_000_000:
        raise ValueError("This example caps its visualization grid at one million points.")
    points = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    basis = integrals.basis_from_spec(list(zip(symbols, coordinates)), basis=basis_name,
                                      unit="Bohr", precompute_eri_groups=False)
    ao = np.asarray(integrals.evaluate_cartesian_ao(basis, jnp.asarray(points), chunk_size=2048))
    if not np.isfinite(ao).all():
        raise RuntimeError("AO field sampling returned nonfinite values.")
    return ao, axes, shape


def density_field(ao, matrix, shape):
    return np.einsum("pi,pi->p", ao@np.asarray(matrix), ao).reshape(shape)


def boundary_maximum(field):
    return float(max(np.max(np.abs(face)) for face in (
        field[0], field[-1], field[:, 0], field[:, -1], field[:, :, 0], field[:, :, -1])))


def extract_surfaces(field, axes, level, *, signed=True):
    """Return existing-library triangle meshes, rejecting clipped surfaces."""
    if boundary_maximum(field) >= level:
        raise RuntimeError("Enlarge the sampling box: an isosurface reaches its boundary.")
    spacing = tuple(axis[1]-axis[0] for axis in axes)
    origin = np.array([axis[0] for axis in axes])
    surfaces = []
    for sign, color in ((1., POSITIVE), (-1., NEGATIVE)) if signed else ((1., POSITIVE),):
        values = sign*field
        if np.max(values) <= level:
            continue
        vertices, faces, _, _ = marching_cubes(values, level, spacing=spacing, allow_degenerate=False)
        surfaces.append(((vertices+origin)*BOHR_TO_ANGSTROM, faces, color))
    return surfaces


def common_limits(coordinates, groups):
    points = [np.asarray(coordinates)*BOHR_TO_ANGSTROM]
    points.extend(surface[0] for group in groups for surface in group)
    all_points = np.vstack(points)
    center = .5*(all_points.min(axis=0)+all_points.max(axis=0))
    radius = .5*np.max(np.ptp(all_points, axis=0))+.25
    return tuple((value-radius, value+radius) for value in center)


def draw_surfaces(axis, surfaces, coordinates, symbols, limits):
    # Keep the molecular skeleton readable as an explicit orientation overlay.
    axis.computed_zorder = False
    light = LightSource(azdeg=315, altdeg=40)
    for vertices, faces, color in surfaces:
        artist = Poly3DCollection(vertices[faces], facecolors=color, linewidths=0,
                                  alpha=.5, shade=True, lightsource=light, zorder=1)
        axis.add_collection3d(artist)
    r = np.asarray(coordinates)*BOHR_TO_ANGSTROM
    for a, b in ((0, 1), (0, 2), (0, 3)):
        axis.plot(*r[[a, b]].T, color=".25", linewidth=2, zorder=2)
    colors = {"C": "#333333", "O": "#C34848", "H": "#eeeeee"}
    sizes = {"C": 60, "O": 70, "H": 30}
    for coordinate, symbol in zip(r, symbols):
        axis.scatter(*coordinate, color=colors[symbol], s=sizes[symbol], edgecolor=".35", linewidth=.5, zorder=3)
    axis.set(xlim=limits[0], ylim=limits[1], zlim=limits[2])
    axis.set_box_aspect((1, 1, 1))
    axis.set_proj_type("ortho")
    axis.view_init(elev=18, azim=25)
    axis.set_axis_off()
