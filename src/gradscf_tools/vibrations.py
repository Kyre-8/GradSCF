"""Molecular harmonic analysis of a supplied, relaxed Cartesian Hessian.

This host-side analysis does not construct force constants or differentiate
individual mode eigenvectors. Electronic response belongs to the SCF/DFPT
modules. Inspect raw rigid-motion residuals before interpreting projected modes.
"""
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from scipy.constants import physical_constants

from gradscf.solvers import EigenSolverConfig, solve_hermitian


AMU_TO_ELECTRON_MASS = (physical_constants["atomic mass constant"][0]
                        / physical_constants["electron mass"][0])
HARTREE_TO_WAVENUMBER = physical_constants["hartree-inverse meter relationship"][0] / 100


@dataclass(frozen=True)
class NormalModes:
    """Columns of ``displacements`` are dR/dQ in electron-mass atomic units.

    ``displacements`` has shape (natom, 3, nmode) and obeys V.T M V = I,
    with M in electron masses. Q has units bohr*sqrt(electron mass).
    Negative displayed wavenumbers denote imaginary harmonic frequencies.
    ``rigid_residual`` is max(abs(D @ rigid_basis)) before projection, with
    D the mass-weighted Hessian. Rotation null vectors require a stationary
    geometry; callers must also verify the nuclear gradient there.
    """
    frequencies_cm1: np.ndarray
    eigenvalues: np.ndarray
    displacements: np.ndarray
    rigid_rank: int
    rigid_residual: float
    hessian_asymmetry: float


def normal_modes(hessian, coordinates, masses_amu, *, symmetry_tolerance=2e-6):
    """Project translations/rotations from a real molecular Hessian.

    Supply coordinates in bohr, Hessian in Hartree/bohr squared and isotope
    masses in atomic mass units. Enable JAX float64 before calling this function.
    Linear molecules have five rigid motions;
    nonlinear molecules have six. No frequency floor or acoustic correction
    is applied. This finite-system projection is not a periodic phonon model.
    """
    if not jax.config.x64_enabled:
        raise ValueError("normal_modes requires JAX float64 enabled.")
    arrays = [np.asarray(x) for x in (hessian, coordinates, masses_amu)]
    if any(np.iscomplexobj(x) or not np.isfinite(x).all() for x in arrays):
        raise ValueError("Hessian, coordinates and masses must be real and finite.")
    hessian, coordinates, masses = [x.astype(float) for x in arrays]
    if (coordinates.ndim != 2 or coordinates.shape[1] != 3
            or len(coordinates) < 2 or masses.shape != (len(coordinates),)):
        raise ValueError("Expected (natom,3) coordinates and (natom,) masses; natom >= 2.")
    if np.any(masses <= 0) or not np.isfinite(symmetry_tolerance) or symmetry_tolerance <= 0:
        raise ValueError("Masses and symmetry_tolerance must be positive.")
    size = coordinates.size
    if hessian.shape != (size, size):
        raise ValueError("Hessian must have shape (3*natom, 3*natom).")
    asymmetry = float(np.max(np.abs(hessian - hessian.T)))
    if asymmetry > symmetry_tolerance:
        raise ValueError(f"Raw Hessian asymmetry {asymmetry:.3e} exceeds tolerance.")
    mass = np.repeat(masses * AMU_TO_ELECTRON_MASS, 3)
    sqrt_mass = np.sqrt(mass).reshape(-1, 3)
    centered = coordinates - np.average(coordinates, axis=0, weights=masses)
    rigid = []
    for axis in np.eye(3):
        rigid.append((np.broadcast_to(axis, coordinates.shape) * sqrt_mass).ravel())
        rigid.append((np.cross(centered, axis) * sqrt_mass).ravel())
    frame, singular, _ = np.linalg.svd(np.asarray(rigid).T, full_matrices=True)
    rank = int(np.count_nonzero(singular > singular[0] * 1e-10))
    if rank not in (5, 6) or rank >= size:
        raise ValueError("Geometry does not define a linear or nonlinear molecular vibration space.")
    raw_dynamic = hessian / np.sqrt(mass[:, None] * mass[None, :])
    residual = float(np.max(np.abs(raw_dynamic @ frame[:, :rank])))
    dynamic = .5 * (raw_dynamic + raw_dynamic.T)
    internal = frame[:, rank:]
    reduced = internal.T @ dynamic @ internal
    # Use the shared eigensolver. Raw roots are appropriate for host-side
    # mode reporting, including degenerate modes with no selected AD gauge.
    solved = solve_hermitian(jnp.asarray(.5 * (reduced + reduced.T)),
                            config=EigenSolverConfig(method="dense", nroots=size-rank,
                                                     atol=1e-10))
    if not bool(jnp.all(solved.converged)):
        raise RuntimeError("The mass-weighted Hessian eigensolve did not converge.")
    values = np.asarray(solved.raw_values)[:size-rank]
    vectors = internal @ np.asarray(solved.raw_vectors)[:, :size-rank]
    displacements = (vectors / np.sqrt(mass)[:, None]).reshape(len(masses), 3, -1)
    frequencies = np.sign(values) * np.sqrt(np.abs(values)) * HARTREE_TO_WAVENUMBER
    return NormalModes(frequencies, values, displacements, rank, residual, asymmetry)
