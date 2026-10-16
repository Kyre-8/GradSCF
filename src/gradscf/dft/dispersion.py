"""Geometry-only dispersion of the original omegaB97X-D functional.

Chai and Head-Gordon, PCCP 10, 6615 (2008), DOI: 10.1039/B810189B,
eqs. (2)-(3) and Table 1: unscaled C6 pair energy, damping exponent 12,
and damping parameter 6. This is the original CHG correction, not D3/D4.
Atomic C6 and radii: Grimme, JCC 27, 1787 (2006), Table 1,
DOI: 10.1002/jcc.20495; cross-checked against ASE's vdWDB_Grimme06jcc.
"""
from __future__ import annotations

from collections.abc import Sequence

import jax.numpy as jnp

from gradscf.data.molecule import ANGSTROM_TO_BOHR


# Original tabulated units: C6 in J nm^6 mol^-1; R0 in Angstrom.
_GRIMME_2006 = {
    "H": (0.14, 1.001),
    "C": (1.75, 1.452),
    "N": (1.23, 1.397),
    "O": (0.70, 1.342),
    "F": (0.75, 1.287),
}
# Avogadro constant (exact SI) and Hartree in joules (CODATA 2018).
_AVOGADRO = 6.02214076e23
_HARTREE_J = 4.3597447222071e-18
_C6_TO_ATOMIC = (10.0 * ANGSTROM_TO_BOHR)**6 / (_AVOGADRO * _HARTREE_J)


def wb97xd_dispersion_energy(symbols: Sequence[str], coords_bohr):
    """Return original CHG dispersion in Hartree for Cartesian bohr positions.

    ``symbols`` is static molecular metadata; H, C, N, O, F are supported.
    Real coordinates of shape ``(len(symbols), 3)`` may be JAX tracers. The
    correction depends only on geometry, so it contributes no density or
    electric-field derivative. It must be added once to the electronic plus
    nuclear energy. No molecular cutoffs or periodic images are included.
    """
    symbols = tuple(symbols)
    unknown = sorted(set(symbols) - _GRIMME_2006.keys())
    if unknown:
        raise ValueError(f"Unsupported omegaB97X-D dispersion elements: {', '.join(unknown)}")
    coordinates = jnp.asarray(coords_bohr)
    if jnp.iscomplexobj(coordinates):
        raise TypeError("coords_bohr must contain real Cartesian coordinates")
    if coordinates.shape != (len(symbols), 3):
        raise ValueError("coords_bohr must have shape (len(symbols), 3)")
    c6 = jnp.asarray([_GRIMME_2006[s][0] * _C6_TO_ATOMIC for s in symbols])
    radii = jnp.asarray([_GRIMME_2006[s][1] * ANGSTROM_TO_BOHR for s in symbols])
    i, j = jnp.triu_indices(len(symbols), k=1)
    displacement = coordinates[i] - coordinates[j]
    r6 = jnp.sum(displacement**2, axis=-1)**3
    reference_radius = radii[i] + radii[j]
    # Algebraically identical to -C6/R^6 / (1 + 6*(R/Rr)^-12),
    # with a smooth zero-distance limit and no singular intermediate norm.
    return -jnp.sum(jnp.sqrt(c6[i] * c6[j]) * r6 /
                    (r6**2 + 6.0 * reference_radius**12))


__all__ = ["wb97xd_dispersion_energy"]
