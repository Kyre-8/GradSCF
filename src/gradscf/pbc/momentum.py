"""Shared momentum-conservation tables for periodic response and correlation.

In a periodic GW calculation the density response is evaluated at every
momentum transfer ``q = kL`` from pair transitions (ki, kj) satisfying

    -ki + kj + kL = G_reciprocal   (i.e. kj = ki - kL modulo the k mesh)

and the self-energy at kn contracts orbital m at the inverse partner
``table[km, q] = kn`` (``km = kn + q``). This module precomputes the mapping on a uniform
k mesh, following the table construction of PySCF ``pbc.gw.krgw_ac``
(``kpti_kptj`` / ``kidx``).

References
----------
- J. Deslippe et al., Comput. Phys. Commun. 183, 1269 (2012).
  DOI:10.1016/j.cpc.2011.12.023
- L. Hedin, Phys. Rev. 139, A796 (1965). DOI:10.1103/PhysRev.139.A796
"""

from __future__ import annotations

import numpy as np


def wrap_to_kmesh(kpts: np.ndarray, tol: float = 1e-8) -> np.ndarray:
    """Wrap k points into [-0.5, 0.5) in fractional (mesh) coordinates."""
    fractional = np.asarray(kpts, dtype=float)
    wrapped = fractional - np.floor(fractional + 0.5)
    wrapped[np.abs(wrapped + 0.5) < tol] = -0.5
    return wrapped


def momentum_transfer_table(kpts: np.ndarray, *, qpts=None, tol: float = 1e-6) -> np.ndarray:
    """Index table ``kj_of_ki_q[ki, q] = kj`` with ``kj = ki - q`` on the mesh.

    Parameters
    ----------
    kpts:
        ``(nk, 3)`` k points in fractional reciprocal coordinates (period one).
    tol:
        Matching tolerance in fractional units.

    Returns
    -------
    Integer array of shape ``(nk, nq)``; indices satisfy ki-q=kj modulo a
    reciprocal vector. qpts defaults to kpts for compatibility with Gamma-
    centered GW meshes; shifted k meshes need a separate unshifted q mesh.
    """
    kpts = np.asarray(kpts, dtype=float)
    qpts = kpts if qpts is None else np.asarray(qpts,dtype=float)
    for values in (kpts,qpts):
        if values.ndim!=2 or values.shape[1]!=3 or not len(values) or not np.isfinite(values).all():
            raise ValueError('Momentum tables require finite fractional (n,3) coordinates')
    nk = kpts.shape[0]
    wrapped = wrap_to_kmesh(kpts, tol=tol)
    transfers = wrap_to_kmesh(qpts, tol=tol)
    table = np.full((nk, len(transfers)), -1, dtype=int)
    for ki in range(nk):
        for q in range(len(transfers)):
            target = wrap_to_kmesh(wrapped[ki] - transfers[q], tol=tol)
            match = np.where(np.all(np.abs(wrap_to_kmesh(wrapped - target)) < tol, axis=1))[0]
            if match.size != 1:
                raise ValueError(
                    f"k mesh is not closed under momentum transfer: "
                    f"kpts[{ki}] - kpts[{q}] matches {match.size} points."
                )
            table[ki, q] = int(match[0])
    return table


__all__ = ["wrap_to_kmesh", "momentum_transfer_table"]
