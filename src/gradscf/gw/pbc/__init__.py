"""Periodic (k-point) GW -- planned Stage 2.

This subpackage will provide KRGW/KUGW with contour deformation on top of
``gradscf.pbc`` mean-field objects, including the q -> 0 head/wing
corrections of the inverse dielectric matrix.

References
----------
- L. Hedin, Phys. Rev. 139, A796 (1965). DOI:10.1103/PhysRev.139.A796
- J. Deslippe et al., "BerkeleyGW: A massively parallel computer package
  for the calculation of the quasiparticle and optical properties of
  materials and nanostructures", Comput. Phys. Commun. 183, 1269 (2012).
  DOI:10.1016/j.cpc.2011.12.023
- q -> 0 treatment: Phys. Rev. B 83, 245122 (2011).
"""

from __future__ import annotations


from .product_basis import gamma_product_factors, kpoint_product_factors
from ...pbc.momentum import momentum_transfer_table, wrap_to_kmesh
from .krgw import KRGW, evgw_cd_gamma, g0w0_cd_gamma
from .kugw import KUGW, g0w0_cd_gamma_unrestricted
from .ksigma import g0w0_cd_kpoints
from .query import g0w0_bands


__all__ = [
    "KRGW",
    "KUGW",
    "g0w0_cd_gamma",
    "g0w0_cd_gamma_unrestricted",
    "g0w0_cd_kpoints",
    "g0w0_bands",
    "evgw_cd_gamma",
    "gamma_product_factors",
    "kpoint_product_factors",
    "momentum_transfer_table",
    "wrap_to_kmesh",
]
