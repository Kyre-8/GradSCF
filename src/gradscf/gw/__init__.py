"""Differentiable GW (Hedin) module for molecular and periodic systems.

Molecular restricted/unrestricted G0W0 uses contour-deformation (CD)
self-energies and implicit/unrolled quasiparticle differentiation. Restricted
GW additionally exposes independent G/W windows and fixed-W evGW0. evGW
and real restricted qsGW provide eager self-consistency loops with opt-in
molecular outer implicit response. Periodic Gamma/k-point drivers are
available with the staged AD coverage documented in their modules.
Restricted scGW provides an eager full-matrix finite-temperature Matsubara
loop with explicit beta and grid controls and opt-in implicit differentiation
of the coupled matrix/charge equations. Analytic continuation is not implemented.
See SCGW.md in this source directory for validation and response-conditioning limits.

References
----------
- L. Hedin, "New Method for Calculating the One-Particle Green's Function
  with Application to the Electron-Gas Problem", Phys. Rev. 139, A796
  (1965). DOI:10.1103/PhysRev.139.A796
- M. S. Hybertsen and S. G. Louie, Phys. Rev. B 34, 5390 (1986).
  DOI:10.1103/PhysRevB.34.5390 (first-principles GW for solids)
- X. Ren et al., New J. Phys. 14, 053020 (2012).
  DOI:10.1088/1367-2630/14/5/053020 (RI/DF formulation of G0W0)
- R. W. Godby and R. J. Needs, Phys. Rev. Lett. 62, 1169 (1989).
  DOI:10.1103/PhysRevLett.62.1169 (contour deformation)
- T. Zhu and G. K.-L. Chan, J. Chem. Theory Comput. 14, 4856 (2018)
  (all-electron GW-CD; conventions match PySCF ``gw_cd``)
- S. V. Faleev, M. van Schilfgaarde and T. Kotani, Phys. Rev. Lett. 93,
  126406 (2004). DOI:10.1103/PhysRevLett.93.126406 (qsGW)
- M. van Schilfgaarde, T. Kotani and S. V. Faleev, Phys. Rev. B 74, 245125
  (2006). DOI:10.1103/PhysRevB.74.245125 (qsGW)
- V. M. Galitskii and A. B. Migdal, Sov. Phys. JETP 7, 96 (1958);
  B. Holm and U. von Barth, Phys. Rev. B 57, 2108 (1998).
  DOI:10.1103/PhysRevB.57.2108 (scGW)
- J. Deslippe et al., Comput. Phys. Commun. 183, 1269 (2012).
  DOI:10.1016/j.cpc.2011.12.023 (periodic GW / BerkeleyGW)
"""

from __future__ import annotations

from .evgw import evgw_cd_restricted, evgw_cd_unrestricted
from .freq import scaled_legendre_grid
from .g0w0 import g0w0_cd_restricted, g0w0_cd_unrestricted
from .polarizability import rho_response_iw, rho_response_real
from .qp import (
    solve_qp_orbital,
)
from .qsgw import qsgw_cd_restricted
from .rgw import GW
from .scgw import SCGWResult, scgw_cd_restricted, scgw_matsubara_restricted
from .screened import screened_w_imag_axis, StaticScreening, build_static_screening, apply_static_screening, solve_static_screening
from .self_energy import (
    sigma_cd,
)
from .types import GWResult
from .ugw import UGW

__all__ = [
    'GW',
    'UGW',
    'GWResult',
    'g0w0_cd_restricted',
    'g0w0_cd_unrestricted',
    'evgw_cd_restricted',
    'evgw_cd_unrestricted',
    'qsgw_cd_restricted',
    'scgw_cd_restricted',
    'scgw_matsubara_restricted',
    'SCGWResult',
    'scaled_legendre_grid',
    'rho_response_iw',
    'rho_response_real',
    'screened_w_imag_axis',
    'StaticScreening',
    'build_static_screening',
    'apply_static_screening',
    'solve_static_screening',
    'sigma_cd',
    'solve_qp_orbital',
]
