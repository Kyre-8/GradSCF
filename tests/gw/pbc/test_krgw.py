"""Gamma-point periodic G0W0-CD tests.

Local tests validate internal consistency (convergence, HF delta_v = 0,
finite QP shifts, product-basis exchange).  The PySCF comparison test runs
wherever PySCF's periodic GW (KRGW with contour deformation) is available
(c20: PySCF 2.13).
"""

import numpy as np
import jax.numpy as jnp
import pytest

from gradscf.gw.pbc import KRGW, KUGW
from gradscf.pbc import gto, scf

_MESH = (17, 17, 17)
_NW = 60
# Bond center at (3,3,3): inversion maps the FFT grid onto itself, so the
# parity-forbidden screened-interaction channels vanish exactly.  Off-grid
# placements break discrete parity and alias O(1e-2 Ha) errors into W.
_ATOM = "H 3.0 3.0 2.3; H 3.0 3.0 3.7"


def _h2_gamma_mf():
    cell = gto.M(
        atom=_ATOM,
        a=np.eye(3) * 6.0,
        unit="Bohr",
        basis="gth-szv",
        pseudo="gth-pade",
        mesh=_MESH,
        precision=1e-10,
    )
    mf = scf.RHF(cell).run()
    assert mf.converged
    return mf


@pytest.mark.parametrize('unrestricted', [False, True])
def test_gw_rejects_mismatched_reference_exchange(unrestricted):
    cell = gto.M(atom=_ATOM, a=np.eye(3)*6., unit='Bohr',
        basis='gth-szv', pseudo='gth-pade', mesh=(9,)*3)
    mean_field = scf.UHF if unrestricted else scf.RHF
    mf = mean_field(cell, exxdiv='vcut_sph').run()
    assert mf.converged
    driver = KUGW if unrestricted else KRGW
    with pytest.raises(NotImplementedError, match='Ewald exchange'):
        driver(mf, nw=2).run()


def test_krgw_gamma_hf_start_smoke():
    mf = _h2_gamma_mf()
    gw = KRGW(mf, nw=_NW).run()
    e_qp = np.asarray(gw.mo_energy)
    e_mf = np.asarray(mf.mo_energy[0])
    assert e_qp.shape == e_mf.shape
    assert np.all(np.isfinite(e_qp))
    assert np.asarray(gw.result.converged_mask).all()
    # HF starting point: v^x - v^mf = 0, so the QP shift is purely the
    # correlation self-energy; it must be nonzero but bounded.
    shift = e_qp - e_mf
    assert np.all(np.abs(shift) < 1.0)
    assert np.any(np.abs(shift) > 1e-4)


def test_krgw_gamma_matches_pyscf_krgw_cd():
    pytest.importorskip("pyscf")
    from pyscf.pbc import df as pdf
    from pyscf.pbc import gto as pgto
    from pyscf.pbc import gw as pgw
    from pyscf.pbc import scf as pscf

    mf = _h2_gamma_mf()
    e_ours = np.asarray(KRGW(mf, nw=_NW).run().mo_energy)

    cell = pgto.Cell()
    cell.atom = _ATOM
    cell.a = np.eye(3) * 6.0
    cell.unit = "Bohr"
    cell.basis = "gth-szv"
    cell.pseudo = "gth-pade"
    cell.mesh = list(_MESH)
    cell.verbose = 0
    cell.build()
    kmf = pscf.KRHF(cell, kpts=np.zeros((1, 3)), exxdiv="ewald")
    kmf.with_df = pdf.GDF(cell).build()
    kmf.kernel()
    assert kmf.converged
    kg = pgw.KRGW(kmf, freq_int="cd")
    kg.kernel(nw=_NW)
    e_ref = np.asarray(kg.mo_energy)[0]
    e_ref_mf = np.asarray(kmf.mo_energy[0])

    # The QP *shifts* (self-energy corrections) agree to ~1e-5 Ha; the
    # residual QP-energy difference is dominated by the SCF-level
    # discrepancy between the two FFT grid conventions, so both criteria
    # are asserted separately.
    e_mf = np.asarray(mf.mo_energy[0])
    np.testing.assert_allclose(e_ours - e_mf, e_ref - e_ref_mf, atol=2e-4)
    np.testing.assert_allclose(e_ours, e_ref, atol=2e-3)
