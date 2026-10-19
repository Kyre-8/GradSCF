"""Restricted G0W0-CD correctness tests against PySCF ``gw_cd``.

System: H2O / def2-svp (cartesian), HF starting point, nw=100, eta=1e-3.
PySCF uses a three-center density-fitting auxiliary basis (mp2fit);
GradSCF uses spectral factorization of the full ERI (tol=1e-12). The
same-branch comparison retains a 3e-4 Ha tolerance. It exercises the public
array driver with cached resolvent screening, not the default dense facade.
Default nearest-on-shell root selection is checked separately: a PySCF
local root need not be the branch chosen by that policy.
"""

import numpy as np
import pytest

pytest.importorskip("pyscf")

from gradscf import dft, gto
from gradscf.gw import g0w0_cd_restricted
from gradscf.integrals.molecular.factorization import eri_pair_matrix_to_df_factors

_H2O_ATOM = "O 0 0 0.117790; H 0 0.755453 -0.471161; H 0 -0.755453 -0.471161"
_BASIS = "def2-svp"
_NW = 100


@pytest.fixture(scope="module")
def water_comparison():
    mol = gto.M(atom=_H2O_ATOM, basis=_BASIS, cart=True)
    mf = dft.RKS(mol, xc="hf").run()
    assert mf.converged
    nocc = int(np.count_nonzero(np.asarray(mf.mo_occ) > 0.0))
    # Only the valence region (occupied + 7 virtuals) is corrected: those
    # are the orbitals compared against PySCF, and skipping the pole-dense
    # high-lying virtuals keeps the QP batch affordable on shared hosts.
    valence = tuple(range(nocc + 7))
    inputs, reference = mf._scf_inputs, mf.scf_result
    factors = getattr(inputs, "df_factors", None)
    if factors is None:
        factors = eri_pair_matrix_to_df_factors(inputs.eri_pair_matrix,
                                                nao=reference.mo_coeff.shape[0], tol=1e-12)
    e_ref = _pyscf_qp_energies()
    arguments = dict(mo_energy=reference.mo_energy, mo_coeff=reference.mo_coeff,
        nocc=nocc, df_factors=factors, fock_matrix=reference.fock_matrix,
        hcore_matrix=reference.hcore_matrix, density_matrix=reference.density_matrix,
        nw=_NW, eta=1e-3, orbs=valence, resolvent_expansion=True)
    default = g0w0_cd_restricted(**arguments)
    # External energies are numerical continuation seeds only; the actual
    # GradSCF Dyson equations and residual checks still determine the roots.
    same_branch = g0w0_cd_restricted(**arguments, qp_initial=e_ref)
    for result in (default, same_branch):
        mask = np.asarray(result.converged_mask)
        assert mask[list(valence)].all(), f"valence QP not converged: {mask}"
        assert result.converged
    return mf, valence, e_ref, default, same_branch


def _pyscf_qp_energies():
    from pyscf import gto as pgto
    from pyscf import scf as pscf
    from pyscf.gw.gw_cd import GWCD

    mol = pgto.M(atom=_H2O_ATOM, basis=_BASIS, cart=True, verbose=0)
    # The shared remote host can report high memory usage; PySCF's DF-GW
    # ao2mo guard would then spuriously raise NotImplementedError.  Force
    # incore behavior so the comparison is environment-independent.
    mol.incore_anyway = True
    mf = pscf.RHF(mol).run()
    gw = GWCD(mf)
    gw.kernel(nw=_NW)
    assert gw.converged
    return np.asarray(gw.mo_energy)


def test_g0w0_cd_restricted_matches_pyscf_water(water_comparison):
    _, valence, e_ref, _, same_branch = water_comparison
    e_ours = np.asarray(same_branch.mo_energy)
    assert e_ours.shape == e_ref.shape
    # Occupied and low-lying virtual orbitals: DF-representation differences
    # (true 3-center DF in PySCF vs spectral factorization here) only enter
    # at the 1e-4 Ha level.
    indices = list(valence)
    np.testing.assert_allclose(e_ours[indices], e_ref[indices], atol=3e-4)
    assert np.max(abs(np.asarray(same_branch.qp_residual)[indices])) < 1e-6
    # Unrequested high virtuals retain finite MF values; this does not certify
    # their QP coverage or a global spectral-peak selection.
    assert np.all(np.isfinite(e_ours))


def test_default_root_policy_can_select_a_different_water_branch(water_comparison):
    _, valence, _, default, same_branch = water_comparison
    indices, orbital = list(valence), valence[-1]
    assert np.max(abs(np.asarray(default.qp_residual)[indices])) < 1e-6
    assert np.all(np.asarray(default.qp_branch_valid)[indices])
    assert np.all(np.asarray(same_branch.qp_branch_valid)[indices])
    assert np.all(np.asarray(default.qp_weight)[indices] > 0)
    assert np.all(np.asarray(same_branch.qp_weight)[indices] > 0)
    assert abs(float(default.mo_energy[orbital]-same_branch.mo_energy[orbital])) > 1e-2
    assert default.qp_weight[orbital] > same_branch.qp_weight[orbital]
    assert max(int(default.qp_candidate_count[orbital]),
               int(same_branch.qp_candidate_count[orbital])) >= 2


def test_g0w0_cd_restricted_hf_delta_v_vanishes(water_comparison):
    # With an HF starting point v^x - v^mf == 0 by construction, so the QP
    # shift must be purely the correlation self-energy.
    mf, valence, _, default, _ = water_comparison
    e_ours = np.asarray(default.mo_energy)
    e_mf = np.asarray(mf.mo_energy)
    # HF orbital energies obey Koopmans; correlation typically shifts HOMO
    # up and LUMO down relative to HF, but we only assert the shift is
    # nonzero and of a sensible magnitude (|Sigma_c| < 1 Ha).
    shift = e_ours - e_mf
    np.testing.assert_allclose(shift[list(valence)],
        np.asarray(default.sigma_qp)[list(valence)].real, atol=1e-6, rtol=0)
    assert np.all(np.abs(shift) < 1.0)
    assert np.any(np.abs(shift) > 1e-4)
