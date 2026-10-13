"""Periodic GW tests: plane-wave product basis and momentum tables."""

import numpy as np
import jax.numpy as jnp

from gradscf.integrals.periodic.fft import get_jk
from gradscf.pbc.momentum import momentum_transfer_table, wrap_to_kmesh
from gradscf.gw.pbc.product_basis import gamma_product_factors
from gradscf.pbc import gto, scf

_MESH = (17, 17, 17)


def _h2_gamma():
    cell = gto.M(
        atom="H 0.2 0.3 0.4; H 1.6 0.3 0.4",
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


def test_gamma_product_factors_reconstruct_exchange():
    """B factors must reproduce the FFT pair-potential exchange matrix.

    For each spin sigma, K^sigma_mn = sum_{i occ sigma} (mi|in) with
    (mi|in) = sum_G conj(B_G(m,i)) B_G(i,n).  The reference is the direct
    real-space FFT contraction ``fft.get_jk`` with exxdiv=None (no Ewald
    correction); both use v(G=0) = 0 consistently.
    """
    mf = _h2_gamma()
    inputs = mf.inputs
    coeff = np.asarray(mf.mo_coeff[0])  # (nao, nmo) at Gamma
    density = np.asarray(mf.result.density_spin[:, 0])  # (2, nao, nao)
    occ = np.asarray(mf.result.mo_occ_spin[:, 0])  # (2, nmo)

    _, k_ref = get_jk(inputs, jnp.asarray(density), exxdiv=None, with_k=True)
    k_ref = np.asarray(k_ref)  # (2, nao, nao)

    b = gamma_product_factors(inputs, jnp.asarray(coeff), mesh=_MESH)  # (ngrid, nmo, nmo)
    for sigma in range(2):
        nocc = int(occ[sigma].sum())
        k_mo_ref = coeff.T @ k_ref[sigma] @ coeff
        k_mo_b = np.zeros_like(k_mo_ref, dtype=np.complex128)
        b_np = np.asarray(b)
        for i in range(nocc):
            k_mo_b += np.einsum("Gm,Gn->mn", b_np[:, :, i].conj(), b_np[:, i, :])
        np.testing.assert_allclose(k_mo_b.real, k_mo_ref, atol=1e-10)
        np.testing.assert_allclose(k_mo_b.imag, 0.0, atol=1e-10)


def test_gamma_product_factors_g0_component_vanishes():
    mf = _h2_gamma()
    coeff = np.asarray(mf.mo_coeff[0])
    b = np.asarray(gamma_product_factors(mf.inputs, jnp.asarray(coeff), mesh=_MESH))
    g2 = np.asarray(mf.inputs.gvectors)
    g0 = np.where(np.sum(g2 * g2, axis=-1) < 1e-24)[0]
    assert g0.size == 1
    np.testing.assert_allclose(b[g0[0]], 0.0, atol=1e-14)


def test_momentum_transfer_table_cubic_mesh():
    # 2x2x2 uniform mesh in fractional coordinates
    fracs = np.array([[i / 2.0, j / 2.0, k / 2.0] for i in range(2) for j in range(2) for k in range(2)])
    table = momentum_transfer_table(fracs)
    assert table.shape == (8, 8)
    for ki in range(8):
        for q in range(8):
            kj = table[ki, q]
            diff = wrap_to_kmesh(fracs[ki] - fracs[q] - fracs[kj])
            np.testing.assert_allclose(diff, 0.0, atol=1e-8)


def test_momentum_transfer_table_rejects_incomplete_mesh():
    fracs = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [0.0, 0.5, 0.0]])
    try:
        momentum_transfer_table(fracs)
    except ValueError:
        return
    raise AssertionError("incomplete k mesh must raise ValueError")
