"""Small discrete Fourier models: k weights, complex gauges, and q orientation.

CPU/float64, Ha and Bohr. These test algebraic invariants, not solid-state
basis/grid accuracy; nine grid points keep the auxiliary solves inexpensive.
"""

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gradscf.gw.freq import scaled_legendre_grid
from gradscf.gw.polarizability import rho_response_iw, rho_response_real
from gradscf.gw.pbc.ksigma import _screened_w_kpoint, _sigma_residue_kpoint, g0w0_cd_kpoints
from gradscf.gw.pbc.krgw import g0w0_cd_gamma
from gradscf.gw.pbc.kugw import g0w0_cd_gamma_unrestricted
from gradscf.pbc.momentum import momentum_transfer_table
from gradscf.gw.pbc.product_basis import kpoint_product_factors
from gradscf.gw.pbc.q0 import q0_residue_correction, screened_w_imag_axis_head_wing
from gradscf.gw.self_energy import sigma_residue_part
from gradscf.integrals.periodic.coulomb import coulomb_kernel
from gradscf.integrals.periodic.fft import get_kpoint_jk


def _toy(nk=3):
    mesh = (9, 1, 1)
    length = 8.0
    volume = length**3
    coords = np.zeros((mesh[0], 3))
    coords[:, 0] = np.arange(mesh[0]) * length / mesh[0]
    gv = np.zeros_like(coords)
    gv[:, 0] = 2 * np.pi * np.fft.fftfreq(mesh[0], d=length / mesh[0])
    frac = np.zeros((nk, 3))
    frac[:, 0] = np.arange(nk) / nk
    kpts = frac * 2 * np.pi / length
    angle = 2 * np.pi * coords[:, 0] / length
    values = np.sqrt(2) * np.column_stack([np.cos(angle), np.sin(angle)]) / np.sqrt(volume)
    gradient = np.sqrt(2) * np.column_stack([-np.sin(angle), np.cos(angle)])
    gradient *= 2 * np.pi / length / np.sqrt(volume)
    ao = np.zeros((nk, 4, mesh[0], 2), dtype=complex)
    for k in range(nk):
        ao[k, 0] = values
        ao[k, 1] = gradient + 1j * kpts[k, 0] * values
    pairs = values[:, :, None] * values[:, None, :]
    kernel = np.asarray(coulomb_kernel(jnp.asarray(gv)))
    potential = np.fft.ifft(np.fft.fft(pairs, axis=0) * kernel[:, None, None], axis=0).real
    inputs = SimpleNamespace(
        ao=jnp.asarray(ao), coords=jnp.asarray(coords), gvectors=jnp.asarray(gv),
        kpoints=jnp.asarray(kpts), weights=jnp.full(mesh[0], volume / mesh[0]),
        overlap=jnp.tile(jnp.eye(2), (nk, 1, 1)), madelung=0.0,
        pair_potential=jnp.asarray(potential),
    )
    coeff = []
    for k in range(nk):
        t = 0.2 + 0.1 * k
        coeff.append([[np.cos(t), 1j * np.sin(t)], [1j * np.sin(t), np.cos(t)]])
    coeff = jnp.asarray(coeff)
    energy = jnp.tile(jnp.array([-0.7, 0.6]), (nk, 1))
    density = jnp.einsum("kpi,kqi->kpq", coeff[:, :, :1], coeff[:, :, :1].conj())
    density = jnp.stack([density, density])
    jmat, kmat = get_kpoint_jk(inputs, density, mesh=mesh)
    fock = jnp.einsum("kpi,ki,kqi->kpq", coeff, energy, coeff.conj())
    hcore = fock - jmat + kmat[0]
    return dict(
        inputs=inputs, kpts_frac=frac, mo_energy_k=energy, mo_coeff_k=coeff,
        nocc=1, fock_k=fock, hcore_k=hcore, density_spin=density, mesh=mesh, nw=16,
    )


@pytest.mark.parametrize("real_axis", [False, True])
def test_complex_response_matches_direct_contraction(real_axis):
    b = jnp.array([[[0.2 + 0.1j]], [[-0.1 + 0.3j]]])
    energy = jnp.array([-0.5, 0.7])
    omega, eta, gap = 0.4, 0.03, -1.2
    if real_axis:
        chi = 1 / (omega + gap + 2j * eta) + 1 / (-omega + gap)
        result = rho_response_real(omega, energy, b, eta=eta, conjugate=True)
        spin = 2.0
    else:
        chi = gap / (omega**2 + gap**2)
        result = rho_response_iw(omega, energy, b, conjugate=True)
        spin = 4.0
    ref = spin * chi * np.outer(np.asarray(b[:, 0, 0]), np.asarray(b[:, 0, 0]).conj())
    np.testing.assert_allclose(result, ref, rtol=1e-13, atol=1e-14)


def _replicated(nk):
    b = jnp.array([[[0.5, 0.3], [0.3, 0.2]]], dtype=jnp.complex128)
    energy = jnp.array([-0.5, 0.5])
    table = (np.arange(nk)[:, None] - np.arange(nk)[None, :]) % nk
    blocks = [jnp.stack([b] * nk)] * nk
    channels = [tuple(((energy[:1], energy[1:]), b[:, :1, 1:], 2.0) for _ in range(nk))] * nk
    return blocks, [energy] * nk, table, channels


def test_replicated_q_blocks_preserve_averaged_screened_interaction():
    freqs, _ = scaled_legendre_grid(8)
    outputs = []
    for nk in (1, 3):
        b, e, table, _ = _replicated(nk)
        w, _, _ = _screened_w_kpoint(b, e, 1, table, freqs, nk)
        outputs.append(sum(wq[:, 0] for wq in w))
    np.testing.assert_allclose(outputs[1], outputs[0], rtol=1e-12, atol=1e-14)


def test_replicated_k_channels_preserve_residue_screening():
    outputs = []
    for nk in (1, 3):
        b, e, table, channels = _replicated(nk)
        outputs.append(_sigma_residue_kpoint(-0.8, 0, 0, e, b, channels, table, 0.0, 0.01, nk))
    np.testing.assert_allclose(outputs[1], outputs[0], rtol=1e-12, atol=1e-14)


def test_three_k_screening_uses_inverse_momentum_partner():
    rng = np.random.default_rng(127)
    nk, naux = 3, 2
    b = 0.1 * (rng.normal(size=(nk, nk, naux, 2, 2)) + 1j * rng.normal(size=(nk, nk, naux, 2, 2)))
    energy = np.array([[-0.5, 0.6], [-0.4, 0.8], [-0.7, 0.9]])
    table = (np.arange(nk)[:, None] - np.arange(nk)[None, :]) % nk
    omega = 0.3
    w, _, _ = _screened_w_kpoint(jnp.asarray(b), list(energy), 1, table, jnp.array([omega]), nk)
    for q in range(nk):
        pi = np.zeros((naux, naux), dtype=complex)
        for ki in range(nk):
            gap = energy[ki, 0] - energy[table[ki, q], 1]
            v = b[q, ki, :, 0, 1]
            pi += 4 / nk * gap / (omega**2 + gap**2) * np.outer(v, v.conj())
        screened = np.linalg.solve(np.eye(naux) - pi, pi)
        for kn in range(nk):
            km = np.flatnonzero(table[:, q] == kn)[0]
            ref = np.einsum("Gmp,GH,Hmp->mp", b[q, km].conj(), screened, b[q, km]) / nk
            np.testing.assert_allclose(w[q][0, kn], ref, rtol=1e-12, atol=1e-14)


def test_product_factors_match_direct_canonical_fourier_sum():
    kw = _toy()
    inputs = kw["inputs"]
    table = momentum_transfer_table(kw["kpts_frac"])
    b = kpoint_product_factors(inputs, kw["mo_coeff_k"], mesh=kw["mesh"], momentum_table=table)
    psi = np.einsum("kgp,kpm->kgm", inputs.ao[:, 0], kw["mo_coeff_k"])
    coords, gv, kpts = map(np.asarray, (inputs.coords, inputs.gvectors, inputs.kpoints))
    volume = float(inputs.weights.sum())
    for q in range(3):
        wavevectors = gv - kpts[q]
        scale = np.sqrt(volume * np.asarray(coulomb_kernel(jnp.asarray(wavevectors)))) / len(coords)
        fourier = np.exp(-1j * wavevectors @ coords.T)
        for ki in range(3):
            ka = table[ki, q]
            pairs = psi[ki].conj()[:, :, None] * psi[ka][:, None, :]
            pairs *= np.exp(1j * coords @ (kpts[ka] - kpts[ki]))[:, None, None]
            ref = scale[:, None, None] * np.einsum("Gg,gmn->Gmn", fourier, pairs)
            np.testing.assert_allclose(b[q][ki], ref, rtol=1e-12, atol=1e-13)


@pytest.mark.parametrize("fc", [False, True])
def test_kpoint_driver_is_invariant_to_independent_band_phases(fc):
    kw = _toy()
    original = g0w0_cd_kpoints(**kw, fc=fc)
    phase = jnp.exp(1j * jnp.array([[0.3, 1.1], [-0.7, 0.4], [0.6, -1.0]]))
    coeff = kw["mo_coeff_k"] * phase[:, None, :]
    transformed = g0w0_cd_kpoints(**{**kw, "mo_coeff_k": coeff}, fc=fc)
    np.testing.assert_allclose(transformed.mo_coeff, coeff, rtol=0, atol=1e-14)
    np.testing.assert_allclose(transformed.mo_energy, original.mo_energy, rtol=0, atol=1e-9)
    np.testing.assert_array_equal(transformed.converged_mask, original.converged_mask)


def test_head_wing_is_invariant_to_auxiliary_unitary():
    rng = np.random.default_rng(91)
    b = jnp.asarray(0.1 * (rng.normal(size=(2, 2, 2)) + 1j * rng.normal(size=(2, 2, 2))))
    u, _ = np.linalg.qr(rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2)))
    e = jnp.array([-0.5, 0.8])
    qij = jnp.array([[2e-4 + 1e-4j]])
    args = (e, qij, jnp.array([1e-3, 0, 0]), 100.0, jnp.array([0.2, 1.0]))
    ref = screened_w_imag_axis_head_wing(b, b[:, :1, 1:], *args)
    rotated = jnp.einsum("PQ,Qmn->Pmn", jnp.asarray(u), b)
    actual = screened_w_imag_axis_head_wing(rotated, rotated[:, :1, 1:], *args)
    for a, r in zip(actual, ref):
        np.testing.assert_allclose(a, r, rtol=1e-11, atol=1e-13)


def test_gamma_matches_single_k_with_complex_orbitals():
    kw = _toy(nk=1)
    result = g0w0_cd_kpoints(**kw, fc=False)
    gamma = g0w0_cd_gamma(
        inputs=kw["inputs"], mo_energy=kw["mo_energy_k"][0], mo_coeff=kw["mo_coeff_k"][0],
        nocc=1, fock_matrix=kw["fock_k"][0], hcore_matrix=kw["hcore_k"][0],
        density_spin=kw["density_spin"][:, 0], mesh=kw["mesh"], nw=kw["nw"], fc=False,
    )
    np.testing.assert_allclose(gamma.mo_coeff, kw["mo_coeff_k"][0], rtol=0, atol=1e-14)
    np.testing.assert_allclose(gamma.mo_energy, result.mo_energy[0], rtol=0, atol=1e-9)


def test_gamma_unrestricted_preserves_independent_spin_phases():
    kw = _toy(nk=1)
    ref = g0w0_cd_kpoints(**kw, fc=False)
    ca = kw["mo_coeff_k"][0] * jnp.exp(1j * jnp.array([0.2, -0.7]))
    cb = kw["mo_coeff_k"][0] * jnp.exp(1j * jnp.array([-0.5, 1.2]))
    result = g0w0_cd_gamma_unrestricted(
        inputs=kw["inputs"], mo_energy=(kw["mo_energy_k"][0],) * 2, mo_coeff=(ca, cb),
        nocc=(1, 1), fock_matrix=(kw["fock_k"][0],) * 2, hcore_matrix=kw["hcore_k"][0],
        density_spin=kw["density_spin"][:, 0], mesh=kw["mesh"], nw=kw["nw"],
    )
    np.testing.assert_allclose(result.mo_energy, jnp.repeat(ref.mo_energy, 2, axis=0), rtol=0, atol=1e-9)


def test_q0_correction_does_not_depend_on_kpoint_order():
    kw = _toy()
    ref = g0w0_cd_kpoints(**kw, fc=True)
    order = np.array([1, 0, 2])  # Gamma is no longer the first transfer.
    permuted = dict(kw)
    for key in ("kpts_frac", "mo_energy_k", "mo_coeff_k", "fock_k", "hcore_k"):
        permuted[key] = kw[key][order]
    permuted["density_spin"] = kw["density_spin"][:, order]
    data = vars(kw["inputs"]).copy()
    for key in ("ao", "overlap", "kpoints"):
        data[key] = data[key][order]
    permuted["inputs"] = SimpleNamespace(**data)
    actual = g0w0_cd_kpoints(**permuted, fc=True)
    np.testing.assert_allclose(actual.mo_energy, ref.mo_energy[order], rtol=0, atol=1e-9)


def test_q0_head_scales_with_cell_sampling_without_double_averaging():
    values = []
    for nk in (1, 3):
        b, energy, table, _ = _replicated(nk)
        b = [jnp.zeros_like(block) for block in b]
        channels = [tuple((e, jnp.zeros((1, 1, 1)), 2.0) for e in energy)] * nk
        qij = tuple(jnp.array([[2e-4 + 1e-4j]]) for _ in range(nk))
        q0 = (qij, jnp.array([1e-3, 0, 0]), 100.0)
        sig = _sigma_residue_kpoint(-0.8, 0, 0, energy, b, channels, table, 0.0, 0.01, nk, q0)
        values.append(sig * nk**(1.0 / 3.0))
    assert abs(values[0]) > 1e-4
    np.testing.assert_allclose(values[1], values[0], rtol=1e-12, atol=1e-14)


@pytest.mark.parametrize("nk", [1, 3])
def test_standalone_q0_residue_matches_multichannel_self_energy(nk):
    energy = jnp.array([-0.5, 0.7])
    b = jnp.array([[0.2 + 0.1j, 0.1 - 0.05j], [-0.1 + 0.3j, 0.15 + 0.1j]])
    channels = (
        (energy, b[:, 1:].reshape(2, 1, 1), 1.0),
        (jnp.array([-0.4, 0.9]), 0.7 * b[:, 1:].reshape(2, 1, 1), 1.0),
    )
    qij = (jnp.array([[2e-4 + 1e-4j]]), jnp.array([[1e-4 - 2e-4j]]))
    qabs = jnp.array([1e-3, 0, 0])
    q0 = dict(qij=qij, q_abs=qabs, volume=100.0, p_index=0, nkpts=nk)
    body = sigma_residue_part(-0.8, energy, b, b, channels, 0.0, 0.01, True, response_scale=1 / nk)
    full = sigma_residue_part(-0.8, energy, b, b, channels, 0.0, 0.01, True, q0, response_scale=1 / nk)
    correction = q0_residue_correction(0.3, b[:, 0], channels, qij, qabs, 100.0, 0.01, nkpts=nk)
    np.testing.assert_allclose(correction, -(full - body), rtol=1e-12, atol=1e-14)


def test_kpoint_driver_rejects_unsupported_differentiation_explicitly():
    kw = _toy()
    with pytest.raises(NotImplementedError, match="k-point GW.*eager"):
        jax.grad(lambda e: g0w0_cd_kpoints(**{**kw, "mo_energy_k": e}, fc=False).mo_energy.sum())(
            kw["mo_energy_k"]
        )


def test_small_h2_cell_gamma_matches_single_k_with_head_wing():
    from gradscf.gw.pbc import KRGW
    from gradscf.pbc import gto, scf

    # A real SCF input, deliberately coarse: this checks driver integration,
    # not basis/grid-converged physical accuracy. Only 125 auxiliary functions.
    mesh = (5, 5, 5)
    cell = gto.M(
        atom="H 3 3 2.3; H 3 3 3.7", a=np.eye(3) * 6, unit="Bohr",
        basis="gth-szv", pseudo="gth-pade", mesh=mesh, precision=1e-10,
    )
    mf = scf.RHF(cell).run()
    assert mf.converged
    result = mf.result
    gamma = KRGW(mf, nw=16, fc=True).run().result
    kpoint = g0w0_cd_kpoints(
        inputs=mf.inputs, kpts_frac=np.zeros((1, 3)),
        mo_energy_k=result.mo_energy_spin[0], mo_coeff_k=result.mo_coeff_spin[0],
        nocc=1, fock_k=result.fock_spin[0], hcore_k=mf.inputs.hcore,
        density_spin=result.density_spin, mesh=mesh, nw=16, fc=True,
    )
    assert gamma.converged and kpoint.converged
    np.testing.assert_allclose(gamma.mo_energy, kpoint.mo_energy[0], rtol=0, atol=1e-8)
