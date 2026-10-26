"""qsGW (Stage 4) tests: self-consistent static potential iteration."""

import numpy as np
import pytest
import jax
import jax.numpy as jnp

from gradscf import dft, gto
from gradscf.gw import qsgw_cd_restricted
from gradscf.gw.freq import scaled_legendre_grid
from gradscf.gw.polarizability import rho_response_iw
from gradscf.gw.qsgw import _static_self_energy
from gradscf.gw.screened import screened_w_imag_axis_matrix
from gradscf.gw.self_energy import sigma_imag_matrix, sigma_residue_matrix

_ATOM = "O 0 0 0.117790; H 0 0.755453 -0.471161; H 0 -0.755453 -0.471161"
_NW = 30  # qsGW evaluates the full off-diagonal self-energy; keep it small


def _small_static_inputs():
    rng = np.random.default_rng(814)
    b = rng.normal(size=(2, 3, 3)) * 0.15
    b = jnp.asarray(0.5 * (b + b.transpose(0, 2, 1)))
    energy = jnp.array([-0.91, 0.28, 1.03])
    freqs, wts = scaled_legendre_grid(24)
    return dict(
        b_mn=b, b_ov=b[:, :1, 1:], mo_energy=energy, nocc=1,
        ef=0.5 * (energy[0] + energy[1]), freqs=freqs, wts=wts, eta=1e-3,
    )


def _endpoint_reference(kw):
    """Direct mode-A definition; independently select both endpoint entries."""
    b, energy = kw["b_mn"], kw["mo_energy"]
    w = screened_w_imag_axis_matrix(
        b, lambda omega: rho_response_iw(omega, energy, kw["b_ov"]), kw["freqs"]
    )

    w_static = screened_w_imag_axis_matrix(
        b, lambda omega: rho_response_iw(omega, energy, kw["b_ov"]), jnp.zeros(1)
    )[0]
    static = dict(wmn_static=w_static,
                  occupation_sign=jnp.where(jnp.arange(len(energy)) < kw["nocc"], -1., 1.))

    def sigma(omega):
        return np.asarray(
            sigma_imag_matrix(omega, w, energy, kw["ef"], kw["freqs"], kw["wts"], kw["eta"], **static)
            + sigma_residue_matrix(omega, energy, b, ((energy, kw["b_ov"], 2.0),), kw["ef"], kw["eta"], **static)
        )

    endpoints = [sigma(e) for e in energy]
    hermitian = [0.5 * (s + s.conj().T) for s in endpoints]
    ref = np.empty((3, 3))
    midpoint = np.empty_like(ref)
    for m in range(3):
        for n in range(3):
            ref[m, n] = 0.5 * (hermitian[m][m, n] + hermitian[n][m, n]).real
            midpoint[m, n] = sigma(0.5 * (energy[m] + energy[n]))[m, n].real
    return ref, midpoint


def test_static_mapping_averages_endpoint_self_energies():
    kw = _small_static_inputs()
    ref, midpoint = _endpoint_reference(kw)
    assert np.max(np.abs(ref - midpoint)) > 1e-5  # Discriminate the old approximation.
    actual = np.asarray(_static_self_energy(**kw))
    np.testing.assert_allclose(actual, ref, rtol=1e-12, atol=1e-13)
    np.testing.assert_allclose(actual, actual.T, rtol=0, atol=1e-14)
    np.testing.assert_allclose(np.diag(actual), np.diag(midpoint), rtol=0, atol=1e-13)


def test_static_mapping_jit_and_factor_gradient():
    kw = _small_static_inputs()

    def mapped(scale):
        return _static_self_energy(**{
            **kw, "b_mn": kw["b_mn"] * scale, "b_ov": kw["b_ov"] * scale,
        })

    np.testing.assert_allclose(jax.jit(mapped)(1.0), mapped(1.0), rtol=1e-12, atol=1e-13)
    loss = lambda scale: jnp.sum(mapped(scale)**2)
    step = 1e-5
    finite_difference = (loss(1.0 + step) - loss(1.0 - step)) / (2 * step)
    np.testing.assert_allclose(jax.jit(jax.grad(loss))(1.0), finite_difference, rtol=1e-6, atol=1e-10)


def test_static_mapping_energy_gradient_moves_evaluation_endpoints():
    kw = _small_static_inputs()
    def loss(energy):
        mapped = _static_self_energy(**{**kw, "mo_energy": energy, "ef": 0.5 * (energy[0] + energy[1])})
        return jnp.sum(mapped**2)

    energy = kw["mo_energy"]
    step = 1e-5
    finite_difference = jnp.array([
        (loss(energy.at[i].add(step)) - loss(energy.at[i].add(-step))) / (2 * step)
        for i in range(len(energy))
    ])
    np.testing.assert_allclose(jax.jit(jax.grad(loss))(energy), finite_difference, rtol=1e-5, atol=1e-10)


def _zero_interaction_inputs():
    return dict(
        mo_energy=jnp.array([-1.0, 1.0]), mo_coeff=jnp.eye(2), nocc=1,
        df_factors=jnp.zeros((1, 2, 2)), hcore_matrix=jnp.diag(jnp.array([-0.6, 0.8])),
        nw=8,
    )


def test_qsgw_damping_cannot_hide_a_large_energy_residual():
    with pytest.raises(ArithmeticError, match="qsGW did not converge"):
        qsgw_cd_restricted(**_zero_interaction_inputs(), max_iter=1, tol=1e-6, damping=0.999999)


def test_qsgw_cannot_converge_on_isospectral_virtual_rotation():
    from gradscf.integrals.molecular.jk import build_j_from_df
    from gradscf.gw.g0w0 import _exchange_mo

    kw = _small_static_inputs()
    b, energy = kw["b_mn"], kw["mo_energy"]
    angle = 0.5
    rotation = jnp.array([
        [1.0, 0.0, 0.0],
        [0.0, np.cos(angle), -np.sin(angle)],
        [0.0, np.sin(angle), np.cos(angle)],
    ])
    density = jnp.diag(jnp.array([2.0, 0.0, 0.0]))
    # The first diagonalization changes virtual orbitals but neither the
    # spectrum nor occupied density. The starting state is not stationary.
    target = rotation @ jnp.diag(energy) @ rotation.T
    hcore = target - build_j_from_df(b, density) - _exchange_mo(b, 1) - _static_self_energy(**kw)
    with pytest.raises(ArithmeticError, match="qsGW did not converge"):
        qsgw_cd_restricted(
            mo_energy=energy, mo_coeff=jnp.eye(3), nocc=1, df_factors=b,
            hcore_matrix=hcore, nw=24, max_iter=1, tol=1e-8, tol_density=1e-8,
        )


@pytest.mark.parametrize("option", [{"max_iter": 0}, {"tol": 0.0}, {"tol_density": 0.0}])
def test_qsgw_rejects_invalid_convergence_controls(option):
    with pytest.raises(ValueError, match="positive"):
        qsgw_cd_restricted(**_zero_interaction_inputs(), **option)


def test_qsgw_outer_differentiation_is_explicitly_unsupported():
    kw = _zero_interaction_inputs()
    with pytest.raises(NotImplementedError, match="qsGW.*eager"):
        jax.grad(lambda e: qsgw_cd_restricted(**{**kw, "mo_energy": e}).mo_energy.sum())(kw["mo_energy"])


def test_qsgw_rejects_complex_orbitals_instead_of_discarding_imaginary_parts():
    kw = _zero_interaction_inputs()
    kw["mo_coeff"] = kw["mo_coeff"] * jnp.array([1.0, 1j])
    with pytest.raises(NotImplementedError, match="qsGW.*real"):
        qsgw_cd_restricted(**kw)


def test_qsgw_water_sto3g_converges_and_is_sane():
    mol = gto.M(atom=_ATOM, basis="sto-3g", cart=True)
    mf = dft.RKS(mol, xc="hf").run()
    res = mf.scf_result
    from gradscf.integrals.molecular.factorization import eri_pair_matrix_to_df_factors

    df = eri_pair_matrix_to_df_factors(
        mf._scf_inputs.eri_pair_matrix, nao=res.mo_coeff.shape[0], tol=1e-12
    )
    out = qsgw_cd_restricted(
        mo_energy=jnp.asarray(res.mo_energy),
        mo_coeff=jnp.asarray(res.mo_coeff),
        nocc=5,
        df_factors=df,
        hcore_matrix=jnp.asarray(res.hcore_matrix),
        nw=_NW,
        max_iter=30,
        tol=1e-7,
        tol_density=1e-7,
        damping=0.3,
    )
    e_qp = np.asarray(out.mo_energy)
    e_mf = np.asarray(res.mo_energy)
    assert np.all(np.isfinite(e_qp))
    # qsGW HOMO (ionization potential) is below the HF Koopmans value and
    # the gap remains open
    homo, lumo = 4, 5
    # The matrix-pole boundary correction changes the small-basis shift by a
    # few mH; keep this as a bounded sanity check rather than a fitted number.
    assert e_qp[homo] < e_mf[homo] + 0.1
    assert e_qp[lumo] > e_qp[homo]
    # orbitals are updated and remain orthonormal w.r.t. the overlap
    s = np.asarray(res.overlap_matrix)
    c = np.asarray(out.mo_coeff)
    np.testing.assert_allclose(c.T @ s @ c, np.eye(c.shape[1]), atol=1e-10)
    # Rebuild the effective Fock at the returned state. This checks the
    # self-consistent equation rather than only changes between iterates.
    from gradscf.integrals.molecular.jk import build_j_from_df
    from gradscf.gw.g0w0 import _exchange_mo, _mo_factors

    coeff = out.mo_coeff
    density = 2.0 * coeff[:, :5] @ coeff[:, :5].T
    b_mn = _mo_factors(df, coeff)
    freqs, wts = scaled_legendre_grid(_NW)
    correlation = _static_self_energy(
        b_mn=b_mn, b_ov=b_mn[:, :5, 5:], mo_energy=out.mo_energy, nocc=5,
        ef=0.5 * (out.mo_energy[4] + out.mo_energy[5]), freqs=freqs, wts=wts, eta=1e-3,
    )
    fock = coeff.T @ (res.hcore_matrix + build_j_from_df(df, density)) @ coeff
    fock = fock + _exchange_mo(b_mn, 5) + correlation
    np.testing.assert_allclose(fock, jnp.diag(out.mo_energy), rtol=0, atol=1e-7)


def _analytic_two_orbital_static(energy, factors):
    """Independent exact one-RPA-pole model; no contour-deformation kernels."""
    gap = energy[1] - energy[0]
    coupling = 4 * factors[0, 0, 1] ** 2 * gap
    pole = jnp.sqrt(gap ** 2 + coupling)
    residue = coupling / (2 * pole)
    b = factors[0]
    rows = []
    for p in range(2):
        sigma = sum(
            residue * jnp.outer(b[:, q], b[q, :])
            / (energy[p] - energy[q] + (pole if q == 0 else -pole))
            for q in range(2)
        )
        rows.append(sigma[p])
    rows = jnp.stack(rows)
    return (rows + rows.T) / 2


def test_qsgw_static_potential_matches_independent_rpa_pole():
    energy = jnp.array([-.5, .5])
    factors = jnp.array([[[.4, .2], [.2, .3]]])
    freqs, weights = scaled_legendre_grid(100)

    def potential(t):
        e = energy + t * jnp.array([.03, -.02])
        b = factors * (1 + .1 * t)
        return _static_self_energy(
            b_mn=b, b_ov=b[:, :1, 1:], mo_energy=e, nocc=1,
            ef=jnp.mean(e), freqs=freqs, wts=weights, eta=1e-7,
        )

    expected = _analytic_two_orbital_static(energy, factors)
    np.testing.assert_allclose(jax.jit(potential)(0.), expected, atol=2e-11, rtol=0)
    exact_loss = lambda t: jnp.sum(_analytic_two_orbital_static(
        energy + t * jnp.array([.03, -.02]), factors * (1 + .1 * t)) ** 2)
    loss = lambda t: jnp.sum(potential(t) ** 2)
    expected_grad = jax.grad(exact_loss)(0.)
    np.testing.assert_allclose(jax.jit(jax.grad(loss))(0.), expected_grad, atol=2e-11, rtol=1e-7)
    np.testing.assert_allclose(jax.jvp(loss, (0.,), (1.,))[1], expected_grad, atol=2e-11, rtol=1e-7)
