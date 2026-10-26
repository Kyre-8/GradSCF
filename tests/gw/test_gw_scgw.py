"""Full matrix finite-temperature scGW on H2 and analytic limits."""

import numpy as np
import jax.numpy as jnp
import pytest
import jax

from gradscf import dft, gto
from gradscf.gw import scgw_cd_restricted

_NW = 40


def _run(**kwargs):
    from gradscf.gw import scgw_matsubara_restricted
    return scgw_matsubara_restricted(**kwargs)


def _h2_inputs():
    from gradscf.integrals.molecular.factorization import eri_pair_matrix_to_df_factors
    mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g", cart=True)
    mf = dft.RKS(mol, xc="hf").run()
    res = mf.scf_result
    df = eri_pair_matrix_to_df_factors(mf._scf_inputs.eri_pair_matrix, nao=res.mo_coeff.shape[0], tol=1e-12)
    return dict(mo_energy=res.mo_energy, mo_coeff=res.mo_coeff, nocc=1, df_factors=df,
                hcore_matrix=res.hcore_matrix, nuclear_repulsion=float(res.nuclear_repulsion)), res


def test_scgw_correlation_energy_is_invariant_to_energy_origin():
    kw, mean_field = _h2_inputs()
    ref = _run(**kw, nw=_NW, beta=80.0, max_iter=300, tol=1e-7, particle_tol=1e-12)
    shift = 0.2  # Ha: h -> h + shift*S, epsilon -> epsilon + shift.
    other = _run(**{
        **kw, "mo_energy": kw["mo_energy"] + shift,
        "hcore_matrix": kw["hcore_matrix"] + shift * mean_field.overlap_matrix,
    }, nw=_NW, beta=80.0, max_iter=300, tol=1e-7, particle_tol=1e-12)
    np.testing.assert_allclose(other.correlation_energy, ref.correlation_energy, rtol=0, atol=1e-8)
    np.testing.assert_allclose(other.chemical_potential, ref.chemical_potential + shift, rtol=0, atol=1e-8)
    np.testing.assert_allclose(other.total_energy, ref.total_energy + 2 * shift, rtol=0, atol=1e-7)
    np.testing.assert_allclose(other.green_iw, ref.green_iw, rtol=0, atol=1e-7)
    np.testing.assert_allclose(other.density_matrix, ref.density_matrix, rtol=0, atol=1e-8)


def test_scgw_h2_sto3g_converges_with_sane_energy():
    mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g", cart=True)
    mf = dft.RKS(mol, xc="hf").run()
    res = mf.scf_result
    from gradscf.integrals.molecular.factorization import eri_pair_matrix_to_df_factors

    df = eri_pair_matrix_to_df_factors(
        mf._scf_inputs.eri_pair_matrix, nao=res.mo_coeff.shape[0], tol=1e-12
    )
    out = _run(
        mo_energy=jnp.asarray(res.mo_energy),
        mo_coeff=jnp.asarray(res.mo_coeff),
        nocc=1,
        df_factors=df,
        hcore_matrix=jnp.asarray(res.hcore_matrix),
        nuclear_repulsion=float(res.nuclear_repulsion),
        nw=_NW,
        max_iter=25,
        tol=1e-5,
        mixing=0.5,
        beta=80.0,
    )
    assert out.converged
    e_c = float(out.correlation_energy)
    e_tot = float(out.total_energy)
    e_hf = float(res.total_energy)
    # correlation energy is negative and bounded for H2
    assert -0.2 < e_c < 0.0
    # total energy sits below the HF energy (GM energy includes E_c)
    assert e_tot < e_hf + 1e-6
    # Imaginary-axis self-consistency does not provide QP poles by itself.
    assert out.mo_energy is None
    np.testing.assert_allclose(np.trace(out.density_mo), 2.0, rtol=0, atol=1e-8)
    assert np.trace(np.asarray(out.density_mo) @ np.asarray(out.density_mo)) < 4 - 1e-4
    np.testing.assert_allclose(out.density_matrix, out.mo_coeff @ out.density_mo @ out.mo_coeff.T, rtol=0, atol=1e-12)
    assert float(out.fixed_point_residual) < 1e-5


def test_returned_state_satisfies_dyson_and_full_gw_map():
    from gradscf.integrals.molecular.jk import build_jk_from_df
    from gradscf.gw.g0w0 import _mo_factors
    from gradscf.gw.matsubara import gw_matsubara_step

    kw, _ = _h2_inputs()
    out = _run(**kw, nw=48, beta=80.0, max_iter=300, tol=1e-7, particle_tol=1e-12)
    inverse = (1j * out.grid.fermion[:, None, None] + out.chemical_potential) * jnp.eye(2)
    inverse = inverse - out.fock_mo - out.self_energy_iw
    np.testing.assert_allclose(inverse @ out.green_iw, jnp.broadcast_to(jnp.eye(2), out.green_iw.shape), rtol=0, atol=1e-11)
    b = _mo_factors(kw["df_factors"], out.mo_coeff)
    rebuilt = gw_matsubara_step(out.green_iw, out.fock_mo, out.chemical_potential, b, out.grid, out.sigma_moment)
    np.testing.assert_allclose(rebuilt["sigma_iw"], out.self_energy_iw, rtol=0, atol=1e-7)
    np.testing.assert_allclose(rebuilt["sigma_iw"], out.mapped_self_energy_iw, rtol=0, atol=1e-12)
    np.testing.assert_allclose(rebuilt["correlation_energy"], out.correlation_energy, rtol=0, atol=1e-12)
    jmat, kmat = build_jk_from_df(b, out.density_mo)
    ref_fock = out.mo_coeff.T @ kw["hcore_matrix"] @ out.mo_coeff + jmat - 0.5 * kmat
    np.testing.assert_allclose(ref_fock, out.fock_mo, rtol=0, atol=1e-7)
    np.testing.assert_allclose(ref_fock, out.mapped_fock_mo, rtol=0, atol=1e-12)


@pytest.mark.parametrize("variable", ["hcore", "beta"])
def test_outer_scgw_ad_is_explicitly_unsupported(variable):
    kw, _ = _h2_inputs()
    with pytest.raises(NotImplementedError, match="scGW.*eager"):
        if variable == "hcore":
            jax.grad(lambda h: _run(**{**kw, "hcore_matrix": h}).total_energy)(kw["hcore_matrix"])
        else:
            jax.grad(lambda beta: _run(**kw, beta=beta).total_energy)(80.0)


def test_scgw_frequency_refinement_controls_energy_error():
    kw, _ = _h2_inputs()
    energies = []
    for nw in (80, 160, 320):
        out = _run(**kw, beta=80.0, nw=nw, max_iter=300, tol=1e-7, particle_tol=1e-12)
        occupations = np.linalg.eigvalsh(out.density_mo)
        assert occupations.min() >= 0.0 and occupations.max() <= 2.0
        energies.append(float(out.total_energy))
    changes = np.abs(np.diff(energies))
    assert changes[1] < 0.5 * changes[0]
    assert changes[1] < 2e-5  # Ha, a discretization check separate from iteration tol.


def test_scgw_low_temperature_limit_at_fixed_time_resolution():
    kw, _ = _h2_inputs()
    energies = []
    for beta, nw in ((40.0, 160), (80.0, 320), (160.0, 640)):
        out = _run(**kw, beta=beta, nw=nw, max_iter=300, tol=1e-7, particle_tol=1e-12)
        energies.append(float(out.total_energy))
        assert abs(float(out.particle_number_error)) < 1e-12
    assert np.ptp(energies) < 1e-7  # Ha; does not establish an infinite-grid limit.


def test_noninteracting_matrix_solution_and_finite_temperature_energy():
    beta = 5.0
    rotation = jnp.array([[0.8, -0.6], [0.6, 0.8]])
    energy = jnp.array([-0.5, 0.7])
    hcore = rotation @ jnp.diag(energy) @ rotation.T
    out = _run(mo_energy=energy, mo_coeff=jnp.eye(2), nocc=1,
               df_factors=jnp.zeros((1, 2, 2)), hcore_matrix=hcore,
               beta=beta, nw=8, max_iter=5, particle_tol=1e-12)
    occupation = jax.nn.sigmoid(-beta * (energy - 0.1))
    ref_density = 2 * (rotation * occupation) @ rotation.T
    np.testing.assert_allclose(out.density_matrix, ref_density, rtol=0, atol=1e-11)
    np.testing.assert_allclose(out.total_energy, 2 * jnp.sum(energy * occupation), rtol=0, atol=1e-11)
    assert float(out.correlation_energy) == 0.0
    np.testing.assert_allclose(out.chemical_potential, 0.1, rtol=0, atol=1e-10)


def test_scgw_cannot_converge_by_mixing_away_the_residual():
    kw, _ = _h2_inputs()
    with pytest.raises(ArithmeticError, match="scGW did not converge"):
        _run(**kw, nw=16, beta=40.0, max_iter=1, tol=1e-7, mixing=1e-8)


def test_legacy_name_warns_and_rejects_real_axis_eta():
    kw, _ = _h2_inputs()
    with pytest.warns(DeprecationWarning, match="Matsubara"):
        out = scgw_cd_restricted(**kw, nw=16, beta=40.0, max_iter=50, tol=1e-5)
    assert out.mo_energy is None
    with pytest.raises(ValueError, match="eta"):
        scgw_cd_restricted(**kw, eta=1e-3)
