"""Transform only requested orbital blocks, including compressed AO data."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest


@pytest.mark.parametrize("representation", ["dense", "packed", "df"])
def test_selected_eri_block_and_derivatives(representation):
    from gradscf.ao2mo import transform_eri_block
    rng = np.random.default_rng(9)
    b = rng.normal(size=(4, 5, 5))
    b = jnp.asarray(b + b.transpose(0, 2, 1))
    g = jnp.einsum("Qpq,Qrs->pqrs", b, b)
    rows, cols = np.tril_indices(5)
    packed = g[rows, cols][:, rows, cols]
    cs = tuple(jnp.asarray(rng.normal(size=(5, n))) for n in (2, 3, 1, 4))
    kwargs = {"dense": {"eri": g}, "packed": {"eri_pair_matrix": packed},
              "df": {"df_factors": b}}[representation]
    expected = lambda x: jnp.einsum("pqrs,pi,qj,rk,sl->ijkl", g, x, *cs[1:])
    actual = lambda x: transform_eri_block((x, *cs[1:]), **kwargs)
    np.testing.assert_allclose(actual(cs[0]), expected(cs[0]), atol=1e-11)
    loss = lambda x: jnp.sum(actual(x)**2)
    ref = lambda x: jnp.sum(expected(x)**2)
    np.testing.assert_allclose(jax.grad(loss)(cs[0]), jax.grad(ref)(cs[0]), atol=1e-8)


@pytest.mark.parametrize("df", [False, True])
@pytest.mark.parametrize("with_t2", [False, True])
def test_rhf_facade_keeps_mp2_transformation_in_selected_blocks(df, with_t2, monkeypatch):
    from gradscf import gto, scf, mp
    from gradscf.ao2mo import reference as mo
    from gradscf.scf.reference import reference_from_source
    mol = gto.M(atom="O 0 0 0; H 0 .75 .58; H 0 -.75 .58", basis="sto-3g")
    mf = scf.RHF(mol, conv_tol=1e-12, conv_tol_grad=1e-10)
    if df:
        mf.density_fit()
    mf.run()
    ref = reference_from_source(mf)
    dense = mp.run_mp(ref.h1, ref.eri, nocc=5)
    def forbidden(*args, **kwargs):
        pytest.fail("MP2 must not transform a full MO ERI tensor")
    monkeypatch.setattr(mo, "_transform_eri", forbidden)
    result = mp.MP2(mf, with_t2=with_t2).run()
    np.testing.assert_allclose(result.e_corr, dense.e2, atol=1e-12)
    if with_t2:
        np.testing.assert_allclose(result.t2, dense.t2, atol=1e-10)
    else:
        assert result.t2 is None
    mf.mo_coeff = mf.mo_coeff.at[:, 0].multiply(-1)
    with pytest.raises(RuntimeError, match="Orbital state changed"):
        result.kernel()
    assert not result.converged
    assert result.result is None and result.e_corr is None and result.t2 is None


@pytest.mark.parametrize("df", [False, True])
def test_uhf_facade_keeps_spin_blocks_without_full_mo_eri(df, monkeypatch):
    from gradscf import gto, scf, mp
    from gradscf.ao2mo import reference as mo
    from gradscf.scf.reference import unrestricted_reference_from_source
    mol = gto.M(atom="H 0 0 0; H 0 0 .85; H 0 0 1.9", basis="sto-3g", spin=1)
    mf = scf.UHF(mol, conv_tol=1e-13, conv_tol_grad=1e-11, max_cycle=150)
    if df:
        mf.density_fit()
    mf.run()
    ref = unrestricted_reference_from_source(mf)
    dense = mp.run_mp(ref.h1, ref.eri, nocc=ref.nocc)
    def forbidden(*args, **kwargs):
        pytest.fail("UMP2 must retain separate occupied/virtual spin factors")
    monkeypatch.setattr(mo, "_transform_eri", forbidden)
    result = mp.MP2(mf).run()
    np.testing.assert_allclose(result.e_corr, dense.e2, atol=1e-12)
    for actual, expected in zip(result.t2, dense.t2):
        np.testing.assert_allclose(actual, expected, atol=1e-10)
