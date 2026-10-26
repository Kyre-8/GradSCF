"""Canonical MP2 energies, amplitudes and AD; CPU float64, Hartree."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest


@pytest.fixture(scope="module")
def water():
    pytest.importorskip("pyscf")
    from pyscf import gto, scf, ao2mo
    mol = gto.M(atom="O 0 0 0; H 0 .75 .58; H 0 -.75 .58",
                basis="6-31g*", cart=True, verbose=0)
    mf = scf.RHF(mol).run(conv_tol=1e-13)
    c = mf.mo_coeff
    h = c.T @ mf.get_hcore() @ c
    g = ao2mo.kernel(mol, c, compact=False).reshape((c.shape[1],) * 4)
    return mf, jnp.asarray(h), jnp.asarray(g)


@pytest.fixture(scope="module")
def radical():
    pytest.importorskip("pyscf")
    from pyscf import gto, scf, ao2mo
    mol = gto.M(atom="H 0 0 0; H 0 0 .85; H 0 0 1.9",
                basis="sto-3g", spin=1, verbose=0)
    mf = scf.UHF(mol).run(conv_tol=1e-14, conv_tol_grad=1e-11)
    ca, cb = mf.mo_coeff
    h = tuple(jnp.asarray(c.T @ mf.get_hcore() @ c) for c in (ca, cb))
    g = tuple(jnp.asarray(ao2mo.general(mol, cs, compact=False).reshape((3,) * 4))
              for cs in ((ca, ca, ca, ca), (ca, ca, cb, cb), (cb, cb, cb, cb)))
    return mf, h, g


def test_namespace_is_public():
    import gradscf
    assert "mp" in gradscf.__all__
    from gradscf import mp
    assert all(hasattr(mp, name) for name in ("MP2", "RMP2", "UMP2", "run_mp"))


@pytest.mark.parametrize("frozen", [None, 1, [0, 18]])
def test_rmp2_matches_pyscf_energy_spin_components_and_amplitudes(water, frozen):
    from pyscf import mp as pmp
    from gradscf.mp import run_mp
    mf, h, g = water
    reference = pmp.MP2(mf, frozen=frozen).run()
    result = run_mp(h, g, nocc=5, frozen=frozen,
                    nuclear_repulsion=mf.mol.energy_nuc())
    assert result.valid
    np.testing.assert_allclose(result.correlation_energy, reference.e_corr, atol=1e-11)
    np.testing.assert_allclose(result.total_energy, reference.e_tot, atol=1e-10)
    np.testing.assert_allclose(result.same_spin_energy, reference.e_corr_ss, atol=1e-11)
    np.testing.assert_allclose(result.opposite_spin_energy, reference.e_corr_os, atol=1e-11)
    np.testing.assert_allclose(result.t2, reference.t2, atol=2e-9)


@pytest.mark.parametrize("frozen", [None, ([0], []), ([], [2])])
def test_ump2_matches_pyscf(radical, frozen):
    from pyscf import mp as pmp
    from gradscf.mp import run_mp
    mf, h, g = radical
    reference = pmp.MP2(mf, frozen=frozen).run()
    result = run_mp(h, g, nocc=(2, 1), frozen=frozen, nuclear_repulsion=mf.mol.energy_nuc())
    assert result.valid
    np.testing.assert_allclose(result.correlation_energy, reference.e_corr, atol=1e-11)
    np.testing.assert_allclose(result.total_energy, reference.e_tot, atol=1e-10)
    for actual, expected in zip(result.t2, reference.t2):
        np.testing.assert_allclose(actual, expected, atol=2e-9)


def test_ump2_gradient_and_hvp(radical):
    from gradscf.mp import run_mp, MPConfig
    mf, h, g = radical
    f = tuple(jnp.diag(jnp.asarray(eps)) for eps in mf.mo_energy)
    interaction = tuple(fs - hs for fs, hs in zip(f, h))
    def energy(scale):
        return run_mp(tuple(fs - scale * v for fs, v in zip(f, interaction)),
                      tuple(scale * x for x in g), nocc=(2, 1),
                      config=MPConfig(with_t2=False)).e2
    value, gradient = jax.jit(jax.value_and_grad(energy))(1.)
    np.testing.assert_allclose(gradient, 2 * value, atol=1e-12)
    np.testing.assert_allclose(jax.jvp(jax.grad(energy), (1.,), (.2,))[1],
                               .4 * value, atol=1e-12)


def test_mp2_energy_only_jit_gradient_and_hvp(water):
    from gradscf.mp import MPConfig, run_mp
    _, h, g = water
    config = MPConfig(with_t2=False)
    # Vary the fluctuation interaction while preserving a diagonal reference F.
    coulomb = 2 * jnp.einsum("pqii->pq", g[:, :, :5, :5])
    exchange = jnp.einsum("piiq->pq", g[:, :5, :5, :])
    f = h + coulomb - exchange
    def energy(scale):
        return run_mp(f - scale * (coulomb - exchange), scale * g,
                      nocc=5, config=config).correlation_energy
    value, derivative = jax.jit(jax.value_and_grad(energy))(1.)
    np.testing.assert_allclose(derivative, 2 * value, atol=1e-11)
    np.testing.assert_allclose(jax.jvp(jax.grad(energy), (1.,), (.3,))[1],
                               .6 * value, atol=1e-11)
    assert run_mp(h, g, nocc=5, config=config).t2 is None


def test_invalid_canonical_and_small_denominator_are_not_regularized(water):
    from gradscf.mp import run_mp
    _, h, g = water
    bad = h.at[0, 5].add(.1).at[5, 0].add(.1)
    result = run_mp(bad, g, nocc=5)
    assert not result.valid and jnp.isnan(result.correlation_energy)
    g0 = jnp.zeros((2,) * 4)
    singular = run_mp(jnp.zeros((2, 2)), g0, nocc=1)
    assert not singular.valid and singular.min_abs_denominator == 0
    assert jnp.isnan(singular.correlation_energy)


def test_no_active_doubles_has_zero_energy():
    from gradscf.mp import run_mp
    h = jnp.diag(jnp.array([-1., .4]))
    g = jnp.zeros((2,) * 4)
    result = run_mp((h, h), (g, g, g), nocc=(1, 0))
    assert result.valid
    assert result.correlation_energy == 0
    assert jnp.isinf(result.min_abs_denominator)


def test_frozen_all_occupied_is_valid_zero(water):
    from gradscf.mp import run_mp
    _, h, g = water
    result = run_mp(h, g, nocc=5, frozen=5)
    assert result.valid and result.correlation_energy == 0


def test_facade_accepts_explicit_reference_and_dispatches_spin(water, radical):
    from gradscf import mp
    from gradscf.scf.reference import RestrictedReference, UnrestrictedReference
    mf, h, g = water
    pt = mp.MP2(RestrictedReference(h, g, 5, mf.mol.energy_nuc())).run()
    assert isinstance(pt, mp.RMP2) and pt.converged
    assert pt.kernel()[0] == pt.e_corr
    umf, uh, ug = radical
    upt = mp.MP2(UnrestrictedReference(uh, ug, (2, 1), umf.mol.energy_nuc())).run()
    assert isinstance(upt, mp.UMP2) and upt.converged
    with pytest.raises(ValueError, match="restricted"):
        mp.RMP2(UnrestrictedReference(uh, ug, (2, 1)))


@pytest.mark.parametrize("kwargs", [{"order": 1}, {"denominator_tol": 0},
                                     {"canonical_tol": float("nan")}])
def test_invalid_configuration(kwargs):
    from gradscf.mp import MPConfig
    with pytest.raises(ValueError):
        MPConfig(**kwargs)
