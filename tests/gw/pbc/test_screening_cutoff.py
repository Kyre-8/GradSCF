"""Independent dielectric cutoffs must never truncate bare Hartree/exchange."""
from dataclasses import replace
import numpy as np
import jax.numpy as jnp


def _model():
    from gradscf.gw.pbc.source import PeriodicGWModel
    energy = jnp.array([[-.4, .3]])
    bare = jnp.array([[[[.3, .12+.05j], [.12-.05j, .2]],
                        [[.2, -.04+.03j], [-.04-.03j, .1]]]])
    return PeriodicGWModel(energy=energy, coeff=jnp.eye(2)[None],
        hcore=jnp.diag(energy[0])[None], reference_fock=jnp.diag(energy[0])[None],
        reference_potential=jnp.zeros((1, 2, 2)), nocc=1,
        kpoints=np.zeros((1, 3)), lattice=np.eye(3), table=np.array([[0]]),
        inverse=np.array([[0]]), factors=(bare,), metadata={'fc': False})


def test_independent_screening_factors_change_w_but_preserve_bare_j_and_k():
    from gradscf.gw.pbc.source import build_correlation, hartree_exchange
    bare = _model()
    low = (bare.factors[0][:, :1],)
    separate = replace(bare, screening_factors=low)
    expected = replace(bare, factors=low)
    rho = jnp.array([[[.8, .05+.03j], [.05-.03j, .2]]])
    for a, b in zip(hartree_exchange(bare, rho), hartree_exchange(separate, rho)):
        np.testing.assert_allclose(a, b, atol=0, rtol=0)
    full = build_correlation(bare, bare.energy, nw=8, eta=.001)
    actual = build_correlation(separate, bare.energy, nw=8, eta=.001)
    reference = build_correlation(expected, bare.energy, nw=8, eta=.001)
    np.testing.assert_allclose(actual['positive'][0], reference['positive'][0], atol=1e-14)
    assert np.max(abs(np.asarray(actual['positive'][0]-full['positive'][0]))) > 1e-7


def test_matsubara_uses_dielectric_factors_without_altering_static_interaction():
    from gradscf.gw.pbc.matsubara import matsubara_grid, dyson_green_and_density, gw_matsubara_step
    bare = _model()
    low = (bare.factors[0][:, :1],)
    separate = replace(bare, screening_factors=low)
    expected = replace(bare, factors=low)
    grid = matsubara_grid(nw=8, beta=12.)
    sigma = jnp.zeros((1, 16, 2, 2), dtype=jnp.complex128)
    green, density = dyson_green_and_density(bare.reference_fock, sigma, 0., grid)
    actual = gw_matsubara_step(green, bare.reference_fock, 0., separate, grid, density=density)
    reference = gw_matsubara_step(green, bare.reference_fock, 0., expected, grid, density=density)
    np.testing.assert_allclose(actual['sigma_iw'], reference['sigma_iw'], atol=1e-13)
    np.testing.assert_allclose(actual['sigma_moment'], reference['sigma_moment'], atol=1e-13)


def test_native_dielectric_cutoff_matches_an_independent_shifted_g_mask():
    from gradscf.pbc import dft, gto
    from gradscf.gw.pbc.source import build_periodic_model, hartree_exchange
    from gradscf.gw.pbc.product_basis import kpoint_product_factor_builder
    cell = gto.M(atom='H 2 2 1.3; H 2 2 2.7', a=np.eye(3)*5., unit='Bohr',
        basis='gth-szv', pseudo='gth-pbe', mesh=(5, 5, 5), precision=1e-10)
    mf = dft.KRKS(cell, kpts=cell.make_kpts((2, 1, 1)), xc='pbe', exxdiv=None,
        temperature=None, conv_tol=1e-11, conv_tol_density=1e-10,
        conv_tol_grad=1e-9, max_cycle=150).run()
    assert mf.converged
    full = build_periodic_model(mf)
    cut = build_periodic_model(mf, screening_cutoff_rydberg=1.5)
    for left, right in zip(hartree_exchange(full, full.reference_density),
                           hartree_exchange(cut, cut.reference_density)):
        np.testing.assert_allclose(left, right, atol=2e-12, rtol=0)
    pairs = np.array([(m, n) for m in range(2) for n in range(2)])
    build = kpoint_product_factor_builder(mf.inputs, cut.coeff, mesh=cell.mesh,
                                          momentum_table=cut.table)
    for q, factor in enumerate(cut.correlation_factors):
        mask = np.sum((np.asarray(mf.inputs.gvectors)-np.asarray(mf.kpts)[q])**2, axis=1) <= 1.5
        raw = np.asarray(build(q, pairs=pairs))[:, mask].transpose(1, 0, 2).reshape(mask.sum(), -1)
        actual = np.asarray(factor).transpose(1, 0, 2, 3).reshape(factor.shape[1], -1)
        np.testing.assert_allclose(actual.conj().T@actual, raw.conj().T@raw, atol=2e-12, rtol=1e-11)
        assert cut.metadata['screening_compression'][q]['plane_waves'] == int(mask.sum())
