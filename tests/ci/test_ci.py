"""Independent molecular references for the restricted CI hierarchy (CPU/x64)."""
import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest


@pytest.fixture(scope="module")
def h4():
    pyscf = pytest.importorskip("pyscf")
    from pyscf import ao2mo, fci

    pyscf.lib.num_threads(1)
    mol = pyscf.gto.M(atom="H 0 0 0; H 0 0 0.8; H 0 0 1.9; H 0 0 3.1",
                      basis="sto-3g", unit="Angstrom", verbose=0)
    mf = mol.RHF().run(conv_tol=1e-12)
    assert mf.converged
    h = mf.mo_coeff.T @ mf.get_hcore() @ mf.mo_coeff
    g = ao2mo.restore(1, ao2mo.kernel(mol, mf.mo_coeff), 4)
    strings = fci.cistring.make_strings(range(4), 2)
    dets = [int(a) | (int(b) << 4) for a in strings for b in strings]
    h2 = fci.direct_spin1.absorb_h1e(h, g, 4, (2, 2), 0.5)
    eye = np.eye(len(dets))
    matrix = np.column_stack([
        fci.direct_spin1.contract_2e(h2, v.reshape(6, 6), 4, (2, 2)).ravel()
        for v in eye
    ])
    return mf, h, g, dets, matrix


def test_public_ci_namespace():
    import gradscf

    assert "ci" in dir(gradscf)
    from gradscf import ci
    assert all(hasattr(ci, name) for name in ("CIS", "CISD", "CISDT", "CISDTQ", "CI"))


@pytest.mark.parametrize("rank", [1, 2, 3, 4])
def test_hamiltonian_matches_projected_pyscf_fci(h4, rank):
    from gradscf.ci import make_ci_space, hamiltonian_matrix, hamiltonian_action

    _, h, g, dets, full = h4
    space = make_ci_space(4, 2, max_excitation=rank)
    idx = [dets.index(d) for d in space.determinants]
    expected = full[np.ix_(idx, idx)]
    actual = hamiltonian_matrix(jnp.asarray(h), jnp.asarray(g), space)
    np.testing.assert_allclose(actual, expected, atol=2e-12)
    vectors = jnp.asarray(np.random.default_rng(12).normal(size=(len(idx), 2)))
    action = jax.jit(lambda x: hamiltonian_action(h, g, space, x))(vectors)
    np.testing.assert_allclose(action, expected @ vectors, atol=2e-12)


def test_space_limits_and_frozen_orbitals(h4):
    from gradscf.ci import make_ci_space, hamiltonian_matrix

    _, h, g, dets, full = h4
    space = make_ci_space(4, 2, max_excitation=4, frozen=[0, 3])
    assert len(space.determinants) == 4
    idx = [dets.index(d) for d in space.determinants]
    np.testing.assert_allclose(hamiltonian_matrix(h, g, space), full[np.ix_(idx, idx)], atol=2e-12)
    with pytest.raises(ValueError, match="max_determinants"):
        make_ci_space(30, 15, max_excitation=4, max_determinants=100)
    for frozen in ([-1], [4], [0, 0]):
        with pytest.raises(ValueError):
            make_ci_space(4, 2, frozen=frozen)


@pytest.mark.parametrize("solver", ["dense", "davidson"])
def test_cisd_matches_pyscf_and_result_is_jittable(h4, solver):
    from gradscf.ci import CIConfig, make_ci_space, solve_ci

    mf, h, g, _, _ = h4
    space = make_ci_space(4, 2, max_excitation=2)
    cfg = CIConfig(solver=solver, conv_tol=1e-10)
    result = jax.jit(lambda x, y: solve_ci(x, y, space, nuclear_repulsion=mf.mol.energy_nuc(), config=cfg))(h, g)
    ref = mf.CISD().run(conv_tol=1e-12)
    np.testing.assert_allclose(result.total_energies[0], ref.e_tot, atol=2e-9)
    np.testing.assert_allclose(result.correlation_energies[0], ref.e_corr, atol=2e-9)
    assert result.converged.shape == (1,)
    assert bool(result.converged[0])
    assert float(result.residual_norms[0]) < 1e-9


def test_hierarchy_variational_and_full_ci_limit(h4):
    from gradscf.ci import CIConfig, make_ci_space, solve_ci

    _, h, g, _, full = h4
    energies = [float(solve_ci(h, g, make_ci_space(4, 2, max_excitation=k),
                              config=CIConfig(solver="dense")).total_energies[0])
                for k in range(1, 5)]
    assert np.all(np.diff(energies) <= 1e-12)
    np.testing.assert_allclose(energies[-1], np.linalg.eigvalsh(full)[0], atol=2e-12)
    # Triples and quadruples must actually be present, not aliases of CISD.
    assert energies[1] - energies[-1] > 1e-6


@pytest.mark.parametrize("singlet", [True, False])
def test_cis_matches_pyscf_tda(h4, singlet):
    from gradscf.ci import CIConfig, solve_cis

    mf, h, g, _, _ = h4
    result = solve_cis(h, g, nocc=2, singlet=singlet,
                       config=CIConfig(nroots=3, solver="dense"))
    td = mf.TDA().set(singlet=singlet, nstates=3, conv_tol=1e-11).run()
    np.testing.assert_allclose(result.excitation_energies, td.e, atol=2e-9)
    np.testing.assert_allclose(np.sum(np.asarray(result.amplitudes)**2, axis=(1, 2)), 1.0, atol=1e-12)


@pytest.mark.parametrize("solver", ["dense", "davidson"])
def test_ci_integral_gradient_matches_finite_difference(h4, solver):
    from gradscf.ci import CIConfig, make_ci_space, solve_ci

    _, h, g, _, _ = h4
    space = make_ci_space(4, 2)
    cfg = CIConfig(solver=solver, conv_tol=1e-11)
    direction = np.random.default_rng(9).normal(size=h.shape)
    direction = (direction + direction.T) / 2

    def energy(t):
        return solve_ci(h + t * direction, g * (1 + 0.03 * t), space, config=cfg).total_energies[0]

    ad = jax.jit(jax.grad(energy))(0.)
    fd = (energy(1e-4) - energy(-1e-4)) / 2e-4
    assert np.isfinite(np.asarray(ad)).all() and np.isfinite(np.asarray(fd)).all()
    if os.environ.get('GRADSCF_TEST_GPU') == '1':
        assert ad.devices() == {jax.devices('gpu')[0]}
    np.testing.assert_allclose(ad, fd, atol=2e-7, rtol=2e-7, equal_nan=False)


def test_facade_frozen_cisd_and_high_rank(h4):
    from gradscf import ci

    mf, h, g, _, _ = h4
    reference = ci.CIReference(h, g, nocc=2, nuclear_repulsion=mf.mol.energy_nuc(),
                                mo_energy=mf.mo_energy)
    obj = ci.CISD(reference, frozen=1, solver="dense").run()
    expected = mf.CISD(frozen=1).run(conv_tol=1e-12)
    np.testing.assert_allclose(obj.e_tot, expected.e_tot, atol=2e-9)
    e, coeff = obj.kernel()
    assert e.shape == () and coeff.ndim == 1
    assert obj.result.coefficients.shape == (coeff.size, 1)
    assert ci.CISDT(reference, solver="dense").run().e_tot >= ci.CISDTQ(reference, solver="dense").run().e_tot - 1e-12


def test_nonconvergence_and_invalid_requests(h4):
    from gradscf.ci import CIConfig, make_ci_space, solve_ci

    _, h, g, _, _ = h4
    space = make_ci_space(4, 2)
    result = solve_ci(h, g, space, config=CIConfig(solver="davidson", max_cycle=1, conv_tol=1e-14))
    assert not bool(result.converged[0])
    with pytest.raises(ValueError, match="nroots"):
        solve_ci(h, g, space, config=CIConfig(nroots=999))
    with pytest.raises((ValueError, NotImplementedError), match="real"):
        solve_ci(h.astype(complex), g, space)


def test_mo_transform_layouts_and_derivative(h4):
    from gradscf.ao2mo import transform_integrals

    mf, h, g, _, _ = h4
    ao = mf.mol.intor("int2e")
    rows, cols = np.tril_indices(4)
    pair = ao[rows[:, None], cols[:, None], rows[None, :], cols[None, :]]
    factors = np.linalg.cholesky(pair)
    unpacked = np.zeros((factors.shape[1], 4, 4))
    unpacked[:, rows, cols] = factors.T
    unpacked[:, cols, rows] = factors.T
    layouts = [{"eri": ao}, {"eri_pair_matrix": pair}, {"df_factors": unpacked}]
    for layout in layouts:
        hm, gm = transform_integrals(mf.get_hcore(), mf.mo_coeff, **layout)
        np.testing.assert_allclose(hm, h, atol=1e-12)
        np.testing.assert_allclose(gm, g, atol=1e-12)
    direction = jnp.asarray(np.random.default_rng(2).normal(size=(4, 4)))
    def value(t):
        hm, gm = transform_integrals(mf.get_hcore(), mf.mo_coeff + t * direction, eri_pair_matrix=pair)
        return jnp.sum(hm**2) + jnp.sum(gm**2)
    np.testing.assert_allclose(jax.jit(jax.grad(value))(0.), (value(1e-5)-value(-1e-5))/2e-5, rtol=1e-7)


def test_gradscf_hf_facade_and_rejected_references():
    from gradscf import ci, dft, gto

    mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g")
    mf = dft.RKS(mol, xc="hf", conv_tol=1e-11)
    with pytest.raises(RuntimeError, match="SCF"):
        ci.CISD(mf).kernel()
    mf.run()
    obj = mf.CISD(solver="dense").run()
    pyscf = pytest.importorskip("pyscf")
    pmf = pyscf.gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g", verbose=0).RHF().run()
    np.testing.assert_allclose(obj.e_tot, pmf.CISD().run().e_tot, atol=1e-8)
    assert mf.CIS(nroots=1).run().e.shape == (1,)
    with pytest.raises((ValueError, NotImplementedError), match="HF"):
        ci.CISD(dft.RKS(mol, xc="pbe")).kernel()


def test_implicit_ci_coefficient_response(h4):
    from gradscf.ci import CIConfig, make_ci_space, solve_ci

    _, h, g, _, _ = h4
    space = make_ci_space(4, 2)
    cfg = CIConfig(solver="davidson", gradient_mode="implicit_eigenvector", conv_tol=1e-11)
    def reference_weight(t):
        c = solve_ci(h, g * (1 + t), space, config=cfg).coefficients[:, 0]
        return c[0]**2
    ad = jax.jit(jax.grad(reference_weight))(0.)
    fd = (reference_weight(1e-4)-reference_weight(-1e-4))/2e-4
    assert np.isfinite(np.asarray(ad)).all() and np.isfinite(np.asarray(fd)).all()
    if os.environ.get('GRADSCF_TEST_GPU') == '1':
        assert ad.devices() == {jax.devices('gpu')[0]}
    np.testing.assert_allclose(ad, fd, atol=2e-7, rtol=2e-6, equal_nan=False)


def test_unconverged_eigenvalue_derivative_is_not_silently_zero(h4):
    from gradscf.ci import CIConfig, make_ci_space, solve_ci

    _, h, g, _, _ = h4
    space = make_ci_space(4, 2)
    cfg = CIConfig(max_cycle=1, conv_tol=1e-14)
    def energy(t):
        return solve_ci(h, g * (1 + t), space, config=cfg).total_energies[0]
    assert np.isfinite(energy(0.))
    assert not np.isfinite(jax.grad(energy)(0.))


@pytest.mark.parametrize("frozen", [None, 1, [0, 5]])
def test_lih_cisd_and_frozen_virtuals(frozen):
    pyscf = pytest.importorskip("pyscf")
    from pyscf import ao2mo
    from gradscf import ci

    mol = pyscf.gto.M(atom="Li 0 0 0; H 0 0 1.6", basis="sto-3g", verbose=0)
    mf = mol.RHF().run(conv_tol=1e-12)
    h = mf.mo_coeff.T @ mf.get_hcore() @ mf.mo_coeff
    g = ao2mo.restore(1, ao2mo.kernel(mol, mf.mo_coeff), h.shape[0])
    ref = ci.CIReference(h, g, 2, mol.energy_nuc())
    obj = ci.CISD(ref, frozen=frozen, conv_tol=1e-10).run()
    expected = mf.CISD(frozen=frozen).run(conv_tol=1e-12)
    assert obj.converged
    np.testing.assert_allclose(obj.e_tot, expected.e_tot, atol=2e-9)
    np.testing.assert_allclose(obj.e_corr, expected.e_corr, atol=2e-9)


def test_multiroot_davidson_and_orbital_rotation(h4):
    from gradscf.ci import CIConfig, make_ci_space, solve_ci
    from gradscf.ao2mo import transform_integrals

    _, h, g, _, _ = h4
    space = make_ci_space(4, 2)
    cfg = CIConfig(nroots=3, conv_tol=1e-10)
    expected = solve_ci(h, g, space, config=CIConfig(nroots=3, solver="dense"))
    result = solve_ci(h, g, space, config=cfg)
    np.testing.assert_allclose(result.total_energies, expected.total_energies, atol=2e-10)
    assert np.all(result.converged)
    rotation = np.zeros((4, 4))
    rotation[:2, :2] = np.linalg.qr(np.random.default_rng(3).normal(size=(2, 2)))[0]
    rotation[2:, 2:] = np.linalg.qr(np.random.default_rng(4).normal(size=(2, 2)))[0]
    hr, gr = transform_integrals(h, rotation, eri=g)
    rotated = solve_ci(hr, gr, space, config=cfg)
    np.testing.assert_allclose(rotated.total_energies, expected.total_energies, atol=2e-10)


def test_cis_facade_rejects_nonstationary_hf_reference(h4):
    from gradscf import ci

    _, h, g, _, _ = h4
    broken_h = h.copy()
    broken_h[0, 3] += 0.1
    broken_h[3, 0] += 0.1
    with pytest.raises(ValueError, match="stationary"):
        ci.CIS(ci.CIReference(broken_h, g, 2)).run()


def test_unconverged_coefficient_response_is_invalid(h4):
    from gradscf.ci import CIConfig, make_ci_space, solve_ci

    _, h, g, _, _ = h4
    space = make_ci_space(4, 2)
    cfg = CIConfig(max_cycle=1, conv_tol=1e-14, gradient_mode="implicit_eigenvector")
    def weight(t):
        c = solve_ci(h, g * (1+t), space, config=cfg).coefficients
        return c[0, 0]**2
    assert not np.isfinite(jax.grad(weight)(0.))
