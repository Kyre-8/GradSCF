"""Open-shell CI: real UHF/ROHF, CPU float64, energies in Hartree."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest


@pytest.mark.parametrize("rank", [1, 2, 3])
def test_unrestricted_hamiltonian_against_fci(radical, rank):
    from gradscf import ci
    from pyscf.fci import direct_uhf, cistring
    mf, h, g = radical
    space = ci.make_uci_space(3, (2, 1), max_excitation=rank)
    aa, bb = cistring.make_strings(range(3), 2), cistring.make_strings(range(3), 1)
    dets = [int(a) | (int(b) << 3) for a in aa for b in bb]
    eff = direct_uhf.absorb_h1e(tuple(map(np.asarray, h)), tuple(map(np.asarray, g)), 3, (2, 1), .5)
    eye = np.eye(len(dets))
    full = np.column_stack([direct_uhf.contract_2e(eff, v.reshape(len(aa), len(bb)), 3, (2, 1)).ravel() for v in eye])
    idx = [dets.index(d) for d in space.determinants]
    actual = jax.jit(lambda h, g: ci.hamiltonian_matrix(h, g, space))(h, g)
    np.testing.assert_allclose(actual, full[np.ix_(idx, idx)], atol=2e-12)


@pytest.mark.parametrize("solver", ["dense", "davidson"])
def test_ucisd_energy_and_gradient(radical, solver):
    from gradscf import ci
    from gradscf.scf.reference import UnrestrictedReference
    mf, h, g = radical
    obj = ci.UCISD(UnrestrictedReference(h, g, (2, 1), mf.mol.energy_nuc()),
                   solver=solver, conv_tol=1e-11).run()
    ref = mf.CISD().set(conv_tol=1e-12).run()
    assert obj.converged
    np.testing.assert_allclose(obj.e_tot, ref.e_tot, atol=2e-9)
    direction = jnp.diag(jnp.array([.2, -.1, .07]))
    def energy(x):
        return ci.solve_ci((h[0]+x*direction, h[1]), g, obj.space,
                            config=ci.CIConfig(solver=solver, conv_tol=1e-11)).total_energies[0]
    np.testing.assert_allclose(jax.jit(jax.grad(energy))(0.),
                               (energy(1e-4)-energy(-1e-4))/2e-4, atol=2e-8)


def test_unrestricted_transform_layouts(radical):
    from gradscf.ao2mo import transform_unrestricted_integrals
    from pyscf import ao2mo
    mf, h, g = radical
    ao = mf.mol.intor("int2e")
    for kwargs in ({"eri": ao}, {"eri_pair_matrix": ao2mo.restore(4, ao, 3)}):
        hh, gg = transform_unrestricted_integrals(mf.get_hcore(), mf.mo_coeff, **kwargs)
        for actual, expected in zip((*hh, *gg), (*h, *g)):
            np.testing.assert_allclose(actual, expected, atol=2e-12)

    rng = np.random.default_rng(51)
    factors = rng.normal(size=(5, 3, 3))
    factors = .5*(factors+factors.transpose(0, 2, 1))
    dense = np.einsum("Lpq,Lrs->pqrs", factors, factors)
    ca, cb = map(jnp.asarray, mf.mo_coeff)
    def value(x, df):
        coeff = (ca+x*jnp.eye(3)*.01, cb)
        _, gg = transform_unrestricted_integrals(mf.get_hcore(), coeff,
                      **({"df_factors": factors} if df else {"eri": dense}))
        return sum(jnp.sum(a*a) for a in gg)
    np.testing.assert_allclose(value(0., True), value(0., False), atol=1e-10)
    np.testing.assert_allclose(jax.grad(lambda x: value(x, True))(0.),
                               (value(1e-4, False)-value(-1e-4, False))/2e-4, rtol=1e-7)


def test_frozen_spaces_and_polarized_sector():
    from gradscf import ci
    space = ci.make_uci_space(4, (2, 1), frozen=([0], [0, 3]))
    assert all(d & 1 and d & (1 << 4) and not d & (1 << 7) for d in space.determinants)
    assert ci.make_uci_space(3, (1, 0)).size == 3
    with pytest.raises(ValueError, match="max_determinants"):
        ci.make_uci_space(30, (15, 14), max_determinants=5)


def test_ucis_against_uhf_tda(radical):
    from gradscf import ci
    mf, h, g = radical
    ref = mf.TDA().set(nstates=2, conv_tol=1e-11).run()
    obj = ci.UCIS(ci.UnrestrictedReference(h, g, (2, 1)), nroots=2,
                  conv_tol=1e-11).run()
    np.testing.assert_allclose(obj.e, ref.e, atol=2e-9)
    assert tuple(a.shape for a in obj.amplitudes) == ((2, 2, 1), (2, 1, 2))
    source = ci.UnrestrictedReference(h, g, (2, 1))
    np.testing.assert_allclose(ci.CIS(source, nroots=2).run().e, obj.e, atol=2e-9)
    with pytest.raises(ValueError, match="singlet/triplet"):
        ci.CIS(source, singlet=False)
    with pytest.raises(NotImplementedError, match="closed-shell"):
        ci.CIS_D(source)


def test_uci_nonempty_triples_against_full_fci():
    from gradscf import ci
    from gradscf.ao2mo import transform_unrestricted_integrals
    from pyscf.fci import direct_uhf, cistring
    rng = np.random.default_rng(32)
    factors = rng.normal(size=(6, 4, 4))*.1
    factors = .5*(factors+factors.transpose(0, 2, 1))
    cb = np.linalg.qr(rng.normal(size=(4, 4)))[0]
    h, g = transform_unrestricted_integrals(np.diag([-1., -.4, .3, .8]),
                                             (np.eye(4), cb), df_factors=factors)
    aa, bb = cistring.make_strings(range(4), 2), cistring.make_strings(range(4), 1)
    dets = [int(a) | (int(b) << 4) for a in aa for b in bb]
    eff = direct_uhf.absorb_h1e(tuple(map(np.asarray, h)), tuple(map(np.asarray, g)), 4, (2, 1), .5)
    full = np.column_stack([direct_uhf.contract_2e(eff, v.reshape(6, 4), 4, (2, 1)).ravel()
                            for v in np.eye(24)])
    energies = []
    for rank in (2, 3, 4):
        space = ci.make_uci_space(4, (2, 1), max_excitation=rank)
        idx = [dets.index(d) for d in space.determinants]
        np.testing.assert_allclose(ci.hamiltonian_matrix(h, g, space), full[np.ix_(idx, idx)], atol=2e-12)
        energies.append(float(ci.solve_ci(h, g, space).total_energies[0]))
        if rank == 3:
            assert 3 in space.ranks and space.size == 24
    assert energies[0] > energies[1]+1e-8
    np.testing.assert_allclose(energies[1:], np.linalg.eigvalsh(full)[0], atol=2e-10)


def test_uci_frozen_and_coefficient_response(radical):
    from gradscf import ci
    mf, h, g = radical
    obj = ci.UCISD(ci.UnrestrictedReference(h, g, (2, 1), mf.mol.energy_nuc()),
                   frozen=1, solver="dense").run()
    oracle = mf.CISD(frozen=1).set(conv_tol=1e-12).run()
    np.testing.assert_allclose(obj.e_tot, oracle.e_tot, atol=2e-9)
    space = ci.make_uci_space(3, (2, 1))
    cfg = ci.CIConfig(gradient_mode="implicit_eigenvector", conv_tol=1e-11)
    def value(x):
        out = ci.solve_ci(h, (g[0], x*g[1], g[2]), space, config=cfg)
        return out.total_energies[0]+.2*jnp.sum(out.coefficients[:, 0]**4)
    np.testing.assert_allclose(jax.jit(jax.grad(value))(1.),
                               (value(1.+1e-4)-value(1.-1e-4))/2e-4, atol=3e-8)


@pytest.mark.parametrize("kind", ["UHF", "ROHF"])
def test_gradscf_open_shell_facade_and_stale_inputs(kind):
    from gradscf import ci, cc, gto, scf
    from gradscf.scf.reference import unrestricted_reference_from_source
    mol = gto.M(atom="H 0 0 0; H 0 0 .85; H 0 0 1.9", basis="sto-3g", spin=1)
    mf = getattr(scf, kind)(mol, conv_tol=1e-11, max_cycle=150)
    with pytest.raises(RuntimeError, match="SCF"):
        ci.UCISD(mf).kernel()
    mf.run()
    ref = unrestricted_reference_from_source(mf)
    assert ref.nocc == (2, 1)
    if kind == "UHF":
        # Post-HF freshness bookkeeping must preserve the existing TDA adapter.
        assert mf._ensure_reference() is mf.reference
    obj = mf.CISD(solver="dense").run()
    coupled = mf.CCSD(residual_tol=1e-10).run()
    assert obj.converged and coupled.converged
    oracle = ci.UCI(ref, max_excitation=3, solver="dense").run()
    # Three electrons: CISD is not assumed equal to FCI or CCSD.
    assert float(obj.e_tot) >= float(oracle.e_tot)-1e-10
    from dataclasses import replace
    mf.mol = replace(mf.mol, spin=-1)
    with pytest.raises(RuntimeError, match="SCF inputs changed"):
        ci.UCISD(mf).kernel()
