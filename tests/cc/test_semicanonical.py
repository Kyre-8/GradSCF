"""General real-reference triples, explicit semicanonical PySCF oracles."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest


@pytest.fixture(scope="module")
def oh_rohf():
    from pyscf import gto, ao2mo
    mf = gto.M(atom="O 0 0 0; H 0 0 .97", basis="sto-3g", spin=1,
                verbose=0).ROHF().run(conv_tol=1e-13)
    c = mf.mo_coeff
    n = c.shape[1]
    h = c.T @ mf.get_hcore() @ c
    g = ao2mo.restore(1, ao2mo.kernel(mf.mol, c), n)
    return mf, (jnp.asarray(h),)*2, (jnp.asarray(g),)*3


def _pyscf_semicanonical(mf, frozen):
    """Diagonalize separate alpha/beta active oo and vv Fock blocks in NumPy."""
    from pyscf import cc, scf
    from gradscf.ao2mo import unrestricted_frozen_indices
    reference = scf.addons.convert_to_uhf(mf)
    coeff = np.asarray(reference.mo_coeff).copy()
    fock = reference.get_fock()
    counts = reference.mol.nelec
    sets = unrestricted_frozen_indices(coeff.shape[2], counts, frozen)
    for spin, (no, fr) in enumerate(zip(counts, sets)):
        f = coeff[spin].T @ fock[spin] @ coeff[spin]
        for indices in ([i for i in range(no) if i not in fr],
                        [i for i in range(no, f.shape[0]) if i not in fr]):
            _, rotation = np.linalg.eigh(f[np.ix_(indices, indices)])
            coeff[spin][:, indices] = coeff[spin][:, indices] @ rotation
    reference.mo_coeff = coeff
    obj = cc.UCCSD(reference, frozen=frozen).set(conv_tol=1e-12, conv_tol_normt=1e-11,
                                                max_cycle=150).run()
    assert obj.converged
    return obj


@pytest.mark.parametrize("frozen", [None, 1, ([0], [])])
def test_rohf_triples_against_explicit_semicanonical_oracle(oh_rohf, frozen):
    from gradscf import cc
    mf, h, g = oh_rohf
    obj = cc.UCCSD(cc.UnrestrictedReference(h, g, (5, 4)), frozen=frozen,
                   conv_tol=1e-12, residual_tol=1e-10).run()
    reference = _pyscf_semicanonical(mf, frozen)
    np.testing.assert_allclose(obj.e_tot+mf.mol.energy_nuc(), reference.e_tot, atol=2e-9)
    with pytest.raises(ValueError, match="canonical"):
        obj.ccsd_t()
    actual = obj.triples(orbital_basis="semicanonical")
    assert actual.valid
    np.testing.assert_allclose(actual.energy, reference.ccsd_t(), atol=2e-10, rtol=0)
    assert abs(actual.energy) > 1e-8
    np.testing.assert_allclose(actual.energy, actual.connected_component+actual.singles_component,
                               atol=1e-14, rtol=0)


def test_semicanonical_closed_shell_and_jit_response():
    from gradscf import cc
    from pyscf import gto, ao2mo
    mf = gto.M(atom="H 0 0 0; H 0 0 .8; H 0 0 1.9; H 0 0 3.1",
                basis="sto-3g", verbose=0).RHF().run(conv_tol=1e-13)
    c = mf.mo_coeff
    h = jnp.asarray(c.T @ mf.get_hcore() @ c)
    g = jnp.asarray(ao2mo.restore(1, ao2mo.kernel(mf.mol, c), 4))
    cfg = cc.CCConfig(conv_tol=1e-12, residual_tol=1e-11)
    state = cc.run_cc(h, g, nocc=2, config=cfg)
    np.testing.assert_allclose(cc.triples_correction(h, g, state, nocc=2,
        orbital_basis="semicanonical"), cc.triples_correction(h, g, state, nocc=2), atol=2e-11)
    direction = jnp.array([[.1, .07, .03, -.02], [.07, -.05, .02, .04],
                          [.03, .02, .08, -.06], [-.02, .04, -.06, -.03]])
    def energy(x):
        hs = h+x*direction
        out = cc.run_cc(hs, g, nocc=2, config=cfg)
        return out.total_energy+cc.triples_correction(hs, g, out, nocc=2,
                                                     orbital_basis="semicanonical")
    derivative = jax.jit(jax.grad(energy))(0.)
    np.testing.assert_allclose(derivative, (energy(1e-4)-energy(-1e-4))/2e-4,
                               atol=2e-7, rtol=0)
    with pytest.raises(ValueError, match="max_triples_elements"):
        cc.triples_correction(h, g, state, nocc=2, orbital_basis="semicanonical",
                               max_triples_elements=1)


def test_semicanonical_orbital_rotation_invariance(oh_rohf):
    from gradscf import cc
    from scipy.linalg import expm
    _, h, g = oh_rohf
    counts, n = (5, 4), 6
    cfg = cc.CCConfig(conv_tol=1e-12, residual_tol=1e-10)
    state = cc.run_ucc(h, g, nocc=counts, config=cfg)
    assert state.converged
    expected = cc.triples_correction(h, g, state, nocc=counts, orbital_basis="semicanonical")
    rng = np.random.default_rng(51)
    rotations = []
    for no in counts:
        q = np.eye(n)
        for start, stop in ((0, no), (no, n)):
            x = rng.normal(size=(stop-start, stop-start))*.3
            q[start:stop, start:stop] = expm(x-x.T)
        rotations.append(q)
    hs = tuple(q.T @ a @ q for a, q in zip(h, rotations))
    transform = lambda a, p, q, r, s: np.einsum('pP,qQ,rR,sS,pqrs->PQRS', p, q, r, s, a, optimize=True)
    qa, qb = rotations
    gs = (transform(g[0], qa, qa, qa, qa), transform(g[1], qa, qa, qb, qb),
          transform(g[2], qb, qb, qb, qb))
    qo = tuple(q[:no, :no] for q, no in zip(rotations, counts))
    qv = tuple(q[no:, no:] for q, no in zip(rotations, counts))
    t1 = tuple(o.T @ t @ v for o, t, v in zip(qo, state.t1, qv))
    t2 = (transform(state.t2[0], qo[0], qo[0], qv[0], qv[0]),
          transform(state.t2[1], qo[0], qo[1], qv[0], qv[1]),
          transform(state.t2[2], qo[1], qo[1], qv[1], qv[1]))
    moved = state._replace(t1=tuple(map(jnp.asarray, t1)), t2=tuple(map(jnp.asarray, t2)))
    out = cc.evaluate_triples(hs, gs, moved, nocc=counts, orbital_basis="semicanonical")
    assert out.valid
    np.testing.assert_allclose(out.energy, expected, atol=2e-12, rtol=0)


@pytest.mark.parametrize("algorithm", ["auto", "streamed"])
def test_semicanonical_cc_response_at_exact_orbital_degeneracy(algorithm):
    from gradscf import cc
    from gradscf.ao2mo import spin_orbital_integrals
    rng = np.random.default_rng(52)
    b = rng.normal(size=(5, 4, 4))*.06
    b += b.transpose(0, 2, 1)
    g0 = jnp.asarray(np.einsum('Lpq,Lrs->pqrs', b, b))
    g = (g0,)*3
    zeros = (jnp.zeros((4, 4)),)*2
    _, full = spin_orbital_integrals(zeros, g)
    occ = jnp.array([0, 1, 4])
    interaction = full[:, :, occ, occ].sum(-1)-full[:, occ, occ, :].sum(1)
    h = (jnp.diag(jnp.array([-1., -1., .5, .5]))-interaction[:4, :4],
         jnp.diag(jnp.array([-.8, .3, .3, .8]))-interaction[4:, 4:])
    direction = jnp.array([[.03, .07, .01, 0.], [.07, -.05, 0., -.02],
                           [.01, 0., -.04, .09], [0., -.02, .09, .02]])
    cfg = cc.CCConfig(conv_tol=1e-12, residual_tol=1e-11)
    def correction(x):
        hs = (h[0]+x*direction, h[1])
        state = cc.run_ucc(hs, g, nocc=(2, 1), config=cfg)
        return cc.triples_correction(hs, g, state, nocc=(2, 1), orbital_basis="semicanonical",
                                     algorithm=algorithm)
    value, derivative = jax.jit(jax.value_and_grad(correction))(0.)
    assert abs(value) > 1e-9 and np.isfinite(derivative)
    np.testing.assert_allclose(derivative, (correction(1e-4)-correction(-1e-4))/2e-4,
                               atol=2e-9, rtol=0)
    curvature = jax.jit(jax.grad(jax.grad(correction)))(0.)
    gradient = jax.jit(jax.grad(correction))
    assert np.isfinite(curvature)
    for step in (3e-4, 1e-4):
        finite = (gradient(step)-gradient(-step))/(2*step)
        np.testing.assert_allclose(curvature, finite, atol=2e-9, rtol=2e-5)


def test_semicanonical_empty_space_and_invalid_state():
    from gradscf import cc
    h = (jnp.diag(jnp.array([-1., .4])),)*2
    g = (jnp.zeros((2,)*4),)*3
    state = cc.run_ucc(h, g, nocc=(1, 0))
    out = cc.evaluate_triples(h, g, state, nocc=(1, 0), orbital_basis="semicanonical")
    assert out.valid and out.energy == 0
    bad = state._replace(converged=jnp.array(False))
    assert not cc.evaluate_triples(h, g, bad, nocc=(1, 0), orbital_basis="semicanonical").valid
    with pytest.raises(ValueError, match="orbital_basis"):
        cc.triples_correction(h, g, state, nocc=(1, 0), orbital_basis="diagonal-only")
    with pytest.raises(NotImplementedError, match="CCSD"):
        cc.triples_correction(h, g, state, nocc=(1, 0), orbital_basis="semicanonical",
                               variant="ccsd+t(ccsd)")
