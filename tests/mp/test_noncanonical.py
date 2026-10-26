"""MP2 response in fixed occupied/virtual frames, including exact MO degeneracy."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest


def _model(unrestricted=False, *, degenerate=False):
    rng = np.random.default_rng(92)
    factors = rng.normal(size=(5, 4, 4))*.06
    factors = .5*(factors+factors.swapaxes(1, 2))
    g = np.einsum("Qpq,Qrs->pqrs", factors, factors)
    alpha = np.array([-1.2, -1.2, .6, .6] if degenerate else [-1.3, -.8, .5, .9])
    if not unrestricted:
        h = (np.diag(alpha)-2*np.einsum("pqii->pq", g[:, :, :2, :2])
             + np.einsum("piiq->pq", g[:, :2, :2, :]))
        return h, g, 2
    beta = np.array([-1.1, .7, .7, .7] if degenerate else [-1.1, .3, .6, 1.])
    ja = np.einsum("pqii->pq", g[:, :, :2, :2])
    jb = np.einsum("pqii->pq", g[:, :, :1, :1])
    ha = np.diag(alpha)-ja-jb+np.einsum("piiq->pq", g[:, :2, :2, :])
    hb = np.diag(beta)-ja-jb+np.einsum("piiq->pq", g[:, :1, :1, :])
    return (ha, hb), (g, g, g), (2, 1)


def _rotate(h, g, rotations):
    def transform(block, a, b):
        return np.einsum("pqrs,pP,qQ,rR,sS->PQRS", block, a, a, b, b, optimize=True)
    if isinstance(h, tuple):
        a, b = rotations
        return ((a.T@h[0]@a, b.T@h[1]@b),
                (transform(g[0], a, a), transform(g[1], a, b), transform(g[2], b, b)))
    return rotations.T@h@rotations, transform(g, rotations, rotations)


def _block_rotation(nocc, seed):
    rng = np.random.default_rng(seed)
    rotation = np.zeros((4, 4))
    for selected in (slice(0, nocc), slice(nocc, 4)):
        size = rotation[selected, selected].shape[0]
        rotation[selected, selected] = np.linalg.qr(rng.normal(size=(size, size)))[0]
    return rotation


def _numpy_mp2(h, g, nocc):
    """Independent semicanonicalization and explicit spin-channel energy sum."""
    if isinstance(nocc, tuple):
        na, nb = nocc
        aa, ab, bb = g
        fa = (h[0]+np.einsum("pqii->pq", aa[:, :, :na, :na])
              - np.einsum("piiq->pq", aa[:, :na, :na, :])
              + np.einsum("pqii->pq", ab[:, :, :nb, :nb]))
        fb = (h[1]+np.einsum("pqii->pq", bb[:, :, :nb, :nb])
              - np.einsum("piiq->pq", bb[:, :nb, :nb, :])
              + np.einsum("iipq->pq", ab[:na, :na]))
        fs, ns = (fa, fb), nocc
    else:
        fs = (h+2*np.einsum("pqii->pq", g[:, :, :nocc, :nocc])
              - np.einsum("piiq->pq", g[:, :nocc, :nocc, :]),)
        ns = (nocc,)
    energies, rotations = [], []
    for f, no in zip(fs, ns):
        eo, co = np.linalg.eigh(f[:no, :no])
        ev, cv = np.linalg.eigh(f[no:, no:])
        c = np.zeros_like(f)
        c[:no, :no], c[no:, no:] = co, cv
        rotations.append(c)
        energies.append(np.concatenate((eo, ev)))
    _, canonical = _rotate(h, g, tuple(rotations) if isinstance(nocc, tuple) else rotations[0])
    pairs = ((0, 0), (0, 1), (1, 1)) if isinstance(nocc, tuple) else ((0, 0),)
    value = 0.
    for index, (a, b) in enumerate(pairs):
        no, nj = ns[a], ns[b]
        block = canonical[index] if isinstance(nocc, tuple) else canonical
        for i in range(no):
            for j in range(nj):
                for x in range(no, 4):
                    for y in range(nj, 4):
                        direct = block[i, x, j, y]
                        denominator = energies[a][i]+energies[b][j]-energies[a][x]-energies[b][y]
                        if not isinstance(nocc, tuple):
                            value += direct*(2*direct-block[i, y, j, x])/denominator
                        elif a == b:
                            value += .25*(direct-block[i, y, j, x])**2/denominator
                        else:
                            value += direct**2/denominator
    return value


@pytest.mark.parametrize("unrestricted", [False, True])
@pytest.mark.parametrize("with_t2", [False, True])
def test_mp2_noncanonical_rotations_preserve_energy_and_spin_components(unrestricted, with_t2):
    from gradscf.mp import MPConfig, run_mp
    h, g, no = _model(unrestricted)
    rotations = (tuple(_block_rotation(n, seed) for n, seed in zip(no, (4, 9)))
                 if unrestricted else _block_rotation(no, 4))
    hr, gr = _rotate(h, g, rotations)
    canonical = run_mp(h, g, nocc=no, config=MPConfig(with_t2=with_t2))
    result = jax.jit(lambda h, g: run_mp(h, g, nocc=no,
        config=MPConfig(orbital_basis="noncanonical", with_t2=with_t2)))(hr, gr)
    assert result.valid
    assert result.canonical_error > 1e-3
    for name in ("e2", "same_spin_energy", "opposite_spin_energy"):
        np.testing.assert_allclose(getattr(result, name), getattr(canonical, name), atol=2e-13, rtol=0)
    np.testing.assert_allclose(result.e2, _numpy_mp2(hr, gr, no), atol=2e-13, rtol=0)
    if with_t2:
        spin_pairs = ((0, 0), (0, 1), (1, 1)) if unrestricted else ((0, 0),)
        ns = no if unrestricted else (no,)
        cs = rotations if unrestricted else (rotations,)
        actual = result.t2 if unrestricted else (result.t2,)
        expected = canonical.t2 if unrestricted else (canonical.t2,)
        for t, reference, (a, b) in zip(actual, expected, spin_pairs):
            na, nb = ns[a], ns[b]
            rotated = np.einsum("ijab,iI,jJ,aA,bB->IJAB", reference,
                cs[a][:na, :na], cs[b][:nb, :nb], cs[a][na:, na:], cs[b][nb:, nb:], optimize=True)
            np.testing.assert_allclose(t, rotated, atol=2e-13, rtol=0)


@pytest.mark.parametrize("unrestricted", [False, True])
def test_exact_degenerate_fock_blocks_have_finite_mp2_response(unrestricted):
    from gradscf.mp import MPConfig, run_mp
    h, g, no = _model(unrestricted, degenerate=True)
    rng = np.random.default_rng(19)
    ns = no if unrestricted else (no,)
    directions = []
    for n in ns:
        d = rng.normal(size=(4, 4))*.04
        d = .5*(d+d.T)
        d[:n, n:] = d[n:, :n] = 0.
        directions.append(d)
    directions = tuple(directions) if unrestricted else directions[0]
    def shifted(x):
        return jax.tree.map(lambda a, d: jnp.asarray(a)+x*jnp.asarray(d), h, directions)
    def energy(x):
        return run_mp(shifted(x), g, nocc=no,
            config=MPConfig(orbital_basis="noncanonical", with_t2=False)).e2
    value, gradient = jax.jit(jax.value_and_grad(energy))(0.)
    assert np.isfinite(value) and np.isfinite(gradient)
    np.testing.assert_allclose(value, _numpy_mp2(h, g, no), atol=2e-13, rtol=0)
    for step in (3e-4, 1e-4):
        upper = jax.tree.map(lambda a, d: np.asarray(a)+step*d, h, directions)
        lower = jax.tree.map(lambda a, d: np.asarray(a)-step*d, h, directions)
        finite = (_numpy_mp2(upper, g, no)-_numpy_mp2(lower, g, no))/(2*step)
        np.testing.assert_allclose(gradient, finite, atol=2e-11, rtol=2e-7)


def test_noncanonical_is_explicit_and_rejects_nonstationary_reference():
    from gradscf.mp import MPConfig, run_mp
    h, g, no = _model()
    hr, gr = _rotate(h, g, _block_rotation(no, 4))
    assert not run_mp(hr, gr, nocc=no).valid
    config = MPConfig(orbital_basis="noncanonical")
    bad = jnp.asarray(hr).at[0, 2].add(.02).at[2, 0].add(.02)
    assert not run_mp(bad, gr, nocc=no, config=config).valid
    invalid_energy = lambda x: run_mp(bad+x*jnp.eye(4), gr, nocc=no, config=config).e2
    assert jnp.isnan(jax.grad(invalid_energy)(0.))
    singular = run_mp(jnp.zeros((2, 2)), jnp.zeros((2,)*4), nocc=1, config=config)
    assert not singular.valid and jnp.isnan(singular.e2)
    with pytest.raises(ValueError, match="orbital_basis"):
        MPConfig(orbital_basis="automatic")


def test_noncanonical_frozen_boundary_cannot_mix_active_and_frozen_orbitals():
    from gradscf.mp import MPConfig, run_mp
    h, g, no = _model()
    config = MPConfig(orbital_basis="noncanonical", with_t2=False)
    reference = run_mp(h, g, nocc=no, frozen=1)
    rotation = _block_rotation(no, 4)
    rotation[:no, :no] = np.eye(no)
    hr, gr = _rotate(h, g, rotation)
    result = run_mp(hr, gr, nocc=no, frozen=1, config=config)
    assert result.valid
    np.testing.assert_allclose(result.e2, reference.e2, atol=2e-13, rtol=0)
    hr, gr = _rotate(h, g, _block_rotation(no, 4))
    assert not run_mp(hr, gr, nocc=no, frozen=1, config=config).valid


@pytest.mark.parametrize("unrestricted", [False, True])
@pytest.mark.parametrize("orbital", [0, 2])
def test_noncanonical_frozen_cannot_split_degenerate_fock_subspace(unrestricted, orbital):
    from gradscf.mp import MPConfig, run_mp
    h, g, no = _model(unrestricted, degenerate=True)
    frozen = ([orbital], []) if unrestricted else [orbital]
    assert run_mp(h, g, nocc=no, frozen=frozen).valid
    result = run_mp(h, g, nocc=no, frozen=frozen,
                    config=MPConfig(orbital_basis="noncanonical", with_t2=False))
    assert not result.valid
    assert np.isnan(result.e2)


def test_noncanonical_mp2_facade_uses_explicit_rotated_reference():
    from gradscf import mp
    from gradscf.scf.reference import RestrictedReference
    h, g, no = _model()
    hr, gr = _rotate(h, g, _block_rotation(no, 4))
    result = mp.MP2(RestrictedReference(hr, gr, no), orbital_basis="noncanonical").run()
    assert result.converged
    np.testing.assert_allclose(result.e_corr, _numpy_mp2(hr, gr, no), atol=2e-13, rtol=0)


def test_noncanonical_mp3_auto_uses_general_fock_series():
    from gradscf.mp import MPConfig, run_mp
    h, g, no = _model()
    hr, gr = _rotate(h, g, _block_rotation(no, 4))
    expected = run_mp(h, g, nocc=no, config=MPConfig(order=3))
    result = run_mp(hr, gr, nocc=no, config=MPConfig(order=3, orbital_basis="noncanonical"))
    assert result.valid
    np.testing.assert_allclose(result.corrections, expected.corrections, atol=2e-12, rtol=0)
    with pytest.raises(ValueError, match="direct"):
        MPConfig(order=3, algorithm="direct", orbital_basis="noncanonical")
