"""MP3 checked against independent Rayleigh--Schrodinger determinant algebra."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest


def determinant_coefficients(h, g, eps, nocc, frozen=()):
    """Use PySCF FCI Hamiltonian actions, not coupled-cluster equations."""
    from pyscf import fci
    n = len(eps)
    occupations = fci.cistring.gen_occslst(range(n), nocc)
    shape = (len(occupations),) * 2
    reference = np.zeros(shape)
    reference[0, 0] = 1
    absorbed = fci.direct_spin1.absorb_h1e(h, g, n, (nocc, nocc), .5)
    action = lambda c: np.asarray(fci.direct_spin1.contract_2e(absorbed, c, n, (nocc, nocc)))
    coupling = action(reference)
    ehf = coupling[0, 0]
    occupied_sum = np.sum(eps[occupations], axis=1)
    denominator = 2 * np.sum(eps[:nocc]) - occupied_sum[:, None] - occupied_sum[None, :]
    allowed = np.array([all((i in occ) == (i < nocc) for i in frozen) for occ in occupations])
    mask = allowed[:, None] & allowed[None, :]
    mask[0, 0] = False
    first = np.where(mask, coupling / np.where(mask, denominator, 1.), 0.)
    e2 = np.sum(coupling * first)
    e3 = np.sum(first * (action(first) - ehf * first + denominator * first))
    return e2, e3


@pytest.fixture(scope="module", params=[("H 0 0 0; H 0 0 .9", "3-21g"),
                                       ("Li 0 0 0; H 0 0 1.6", "sto-3g")])
def molecule(request):
    pytest.importorskip("pyscf")
    from pyscf import gto, scf, ao2mo
    atom, basis = request.param
    mol = gto.M(atom=atom, basis=basis, verbose=0, cart=True)
    mf = scf.RHF(mol).run(conv_tol=1e-14, conv_tol_grad=1e-11)
    c = mf.mo_coeff
    h = c.T @ mf.get_hcore() @ c
    g = ao2mo.kernel(mol, c, compact=False).reshape((c.shape[1],) * 4)
    return mf, jnp.asarray(h), jnp.asarray(g)


@pytest.mark.parametrize("frozen", [None, 1, [0, 3]])
def test_mp3_matches_determinant_perturbation(molecule, frozen):
    from gradscf.mp import run_mp, MPConfig
    mf, h, g = molecule
    no = mf.mol.nelectron // 2
    frozen_indices = range(frozen or 0) if frozen is None or isinstance(frozen, int) else frozen
    e2, e3 = determinant_coefficients(np.asarray(h), np.asarray(g), mf.mo_energy,
                                      no, frozen_indices)
    result = run_mp(h, g, nocc=no, frozen=frozen, config=MPConfig(order=3))
    assert result.valid
    np.testing.assert_allclose(result.e2, e2, atol=1e-11)
    np.testing.assert_allclose(result.e3, e3, atol=1e-11)
    np.testing.assert_allclose(result.correlation_energy, e2 + e3, atol=1e-11)


def test_mp3_jit_ad_and_energy_only(molecule):
    from gradscf.mp import run_mp, MPConfig
    mf, h, g = molecule
    no = mf.mol.nelectron // 2
    interaction = (2 * jnp.einsum("pqii->pq", g[:, :, :no, :no])
                   - jnp.einsum("piiq->pq", g[:, :no, :no, :]))
    config = MPConfig(order=3, with_t2=False)
    energy = lambda s: run_mp(h + (1 - s) * interaction, s * g,
                              nocc=no, config=config).correlation_energy
    result = run_mp(h, g, nocc=no, config=config)
    value, derivative = jax.jit(jax.value_and_grad(energy))(1.)
    np.testing.assert_allclose(value, result.e2 + result.e3, atol=1e-12)
    np.testing.assert_allclose(derivative, 2 * result.e2 + 3 * result.e3, atol=1e-11)
    hvp = jax.jvp(jax.grad(energy), (1.,), (.2,))[1]
    np.testing.assert_allclose(hvp, .2 * (2 * result.e2 + 6 * result.e3), atol=1e-11)
    step = 1e-4
    np.testing.assert_allclose(derivative, (energy(1 + step) - energy(1 - step)) / (2 * step), atol=1e-8)
    assert result.t2 is None


def test_mp3_facade(molecule):
    from gradscf import mp
    from gradscf.scf.reference import RestrictedReference
    mf, h, g = molecule
    pt = mp.MP3(RestrictedReference(h, g, mf.mol.nelectron // 2, mf.mol.energy_nuc())).run()
    assert pt.converged
    np.testing.assert_allclose(pt.e_corr, pt.e2 + pt.e3, atol=1e-13)


def test_direct_ump3_is_not_silently_rmp3():
    from gradscf.mp import run_mp, MPConfig
    h = jnp.diag(jnp.array([-1., .4]))
    g = jnp.zeros((2,) * 4)
    with pytest.raises(NotImplementedError, match="restricted"):
        run_mp((h, h), (g, g, g), nocc=(1, 0), config=MPConfig(order=3, algorithm="direct"))
