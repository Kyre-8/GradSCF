"""Restricted EOM Dyson residues: independent endpoints and the H2 FCI limit."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest


def test_vector_response_contract_is_static_across_jit():
    from gradscf.cc.eom import EOMResult
    # Test the result container independently of any converged chemistry state.
    result = EOMResult(*(jnp.zeros(1) for _ in EOMResult._fields[:-2]),
                       vector_response=True, sector="ip")
    returned = jax.jit(lambda state: state)(result)
    assert returned.vector_response is True
    assert returned.sector == "ip"
    assert len(jax.tree_util.tree_leaves(returned)) == len(EOMResult._fields)-2


@pytest.mark.parametrize("sector", ["ip", "ea"])
def test_endpoints_match_independent_pyscf_contractions(sector):
    from pyscf.cc import eom_rccsd, momgfccsd
    from gradscf.cc.eom.properties import _endpoints

    rng = np.random.default_rng(1047)
    t1, l1 = (rng.normal(scale=.04, size=(2, 3)) for _ in range(2))
    t2, l2 = (rng.normal(scale=.04, size=(2, 2, 3, 3)) for _ in range(2))
    t2 = .5*(t2+t2.transpose(1, 0, 3, 2))
    l2 = .5*(l2+l2.transpose(1, 0, 3, 2))
    # Packing methods require no initialized PySCF CC object.
    from types import SimpleNamespace
    pack = eom_rccsd.amplitudes_to_vector_ip if sector == "ip" else eom_rccsd.amplitudes_to_vector_ea
    space = SimpleNamespace(amplitudes_to_vector=pack)
    bra_function = momgfccsd.build_bra_hole if sector == "ip" else momgfccsd.build_bra_part
    ket_function = momgfccsd.build_ket_hole if sector == "ip" else momgfccsd.build_ket_part
    expected_bra = np.stack([bra_function(None, space, t1, t2, l1, l2, p) for p in range(5)])
    expected_ket = np.stack([ket_function(None, space, t1, t2, p) for p in range(5)])
    if sector == "ea":
        expected_ket *= -1.  # Upstream particle moments carry this outside.
    bra, ket = jax.jit(lambda a, b, c, d: _endpoints(a, b, c, d, sector))(t1, t2, l1, l2)
    np.testing.assert_allclose(bra, expected_bra, atol=2e-14)
    np.testing.assert_allclose(ket, expected_ket, atol=2e-14)


@pytest.fixture(scope="module")
def h2():
    from pyscf import gto, scf, ao2mo
    mol = gto.M(atom="H 0 0 0; H 0 0 1.4", basis="sto-3g", unit="Bohr", verbose=0)
    mf = scf.RHF(mol).run(conv_tol=1e-13)
    c = mf.mo_coeff
    h = c.T @ mf.get_hcore() @ c
    eri = ao2mo.restore(1, ao2mo.kernel(mol, c), 2)
    return jnp.asarray(h), jnp.asarray(eri)


def properties(h, eri, sector, *, vector_response=True):
    from gradscf import cc
    from gradscf.cc.eom import dyson_orbitals
    from gradscf.solvers import LinearSolverConfig
    config = cc.CCConfig(conv_tol=1e-12, residual_tol=1e-11)
    ground = cc.run_cc(h, eri, nocc=1, config=config)
    left = cc.solve_lambda(h, eri, ground, nocc=1, config=config)
    cfg = cc.EOMConfig(sector=sector, nroots=2, conv_tol=1e-11,
                      vector_response=vector_response,
                      response_config=(LinearSolverConfig(method="direct", max_dense=3,
                                                           rtol=1e-11)
                                       if vector_response else None))
    eom = cc.run_eom(h, eri, ground, nocc=1, config=cfg, cc_config=config)
    return eom, dyson_orbitals(ground, left, eom, sector=sector), ground, left


def fci_reference(h, eri, sector):
    from pyscf import fci
    h, eri = np.asarray(h), np.asarray(eri)
    e0, c0 = fci.direct_spin1.kernel(h, eri, 2, (1, 1), tol=1e-13)
    nelec = (0, 1) if sector == "ip" else (2, 1)
    energies, charged = fci.direct_spin1.kernel(h, eri, 2, nelec, nroots=2, tol=1e-13)
    apply = fci.addons.des_a if sector == "ip" else fci.addons.cre_a
    vectors = np.asarray([[np.vdot(c, apply(c0, 2, (1, 1), p))
                           for c in charged] for p in range(2)])
    return np.asarray(energies)-e0, vectors[:, None, :]*vectors[None, :, :]


@pytest.mark.parametrize("sector", ["ip", "ea"])
def test_h2_complete_ip_ea_spectrum_and_residues_equal_fci(h2, sector):
    h, eri = h2
    eom, dyson, ground, left = properties(h, eri, sector)
    assert ground.converged and left.converged
    assert np.all(eom.converged & eom.response_valid & dyson.valid & dyson.response_valid)
    energies, residues = fci_reference(h, eri, sector)
    actual = dyson.right[:, None, :]*dyson.left[None, :, :]
    np.testing.assert_allclose(eom.energies, energies, atol=2e-10)
    np.testing.assert_allclose(actual, residues, atol=2e-9)
    np.testing.assert_allclose(dyson.weights, np.einsum("ppk->k", residues), atol=2e-9)


def test_full_root_zeroth_moments_and_biorthogonal_rescaling(h2):
    from gradscf.cc.eom import dyson_orbitals
    h, eri = h2
    matrices = []
    for sector in ("ip", "ea"):
        eom, result, ground, left = properties(h, eri, sector)
        matrices.append(result.right @ result.left.T)
        scale = jnp.array([2.3, -.7])
        changed = eom._replace(right_vectors=eom.right_vectors*scale,
                               left_vectors=eom.left_vectors/scale)
        other = dyson_orbitals(ground, left, changed, sector=sector)
        np.testing.assert_allclose(other.weights, result.weights, atol=2e-12)
        np.testing.assert_allclose(other.right[:, None, :]*other.left[None, :, :],
            result.right[:, None, :]*result.left[None, :, :], atol=2e-12)
    np.testing.assert_allclose(matrices[0]+matrices[1], np.eye(2), atol=2e-10)
    np.testing.assert_allclose(np.trace(matrices[0]), 1., atol=2e-10)
    np.testing.assert_allclose(np.trace(matrices[1]), 1., atol=2e-10)


@pytest.mark.parametrize("sector", ["ip", "ea"])
def test_fixed_mo_weight_response_includes_t_lambda_and_both_vectors(h2, sector):
    h, eri = h2
    perturbation = jnp.array([[.3, .2], [.2, -.1]])
    def function(t):
        eom, dyson, _, _ = properties(h+t*perturbation, eri, sector)
        return jnp.concatenate((eom.energies, dyson.weights))
    inner = jax.jit(function)
    value, derivative = jax.jit(lambda t: jax.jvp(inner, (t,), (jnp.ones_like(t),)))(0.)
    step = 1e-4
    finite_difference = (function(step)-function(-step))/(2*step)
    np.testing.assert_allclose(derivative, finite_difference, atol=2e-7, rtol=2e-6)
    cotangent = jnp.array([.3, -.2, .7, -.4])
    reverse = jax.jit(jax.grad(lambda t: cotangent @ inner(t)))(0.)
    np.testing.assert_allclose(reverse, cotangent @ derivative, atol=2e-9)
    fci_plus = fci_reference(h+step*perturbation, eri, sector)
    fci_minus = fci_reference(h-step*perturbation, eri, sector)
    independent = np.concatenate(((fci_plus[0]-fci_minus[0])/(2*step),
        np.einsum("ppk->k", fci_plus[1]-fci_minus[1])/(2*step)))
    np.testing.assert_allclose(derivative, independent, atol=2e-7, rtol=2e-6)
    assert np.linalg.norm(derivative[2:]) > 1e-4
    assert np.all(np.isfinite(value))


def test_forward_only_dyson_rejects_incomplete_response(h2):
    h, eri = h2
    direction = jnp.array([[.3, .2], [.2, -.1]])
    def function(t):
        _, dyson, _, _ = properties(h+t*direction, eri, "ip", vector_response=False)
        return dyson.weights[0]
    assert np.isfinite(function(0.))
    assert not np.isfinite(jax.jvp(function, (0.,), (1.,))[1])
    assert not np.isfinite(jax.grad(function)(0.))


def test_dyson_capacity_and_sector_validation(h2):
    from gradscf.cc.eom import dyson_orbitals
    h, eri = h2
    eom, _, ground, left = properties(h, eri, "ip")
    inferred = dyson_orbitals(ground, left, eom)
    explicit = dyson_orbitals(ground, left, eom, sector="ip")
    np.testing.assert_allclose(inferred.weights, explicit.weights, atol=0.)
    with pytest.raises(ValueError, match="sector"):
        dyson_orbitals(ground, left, eom._replace(sector=None))
    with pytest.raises(ValueError, match="sector"):
        dyson_orbitals(ground, left, eom, sector="ee")
    with pytest.raises(ValueError, match="max_elements"):
        dyson_orbitals(ground, left, eom, sector="ip", max_elements=3)


def test_equal_dimension_ee_result_cannot_be_used_as_ip(h2):
    from gradscf import cc
    h, eri = h2
    config = cc.CCConfig(conv_tol=1e-12, residual_tol=1e-11)
    ground = cc.run_cc(h, eri, nocc=1, config=config)
    left = cc.solve_lambda(h, eri, ground, nocc=1, config=config)
    ee = cc.run_eom(h, eri, ground, nocc=1,
        config=cc.EOMConfig(sector="ee", nroots=2), cc_config=config)
    assert ee.right_vectors.shape == (2, 2)  # Same dimension as H2 IP and EA.
    with pytest.raises(ValueError, match="sector"):
        cc.dyson_orbitals(ground, left, ee, sector="ip")
