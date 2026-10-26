"""Independent FCI forward references in real, common spatial orbitals.

PySCF supplies determinant contractions and densities; no GradSCF internal
Hamiltonian construction is used to produce a reference. Energies are Hartree.
"""

import math

import numpy as np
import pytest


pyscf = pytest.importorskip("pyscf")
from pyscf import ao2mo, gto, scf
from pyscf.fci import cistring, direct_spin1, spin_op

from gradscf import fci
from gradscf.solvers import EigenResponseConfig, EigenSolverConfig


SECTORS = [(0, 0), (4, 4), (1, 0), (0, 1), (2, 1), (1, 1)]


def _integrals(norb, seed=107):
    """Real Hermitian h and positive Coulomb tensor with eightfold symmetry."""
    rng = np.random.default_rng(seed)
    h1 = rng.normal(size=(norb, norb))
    h1 = (h1 + h1.T) / 2
    factors = rng.normal(size=(norb + 2, norb, norb))
    factors = (factors + factors.transpose(0, 2, 1)) / (2 * norb)
    eri = np.einsum("Lpq,Lrs->pqrs", factors, factors)
    return h1, eri


def _vector(shape, seed):
    c = np.random.default_rng(seed).normal(size=shape)
    return c / np.linalg.norm(c)


def _pyscf_action(h1, eri, c, norb, nelec):
    effective = direct_spin1.absorb_h1e(h1, eri, norb, nelec, fac=0.5)
    return np.asarray(direct_spin1.contract_2e(effective, c, norb, nelec))


@pytest.mark.parametrize("nelec", SECTORS)
def test_contraction_and_diagonal_against_pyscf(nelec):
    norb = 4
    space = fci.make_fci_space(norb, nelec)
    assert space.norb == norb
    assert space.nelec == nelec
    assert space.shape == tuple(math.comb(norb, n) for n in nelec)
    assert space.size == math.prod(space.shape)
    h1, eri = _integrals(norb)
    c = _vector(space.shape, 2026)

    actual = fci.contract_hamiltonian(h1, eri, c, space)
    assert actual.shape == space.shape
    np.testing.assert_allclose(
        actual, _pyscf_action(h1, eri, c, norb, nelec), atol=2e-12, rtol=2e-12
    )
    diagonal = fci.make_hdiag(h1, eri, space)
    assert diagonal.shape == (space.size,)
    np.testing.assert_allclose(
        diagonal, direct_spin1.make_hdiag(h1, eri, norb, nelec),
        atol=2e-12, rtol=2e-12,
    )


@pytest.mark.parametrize("nelec", SECTORS)
def test_normalized_rdms_and_spin_against_pyscf(nelec):
    norb = 4
    space = fci.make_fci_space(norb, nelec)
    c = _vector(space.shape, 31)
    # Diagonal observables explicitly normalize arbitrary coefficient norms.
    unnormalized = -2.3 * c
    dm1, dm2 = fci.make_rdm12(unnormalized, space)
    ref1, ref2 = direct_spin1.make_rdm12(c, norb, nelec)
    np.testing.assert_allclose(dm1, ref1, atol=3e-12)
    np.testing.assert_allclose(dm2, ref2, atol=3e-12)
    np.testing.assert_allclose(fci.make_rdm1(unnormalized, space), ref1, atol=3e-12)

    dm1s, dm2s = fci.make_rdm12s(unnormalized, space)
    ref1s, ref2s = direct_spin1.make_rdm12s(c, norb, nelec)
    np.testing.assert_allclose(dm1s, ref1s, atol=3e-12)
    np.testing.assert_allclose(dm2s, ref2s, atol=3e-12)
    np.testing.assert_allclose(fci.make_rdm1s(unnormalized, space), ref1s, atol=3e-12)
    np.testing.assert_allclose(
        fci.spin_square(unnormalized, space), spin_op.spin_square(c, norb, nelec),
        atol=3e-12,
    )

    n = sum(nelec)
    np.testing.assert_allclose(np.trace(dm1), n, atol=3e-12)
    np.testing.assert_allclose(np.einsum("pqrr->pq", dm2), (n - 1) * dm1.T, atol=3e-12)
    np.testing.assert_allclose([np.trace(d) for d in dm1s], nelec, atol=3e-12)
    np.testing.assert_allclose(dm1, np.asarray(dm1s).sum(axis=0), atol=3e-12)
    aa, ab, bb = map(np.asarray, dm2s)
    np.testing.assert_allclose(dm2, aa + ab + ab.transpose(2, 3, 0, 1) + bb, atol=3e-12)

    h1, eri = _integrals(norb)
    density_energy = np.einsum("pq,qp", h1, dm1) + 0.5 * np.einsum("pqrs,pqrs", eri, dm2)
    reference_energy = np.vdot(c, _pyscf_action(h1, eri, c, norb, nelec))
    np.testing.assert_allclose(density_energy, reference_energy, atol=3e-12)


@pytest.mark.parametrize("nelec", SECTORS)
def test_bilinear_transition_rdms_against_pyscf(nelec):
    norb = 4
    space = fci.make_fci_space(norb, nelec)
    # Unequal vectors expose transpose errors; unequal norms expose accidental
    # normalization. Transition densities need not be symmetric.
    bra = 1.7 * _vector(space.shape, 19)
    ket = -0.6 * _vector(space.shape, 23)
    dm1, dm2 = fci.trans_rdm12(bra, ket, space)
    ref1, ref2 = direct_spin1.trans_rdm12(bra, ket, norb, nelec)
    np.testing.assert_allclose(dm1, ref1, atol=3e-12)
    np.testing.assert_allclose(dm2, ref2, atol=3e-12)
    np.testing.assert_allclose(fci.trans_rdm1(bra, ket, space), ref1, atol=3e-12)
    actual1,actual2=fci.trans_rdm12s(bra,ket,space)
    expected1,expected2=direct_spin1.trans_rdm12s(bra,ket,norb,nelec)
    assert len(actual2)==4
    for actual,expected in zip((*actual1,*actual2),(*expected1,*expected2)):
        np.testing.assert_allclose(actual,expected,atol=3e-12)
    n = sum(nelec)
    np.testing.assert_allclose(np.trace(dm1), n * np.vdot(bra, ket), atol=3e-12)
    np.testing.assert_allclose(np.einsum("pqrr->pq", dm2), (n - 1) * dm1.T, atol=3e-12)
    h1, eri = _integrals(norb)
    density_element = np.einsum("pq,qp", h1, dm1) + 0.5 * np.einsum("pqrs,pqrs", eri, dm2)
    reference_element = np.vdot(bra, _pyscf_action(h1, eri, ket, norb, nelec))
    np.testing.assert_allclose(density_element, reference_element, atol=3e-12)


@pytest.fixture(scope="module", params=[
    pytest.param("H 0 0 0; H 0 0 0.74", id="H2"),
    pytest.param("Li 0 0 0; H 0 0 1.6", id="LiH"),
    pytest.param("H 0 0 0; H 0 0 1.0; H 0 0 2.1; H 0 0 3.3", id="H4"),
])
def molecule_integrals(request):
    """STO-3G, Angstrom, restricted MOs, RHF tolerance 1e-12 Hartree."""
    mol = gto.M(atom=request.param, basis="sto-3g", unit="Angstrom", verbose=0)
    mf = scf.RHF(mol).run(conv_tol=1e-12)
    assert mf.converged
    mo = mf.mo_coeff
    norb = mo.shape[1]
    h1 = mo.T @ mf.get_hcore() @ mo
    eri = ao2mo.restore(1, ao2mo.kernel(mol, mo), norb)
    reference_energy, reference_c = direct_spin1.kernel(
        h1, eri, norb, mol.nelec, ecore=mol.energy_nuc(),
        tol=1e-13, max_cycle=200,
    )
    return h1, eri, norb, mol.nelec, mol.energy_nuc(), reference_energy, reference_c


@pytest.mark.parametrize("method", ["dense", "davidson"])
def test_molecular_ground_state_against_pyscf(molecule_integrals, method):
    h1, eri, norb, nelec, ecore, ref_energy, ref_c = molecule_integrals
    space = fci.make_fci_space(norb, nelec)
    assert space.shape == tuple(cistring.num_strings(norb, n) for n in nelec)
    result = fci.solve_fci(
        h1, eri, space, ecore=ecore,
        config=EigenSolverConfig(
            method=method, nroots=1, atol=1e-10, maxiter=150,
            max_subspace=min(space.size, 40), max_dense=225,
        ),
        response=EigenResponseConfig(target="eigenpairs"),
    )
    assert result.total_energies.shape == (1,)
    assert result.coefficients.shape == (1, *space.shape)
    assert np.all(result.converged)
    assert np.all(result.response_valid)
    assert np.max(result.residual_norms) < 2e-9
    np.testing.assert_allclose(result.total_energies, [ref_energy], atol=2e-10, rtol=0)
    c = np.asarray(result.coefficients[0])
    np.testing.assert_allclose(np.linalg.norm(c), 1, atol=2e-12)
    np.testing.assert_allclose(abs(np.vdot(c, ref_c)), 1, atol=2e-9)
    independent_residual = _pyscf_action(h1, eri, c, norb, nelec) - (float(result.total_energies[0]) - ecore) * c
    assert np.linalg.norm(independent_residual) < 2e-9
    np.testing.assert_allclose(
        fci.make_rdm1(c, space), direct_spin1.make_rdm1(ref_c, norb, nelec),
        atol=2e-8,
    )
