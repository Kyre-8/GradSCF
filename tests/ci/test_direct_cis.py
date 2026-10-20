"""SCF-backed CIS must use AO J/K actions, with the spatial CI convention."""
import numpy as np
import pytest


@pytest.fixture(scope="module")
def water_direct_hf():
    pytest.importorskip("pyscf")
    from gradscf import gto, scf

    mol = gto.M(atom="O 0 0 0; H 0 .757 .587; H 0 -.757 .587",
                basis="sto-3g", cart=True)
    mf = scf.RHF(mol, integral_backend="native", jk_backend="direct",
                 conv_tol=1e-13, conv_tol_grad=1e-10,
                 conv_tol_density=1e-11).run()
    assert mf.converged
    return mf


def _explicit_reference(mf):
    """Independent PySCF integrals, transformed to the exact GradSCF frame."""
    from pyscf import ao2mo, gto, scf
    from gradscf.ci import CIReference

    mol = gto.M(atom="O 0 0 0; H 0 .757 .587; H 0 -.757 .587",
                basis="sto-3g", cart=True, verbose=0)
    c = np.asarray(mf.mo_coeff)
    h = c.T @ scf.RHF(mol).get_hcore() @ c
    eri = ao2mo.kernel(mol, c, compact=False).reshape((c.shape[1],)*4)
    return CIReference(h, eri, mol.nelectron//2, mol.energy_nuc(), mf.mo_energy)


@pytest.mark.parametrize("singlet", [True, False])
@pytest.mark.parametrize("frozen", [None, [0, 6]])
def test_direct_cis_matches_explicit_without_materializing_mo_eri(
        water_direct_hf, monkeypatch, singlet, frozen):
    from gradscf import ci
    import gradscf.ci.api as api

    mf = water_direct_hf
    expected = ci.CIS(_explicit_reference(mf), singlet=singlet, frozen=frozen,
                      solver="dense", nroots=2, conv_tol=1e-10).run()

    def forbidden_reference(*args, **kwargs):
        raise AssertionError("SCF-backed CIS materialized the MO ERI reference")

    def forbidden_pair(*args, **kwargs):
        raise AssertionError("SCF-backed CIS materialized the AO pair ERI")

    monkeypatch.setattr(api, "reference_from_source", forbidden_reference)
    monkeypatch.setattr(type(mf._scf_inputs), "response_eri_pair_matrix", forbidden_pair)
    actual = ci.CIS(mf, singlet=singlet, frozen=frozen, nroots=2,
                    conv_tol=1e-10, max_cycle=100).run()
    assert actual.result.singlet is singlet
    assert np.all(actual.converged)
    np.testing.assert_allclose(actual.e, expected.e, atol=2e-9, rtol=0)
    vectors = np.asarray(actual.amplitudes).reshape(2, -1)
    reference_vectors = np.asarray(expected.amplitudes).reshape(2, -1)
    # CIS spatial amplitudes have norm 1; restricted TDA amplitudes have norm 1/2.
    np.testing.assert_allclose(np.sum(vectors**2, axis=1), 1., atol=2e-11, rtol=0)
    for vector, reference in zip(vectors, reference_vectors):
        np.testing.assert_allclose(np.outer(vector, vector), np.outer(reference, reference),
                                   atol=3e-8, rtol=0)
    assert actual.reference.eri_pair_matrix is None
    assert actual.reference.direct_basis is not None


def test_cis_d_retains_mo_reference_path(water_direct_hf, monkeypatch):
    from gradscf import ci
    import gradscf.ci.api as api

    reference = _explicit_reference(water_direct_hf)
    expected = ci.CIS_D(reference, solver="dense", nroots=1).run()
    calls = []

    def explicit_reference(source):
        assert source is water_direct_hf
        calls.append(source)
        return reference

    monkeypatch.setattr(api, "reference_from_source", explicit_reference)
    actual = ci.CIS_D(water_direct_hf, solver="dense", nroots=1).run()
    assert calls == [water_direct_hf]
    np.testing.assert_allclose(actual.e, expected.e, atol=2e-11, rtol=0)
    assert actual.reference is reference


@pytest.mark.parametrize("singlet", [True, False])
def test_direct_cis_uses_noncanonical_fock_blocks(water_direct_hf, singlet):
    from dataclasses import replace
    from gradscf import ci, scf
    from gradscf.ci.solver import solve_cis_response

    mf = water_direct_hf
    explicit = _explicit_reference(mf)
    nmo, no = explicit.h1.shape[0], explicit.nocc
    rotation = np.eye(nmo)
    for p, q, angle in ((1, 3, .37), (no, no+1, -.29)):
        rotation[np.ix_([p, q], [p, q])] = [
            [np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    h = rotation.T @ explicit.h1 @ rotation
    eri = np.einsum("pi,qj,rk,sl,pqrs->ijkl", rotation, rotation, rotation,
                    rotation, explicit.eri, optimize=True)
    config = ci.CIConfig(nroots=2, solver="dense", conv_tol=1e-10)
    expected = ci.solve_cis(h, eri, nocc=no, singlet=singlet, config=config)
    reference = replace(scf.as_reference(mf), mo_coeff=np.asarray(mf.mo_coeff) @ rotation,
                        mo_occ=mf.mo_occ, mo_energy=mf.mo_energy,
                        fock_matrix=mf.scf_result.fock_matrix)
    actual = solve_cis_response(reference, singlet=singlet, config=config)
    np.testing.assert_allclose(actual.excitation_energies, expected.excitation_energies,
                               atol=2e-9, rtol=0)
    for vector, target in zip(np.asarray(actual.amplitudes).reshape(2, -1),
                               np.asarray(expected.amplitudes).reshape(2, -1)):
        np.testing.assert_allclose(np.outer(vector, vector), np.outer(target, target),
                                   atol=3e-8, rtol=0)


def test_direct_cis_preserves_energy_and_amplitude_response(water_direct_hf):
    from dataclasses import replace
    import jax
    import jax.numpy as jnp
    from gradscf import ci, scf
    from gradscf.ci.solver import solve_cis_response

    mf = water_direct_hf
    explicit = _explicit_reference(mf)
    c = np.asarray(mf.mo_coeff)
    overlap = np.asarray(mf._scf_inputs.overlap)
    no = explicit.nocc
    direction = np.zeros_like(explicit.h1)
    direction[1:3, 1:3] = [[.11, .07], [.07, -.09]]
    direction[no:, no:] = [[.13, -.04], [-.04, -.08]]
    ao_direction = jnp.asarray(overlap @ c @ direction @ c.T @ overlap)
    reference = replace(scf.as_reference(mf), fock_matrix=mf.scf_result.fock_matrix)
    config = ci.CIConfig(solver="dense", gradient_mode="implicit_eigenvector")
    weights = jnp.arange(no*(c.shape[1]-no), dtype=jnp.float64).reshape(no, -1)

    def observable(result):
        assert result.amplitude_response
        return (result.excitation_energies[0]
                + .03*jnp.sum(weights*result.amplitudes[0]**2))

    def actual(t):
        return observable(solve_cis_response(
            replace(reference, fock_matrix=reference.fock_matrix+t*ao_direction),
            config=config))

    def expected(t):
        return observable(ci.solve_cis(explicit.h1+t*direction, explicit.eri,
                                       nocc=no, config=config))

    np.testing.assert_allclose(jax.jit(jax.value_and_grad(actual))(0.),
                               jax.jit(jax.value_and_grad(expected))(0.),
                               atol=2e-8, rtol=0)
    step = 1e-4
    finite_difference = (actual(step)-actual(-step))/(2*step)
    np.testing.assert_allclose(jax.grad(actual)(0.), finite_difference,
                               atol=2e-8, rtol=2e-6)
