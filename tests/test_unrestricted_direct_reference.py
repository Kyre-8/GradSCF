"""UHF/UKS keeps the native AO action from SCF through its response reference."""

import jax.numpy as jnp
import numpy as np
import pytest

from gradscf import dft, gto, integrals, scf
from gradscf.integrals.backends.native.jk import NativeDirectBasis
from gradscf.integrals.plan import IntegralPlan


@pytest.fixture(params=["hf", "pbe"], ids=["UHF", "UKS-PBE"])
def lithium_reference(request):
    molecule = gto.M(atom="Li 0 0 0", basis="sto-3g", spin=1, cart=True)
    options = dict(integral_backend="native", jk_backend="full", grids_level=0,
                   conv_tol=1e-11, conv_tol_density=1e-10, conv_tol_grad=1e-9,
                   max_cycle=120)
    mean_field = (scf.UHF(molecule, **options) if request.param == "hf"
                  else dft.UKS(molecule, xc=request.param, **options)).run()
    assert mean_field.converged
    return request.param, molecule, options, mean_field


def test_unrestricted_direct_scf_and_reference_avoid_eri_arrays(lithium_reference, monkeypatch):
    xc, molecule, options, full = lithium_reference
    original = IntegralPlan.evaluate

    def no_eri(self, operator, *args, **kwargs):
        if operator == "eri":
            raise AssertionError("Native direct UHF/UKS must not materialize AO ERIs")
        return original(self, operator, *args, **kwargs)

    def no_pair(*args, **kwargs):
        raise AssertionError("The unrestricted response reference must keep its AO provider")

    monkeypatch.setattr(IntegralPlan, "evaluate", no_eri)
    monkeypatch.setattr(NativeDirectBasis, "response_eri_pair_matrix", no_pair)
    options = dict(options, jk_backend="direct", direct_scf_workers=2, direct_scf_tol=0.)
    direct = (scf.UHF(molecule, **options) if xc == "hf"
              else dft.UKS(molecule, xc=xc, **options)).run()
    assert direct.converged
    np.testing.assert_allclose(direct.e_tot, full.e_tot, atol=2e-10, rtol=0.)
    np.testing.assert_allclose(direct.make_rdm1(), full.make_rdm1(), atol=3e-9, rtol=0.)

    reference = scf.as_reference(direct)
    assert isinstance(reference.direct_basis, NativeDirectBasis)
    assert reference.rep_tensor.size == 0 and reference.eri_pair_matrix is None
    assert reference.df_factors is None
    assert (reference.nocc_alpha, reference.nocc_beta) == (2, 1)
    provider = reference.direct_basis
    assert provider.plan.topology.nao == reference.mo_coeff.shape[-2]
    np.testing.assert_array_equal(provider.parameters.nuclear_coords, reference.atom_coords)
    np.testing.assert_array_equal(provider.plan.topology.nuclear_charges, reference.atom_charges)


@pytest.mark.parametrize("with_k", [False, True])
def test_direct_spin_jk_batches_spin_densities_and_skips_unrequested_exchange(monkeypatch, with_k):
    from gradscf.scf.uks import _spin_jk_matrices

    topology, parameters = integrals.prepare_basis("Li 0 0 0", "sto-3g", spin=1, cart=True)
    plan = integrals.make_plan(topology)
    eri = plan.evaluate("eri", parameters)
    provider = NativeDirectBasis(plan, parameters)
    rng = np.random.default_rng(813)
    coefficients = rng.normal(size=(2, topology.nao, topology.nao))
    densities = jnp.asarray(coefficients @ coefficients.transpose(0, 2, 1) / 5.)
    original = NativeDirectBasis.get_jk
    calls = []

    def observed(self, density, **options):
        calls.append((density.shape, options))
        return original(self, density, **options)

    monkeypatch.setattr(NativeDirectBasis, "get_jk", observed)
    total, coulomb, exchange_a, exchange_b = _spin_jk_matrices(
        jnp.zeros((0, 0, 0, 0)), densities[0], densities[1],
        direct_basis=provider, with_k=with_k,
    )
    np.testing.assert_allclose(total, densities.sum(axis=0), atol=1e-12)
    np.testing.assert_allclose(coulomb, jnp.einsum("pqrs,rs->pq", eri, total), atol=2e-11)
    for observed_exchange, density in zip((exchange_a, exchange_b), densities):
        expected = (jnp.einsum("prqs,rs->pq", eri, density)
                    if with_k else jnp.zeros_like(density))
        np.testing.assert_allclose(observed_exchange, expected, atol=2e-11)
    assert len(calls) == 1
    assert calls[0][1]["with_k"] is with_k
    assert calls[0][0] == ((2, topology.nao, topology.nao) if with_k
                           else (topology.nao, topology.nao))
