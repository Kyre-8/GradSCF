"""Direct response references retain an AO action instead of AO ERI arrays."""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gradscf import gto, integrals, scf
from gradscf.integrals.backends.native.jk import (
    NativeDirectBasis,
    ProjectedNativeDirectBasis,
)
from gradscf.scf.builders import _build_restricted_reference
from gradscf.scf.inputs.types import RKSIntegralInputs


def test_direct_reference_does_not_materialize_eri(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("A direct response must not materialize AO ERIs")

    monkeypatch.setattr(RKSIntegralInputs, "response_eri_pair_matrix", forbidden)
    mol = gto.M(atom="H 0 0 0; H 0 0 .8", basis="6-31g")
    mf = scf.RHF(mol, jk_backend="direct", conv_tol=1e-12).run()
    reference = scf.as_reference(mf)
    assert reference.direct_basis is mf._scf_inputs.direct_basis
    assert reference.eri_pair_matrix is None
    assert reference.rep_tensor.size == 0
    assert reference.df_factors is None
    assert mf._scf_inputs.eri is None
    assert mf._scf_inputs.eri_pair_matrix is None


def test_direct_provider_is_dynamic_pytree_with_density_ad():
    top, parameters = integrals.prepare_basis(
        "H 0 0 0; H 0 0 .8", "6-31g", cart=True
    )
    plan = integrals.make_plan(top)
    provider = NativeDirectBasis(plan, parameters)
    density = jnp.arange(top.nao**2, dtype=jnp.float64).reshape(top.nao, top.nao) / 19
    direction = jnp.flip(density, axis=0) + jnp.eye(top.nao) / 7
    eri = plan.evaluate("eri", parameters)

    def explicit(dm):
        return (
            jnp.einsum("pqrs,rs->pq", eri, dm),
            jnp.einsum("prqs,rs->pq", eri, dm),
        )
    apply = jax.jit(lambda bound, dm: bound.get_jk(dm))
    actual = apply(provider, density)
    expected = explicit(density)
    for value, oracle in zip(actual, expected, strict=True):
        np.testing.assert_allclose(value, oracle, atol=2e-12, rtol=0)
    _, tangent = jax.jvp(lambda dm: apply(provider, dm), (density,), (direction,))
    for value, oracle in zip(tangent, explicit(direction), strict=True):
        np.testing.assert_allclose(value, oracle, atol=2e-12, rtol=0)
    weights = (direction, density.T)
    _, actual_vjp = jax.vjp(lambda dm: apply(provider, dm), density)
    _, expected_vjp = jax.vjp(explicit, density)
    np.testing.assert_allclose(
        actual_vjp(weights)[0], expected_vjp(weights)[0], atol=2e-12, rtol=0
    )
    assert len(jax.tree.leaves(provider)) == len(jax.tree.leaves(parameters))
    with pytest.raises(NotImplementedError, match="density AD only"):
        jax.jvp(
            lambda centers: replace(
                provider, parameters=replace(parameters, centers=centers)
            ).get_jk(density)[0],
            (parameters.centers,),
            (jnp.ones_like(parameters.centers),),
        )


def test_explicit_local_pt2_keeps_its_existing_array_reference():
    mol = gto.M(atom="H 0 0 0; H 0 0 .8", basis="6-31g")
    mf = scf.RHF(mol, jk_backend="direct", conv_tol=1e-12).run()
    reference = _build_restricted_reference(
        atom=mf._spec(),
        basis=mol.basis,
        xc_spec="hf",
        rks_config=mf._config(),
        scf_inputs=mf._scf_inputs,
        scf_result=mf.scf_result,
        compute_local_pt2_features=True,
    )
    assert reference.direct_basis is None
    assert reference.eri_pair_matrix.size > 0
    assert reference.pt2_local is not None and np.isfinite(reference.pt2_local).all()


def test_projected_direct_provider_retains_dynamic_contraction():
    top, parameters = integrals.prepare_basis(
        "H 0 0 0; H 0 0 .8", "6-31g", cart=True
    )
    plan = integrals.make_plan(top)
    eri = plan.evaluate("eri", parameters)
    transform = jnp.arange(top.nao * 2, dtype=jnp.float64).reshape(top.nao, 2) / 13
    density = jnp.asarray([[.8, -.3], [.2, .5]])
    provider = ProjectedNativeDirectBasis(NativeDirectBasis(plan, parameters), transform)

    def actual(bound):
        return sum(jnp.sum(x**2) for x in bound.get_jk(density))

    def explicit(t):
        dm = t @ density @ t.T
        j = t.T @ jnp.einsum("pqrs,rs->pq", eri, dm) @ t
        k = t.T @ jnp.einsum("prqs,rs->pq", eri, dm) @ t
        return jnp.sum(j**2) + jnp.sum(k**2)

    np.testing.assert_allclose(jax.jit(actual)(provider), explicit(transform), atol=2e-12)
    np.testing.assert_allclose(
        jax.jit(jax.grad(lambda t: actual(replace(provider, transform=t))))(transform),
        jax.grad(explicit)(transform),
        atol=2e-11,
        rtol=0,
    )


def test_external_xc_scf_keeps_its_existing_array_contract():
    from gradscf import dft

    mol = gto.M(atom="H 0 0 0; H 0 0 .8", basis="sto-3g")
    functional = dft.custom_xc(
        input_fn=lambda state: (state.rho, state.weights),
        energy_fn=lambda params, data: params["scale"] * jnp.dot(data[1], .08 * data[0]**2))
    options = dict(xc=functional, params={"scale": jnp.asarray(.3)}, grids_level=0,
                   conv_tol=1e-11, conv_tol_grad=1e-8, max_cycle=60)
    actual = dft.RKS(mol, jk_backend="direct", **options).run()
    reference = dft.RKS(mol, jk_backend="full", **options).run()
    assert actual.converged and reference.converged
    np.testing.assert_allclose(actual.e_tot, reference.e_tot, atol=2e-10, rtol=0)
