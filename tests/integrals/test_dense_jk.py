"""Exchange contraction, including CPU transpose-index overflow regression."""
import os
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from gradscf.scf.rks import _build_jk


@pytest.mark.skipif(os.environ.get('GRADSCF_LARGE_ERI_TEST')!='1',reason='Opt-in: large real/complex ERI test; allow 160 GiB CPU memory')
def test_exchange_above_uint32_tensor_size_is_not_zero():
    n=257
    eri=jnp.ones((n,)*4,dtype=jnp.float64)
    density=jnp.eye(n,dtype=jnp.float64)
    j,k=jax.jit(_build_jk)(eri,density)
    np.testing.assert_allclose(j,np.full((n,n),n),atol=1e-12,rtol=0)
    np.testing.assert_allclose(k,np.full((n,n),n),atol=1e-12,rtol=0)
    from gradscf.integrals.basis.contraction import contract_integrals
    t=jnp.eye(n,dtype=jnp.float64)[:,[0,n//2,n-1]]
    contracted=jax.jit(lambda e,t:contract_integrals({'eri':e},t)['eri'])(eri,t)
    np.testing.assert_allclose(contracted,np.ones((3,)*4),atol=1e-12,rtol=0)
    from gradscf.scf.gks import generalized_jk
    spin=jnp.asarray([[1.,.2j],[-.2j,.5]],dtype=jnp.complex128)
    spin_density=jnp.kron(spin,density)
    gj,gk=jax.jit(generalized_jk)(eri,spin_density)
    np.testing.assert_allclose(gj,np.kron(np.eye(2),np.full((n,n),1.5*n)),atol=1e-10,rtol=0)
    np.testing.assert_allclose(gk,np.kron(np.asarray(spin),np.full((n,n),n)),atol=1e-10,rtol=0)
    # The public restricted facade can choose packed integrals instead.
    from gradscf.integrals.molecular.jk import build_jk_from_eri_pair_matrix
    del eri,gj,gk,contracted,t
    npair=n*(n+1)//2
    pair=jnp.ones((npair,npair),dtype=jnp.float64)
    pj,pk=build_jk_from_eri_pair_matrix(pair,density)
    np.testing.assert_allclose(pj,np.full((n,n),n),atol=1e-12,rtol=0)
    np.testing.assert_allclose(pk,np.full((n,n),n),atol=1e-12,rtol=0)


@pytest.mark.parametrize('complex_density',[False,True])
def test_blocked_exchange_matches_numpy_and_supports_higher_ad(complex_density):
    from gradscf.integrals.molecular.jk import exchange_matrix
    rng=np.random.default_rng(27);n=5
    eri=jnp.asarray(rng.normal(size=(n,)*4))
    density=rng.normal(size=(2,n,n))
    if complex_density:density=density+1j*rng.normal(size=density.shape)
    density=jnp.asarray(density)
    actual=jax.jit(lambda d:exchange_matrix(eri,d,block_size=2))(density)
    expected=np.einsum('prqs,brs->bpq',np.asarray(eri),np.asarray(density))
    np.testing.assert_allclose(actual,expected,atol=1e-12)
    f=lambda d:jnp.sum(jnp.abs(exchange_matrix(eri,d,block_size=2))**2)
    ref=lambda d:jnp.sum(jnp.abs(jnp.einsum('prqs,brs->bpq',eri,d))**2)
    tangent=jnp.ones_like(density)
    np.testing.assert_allclose(jax.grad(f)(density),jax.grad(ref)(density),atol=1e-10)
    np.testing.assert_allclose(jax.jvp(jax.grad(f),(density,),(tangent,))[1],
                               jax.jvp(jax.grad(ref),(density,),(tangent,))[1],atol=1e-10)
    fe=lambda e:jnp.sum(jnp.abs(exchange_matrix(e,density,block_size=2))**2)
    re=lambda e:jnp.sum(jnp.abs(jnp.einsum('prqs,brs->bpq',e,density))**2)
    np.testing.assert_allclose(jax.grad(fe)(eri),jax.grad(re)(eri),atol=1e-10)
