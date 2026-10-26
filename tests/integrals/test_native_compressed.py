"""Native compressed output must be produced without a dense ERI intermediate."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from gradscf import integrals


@pytest.mark.parametrize('cart',[False,True])
@pytest.mark.parametrize('layout',['s4','s8'])
def test_packed_values_and_jit_against_dense(cart,layout):
    top,p=integrals.prepare_basis('O 0 0 0; H 0 .75 .58; H 0 -.75 .58','6-31g(d)',cart=cart)
    plan=integrals.make_plan(top)
    full=np.asarray(plan.evaluate('eri',p));rows,cols=np.tril_indices(top.nao)
    pair=full[rows[:,None],cols[:,None],rows[None,:],cols[None,:]]
    expected=pair if layout=='s4' else pair[np.tril_indices(len(rows))]
    actual=jax.jit(lambda p:plan.evaluate('eri',p,aosym=layout))(p)
    np.testing.assert_allclose(actual,expected,atol=2e-11,rtol=1e-11)
    assert actual.shape==expected.shape


@pytest.mark.parametrize('layout',['s4','s8'])
def test_packed_jk_complex_density_and_density_ad(layout):
    from gradscf.integrals.molecular.jk import build_jk_from_packed
    top,p=integrals.prepare_basis('H 0 0 0; H .1 .2 .8','6-31g',cart=False)
    plan=integrals.make_plan(top);full=plan.evaluate('eri',p)
    packed=plan.evaluate('eri',p,aosym=layout)
    rng=np.random.default_rng(6);n=top.nao
    d=jnp.asarray(rng.normal(size=(2,n,n))+1j*rng.normal(size=(2,n,n)))
    def expected(d):return (jnp.einsum('pqrs,...rs->...pq',full,d),jnp.einsum('prqs,...rs->...pq',full,d))
    def actual(d):return build_jk_from_packed(packed,d)
    for a,b in zip(jax.jit(actual)(d),expected(d)):np.testing.assert_allclose(a,b,atol=1e-11)
    loss=lambda d:sum(jnp.sum(jnp.abs(x)**2) for x in actual(d))
    ref=lambda d:sum(jnp.sum(jnp.abs(x)**2) for x in expected(d))
    np.testing.assert_allclose(jax.grad(loss)(d),jax.grad(ref)(d),atol=1e-10)
    np.testing.assert_allclose(jax.jvp(jax.grad(loss),(d,),(jnp.ones_like(d),))[1],
                               jax.jvp(jax.grad(ref),(d,),(jnp.ones_like(d),))[1],atol=1e-10)


def test_native_direct_jk_matches_integrals():
    top,p=integrals.prepare_basis('O 0 0 0; H 0 .75 .58; H 0 -.75 .58','sto-3g',cart=False)
    plan=integrals.make_plan(top);eri=plan.evaluate('eri',p)
    d=jnp.asarray(np.random.default_rng(1).normal(size=(top.nao,top.nao)))
    j,k=plan.get_jk(p,d)
    np.testing.assert_allclose(j,jnp.einsum('pqrs,rs->pq',eri,d),atol=1e-11)
    np.testing.assert_allclose(k,jnp.einsum('prqs,rs->pq',eri,d),atol=1e-11)
    tangent=jnp.eye(top.nao)
    _,(dj,dk)=jax.jvp(lambda d:plan.get_jk(p,d),(d,),(tangent,))
    rj,rk=plan.get_jk(p,tangent)
    np.testing.assert_allclose(dj,rj,atol=1e-11);np.testing.assert_allclose(dk,rk,atol=1e-11)
    loss=lambda d:sum(jnp.sum(x*x) for x in plan.get_jk(p,d))
    ref=lambda d:jnp.sum(jnp.einsum('pqrs,rs->pq',eri,d)**2)+jnp.sum(jnp.einsum('prqs,rs->pq',eri,d)**2)
    np.testing.assert_allclose(jax.grad(loss)(d),jax.grad(ref)(d),atol=1e-10)


def test_projected_native_direct_jk_differentiates_contraction():
    from gradscf.integrals.backends.native.jk import NativeDirectBasis, ProjectedNativeDirectBasis
    from gradscf.integrals.basis.contraction import primitive_basis, contraction_matrix
    top,p=integrals.prepare_basis('H 0 0 0; H 0 0 .8','sto-3g',cart=False)
    primitive_top,primitive_parameters=primitive_basis(top,p)
    plan=integrals.make_plan(primitive_top)
    eri=plan.evaluate('eri',primitive_parameters)
    transform=contraction_matrix(top,p)
    density=jnp.asarray([[1.,.2],[.2,.7]])

    def actual(t):
        j,k=ProjectedNativeDirectBasis(
            NativeDirectBasis(plan,primitive_parameters),t
        ).get_jk(density)
        return jnp.sum(j*j)+.3*jnp.sum(k*k)

    def reference(t):
        primitive_density=t@density@t.T
        primitive_j=jnp.einsum('pqrs,rs->pq',eri,primitive_density)
        primitive_k=jnp.einsum('prqs,rs->pq',eri,primitive_density)
        j=t.T@primitive_j@t;k=t.T@primitive_k@t
        return jnp.sum(j*j)+.3*jnp.sum(k*k)

    np.testing.assert_allclose(actual(transform),reference(transform),atol=1e-11)
    np.testing.assert_allclose(jax.grad(actual)(transform),jax.grad(reference)(transform),atol=1e-10)


def test_packed_geometry_ad_is_not_silently_zero():
    top,p=integrals.prepare_basis('H 0 0 0; H 0 0 .8','sto-3g',cart=False)
    plan=integrals.make_plan(top)
    with pytest.raises((ValueError,NotImplementedError)):
        jax.jvp(lambda p:plan.evaluate('eri',p,aosym='s8'),(p,),
                (jax.tree.map(jnp.ones_like,p),))


@pytest.mark.parametrize('cart',[False,True])
def test_general_contractions_and_direct_screening(cart):
    basis=[[0,[2.,.8,-.2],[.4,.3,1.]], [1,[.8,1.]]]
    top,p=integrals.prepare_basis('H 0 0 0; H 0 0 1.1',basis,cart=cart)
    plan=integrals.make_plan(top);eri=plan.evaluate('eri',p)
    packed=plan.evaluate('eri',p,aosym='s8')
    from gradscf.integrals.molecular.jk import build_jk_from_packed
    d=jnp.asarray(np.random.default_rng(2).normal(size=(top.nao,top.nao)))
    refj=jnp.einsum('pqrs,rs->pq',eri,d);refk=jnp.einsum('prqs,rs->pq',eri,d)
    for actual in [build_jk_from_packed(packed,d),plan.get_jk(p,d,screening_threshold=1e-14)]:
        np.testing.assert_allclose(actual[0],refj,atol=1e-11)
        np.testing.assert_allclose(actual[1],refk,atol=1e-11)
    j,k=plan.get_jk(p,d,screening_threshold=1e9)
    np.testing.assert_array_equal(j,0.);np.testing.assert_array_equal(k,0.)


def test_direct_complex_density_hessian_and_basis_ad_guard():
    top,p=integrals.prepare_basis('H 0 0 0; H 0 0 .8','sto-3g',cart=False)
    plan=integrals.make_plan(top);eri=plan.evaluate('eri',p)
    d=jnp.asarray([[[1.,.2j],[-.1j,.7]],[[.3,.1],[.2,.8]]],dtype=jnp.complex128)
    def loss(d):return sum(jnp.sum(jnp.abs(x)**2) for x in plan.get_jk(p,d))
    def ref(d):return jnp.sum(jnp.abs(jnp.einsum('pqrs,brs->bpq',eri,d))**2)+jnp.sum(jnp.abs(jnp.einsum('prqs,brs->bpq',eri,d))**2)
    np.testing.assert_allclose(jax.jit(jax.grad(loss))(d),jax.grad(ref)(d),atol=1e-10)
    tangent=jnp.ones_like(d)
    np.testing.assert_allclose(jax.jvp(jax.grad(loss),(d,),(tangent,))[1],
                               jax.jvp(jax.grad(ref),(d,),(tangent,))[1],atol=1e-10)
    with pytest.raises((NotImplementedError, RuntimeError),match='(?i)(density AD|basis|exponent|coefficient)'):
        jax.jvp(lambda p:plan.get_jk(p,d),(p,),(jax.tree.map(jnp.ones_like,p),))


def test_s8_eri_vjp_and_mixed_second_derivative():
    from gradscf.integrals.molecular.jk import build_jk_from_packed, _build_jk_from_packed_jax
    n=3;npair=n*(n+1)//2
    e=jnp.arange(npair*(npair+1)//2,dtype=jnp.float64)*.03
    d=jnp.asarray(np.random.default_rng(8).normal(size=(n,n)))
    def energy(fn,e,d):
        j,k=fn(e,d)
        return jnp.sum(j*j)+.7*jnp.sum(k*k)
    f=lambda e,d:energy(build_jk_from_packed,e,d)
    ref=lambda e,d:energy(_build_jk_from_packed_jax,e,d)
    for actual,expected in zip(jax.grad(f,(0,1))(e,d),jax.grad(ref,(0,1))(e,d)):
        np.testing.assert_allclose(actual,expected,atol=1e-10)
    a=jax.jvp(lambda d:jax.grad(f,0)(e,d),(d,),(jnp.ones_like(d),))[1]
    b=jax.jvp(lambda d:jax.grad(ref,0)(e,d),(d,),(jnp.ones_like(d),))[1]
    np.testing.assert_allclose(a,b,atol=1e-10)


def test_s8_parallel_accumulation_with_batched_nonsymmetric_density():
    from gradscf.integrals.molecular.jk import build_jk_from_packed, _build_jk_from_packed_jax

    # Large enough to exercise multiple native AO-pair work partitions.
    n = 48
    pairs = n * (n + 1) // 2
    rng = np.random.default_rng(72)
    eri = jnp.asarray(rng.normal(size=pairs * (pairs + 1) // 2))
    density = jnp.asarray(rng.normal(size=(2, n, n)))
    expected = _build_jk_from_packed_jax(eri, density)
    for _ in range(3):
        for actual, reference in zip(build_jk_from_packed(eri, density), expected):
            np.testing.assert_allclose(actual, reference, atol=2e-10, rtol=1e-10)
