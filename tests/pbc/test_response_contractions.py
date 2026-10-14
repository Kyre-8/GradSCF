"""Bound contraction scratch space while preserving complex response and AD."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from gradscf.pbc.response import _fields, _project


def _dense_fields(d, a, b, w, kind):
    rho = jnp.einsum('k,kgp,skpq,kgq->sg', w, b[:,0], d, a[:,0].conj())
    if kind == 'LDA':
        return rho.T
    grad = jnp.einsum('k,kxgp,skpq,kgq->sgx', w, b[:,1:4], d, a[:,0].conj())
    grad += jnp.einsum('k,kgp,skpq,kxgq->sgx', w, b[:,0], d, a[:,1:4].conj())
    return jnp.concatenate([rho.T, grad[0], grad[1]], axis=1)


def _dense_project(a, b, w, v, kind):
    blocks = []
    for s in range(2):
        block = jnp.einsum('g,kgp,kgq->kpq', w*v[:,s], b[:,0].conj(), a[:,0])
        if kind == 'GGA':
            vg = w[:,None]*v[:,2+3*s:5+3*s]
            block += jnp.einsum('gx,kxgp,kgq->kpq', vg, b[:,1:4].conj(), a[:,0])
            block += jnp.einsum('gx,kgp,kxgq->kpq', vg, b[:,0].conj(), a[:,1:4])
        blocks.append(block)
    return jnp.stack(blocks)


@pytest.mark.parametrize('kind', ['LDA', 'GGA'])
@pytest.mark.parametrize('name', ['fields', 'project'])
def test_complex_contractions_and_derivatives(kind, name):
    rng = np.random.default_rng(71)
    def random(shape):
        return jnp.asarray(rng.normal(size=shape)+1j*rng.normal(size=shape))
    a, b = random((5,4,7,3)), random((5,4,7,3))
    if name == 'fields':
        functions = _dense_fields, _fields
        arrays = (random((2,5,3,3)), a, b, jnp.asarray([.1,.15,.2,.25,.3]))
    else:
        functions = _dense_project, _project
        arrays = (a, b, jnp.asarray(rng.random(7)), random((7,2 if kind=='LDA' else 8)))
    reference, actual = (lambda *x, fn=f:fn(*x,kind) for f in functions)
    tangents = tuple(jnp.ones_like(x)*.13 for x in arrays)
    for expected, observed in zip(jax.jvp(reference,arrays,tangents),
                                  jax.jvp(actual,arrays,tangents)):
        np.testing.assert_allclose(observed, expected, atol=2e-11, rtol=2e-12)
    def gradient(fn):
        return jax.grad(lambda *x:jnp.sum(jnp.abs(fn(*x))**2),
                        argnums=tuple(range(len(arrays))))(*arrays)
    for expected, observed in zip(gradient(reference),gradient(actual)):
        np.testing.assert_allclose(observed, expected, atol=2e-9, rtol=2e-12)


@pytest.mark.parametrize('name', ['fields', 'project'])
def test_cpu_scratch_does_not_scale_like_full_k_grid(name):
    if jax.default_backend() != 'cpu':
        pytest.skip('CPU compiler workspace regression')
    def scratch(nk):
        a = jax.ShapeDtypeStruct((nk,4,127,8), jnp.complex128)
        if name == 'fields':
            fn = lambda *x:_fields(*x,'GGA')
            arrays = (jax.ShapeDtypeStruct((2,nk,8,8),jnp.complex128), a, a,
                      jax.ShapeDtypeStruct((nk,),jnp.float64))
        else:
            fn = lambda *x:_project(*x,'GGA')
            arrays = (a, a, jax.ShapeDtypeStruct((127,),jnp.float64),
                      jax.ShapeDtypeStruct((127,8),jnp.complex128))
        stats = jax.jit(fn).lower(*arrays).compile().memory_analysis()
        if stats is None:
            pytest.skip('Compiler does not expose memory analysis')
        return stats.temp_size_in_bytes
    small, large = scratch(4), scratch(32)
    # Allow small matrix/loop bookkeeping growth; reject grid-sized k buffers.
    assert large <= small + 262144, (small, large)


@pytest.mark.parametrize('ao_dtype,weight_dtype',
                         [(jnp.complex64,jnp.float64),(jnp.complex128,jnp.float32)])
def test_fields_common_dtype_promotes_contraction_precision(ao_dtype,weight_dtype):
    rng = np.random.default_rng(17)
    def random(shape,dtype):
        return jnp.asarray(rng.normal(size=shape)+1j*rng.normal(size=shape),dtype=dtype)
    arrays = (random((2,5,8,8),jnp.complex64), random((5,4,13,8),ao_dtype),
              random((5,4,13,8),ao_dtype),jnp.asarray([.1,.15,.2,.25,.3],dtype=weight_dtype))
    expected = _dense_fields(*arrays,'GGA')
    actual = _fields(*arrays,'GGA')
    assert actual.dtype == expected.dtype == jnp.complex128
    np.testing.assert_allclose(actual, expected, atol=2e-11, rtol=2e-12)
