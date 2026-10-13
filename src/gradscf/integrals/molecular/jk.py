"""DF and packed density contractions, with explicit legacy direct dispatch."""
from functools import partial
import math
import jax
import jax.numpy as jnp
import numpy as np
from jax.lax import Precision
from jaxtyping import Array
from .eri import packed_eri_shape, _metadata_arrays, _pair_metadata


def __getattr__(name):
    # CartesianBasis direct calls remain a reference-backend contract until
    # their callers migrate to IntegralPlan.get_jk / NativeDirectBasis.
    if name not in {'DirectJKResult', 'build_direct_jk_from_basis',
                    'build_direct_jk_incremental', '_DIRECT_PACKED_JK_MAX_NAO'}:
        raise AttributeError(name)
    from gradscf.integrals.backends.jax_reference import direct_jk
    value = getattr(direct_jk, name)
    globals()[name] = value
    return value

@jax.jit
def build_jk_from_df(df_factors: Array, density: Array) -> tuple[Array, Array]:
    """Build Coulomb and exchange matrices from DF factors.

    Uses a vmapped contraction over auxiliary channels so the J/K path stays in
    pure JAX and remains jit-friendly on CPU/GPU.
    """

    factors = jnp.asarray(df_factors)
    density = jnp.asarray(density)
    if factors.shape[0] == 0:
        zeros = jnp.zeros_like(density)
        return zeros, zeros

    projected = jnp.matmul(factors, density, precision=Precision.HIGHEST)
    rho_aux = jnp.einsum("Qpq,pq->Q", factors, density, precision=Precision.HIGHEST)
    j_mat = jnp.einsum(
        "Q,Qpq->pq",
        rho_aux,
        factors,
        precision=Precision.HIGHEST,
    )
    k_mat = jnp.einsum(
        "Qps,Qqs->pq",
        projected,
        factors,
        precision=Precision.HIGHEST,
    )
    return 0.5 * (j_mat + j_mat.T), 0.5 * (k_mat + k_mat.T)

@partial(jax.jit, static_argnames=("nocc",))
def build_jk_from_df_orbitals(
    df_factors: Array,
    density: Array,
    mo_coeff: Array,
    mo_occ: Array,
    *,
    nocc: int,
) -> tuple[Array, Array]:
    """Build DF J/K using occupied orbitals for the exchange contraction.

    For a density matrix assembled from orbitals, the K contraction can use
    ``B_Q C_occ`` instead of ``B_Q D``. This
    reduces the large exchange intermediate from ``naux * nao * nao`` to
    ``naux * nao * nocc`` for closed-shell RKS.
    """

    factors = jnp.asarray(df_factors)
    density = jnp.asarray(density)
    coeff = jnp.asarray(mo_coeff, dtype=factors.dtype)
    occ = jnp.asarray(mo_occ, dtype=factors.dtype)
    nocc_int = int(nocc)
    if factors.shape[0] == 0:
        zeros = jnp.zeros_like(density)
        return zeros, zeros
    if nocc_int <= 0 or nocc_int > coeff.shape[-1]:
        raise ValueError("nocc must be in the range [1, nmo].")

    rho_aux = jnp.einsum("Qpq,pq->Q", factors, density, precision=Precision.HIGHEST)
    j_mat = jnp.einsum(
        "Q,Qpq->pq",
        rho_aux,
        factors,
        precision=Precision.HIGHEST,
    )
    coeff_occ = coeff[:, :nocc_int]
    occ_scale = jnp.sqrt(jnp.maximum(occ[:nocc_int], 0.0))
    weighted_occ = coeff_occ * occ_scale[None, :]
    b_occ = jnp.einsum(
        "Qpr,ri->Qpi",
        factors,
        weighted_occ,
        precision=Precision.HIGHEST,
    )
    k_mat = jnp.einsum(
        "Qpi,Qqi->pq",
        b_occ,
        b_occ,
        precision=Precision.HIGHEST,
    )
    return 0.5 * (j_mat + j_mat.T), 0.5 * (k_mat + k_mat.T)

@jax.jit
def build_j_from_df(df_factors: Array, density: Array) -> Array:
    """Build Coulomb matrix only from DF factors.

    Useful for pure semilocal functionals where HF exchange is absent.
    """

    factors = jnp.asarray(df_factors)
    density = jnp.asarray(density)
    if factors.shape[0] == 0:
        return jnp.zeros_like(density)

    rho_aux = jnp.einsum("Qpq,pq->Q", factors, density, precision=Precision.HIGHEST)
    j_mat = jnp.einsum(
        "Q,Qpq->pq",
        rho_aux,
        factors,
        precision=Precision.HIGHEST,
    )
    return 0.5 * (j_mat + j_mat.T)


def exchange_matrix(eri, density, *, block_size=None):
    """K[...,p,q] = sum_rs (pr|qs) D[...,r,s], with bounded transposes.

    Some CPU XLA versions silently return zero for the full-tensor exchange
    transpose when the ERI exceeds 32-bit indexing. Slice its first AO axis
    before contracting; all arithmetic remains in JAX, including higher AD.
    ``block_size`` can force the bounded path on small regression inputs.
    """
    eri,density=jnp.asarray(eri),jnp.asarray(density)
    n=eri.shape[0]
    if eri.shape!=(n,)*4 or density.shape[-2:]!=(n,n):
        raise ValueError('Exchange requires (n,n,n,n) ERI and (...,n,n) density.')
    if block_size is None and math.prod(eri.shape)<2**31:
        return jnp.einsum('prqs,...rs->...pq',eri,density,precision=jax.lax.Precision.HIGHEST)
    width=max(1,min(n,(2**26)//n**3)) if block_size is None else int(block_size)
    if width<1:raise ValueError('block_size must be positive.')
    width=min(width,n)
    result=jnp.zeros(density.shape[:-2]+(n,n),dtype=jnp.result_type(eri,density))
    def contract(block):
        return jnp.einsum('prqs,...rs->...pq',block,density,precision=jax.lax.Precision.HIGHEST)
    def advance(i,out):
        block=jax.lax.dynamic_slice_in_dim(eri,i*width,width,axis=0)
        return jax.lax.dynamic_update_slice(out,contract(block),(0,)*(density.ndim-2)+(i*width,0))
    result=jax.lax.fori_loop(0,n//width,advance,result)
    start=(n//width)*width
    if start<n:result=result.at[...,start:,:].set(contract(eri[start:]))
    return result


@jax.jit
def _build_jk_from_packed_jax(eri: Array, density: Array) -> tuple[Array,Array]:
    """J/K from s4 or s8 real spatial ERIs, including arbitrary complex D.

    Bounded pair/row gathers avoid expanding the packed buffer to N^4.
    Unlike a Hermitian-only shortcut, this preserves all spin-offdiagonal
    exchange blocks. The entire contraction is JAX differentiable.
    """
    eri,density=jnp.asarray(eri),jnp.asarray(density);n=density.shape[-1]
    if eri.ndim not in (1,2) or eri.shape!=packed_eri_shape(n,eri.ndim) or density.shape[-2:]!=(n,n):
        raise ValueError('Packed ERI/density shapes do not match.')
    rows,cols,pair,_=_pair_metadata(n)
    rows,cols,pair=jnp.asarray(rows),jnp.asarray(cols),jnp.asarray(pair,dtype=jnp.int64)
    npair=len(rows)
    def fetch(a,b):
        if eri.ndim==2:return eri[a,b]
        hi=jnp.maximum(a,b);lo=jnp.minimum(a,b)
        return eri[hi*(hi+1)//2+lo]
    d= density[...,rows,cols]+jnp.where(rows!=cols,density[...,cols,rows],0.)
    if eri.ndim==2 and eri.size<2**31:
        jp=jnp.einsum('ab,...b->...a',eri,d,precision=Precision.HIGHEST)
    else:
        width=min(128,npair);ids=jnp.arange(npair,dtype=jnp.int64)
        jp=jnp.zeros(d.shape,dtype=jnp.result_type(eri,density))
        def step(i,out):
            start=i*width
            index=start+jnp.arange(width,dtype=jnp.int64)
            values=fetch(index[:,None],ids[None,:])
            value=jnp.einsum('ab,...b->...a',values,d,precision=Precision.HIGHEST)
            return jax.lax.dynamic_update_slice(out,value,(0,)*(out.ndim-1)+(start,))
        jp=jax.lax.fori_loop(0,npair//width,step,jp)
        start=(npair//width)*width
        if start<npair:
            value=jnp.einsum('ab,...b->...a',fetch(ids[start:,None],ids[None,:]),d,precision=Precision.HIGHEST)
            jp=jp.at[...,start:].set(value)
    j=jnp.zeros(density.shape,dtype=jp.dtype)
    j=j.at[...,rows,cols].set(jp);j=j.at[...,cols,rows].set(jp)
    def k_row(p):
        block=fetch(pair[p,:][None,:,None],pair[:,None,:])
        return jnp.einsum('qrs,...rs->...q',block,density,precision=Precision.HIGHEST)
    k=jnp.moveaxis(jax.lax.map(k_row,jnp.arange(n)),0,-2)
    return j,k

@jax.jit
def build_jk_from_packed(eri: Array, density: Array) -> tuple[Array,Array]:
    """Consume s4/s8 directly; s8 uses a streaming native CPU J/K kernel.

    S4 and non-float64 inputs use the portable bounded JAX path. GPU lowering
    also uses that path. Density and ERI derivatives remain available.
    """
    eri,density=jnp.asarray(eri),jnp.asarray(density)
    if eri.ndim==1 and eri.dtype==jnp.float64 and density.dtype in (jnp.float64,jnp.complex128):
        n=density.shape[-1]
        if eri.shape!=packed_eri_shape(n,1) or density.shape[-2:]!=(n,n):raise ValueError('Packed ERI/density shapes do not match.')
        from gradscf.integrals.backends.native.jk import packed_jk
        return packed_jk(eri,density)
    return _build_jk_from_packed_jax(eri,density)

@jax.jit
def build_j_from_eri_pair_matrix(eri_pair_matrix: Array, density: Array) -> Array:
    """Build exact Coulomb matrix from an AO-pair packed ERI matrix."""

    pair = jnp.asarray(eri_pair_matrix)
    density = 0.5 * (jnp.asarray(density) + jnp.asarray(density).T)
    nao = int(density.shape[0])
    rows, cols, _, multiplicity = _metadata_arrays(nao, density.dtype)
    density_pair = density[rows, cols] * multiplicity
    j_pair = pair @ density_pair
    # Each triangle has unique indices. Halve the diagonal before adding both
    # triangles: equivalent to assignment, with a Taylor-compatible primitive.
    values = j_pair*jnp.where(rows==cols, .5, 1.)
    j_mat = jnp.zeros_like(density)
    j_mat = j_mat.at[rows, cols].add(values)
    j_mat = j_mat.at[cols, rows].add(values)
    return 0.5 * (j_mat + j_mat.T)

@jax.jit
def build_jk_from_eri_pair_matrix(eri_pair_matrix: Array, density: Array) -> tuple[Array, Array]:
    """Build exact Coulomb and exchange matrices from packed no-DF ERIs.

    ``eri_pair_matrix`` follows PySCF's ``aosym='s4'`` lower-triangle AO-pair
    layout: ``M[pair(p,q), pair(r,s)] = (pq|rs)``.
    """

    return build_jk_from_packed(eri_pair_matrix,density)
