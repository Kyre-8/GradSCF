"""Shared AO->MO blocks and real post-HF reference transformations."""
from functools import partial
from math import prod
from numbers import Integral
import jax
import jax.numpy as jnp
from jax.lax import Precision
from jaxtyping import Array
from gradscf._memory import DEFAULT_MAX_MEMORY_MB, array_storage_bytes, workspace_bytes
from .eri import _metadata_arrays


def validate_integrals(h1, eri, nmo=None):
    h1, eri = jnp.asarray(h1), jnp.asarray(eri)
    if jnp.iscomplexobj(h1) or jnp.iscomplexobj(eri):
        raise NotImplementedError(
            "MO currently requires real integrals and real orbitals"
        )
    if not jnp.issubdtype(h1.dtype, jnp.floating) or not jnp.issubdtype(
        eri.dtype, jnp.floating
    ):
        raise ValueError("MO integrals must have floating-point dtype")
    if h1.ndim != 2 or h1.shape[0] != h1.shape[1]:
        raise ValueError("h1 must be square")
    if nmo is not None and h1.shape != (nmo, nmo):
        raise ValueError("MO integral dimensions do not match the MO space")
    if eri.shape != h1.shape * 2:
        raise ValueError(
            "eri must have shape (nmo, nmo, nmo, nmo), in chemists' notation"
        )
    return h1, eri


def transform_integrals(
    hcore, mo_coeff, *, eri=None, eri_pair_matrix=None, df_factors=None
):
    """Transform full, s4 pair, or density-fitted AO integrals using JAX.

    Exactly one ERI representation is required. The resulting full MO ERI tensor
    uses O(nmo**4) memory; this is a small-system reference implementation.
    """
    c, h = jnp.asarray(mo_coeff), jnp.asarray(hcore)
    if jnp.iscomplexobj(c) or jnp.iscomplexobj(h):
        raise NotImplementedError(
            "Post-HF transformation currently requires real orbitals and integrals"
        )
    if c.ndim != 2 or h.shape != (c.shape[0], c.shape[0]):
        raise ValueError("Inconsistent AO Hamiltonian and MO coefficient dimensions")
    g = _transform_eri(c, c, eri=eri, eri_pair_matrix=eri_pair_matrix, df_factors=df_factors)
    return validate_integrals(c.T @ h @ c, g)


def _transform_eri(c, d, *, eri=None, eri_pair_matrix=None, df_factors=None):
    return transform_eri_block((c, c, d, d), eri=eri,
                               eri_pair_matrix=eri_pair_matrix, df_factors=df_factors)


def transform_eri_block(
    coefficients, *, eri=None, eri_pair_matrix=None, df_factors=None,
    max_memory=DEFAULT_MAX_MEMORY_MB,
):
    """Transform a requested (p q|r s) block, without constructing full MO ERIs.

    Four real coefficient matrices may select different orbital subspaces.
    Dense, s4 packed and DF AO representations share the same convention.

    ``max_memory`` is a static planning budget in decimal MB. Known resident
    inputs, output buffers and transformation scratch share the budget after the
    common 20% runtime reserve. DF accumulation reserves three output buffers;
    dense and s4 reserve two. DF scratch is tiled over the auxiliary axis; dense
    and s4 inputs remain small-system, resident-AO reference paths.
    Checkpointing recomputes DF intermediates for differentiation. This plans
    forward arrays, not total AD-process storage or the required input cotangent.
    A traced input may not expose its target device. Without an explicitly
    selected default device, only the shared helper's host-capacity cap applies.
    """
    if len(coefficients) != 4:
        raise ValueError("Supply four orbital coefficient matrices")
    c, d, e, f = map(jnp.asarray, coefficients)
    if any(x.ndim != 2 or x.shape[0] != c.shape[0] for x in (c, d, e, f)):
        raise ValueError("Coefficient matrices must share the AO dimension")
    if any(jnp.iscomplexobj(x) for x in (c, d, e, f)):
        raise NotImplementedError("MO block transformation requires real coefficients")
    if sum(value is not None for value in (eri, eri_pair_matrix, df_factors)) != 1:
        raise ValueError("Supply exactly one of eri, eri_pair_matrix, df_factors")
    nao = c.shape[0]
    if eri is not None:
        representation = "dense"
        source = jnp.asarray(eri)
        if source.shape != (nao,) * 4:
            raise ValueError("AO eri must have shape (nao, nao, nao, nao)")
    elif eri_pair_matrix is not None:
        representation = "packed"
        source = jnp.asarray(eri_pair_matrix)
        npair = nao * (nao + 1) // 2
        if source.shape != (npair, npair):
            raise ValueError("eri_pair_matrix must be a square s4 AO pair matrix")
    else:
        representation = "df"
        source = jnp.asarray(df_factors)
        if source.ndim != 3 or source.shape[1:] != (nao, nao):
            raise ValueError("df_factors must have shape (naux, nao, nao)")

    coefficients = (c, d, e, f)
    shape = tuple(x.shape[1] for x in coefficients)
    dtype = jnp.result_type(source, *coefficients)
    block_size = _eri_block_size(source, coefficients, representation, max_memory)
    if not prod(shape) or (representation == "df" and source.shape[0] == 0):
        return jnp.zeros(shape, dtype=dtype)
    if representation == "dense":
        return jnp.einsum(
            "pqrs,pP,qQ,rR,sS->PQRS", source, c, d, e, f,
            precision="highest", optimize="optimal",
        )
    if representation == "packed":
        rows, cols, _, _ = _metadata_arrays(nao, c.dtype)
        products = _mo_pair_products(c, d, rows, cols)
        other = _mo_pair_products(e, f, rows, cols)
        return jnp.einsum(
            "pqP,PQ,rsQ->pqrs", products, source, other, precision="highest"
        )
    return _transform_df_eri_block(source, coefficients, shape, dtype, block_size)


def _eri_block_size(source, coefficients, representation, max_memory):
    """Check static forward-array storage and choose the DF auxiliary tile."""
    arrays = (source, *coefficients)
    device = None
    for array in arrays:
        device = getattr(array, "device", None)
        if callable(device):
            device = device()
        if device is not None:
            break
    if device is None:
        device = getattr(jax.config, "jax_default_device", None)
    available = workspace_bytes(
        max_memory, resident_bytes=array_storage_bytes(*arrays), device=device,
    )
    nao = coefficients[0].shape[0]
    shape = tuple(x.shape[1] for x in coefficients)
    dtype = jnp.result_type(*arrays)
    itemsize = dtype.itemsize
    # DF can simultaneously retain the old accumulator, a tile contribution
    # and the new accumulator, including during eager tail accumulation.
    output_buffers = 3 if representation == "df" else 2
    available -= output_buffers * prod(shape) * itemsize
    if available < 0:
        raise ValueError("AO->MO output buffers exceed the working-memory budget")
    if not prod(shape) or (representation == "df" and source.shape[0] == 0):
        return 0
    # Promotion may create coefficient copies inside contractions. In the DF
    # branch the AO factors themselves are promoted only after taking a tile.
    coefficient_copies = sum(x.size * itemsize for x in coefficients
                             if x.dtype != dtype)
    available -= coefficient_copies

    if representation == "dense":
        # Bound the three successive one-leg intermediates of the existing
        # optimized contraction, plus a possible promoted resident-AO copy.
        width = max(shape)
        scratch = itemsize * sum(nao ** (4 - n) * width ** n for n in (1, 2, 3))
        if source.dtype != dtype:
            scratch += source.size * itemsize
        if available < scratch:
            raise ValueError("Dense AO->MO scratch exceeds the working-memory budget")
        return None
    if representation == "packed":
        npair = source.shape[0]
        left_size, right_size = shape[0] * shape[1], shape[2] * shape[3]
        # Pair metadata, gathered coefficients, pair-product construction and
        # the larger pair-contraction intermediate remain explicit scratch.
        metadata = 8 * npair + 4 * nao ** 2 + itemsize * npair
        scratch = metadata + itemsize * npair * (
            2 * sum(shape) + 4 * (left_size + right_size)
            + max(left_size, right_size) + 1
        )
        if source.dtype != dtype:
            scratch += source.size * itemsize
        if available < scratch:
            raise ValueError("Packed AO->MO scratch exceeds the working-memory budget")
        return None
    per_aux = itemsize * (
        nao ** 2 + nao * (min(shape[:2]) + min(shape[2:]))
        + shape[0] * shape[1] + shape[2] * shape[3]
    )
    if any(x.dtype != dtype for x in arrays):
        # Retain room for original and promoted versions of tile-local
        # arrays without casting the full resident factor tensor.
        per_aux *= 2
    if available < per_aux:
        raise ValueError("Minimum DF AO->MO tile exceeds the working-memory budget")
    return min(source.shape[0], available // per_aux)


def _transform_df_eri_block(source, coefficients, shape, dtype, block_size):
    """Accumulate exact MO blocks using bounded, recomputed DF intermediates."""
    c, d, e, f = coefficients

    def pair_transform(block, left, right):
        if left.shape[1] <= right.shape[1]:
            half = jnp.einsum("Lpq,pP->LPq", block, left, precision="highest")
            return jnp.einsum("LPq,qQ->LPQ", half, right, precision="highest")
        half = jnp.einsum("Lpq,qQ->LpQ", block, right, precision="highest")
        return jnp.einsum("LpQ,pP->LPQ", half, left, precision="highest")

    def contribution(block, c, d, e, f):
        left = pair_transform(block, c, d)
        right = pair_transform(block, e, f)
        return jnp.einsum("Lpq,Lrs->pqrs", left, right, precision="highest")

    @jax.checkpoint
    def accumulate(index, value, factors, c, d, e, f):
        block = jax.lax.dynamic_slice_in_dim(
            factors, index * block_size, block_size, axis=0,
        )
        return value + contribution(block, c, d, e, f)

    nblocks, tail = divmod(source.shape[0], block_size)
    g = jax.lax.fori_loop(
        0, nblocks, lambda i, value: accumulate(i, value, source, c, d, e, f),
        jnp.zeros(shape, dtype=dtype),
    )
    if tail:
        g = g + jax.checkpoint(contribution)(
            source[nblocks * block_size:], c, d, e, f,
        )
    return g


def validate_unrestricted_integrals(h1, eri):
    """h1=(ha,hb), eri=(gaa,gab,gbb), with gab[pqrs]=(pa qa|rb sb)."""
    if len(h1) != 2 or len(eri) != 3:
        raise ValueError("Unrestricted integrals require two h1 and three ERI blocks")
    ha, gaa = validate_integrals(h1[0], eri[0])
    hb, gbb = validate_integrals(h1[1], eri[2], ha.shape[0])
    _, gab = validate_integrals(ha, eri[1], ha.shape[0])
    return (ha, hb), (gaa, gab, gbb)


def transform_unrestricted_integrals(hcore, mo_coeff, **kwargs):
    """Real alpha/beta MO transform from full, s4 or density-fitted AO ERIs."""
    if len(mo_coeff) != 2:
        raise ValueError("Supply alpha and beta MO coefficients")
    ca, cb = map(jnp.asarray, mo_coeff)
    if ca.shape != cb.shape:
        raise ValueError("Alpha and beta MO spaces must have equal dimensions")
    ha, gaa = transform_integrals(hcore, ca, **kwargs)
    hb, gbb = transform_integrals(hcore, cb, **kwargs)
    gab = _transform_eri(ca, cb, **kwargs)
    return validate_unrestricted_integrals((ha, hb), (gaa, gab, gbb))


def unrestricted_frozen_indices(nmo, nocc, frozen=None):
    """A core count/shared list, or (alpha_list, beta_list) for separate spaces."""
    if len(nocc) != 2 or any(not isinstance(n, Integral) or isinstance(n, bool)
                             or not 0 <= n <= nmo for n in nocc):
        raise ValueError("nocc must be (nalpha, nbeta) with integer occupations")
    separate = (isinstance(frozen, (tuple, list)) and len(frozen) == 2
                and all(isinstance(f, (tuple, list)) for f in frozen))
    choices = frozen if separate else (frozen, frozen)
    return tuple(frozen_indices(nmo, n, f) for n, f in zip(nocc, choices))


def spin_orbital_integrals(h1, eri):
    """Dense real chemists' spin integrals; all alpha orbitals precede beta.

    Reference implementation: O((2*nmo)**4) storage. No spin-flip matrix
    elements are inserted. The beta-alpha block is the pair transpose of gab.
    """
    (ha, hb), (gaa, gab, gbb) = validate_unrestricted_integrals(h1, eri)
    n = ha.shape[0]
    dtype = jnp.result_type(ha, hb, gaa, gab, gbb)
    h = jnp.zeros((2*n, 2*n), dtype=dtype).at[:n, :n].set(ha).at[n:, n:].set(hb)
    g = jnp.zeros((2*n,)*4, dtype=dtype)
    g = g.at[:n, :n, :n, :n].set(gaa).at[:n, :n, n:, n:].set(gab)
    g = g.at[n:, n:, :n, :n].set(gab.transpose(2, 3, 0, 1))
    return h, g.at[n:, n:, n:, n:].set(gbb)


def frozen_indices(nmo, nocc, frozen):
    if frozen is None:
        return ()
    if isinstance(frozen, Integral) and not isinstance(frozen, bool):
        if frozen < 0 or frozen > nocc:
            raise ValueError("frozen core count must lie between 0 and nocc")
        return tuple(range(frozen))
    try:
        indices = tuple(frozen)
    except TypeError as error:
        raise ValueError("frozen must be a core count or orbital index list") from error
    if any(
        not isinstance(i, Integral) or isinstance(i, bool) or i < 0 or i >= nmo
        for i in indices
    ):
        raise ValueError("Frozen orbital indices must be integers in [0, nmo)")
    if len(set(indices)) != len(indices):
        raise ValueError("Duplicate frozen orbital indices")
    return tuple(sorted(int(i) for i in indices))


def df_factors_to_mo_eri_slices(
    df_factors: Array,
    mo_coeff: Array,
    nocc: int,
    *,
    include_oovv: bool = True,
) -> tuple[Array, Array, Array | None]:
    coeff = jnp.asarray(mo_coeff)
    nocc_int = int(nocc)
    orbo = coeff[:, :nocc_int]
    orbv = coeff[:, nocc_int:]
    factors = jnp.asarray(df_factors)
    b_ov = jnp.einsum("Qpq,pi,qa->Qia", factors, orbo, orbv, precision=Precision.HIGHEST)
    b_vo = jnp.einsum("Qpq,pa,qi->Qai", factors, orbv, orbo, precision=Precision.HIGHEST)
    eri_ovov = jnp.einsum("Qia,Qjb->iajb", b_ov, b_ov, precision=Precision.HIGHEST)
    eri_ovvo = jnp.einsum("Qia,Qbj->iabj", b_ov, b_vo, precision=Precision.HIGHEST)
    if not include_oovv:
        return eri_ovov, eri_ovvo, None
    b_oo = jnp.einsum("Qpq,pi,qj->Qij", factors, orbo, orbo, precision=Precision.HIGHEST)
    b_vv = jnp.einsum("Qpq,pa,qb->Qab", factors, orbv, orbv, precision=Precision.HIGHEST)
    eri_oovv = jnp.einsum("Qij,Qab->ijab", b_oo, b_vv, precision=Precision.HIGHEST)
    return eri_ovov, eri_ovvo, eri_oovv


def _mo_pair_products(left: Array, right: Array, rows: Array, cols: Array) -> Array:
    left_rows = left[rows]
    left_cols = left[cols]
    right_rows = right[rows]
    right_cols = right[cols]
    products = jnp.einsum(
        "Pi,Pa->iaP",
        left_rows,
        right_cols,
        precision=Precision.HIGHEST,
    )
    swapped = jnp.einsum(
        "Pi,Pa->iaP",
        left_cols,
        right_rows,
        precision=Precision.HIGHEST,
    )
    offdiag = (rows != cols).astype(products.dtype)
    return products + swapped * offdiag[None, None, :]

@partial(jax.jit, static_argnames=("nocc", "include_oovv"))
def eri_pair_matrix_to_mo_eri_slices(
    eri_pair_matrix: Array,
    mo_coeff: Array,
    *,
    nocc: int,
    include_oovv: bool = True,
) -> tuple[Array, Array, Array | None]:
    """Transform packed no-DF AO ERIs into restricted TDDFT MO slices."""

    pair = jnp.asarray(eri_pair_matrix)
    coeff = jnp.asarray(mo_coeff)
    nocc_int = int(nocc)
    rows, cols, _, _ = _metadata_arrays(int(coeff.shape[0]), coeff.dtype)
    orbo = coeff[:, :nocc_int]
    orbv = coeff[:, nocc_int:]
    ov = _mo_pair_products(orbo, orbv, rows, cols)
    vo = _mo_pair_products(orbv, orbo, rows, cols)
    eri_ovov = jnp.einsum("iaP,PQ,jbQ->iajb", ov, pair, ov, precision=Precision.HIGHEST)
    eri_ovvo = jnp.einsum("iaP,PQ,bjQ->iabj", ov, pair, vo, precision=Precision.HIGHEST)
    if not include_oovv:
        return eri_ovov, eri_ovvo, None
    oo = _mo_pair_products(orbo, orbo, rows, cols)
    vv = _mo_pair_products(orbv, orbv, rows, cols)
    eri_oovv = jnp.einsum("ijP,PQ,abQ->ijab", oo, pair, vv, precision=Precision.HIGHEST)
    return eri_ovov, eri_ovvo, eri_oovv
