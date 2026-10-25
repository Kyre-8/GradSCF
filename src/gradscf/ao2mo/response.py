"""Occupied/virtual MO slices used by the response solvers."""
from functools import partial

import jax
import jax.numpy as jnp
from jax.lax import Precision
from jaxtyping import Array

from gradscf.ao2mo.blocks import _mo_pair_products
from gradscf.integrals.molecular.eri import _metadata_arrays


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
