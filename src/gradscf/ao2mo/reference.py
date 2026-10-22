"""Complete real MO integral tensors for small-system reference calculations."""
import jax.numpy as jnp

from gradscf._memory import (DEFAULT_MAX_MEMORY_MB, tree_storage_bytes,
    validate_memory_mb, validate_reserved_bytes, workspace_bytes)
from gradscf.ao2mo.blocks import transform_eri_block


def transform_fock(fock, mo_coeff):
    """Transport a real Hermitian SCF Fock, retaining its physical response."""
    c, f = jnp.asarray(mo_coeff), jnp.asarray(fock)
    transformed = c.T @ f @ c
    return .5*(transformed+transformed.T)


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
    hcore, mo_coeff, *, eri=None, eri_pair_matrix=None, df_factors=None, direct_basis=None,
    max_memory=DEFAULT_MAX_MEMORY_MB, reserved_bytes=0,
):
    """Transform full, packed, density-fitted or native-direct AO integrals.

    Exactly one ERI representation is required. The resulting full MO ERI tensor
    uses O(nmo**4) memory; this is a small-system reference implementation.
    ``max_memory`` budgets each transform through the shared block planner;
    caller-owned ``reserved_bytes`` and the one-body input/output/workspace
    remain charged while that child constructs the MO ERIs.
    """
    max_memory = validate_memory_mb(max_memory)
    reserved_bytes = validate_reserved_bytes(reserved_bytes)
    c, h = jnp.asarray(mo_coeff), jnp.asarray(hcore)
    if jnp.iscomplexobj(c) or jnp.iscomplexobj(h):
        raise NotImplementedError(
            "Post-HF transformation currently requires real orbitals and integrals"
        )
    if c.ndim != 2 or h.shape != (c.shape[0], c.shape[0]):
        raise ValueError("Inconsistent AO Hamiltonian and MO coefficient dimensions")
    dtype = jnp.result_type(c, h)
    one_body = (c.shape[1]*(c.shape[0]+c.shape[1])
                +sum(a.size for a in (c, h) if a.dtype != dtype))*dtype.itemsize
    resident = tree_storage_bytes(h, exclude=(c, eri, eri_pair_matrix, df_factors))
    reservation = reserved_bytes+resident+one_body
    device = getattr(c, "device", None)
    workspace_bytes(max_memory, resident_bytes=reservation,
                    device=device() if callable(device) else device)
    g = _transform_eri(c, c, eri=eri, eri_pair_matrix=eri_pair_matrix,
                       df_factors=df_factors, direct_basis=direct_basis, max_memory=max_memory,
                       reserved_bytes=reservation)
    return validate_integrals(c.T @ h @ c, g)


def _transform_eri(c, d, *, eri=None, eri_pair_matrix=None, df_factors=None, direct_basis=None,
                   max_memory=DEFAULT_MAX_MEMORY_MB, reserved_bytes=0):
    return transform_eri_block((c, c, d, d), eri=eri,
                               eri_pair_matrix=eri_pair_matrix, df_factors=df_factors, direct_basis=direct_basis,
                               max_memory=max_memory, reserved_bytes=reserved_bytes)


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
