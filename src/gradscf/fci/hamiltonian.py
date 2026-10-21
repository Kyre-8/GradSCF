"""Real spin-free FCI Hamiltonian on an alpha-string x beta-string tensor.

H = h[pq] E_pq + .5 (pq|rs) (E_pq E_rs - delta_qr E_ps).
No determinant-pair connection table or dense FCI Hamiltonian is constructed.
"""
from numbers import Integral
import jax
import jax.numpy as jnp
from gradscf.ao2mo import validate_integrals
from .._memory import DEFAULT_MAX_MEMORY_MB, array_storage_bytes, workspace_bytes
from ..solvers import LinearOperator
from .cistring import FCISpace, _link_maps


DEFAULT_WORKSPACE = 8_000_000


def _action_tiles(space, dtype, nrhs, max_workspace_elements, max_memory, *,
                  arrays=(), extra_bytes=0):
    """Plan a string tile before enumerating links or gathering coefficients.

    The legacy element cap bounds P*A*B*R for one excitation tile. Real integral
    tensors are resident inputs charged to max_memory, rather than tile arrays.
    Six tile arrays and three immutable output versions are conservatively
    charged in addition to the shared runtime reserve.
    """
    if not isinstance(space, FCISpace):
        raise TypeError("Expected an FCISpace")
    if (not isinstance(max_workspace_elements, Integral)
            or isinstance(max_workspace_elements, bool) or max_workspace_elements < 1):
        raise ValueError("max_workspace_elements must be a positive integer")
    pairs = space.norb**2
    itemsize = jnp.dtype(dtype).itemsize
    # Source int32, numerical signs, cached int8 signs and occupations.
    maps = sum(space.shape) * (pairs * (4 + itemsize + 1) + space.norb)
    outputs = 3 * space.size * nrhs * itemsize
    device = next((getattr(a, "device", None) for a in arrays
                   if getattr(a, "device", None) is not None), None)
    available = workspace_bytes(max_memory, device=device,
        resident_bytes=array_storage_bytes(*arrays) + extra_bytes + maps + outputs)
    if not pairs or not nrhs:
        return 1, 1, max(1, nrhs)
    limit = min(max_workspace_elements, available // (6 * itemsize))
    rhs_width = min(nrhs, limit // pairs)
    if rhs_width < 1:
        raise ValueError("FCI smallest string tile exceeds the memory/workspace budget")
    na, nb = space.shape
    beta_width = min(nb, limit // (pairs * rhs_width))
    alpha_width = min(na, limit // (pairs * beta_width * rhs_width))
    # Keep the scalable action free of complete E|c> stacks even when a small
    # determinant space fits the budget. A single determinant is unavoidable.
    if alpha_width == na and beta_width == nb and space.size > 1:
        if na > 1:
            alpha_width = (na + 1) // 2
        else:
            beta_width = (nb + 1) // 2
    return alpha_width, beta_width, rhs_width


def coefficient_array(coefficients, space):
    c = jnp.asarray(coefficients)
    if c.shape not in (space.shape, (space.size,)):
        raise ValueError("FCI coefficients must have alpha-by-beta string shape or flat product shape")
    if jnp.iscomplexobj(c):
        raise NotImplementedError("FCI currently requires real coefficients")
    return c.astype(jnp.result_type(c, 1.)).reshape(space.shape)


def excitation_actions(c, space, spin=None):
    """Stack E_pq|c> over orbital pairs, optionally for one spin only."""
    out = None
    for channel in ((0, 1) if spin is None else (spin,)):
        source, signs, _ = _link_maps(space.norb, space.nelec[channel])
        if channel == 0:
            weights = jnp.asarray(signs, c.dtype).reshape(signs.shape + (1,)*(c.ndim-1))
            value = c[source] * weights
        else:
            weights = jnp.asarray(signs, c.dtype).reshape((signs.shape[0],1,signs.shape[1]) + (1,)*(c.ndim-2))
            value = jnp.moveaxis(c[:, source], 1, 0) * weights
        out = value if out is None else out + value
    return out


def _excitation_transpose(values, space):
    out = jnp.zeros(space.shape + values.shape[3:], values.dtype)
    for channel in (0, 1):
        source, signs, _ = _link_maps(space.norb, space.nelec[channel])
        if channel == 0:
            weights = jnp.asarray(signs).reshape(signs.shape + (1,)*(values.ndim-2))
            out = out.at[source].add(values * weights)
        else:
            weights = jnp.asarray(signs).reshape((signs.shape[0],1,signs.shape[1]) + (1,)*(values.ndim-3))
            update = (values * weights).swapaxes(1,2)
            out = out.swapaxes(0,1).at[source].add(update).swapaxes(0,1)
    return out


def contract_1e(h1, coefficients, space, *, max_workspace_elements=DEFAULT_WORKSPACE,
                max_memory=DEFAULT_MAX_MEMORY_MB):
    """Physical one-electron action with P*A*B*R-bounded string workspace."""
    if not isinstance(space, FCISpace):
        raise TypeError("Expected an FCISpace")
    c = coefficient_array(coefficients, space)
    h = jnp.asarray(h1)
    if h.shape != (space.norb, space.norb) or jnp.iscomplexobj(h):
        raise ValueError("h1 must be a real (norb,norb) matrix")
    c = c.astype(jnp.result_type(h, c))
    tiles = _action_tiles(space, c.dtype, 1, max_workspace_elements, max_memory,
        arrays=(h, c, coefficients), extra_bytes=h.size * c.dtype.itemsize if h.dtype != c.dtype else 0)
    return _contract_tiles(h.astype(c.dtype), c, space, tiles, two_electron=False)


def contract_2e(eri, coefficients, space, *, max_workspace_elements=DEFAULT_WORKSPACE,
                max_memory=DEFAULT_MAX_MEMORY_MB):
    """PySCF-style E E contraction, NOT the bare physical two-electron action.

    Use absorb_h1e(h1,eri,space,fac=.5) first for a complete Hamiltonian.
    Integrals must have real chemists' pair symmetries; no spin tensor is built.
    max_workspace_elements limits one P*A*B*R tile, and max_memory is decimal MB.
    """
    if not isinstance(space, FCISpace):
        raise TypeError("Expected an FCISpace")
    g = jnp.asarray(eri)
    n = space.norb
    if g.shape != (n,)*4 or jnp.iscomplexobj(g):
        raise ValueError("eri must be a real (norb,norb,norb,norb) tensor")
    c = coefficient_array(coefficients, space)
    c = c.astype(jnp.result_type(g, c))
    return _contract_ee(g, c, space, max_workspace_elements=max_workspace_elements,
                        max_memory=max_memory, resident_arrays=(coefficients,))


def _contract_ee(g, c, space, *, max_workspace_elements=DEFAULT_WORKSPACE,
                 max_memory=DEFAULT_MAX_MEMORY_MB, resident_arrays=()):
    nrhs = c.shape[2] if c.ndim == 3 else 1
    tiles = _action_tiles(space, c.dtype, nrhs, max_workspace_elements, max_memory,
        arrays=(g, c, *resident_arrays), extra_bytes=g.size * c.dtype.itemsize if g.dtype != c.dtype else 0)
    return _contract_tiles(g.astype(c.dtype), c, space, tiles, two_electron=True)


def _contract_tiles(kernel, c, space, tiles, *, two_electron):
    """Visit each intermediate determinant once per RHS tile, without E stacks."""
    original_shape = c.shape
    nrhs = c.shape[2] if c.ndim == 3 else 1
    c = c.reshape(space.shape + (nrhs,))
    if not space.norb or not nrhs:
        return jnp.zeros_like(c).reshape(original_shape)
    pairs = space.norb**2
    alpha_width, beta_width, rhs_width = tiles
    na, nb = space.shape
    alpha_tiles = (na + alpha_width - 1) // alpha_width
    beta_tiles = (nb + beta_width - 1) // beta_width
    rhs_tiles = (nrhs + rhs_width - 1) // rhs_width
    whole_beta_rhs = beta_width == nb and rhs_width == nrhs
    source_a, signs_a, _ = _link_maps(space.norb, space.nelec[0])
    source_b, signs_b, _ = _link_maps(space.norb, space.nelec[1])
    source_a, source_b = jnp.asarray(source_a), jnp.asarray(source_b)
    signs_a, signs_b = jnp.asarray(signs_a, c.dtype), jnp.asarray(signs_b, c.dtype)
    matrix = kernel.reshape((pairs, pairs) if two_electron else (pairs,))

    @jax.checkpoint
    def body(index, out):
        alpha = (index // (beta_tiles * rhs_tiles)) * alpha_width + jnp.arange(alpha_width)
        beta = (index // rhs_tiles % beta_tiles) * beta_width + jnp.arange(beta_width)
        rhs = (index % rhs_tiles) * rhs_width + jnp.arange(rhs_width)
        valid = ((alpha[:, None, None] < na) & (beta[None, :, None] < nb)
                 & (rhs[None, None, :] < nrhs))
        alpha = jnp.minimum(alpha, na - 1)
        beta, rhs = jnp.minimum(beta, nb - 1), jnp.minimum(rhs, nrhs - 1)
        alpha_source, beta_source = source_a[:, alpha], source_b[:, beta]
        alpha_sign = signs_a[:, alpha, None, None]
        beta_sign = signs_b[:, None, beta, None]
        if whole_beta_rhs:
            # Gather complete rows without expanding three scalar index arrays.
            t = (alpha_sign * c[alpha_source]
                 + beta_sign * jnp.moveaxis(c[alpha][:, source_b], 1, 0))
        else:
            t = (alpha_sign * c[alpha_source[:, :, None, None], beta[None, None, :, None], rhs]
                 + beta_sign * c[alpha[None, :, None, None], beta_source[:, None, :, None], rhs])
        t = jnp.where(valid[None, :, :, :], t, 0.)
        weighted = matrix @ t.reshape(pairs, alpha_width * beta_width * rhs_width)
        if not two_electron:
            update = weighted.reshape(alpha_width, beta_width, rhs_width)
            if whole_beta_rhs:
                return out.at[alpha].add(jnp.where(valid, update, 0.))
            return out.at[alpha[:, None, None], beta[None, :, None], rhs].add(
                jnp.where(valid, update, 0.))
        weighted = weighted.reshape(pairs, alpha_width, beta_width, rhs_width)
        # For pair-symmetric ERIs this is the required E transpose scatter.
        if whole_beta_rhs:
            values = jnp.where(valid[None, :, :, :], weighted, 0.)
            out = out.at[alpha_source].add(values * alpha_sign)
            beta_update = jnp.zeros((nb, alpha_width, nrhs), c.dtype).at[source_b].add(
                (values * beta_sign).swapaxes(1, 2))
            return out.at[alpha].add(beta_update.swapaxes(0, 1))
        out = out.at[alpha_source[:, :, None, None], beta[None, None, :, None], rhs].add(
            jnp.where(valid[None, :, :, :], weighted * alpha_sign, 0.))
        return out.at[alpha[None, :, None, None], beta_source[:, None, :, None], rhs].add(
            jnp.where(valid[None, :, :, :], weighted * beta_sign, 0.))

    out = jax.lax.fori_loop(0, alpha_tiles * beta_tiles * rhs_tiles, body, jnp.zeros_like(c))
    return out.reshape(original_shape)


def absorb_h1e(h1, eri, space, fac=1.):
    """Fold one-electron terms into the E E tensor, matching PySCF convention."""
    h, g = validate_integrals(h1, eri, space.norb)
    nelec = sum(space.nelec)
    if nelec == 0:
        return fac * g
    f = (h - .5*jnp.einsum('pqqs->ps', g)) / nelec
    eye = jnp.eye(space.norb, dtype=jnp.result_type(h, g))
    return fac * (g + jnp.einsum('pq,rs->pqrs', eye, f) + jnp.einsum('pq,rs->pqrs', f, eye))


def contract_hamiltonian(h1, eri, coefficients, space, *, max_workspace_elements=DEFAULT_WORKSPACE,
                         max_memory=DEFAULT_MAX_MEMORY_MB):
    if not isinstance(space, FCISpace):
        raise TypeError("Expected an FCISpace")
    h, g = validate_integrals(h1, eri, space.norb)
    c = coefficient_array(coefficients, space)
    c = c.astype(jnp.result_type(h, g, c))
    tiles = _action_tiles(space, c.dtype, 1, max_workspace_elements, max_memory,
                          arrays=(h, g, c, coefficients), extra_bytes=3 * g.size * c.dtype.itemsize)
    effective = absorb_h1e(h, g, space, fac=.5).astype(c.dtype)
    return _contract_tiles(effective, c, space, tiles, two_electron=True)


def make_hdiag(h1, eri, space):
    """Electronic diagonal in flattened alpha-major/beta-minor order."""
    h, g = validate_integrals(h1, eri, space.norb)
    a, b = [jnp.asarray(_link_maps(space.norb, n)[2], dtype=h.dtype) for n in space.nelec]
    coulomb = jnp.einsum('ppqq->pq', g)
    exchange = jnp.einsum('pqqp->pq', g)
    def same(occ):
        return occ @ jnp.diag(h) + .5*jnp.einsum('ap,pq,aq->a', occ, coulomb-exchange, occ)
    return (same(a)[:, None] + same(b)[None, :] + a @ coulomb @ b.T).reshape(-1)


def build_hamiltonian(h1, eri, space, *, max_workspace_elements=DEFAULT_WORKSPACE,
                      max_memory=DEFAULT_MAX_MEMORY_MB):
    """Matrix-free action; string tiles share the decimal-MB array budget."""
    if not isinstance(space, FCISpace):
        raise TypeError("Expected an FCISpace")
    h, g = validate_integrals(h1, eri, space.norb)
    dtype = jnp.result_type(h, g)
    itemsize = jnp.dtype(dtype).itemsize
    # Eager absorption can retain two outer products/sums and a new result.
    # Charge three extra ERIs as well as the diagonal and one input vector.
    _action_tiles(space, dtype, 1, max_workspace_elements, max_memory, arrays=(h, g),
                  extra_bytes=(3 * g.size + 2 * space.size) * itemsize)
    effective = absorb_h1e(h, g, space, fac=.5)
    diagonal = make_hdiag(h, g, space)

    def block(c):
        input_c = c
        c = c.astype(jnp.result_type(dtype, c))
        return _contract_ee(effective, c, space,
            max_workspace_elements=max_workspace_elements, max_memory=max_memory,
            resident_arrays=(h, g, diagonal, input_c))

    def apply(v):
        return block(v.reshape(space.shape)).reshape(-1)

    def apply_columns(v):
        return block(v.reshape(space.shape + (v.shape[1],))).reshape(v.shape)

    return LinearOperator((space.size, space.size), dtype, apply,
                          diagonal=diagonal, matmat=apply_columns,
                          transpose_matvec=apply)


def energy(h1, eri, coefficients, space, *, ecore=0., max_workspace_elements=DEFAULT_WORKSPACE,
            max_memory=DEFAULT_MAX_MEMORY_MB):
    c = coefficient_array(coefficients, space)
    norm = jnp.sum(c*c)
    hc = contract_hamiltonian(h1, eri, c, space, max_workspace_elements=max_workspace_elements,
                              max_memory=max_memory)
    value = jnp.sum(c*hc)/jnp.where(norm > 0, norm, 1.) + ecore
    return jnp.where((norm > 0) & jnp.isfinite(norm), value, jnp.nan)
