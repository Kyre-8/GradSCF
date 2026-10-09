"""Real spin-free FCI Hamiltonian on an alpha-string x beta-string tensor.

H = h[pq] E_pq + .5 (pq|rs) (E_pq E_rs - delta_qr E_ps).
No determinant-pair connection table or dense FCI Hamiltonian is constructed.
"""
from numbers import Integral
import jax
import jax.numpy as jnp
from ..integrals.mo import validate_integrals
from ..solvers import LinearOperator
from .cistring import FCISpace, _link_maps


DEFAULT_WORKSPACE = 8_000_000


def _check_workspace(space, max_workspace_elements, *, include_eri=False):
    if not isinstance(space, FCISpace):
        raise TypeError("Expected an FCISpace")
    if (not isinstance(max_workspace_elements, Integral) or isinstance(max_workspace_elements,bool)
            or max_workspace_elements < 1 or space.norb**2 * space.size > max_workspace_elements):
        raise ValueError("FCI excitation tensor exceeds max_workspace_elements")
    if include_eri and space.norb**4 > max_workspace_elements:
        raise ValueError("FCI integral tensor exceeds max_workspace_elements")


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


def contract_1e(h1, coefficients, space, *, max_workspace_elements=DEFAULT_WORKSPACE):
    """Physical one-electron Hamiltonian action, sum h[pq] E_pq."""
    _check_workspace(space, max_workspace_elements)
    c = coefficient_array(coefficients, space)
    h = jnp.asarray(h1)
    if h.shape != (space.norb, space.norb) or jnp.iscomplexobj(h):
        raise ValueError("h1 must be a real (norb,norb) matrix")
    return jnp.einsum('p,pab->ab', h.reshape(-1), excitation_actions(c, space))


def contract_2e(eri, coefficients, space, *, max_workspace_elements=DEFAULT_WORKSPACE):
    """PySCF-style E E contraction, NOT the bare physical two-electron action.

    Use absorb_h1e(h1,eri,space,fac=.5) first for a complete Hamiltonian.
    Integrals must have real chemists' pair symmetries; no spin tensor is built.
    """
    _check_workspace(space, max_workspace_elements, include_eri=True)
    g = jnp.asarray(eri)
    n = space.norb
    if g.shape != (n,)*4 or jnp.iscomplexobj(g):
        raise ValueError("eri must be a real (norb,norb,norb,norb) tensor")
    c = coefficient_array(coefficients, space)
    c = c.astype(jnp.result_type(g, c))
    return _contract_ee(g, c, space)


def _contract_ee(g, c, space):
    t = excitation_actions(c, space)
    pairs = space.norb**2
    weighted = (g.reshape(pairs, pairs) @ t.reshape(pairs, c.size)).reshape(t.shape)
    # For pair-symmetric ERIs, sum g[pqrs] E_pq equals sum g[pqrs] E_qp.
    return _excitation_transpose(weighted, space)


def absorb_h1e(h1, eri, space, fac=1.):
    """Fold one-electron terms into the E E tensor, matching PySCF convention."""
    h, g = validate_integrals(h1, eri, space.norb)
    nelec = sum(space.nelec)
    if nelec == 0:
        return fac * g
    f = (h - .5*jnp.einsum('pqqs->ps', g)) / nelec
    eye = jnp.eye(space.norb, dtype=jnp.result_type(h, g))
    return fac * (g + jnp.einsum('pq,rs->pqrs', eye, f) + jnp.einsum('pq,rs->pqrs', f, eye))


def contract_hamiltonian(h1, eri, coefficients, space, *, max_workspace_elements=DEFAULT_WORKSPACE):
    return contract_2e(absorb_h1e(h1, eri, space, fac=.5), coefficients, space,
                        max_workspace_elements=max_workspace_elements)


def make_hdiag(h1, eri, space):
    """Electronic diagonal in flattened alpha-major/beta-minor order."""
    h, g = validate_integrals(h1, eri, space.norb)
    a, b = [jnp.asarray(_link_maps(space.norb, n)[2], dtype=h.dtype) for n in space.nelec]
    coulomb = jnp.einsum('ppqq->pq', g)
    exchange = jnp.einsum('pqqp->pq', g)
    def same(occ):
        return occ @ jnp.diag(h) + .5*jnp.einsum('ap,pq,aq->a', occ, coulomb-exchange, occ)
    return (same(a)[:, None] + same(b)[None, :] + a @ coulomb @ b.T).reshape(-1)


def build_hamiltonian(h1, eri, space, *, max_workspace_elements=DEFAULT_WORKSPACE):
    _check_workspace(space, max_workspace_elements, include_eri=True)
    h, g = validate_integrals(h1, eri, space.norb)
    effective = absorb_h1e(h, g, space, fac=.5)
    @jax.checkpoint
    def block(c):
        return _contract_ee(effective, c, space)

    def apply(v):
        return block(v.reshape(space.shape)).reshape(-1)

    def apply_columns(v):
        width = max(1, max_workspace_elements // max(1, space.norb**2 * space.size))
        pieces = []
        for start in range(0, v.shape[1], width):
            columns = min(width, v.shape[1]-start)
            c = v[:,start:start+columns].reshape(space.shape+(columns,))
            pieces.append(block(c).reshape(space.size,columns))
        return jnp.concatenate(pieces,axis=1) if pieces else jnp.zeros_like(v)

    return LinearOperator((space.size, space.size), jnp.result_type(h, g), apply,
                          diagonal=make_hdiag(h, g, space), matmat=apply_columns,
                          transpose_matvec=apply)


def energy(h1, eri, coefficients, space, *, ecore=0., max_workspace_elements=DEFAULT_WORKSPACE):
    c = coefficient_array(coefficients, space)
    norm = jnp.sum(c*c)
    hc = contract_hamiltonian(h1, eri, c, space, max_workspace_elements=max_workspace_elements)
    value = jnp.sum(c*hc)/jnp.where(norm > 0, norm, 1.) + ecore
    return jnp.where((norm > 0) & jnp.isfinite(norm), value, jnp.nan)
