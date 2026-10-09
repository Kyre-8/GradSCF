"""True fermionic FCI RDMs and transition RDMs in PySCF index convention.

dm1[p,q]=<q^+ p>, dm2[p,q,r,s]=<p^+ r^+ s q>. Transition matrix
entries are bilinear <bra|O|ket>, without normalization or phase matching.
"""
import jax.numpy as jnp
from .hamiltonian import (
    DEFAULT_WORKSPACE, _check_workspace, coefficient_array, excitation_actions,
)


def _normalized(c, space):
    c = coefficient_array(c, space)
    norm = jnp.sum(c*c)
    c = c / jnp.sqrt(jnp.where(norm > 0, norm, 1.))
    return jnp.where((norm > 0) & jnp.isfinite(norm), c, jnp.nan)


def trans_rdm1s(bra, ket, space, *, max_workspace_elements=DEFAULT_WORKSPACE):
    _check_workspace(space, max_workspace_elements)
    bra, ket = coefficient_array(bra, space), coefficient_array(ket, space)
    n = space.norb
    return tuple(jnp.einsum('ab,pab->p', bra, excitation_actions(ket, space, spin)).reshape(n,n).T
                 for spin in (0,1))


def trans_rdm1(bra, ket, space, **kwargs):
    a, b = trans_rdm1s(bra, ket, space, **kwargs)
    return a+b


def trans_rdm12s(bra, ket, space, *, max_workspace_elements=DEFAULT_WORKSPACE):
    _check_workspace(space, max_workspace_elements)
    if space.norb**4 > max_workspace_elements:
        raise ValueError("FCI density tensor exceeds max_workspace_elements")
    same = bra is ket
    left = coefficient_array(bra, space)
    right = left if same else coefficient_array(ket, space)
    n, d = space.norb, space.size
    tl = tuple(excitation_actions(left, space, spin) for spin in (0,1))
    tr = tl if same else tuple(excitation_actions(right, space, spin) for spin in (0,1))
    one = tuple(jnp.einsum('ab,pab->p', left, t).reshape(n,n) for t in tr)

    def pair(a, b):
        # <bra|E_pq E_rs|ket> = <E_qp bra|E_rs ket>.
        first = a.reshape(n,n,d).transpose(1,0,2).reshape(n*n,d)
        return (first @ b.reshape(n*n,d).T).reshape((n,)*4)

    eye = jnp.eye(n, dtype=jnp.result_type(left,right))
    aa = pair(tl[0], tr[0]) - jnp.einsum('qr,ps->pqrs', eye, one[0])
    ab = pair(tl[0], tr[1])
    bb = pair(tl[1], tr[1]) - jnp.einsum('qr,ps->pqrs', eye, one[1])
    return tuple(x.T for x in one), (aa,ab,ab.transpose(2,3,0,1),bb)


def trans_rdm12(bra, ket, space, **kwargs):
    (a,b), (aa,ab,ba,bb) = trans_rdm12s(bra, ket, space, **kwargs)
    return a+b, aa+ab+ba+bb


def make_rdm1s(c, space, **kwargs):
    c = _normalized(c, space)
    return trans_rdm1s(c, c, space, **kwargs)


def make_rdm1(c, space, **kwargs):
    a,b = make_rdm1s(c, space, **kwargs)
    return a+b


def make_rdm12s(c, space, **kwargs):
    c = _normalized(c, space)
    one,(aa,ab,_,bb) = trans_rdm12s(c, c, space, **kwargs)
    return one,(aa,ab,bb)


def make_rdm12(c, space, **kwargs):
    c = _normalized(c, space)
    return trans_rdm12(c, c, space, **kwargs)


def make_rdm2(c, space, **kwargs):
    return make_rdm12(c, space, **kwargs)[1]


def spin_square(c, space, *, max_workspace_elements=DEFAULT_WORKSPACE):
    """Return (<S^2>, effective multiplicity) in a common spatial MO frame."""
    _check_workspace(space, max_workspace_elements)
    c = _normalized(c, space)
    a, b = (excitation_actions(c, space, spin) for spin in (0,1))
    na, nb = space.nelec
    ss = .25*(na-nb)**2 + .5*(na+nb) - jnp.sum(a*b)
    ss = jnp.where(jnp.all(jnp.isfinite(c)),ss,jnp.nan)
    return ss, jnp.sqrt(1+4*ss)
