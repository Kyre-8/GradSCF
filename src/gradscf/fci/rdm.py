"""True fermionic FCI RDMs and transition RDMs in PySCF index convention.

dm1[p,q]=<q^+ p>, dm2[p,q,r,s]=<p^+ r^+ s q>. Transition matrix
entries are bilinear <bra|O|ket>, without normalization or phase matching.
"""
import jax
import jax.numpy as jnp
from math import prod
from numbers import Integral
from .hamiltonian import (
    DEFAULT_WORKSPACE, _action_tiles, coefficient_array,
)
from .._memory import DEFAULT_MAX_MEMORY_MB, tree_storage_bytes, validate_reserved_bytes
from .cistring import FCISpace, _link_maps


def _normalized(c, space):
    c = coefficient_array(c, space)
    norm = jnp.sum(c*c)
    c = c / jnp.sqrt(jnp.where(norm > 0, norm, 1.))
    return jnp.where((norm > 0) & jnp.isfinite(norm), c, jnp.nan)


def _normalized_property(c, space, options):
    options['reserved_bytes'] = (validate_reserved_bytes(options.get('reserved_bytes', 0))
                                 +tree_storage_bytes(c))
    return _normalized(c, space)


def _moments(bra, ket, space, max_workspace_elements, max_memory, *, kind, indices=None,
              reserved_bytes=0):
    """Reduce determinant tiles into one-/two-body moments or spin exchange.

    Only the requested RDM outputs are dense. The whole additive body is
    recomputed in reverse AD; no tile history of full CI states is retained.
    """
    reserved_bytes = validate_reserved_bytes(reserved_bytes)
    if not isinstance(space, FCISpace):
        raise TypeError("Expected an FCISpace")
    same = bra is ket
    left = coefficient_array(bra, space)
    right = left if same else coefficient_array(ket, space)
    dtype = jnp.result_type(left, right)
    n = space.norb
    pairs = n*n
    output_size = n**4 if indices is None else prod(map(len, indices))
    if kind in ('two', 'block') and output_size > max_workspace_elements:
        raise ValueError("FCI density tensor exceeds max_workspace_elements")
    output_elements = (16 if kind == 'block' else 12)*output_size+6*pairs if kind in ('two', 'block') else 6*pairs
    promotion = sum(x.size for x in (left, right) if x.dtype != dtype)
    aw, bw, _ = _action_tiles(space, dtype, 1, max_workspace_elements, max_memory,
        arrays=(bra, ket, left, right),
        extra_bytes=reserved_bytes+(output_elements+promotion)*dtype.itemsize)
    left, right = left.astype(dtype), right.astype(dtype)
    one = (jnp.zeros(pairs, dtype), jnp.zeros(pairs, dtype))
    if kind == 'two':
        initial = one + tuple(jnp.zeros((pairs, pairs), dtype) for _ in range(3))
    elif kind == 'block':
        left_pairs, right_pairs = len(indices[0])*len(indices[1]), len(indices[2])*len(indices[3])
        initial = one+tuple(jnp.zeros((left_pairs, right_pairs), dtype) for _ in range(4))
        left_indices = jnp.asarray([p*n+q for p in indices[0] for q in indices[1]], dtype=jnp.int32)
        right_indices = jnp.asarray([r*n+s for r in indices[2] for s in indices[3]], dtype=jnp.int32)
    elif kind == 'spin':
        initial = jnp.zeros((), dtype)
    else:
        initial = one
    if not n:
        return initial
    na, nb = space.shape
    n_beta = (nb+bw-1)//bw
    whole_beta = bw == nb
    maps = tuple(_link_maps(n, electrons) for electrons in space.nelec)
    sa, sb = (jnp.asarray(x[0]) for x in maps)
    wa, wb = (jnp.asarray(x[1], dtype) for x in maps)

    def actions(c, alpha, beta, valid):
        if whole_beta:
            a = c[sa[:, alpha]]*wa[:, alpha, None]
            b = jnp.moveaxis(c[alpha][:, sb], 1, 0)*wb[:, None, :]
        else:
            a = c[sa[:, alpha, None], beta[None, None, :]]*wa[:, alpha, None]
            b = c[alpha[None, :, None], sb[:, None, beta]]*wb[:, None, beta]
        return jnp.where(valid, a, 0.), jnp.where(valid, b, 0.)

    @jax.checkpoint
    def body(index, state):
        alpha = index//n_beta*aw + jnp.arange(aw)
        beta = index%n_beta*bw + jnp.arange(bw)
        valid = (alpha[None, :, None] < na) & (beta[None, None, :] < nb)
        alpha, beta = jnp.minimum(alpha, na-1), jnp.minimum(beta, nb-1)
        ra, rb = actions(right, alpha, beta, valid)
        if kind == 'spin':
            return state+jnp.sum(ra*rb)
        bra_tile = left[alpha[:, None], beta[None, :]]
        one = tuple(jnp.sum(bra_tile[None]*r, axis=(1, 2)) for r in (ra, rb))
        if kind == 'one':
            return tuple(x+y for x, y in zip(state, one))
        la, lb = (ra, rb) if same else actions(left, alpha, beta, valid)
        # <bra|E_pq E_rs|ket> = <E_qp bra|E_rs ket>.
        a, b = (x.reshape(n, n, aw*bw).swapaxes(0, 1).reshape(pairs, aw*bw)
                for x in (la, lb))
        ra, rb = ra.reshape(pairs, aw*bw), rb.reshape(pairs, aw*bw)
        if kind == 'block':
            a, b = a[left_indices], b[left_indices]
            ra, rb = ra[right_indices], rb[right_indices]
            updates = one+(a@ra.T, a@rb.T, b@ra.T, b@rb.T)
        else:
            updates = one+(a@ra.T, a@rb.T, b@rb.T)
        return tuple(x+y for x, y in zip(state, updates))

    return jax.lax.fori_loop(0, (na+aw-1)//aw*n_beta, body, initial)


def trans_rdm1s(bra, ket, space, *, max_workspace_elements=DEFAULT_WORKSPACE,
                max_memory=DEFAULT_MAX_MEMORY_MB, reserved_bytes=0):
    one = _moments(bra, ket, space, max_workspace_elements, max_memory, kind='one',
                   reserved_bytes=reserved_bytes)
    return tuple(x.reshape(space.norb, space.norb).T for x in one)


def trans_rdm1(bra, ket, space, **kwargs):
    a, b = trans_rdm1s(bra, ket, space, **kwargs)
    return a+b


def trans_rdm12s(bra, ket, space, *, max_workspace_elements=DEFAULT_WORKSPACE,
                 max_memory=DEFAULT_MAX_MEMORY_MB, reserved_bytes=0):
    a,b,aa,ab,bb = _moments(bra, ket, space, max_workspace_elements, max_memory,
                            kind='two', reserved_bytes=reserved_bytes)
    n = space.norb
    a,b = a.reshape(n,n),b.reshape(n,n)
    aa,ab,bb = (x.reshape((n,)*4) for x in (aa,ab,bb))
    eye = jnp.eye(n, dtype=a.dtype)
    aa -= jnp.einsum('qr,ps->pqrs', eye, a)
    bb -= jnp.einsum('qr,ps->pqrs', eye, b)
    return (a.T,b.T), (aa,ab,ab.transpose(2,3,0,1),bb)


def trans_rdm12(bra, ket, space, **kwargs):
    (a,b), (aa,ab,ba,bb) = trans_rdm12s(bra, ket, space, **kwargs)
    return a+b, aa+ab+ba+bb


def make_rdm1s(c, space, **kwargs):
    c = _normalized_property(c, space, kwargs)
    return trans_rdm1s(c, c, space, **kwargs)


def make_rdm1(c, space, **kwargs):
    a,b = make_rdm1s(c, space, **kwargs)
    return a+b


def make_rdm12s(c, space, **kwargs):
    c = _normalized_property(c, space, kwargs)
    one,(aa,ab,_,bb) = trans_rdm12s(c, c, space, **kwargs)
    return one,(aa,ab,bb)


def make_rdm12(c, space, **kwargs):
    c = _normalized_property(c, space, kwargs)
    return trans_rdm12(c, c, space, **kwargs)


def make_rdm2(c, space, **kwargs):
    return make_rdm12(c, space, **kwargs)[1]


def trans_rdm2_block(bra, ket, space, indices, *, spin_resolved=False,
                     max_workspace_elements=DEFAULT_WORKSPACE,
                     max_memory=DEFAULT_MAX_MEMORY_MB, reserved_bytes=0):
    """Export only dm2[p,q,r,s] for four static ordered orbital selections.

    No complete spatial or spin n**4 density is created. Transition spin blocks
    are (aa,ab,ba,bb), including independent ba for unequal requested axes.
    Repeated orbital indices are allowed; bra/ket remain unnormalized.
    """
    try:
        indices = tuple(tuple(axis) for axis in indices)
    except TypeError as error:
        raise ValueError('indices must contain four orbital index sequences') from error
    if len(indices) != 4 or any(not isinstance(i, Integral) or isinstance(i, bool)
            or not 0 <= i < space.norb for axis in indices for i in axis):
        raise ValueError('indices must contain four valid orbital index sequences')
    if not isinstance(spin_resolved, bool):
        raise TypeError('spin_resolved must be a static bool')
    a,b,*blocks = _moments(bra, ket, space, max_workspace_elements, max_memory,
                          kind='block', indices=indices, reserved_bytes=reserved_bytes)
    p,q,r,s = (jnp.asarray(axis, dtype=jnp.int32) for axis in indices)
    shape = tuple(map(len, indices))
    aa,ab,ba,bb = (x.reshape(shape) for x in blocks)
    delta = (q[:,None] == r[None,:])[None,:,:,None]
    aa -= delta*a.reshape(space.norb,space.norb)[jnp.ix_(p,s)][:,None,None,:]
    bb -= delta*b.reshape(space.norb,space.norb)[jnp.ix_(p,s)][:,None,None,:]
    return (aa,ab,ba,bb) if spin_resolved else aa+ab+ba+bb


def make_rdm2_block(c, space, indices, **kwargs):
    """Normalized counterpart of trans_rdm2_block; all state AD stays live."""
    c = _normalized_property(c, space, kwargs)
    return trans_rdm2_block(c, c, space, indices, **kwargs)


def spin_square(c, space, *, max_workspace_elements=DEFAULT_WORKSPACE,
                max_memory=DEFAULT_MAX_MEMORY_MB, reserved_bytes=0):
    """Return (<S^2>, effective multiplicity) without orbital-pair CI stacks."""
    options = dict(reserved_bytes=reserved_bytes)
    c = _normalized_property(c, space, options)
    exchange = _moments(c, c, space, max_workspace_elements, max_memory, kind='spin',
                        **options)
    na, nb = space.nelec
    ss = .25*(na-nb)**2 + .5*(na+nb) - exchange
    ss = jnp.where(jnp.all(jnp.isfinite(c)),ss,jnp.nan)
    return ss, jnp.sqrt(1+4*ss)
