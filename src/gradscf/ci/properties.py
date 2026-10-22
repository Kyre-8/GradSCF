"""True fermionic CI densities and direct one-/two-body observables.

Only requested spatial densities are exported. Small tables use the original
fermionic connection formula when affordable; otherwise their coefficients
come from the physical sigma action. No full spin-orbital density is built.
"""
import jax
import jax.numpy as jnp

from gradscf._memory import (DEFAULT_MAX_MEMORY_MB, array_storage_bytes,
                             resolve_memory_mb, tree_storage_bytes,
                             validate_reserved_bytes, workspace_bytes)
from gradscf.ao2mo import validate_integrals, validate_unrestricted_integrals
from .space import CISpace, UCISpace
from .direct import build_direct_hamiltonian, build_one_body
from . import _rdm


def _coefficients(coefficients, space):
    if not isinstance(space, (CISpace, UCISpace)):
        raise TypeError("A determinant CISpace or UCISpace is required")
    c = jnp.asarray(coefficients)
    if c.shape != (space.size,):
        raise ValueError("CI coefficients must have shape (ndeterminants,)")
    if jnp.iscomplexobj(c):
        raise NotImplementedError("CI densities currently require real coefficients")
    c = c.astype(jnp.result_type(c, 1.0))
    norm = jnp.vdot(c, c).real
    c = c/jnp.sqrt(jnp.where(norm > 0, norm, 1.))
    return jnp.where((norm > 0) & jnp.isfinite(norm), c, jnp.nan)


def expectation_value(coefficients, space, h1, eri=None, *, max_memory=DEFAULT_MAX_MEMORY_MB,
                       reserved_bytes=0):
    """Return normalized <C|H(h1,eri)|C> without exporting an RDM.

    h1 uses ordinary Hamiltonian indices; its energy contraction is h1:dm1.T.
    Two-electron coefficients use chemists order. Unrestricted inputs are two
    h1 and three (aa,ab,bb) ERI blocks, with the existing beta-alpha chart.
    """
    reserved_bytes = validate_reserved_bytes(reserved_bytes)
    c = _coefficients(coefficients, space)
    maximum = resolve_memory_mb(max_memory)
    held = reserved_bytes+tree_storage_bytes(coefficients, c)+4*c.nbytes
    workspace_bytes(maximum, resident_bytes=held+tree_storage_bytes(h1, eri),
        device=getattr(c, "device", None))
    options = dict(max_memory=maximum, reserved_bytes=held)
    operator = (build_one_body(h1, space, **options) if eri is None else
                build_direct_hamiltonian(h1, eri, space, **options))
    value = jnp.vdot(c, operator.matvec(c)).real
    del operator  # Its forward kernels are reduced before the correction.
    if eri is not None:
        # Sigma's physical-ERI normal-ordering chart uses one representative
        # for equivalent contractions. A true RDM needs the complete raw
        # coefficient chart, including nonsymmetric ERI directions. Restore
        # the other contractions as a one-body normal-ordering correction.
        if isinstance(space, UCISpace):
            hs, (aa, ab, bb) = validate_unrestricted_integrals(h1, eri)
            gs, ns = ((aa, ab), (ab.transpose(2, 3, 0, 1), bb)), space.nocc
        else:
            h, g = validate_integrals(h1, eri, space.nmo)
            hs, gs, ns = (h, h), ((g, g), (g, g)), (space.nocc,)*2
        deltas = []
        for spin in (0, 1):
            delta = jnp.zeros_like(hs[spin])
            for other, no in enumerate(ns):
                g, reverse = gs[spin][other], gs[other][spin]
                delta = delta+.5*(jnp.einsum('iipq->pq', reverse[:no, :no])
                                  -jnp.einsum('pqii->pq', g[:, :, :no, :no]))
                if spin == other:
                    delta = delta+.5*(jnp.einsum('piiq->pq', g[:, :no, :no])
                                      -jnp.einsum('iqpi->pq', g[:no, :, :, :no]))
            deltas.append(delta)
        correction = tuple(deltas) if isinstance(space, UCISpace) else deltas[0]
        correction_held = held+tree_storage_bytes(h1, eri, value, deltas,
                                                   exclude=(correction,))
        value += jnp.vdot(c, build_one_body(correction, space,
            max_memory=maximum, reserved_bytes=correction_held).matvec(c)).real
        value -= sum(jnp.trace(d[:no, :no]) for d, no in zip(deltas, ns))
    return value


def _density(coefficients, space, rank, max_memory, reserved_bytes=0):
    reserved_bytes = validate_reserved_bytes(reserved_bytes)
    c = _coefficients(coefficients, space)
    n = space.nmo
    unrestricted = isinstance(space, UCISpace)
    blocks = rank+1 if unrestricted else 1
    # Zero inputs, exported gradients and immutable update versions are known
    # forward/output storage. This does not cap complete nested-AD residency.
    output_bytes = blocks*n**(2*rank)*c.dtype.itemsize
    known = reserved_bytes+4*output_bytes+tree_storage_bytes(coefficients, c)+4*c.nbytes
    maximum = resolve_memory_mb(max_memory)
    workspace_bytes(maximum, resident_bytes=known,
                    device=getattr(c, "device", None))
    if _rdm.affordable(space,rank,c.dtype,maximum,
        resident_bytes=known-4*output_bytes,device=getattr(c,"device",None)):
        return _rdm.density(coefficients,space,rank,c.dtype)
    h = jnp.zeros((n, n), c.dtype)
    one = (h, h) if unrestricted else h
    # The action owns its zero coefficient tensors. Keep the returned gradient
    # and update versions, plus the caller's original vector, reserved there.
    held = reserved_bytes+3*output_bytes+tree_storage_bytes(coefficients, exclude=(c,))
    if rank == 1:
        gradient = jax.grad(lambda hs: expectation_value(c, space, hs,
            max_memory=max_memory, reserved_bytes=held))(one)
        return jax.tree.map(lambda x: jnp.where(jnp.all(jnp.isfinite(c)),x.T,jnp.nan),gradient)
    g = jnp.zeros((n,)*4, c.dtype)
    two = (g, g, g) if unrestricted else g
    gradient = jax.grad(lambda gs: expectation_value(c, space, one, gs,
        max_memory=max_memory, reserved_bytes=held))(two)
    if unrestricted:
        gradient = 2*gradient[0],gradient[1],2*gradient[2]
    else:
        gradient = 2*gradient
    return jax.tree.map(lambda x:jnp.where(jnp.all(jnp.isfinite(c)),x,jnp.nan),gradient)


def make_rdm1(coefficients, space, *, max_memory=DEFAULT_MAX_MEMORY_MB, reserved_bytes=0):
    """Normalized spatial 1-RDM, dm1[p,q]=<q^+ p>, including frozen electrons.

    A zero/nonfinite vector yields NaNs. Coefficient response remains live;
    integral response needs coefficients with an enabled eigenvector response.
    """
    return _density(coefficients, space, 1, max_memory, reserved_bytes)


def make_rdm2(coefficients, space, *, max_memory=DEFAULT_MAX_MEMORY_MB, reserved_bytes=0):
    """Budgeted true spatial 2-RDM, dm2[p,q,r,s]=<p^+ r^+ s q>.

    Restricted output is spin summed; unrestricted output contains (aa,ab,bb).
    Energy contraction weights are respectively .5 and (.5,1,.5). This is the
    fermionic density, without imposing eightfold ERI derivative symmetry.
    """
    return _density(coefficients, space, 2, max_memory, reserved_bytes)


def make_rdm12(coefficients, space, *, max_memory=DEFAULT_MAX_MEMORY_MB, reserved_bytes=0):
    """Return the requested spatial one-/two-RDMs under one MB budget."""
    reserved_bytes = validate_reserved_bytes(reserved_bytes)
    one = make_rdm1(coefficients, space, max_memory=max_memory, reserved_bytes=reserved_bytes)
    # The first returned density stays live during the second export.
    reserved = reserved_bytes+array_storage_bytes(*jax.tree.leaves(one))
    workspace_bytes(resolve_memory_mb(max_memory), resident_bytes=reserved)
    return one, _density(coefficients, space, 2, max_memory, reserved)
