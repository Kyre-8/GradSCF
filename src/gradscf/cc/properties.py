"""CC one- and two-particle properties including the stationary left state."""

import jax
import jax.numpy as jnp
from .amplitudes import AmplitudeSpace
from .integrals import prepare_integrals
from .rccsd import residual, correlation_energy
from .lambda_equations import solve_lambda
from .types import CCConfig
from .uccsd import prepare_ucc_integrals
from .spin_amplitudes import SpinAmplitudeSpace
from . import _spin_equations
from gradscf.ao2mo import frozen_indices, unrestricted_frozen_indices
from ._spin_density import active_density_parts
from ._memory import check_payload, preparation_reservation


def _density_from_lambda(h1, eri, result, left, *, nocc, frozen, config,
                          max_memory=None, reserved_bytes=0):
    if isinstance(nocc, (tuple, list)):
        h = tuple(map(jnp.asarray, h1))
        reserved_bytes = preparation_reservation(lambda a, b: prepare_ucc_integrals(
            tuple(.5*(x+x.T) for x in a), b, nocc=nocc, frozen=frozen),
            (h, eri), max_memory, reserved_bytes, consumer_owned=(result, left))
        ints, _, _ = prepare_ucc_integrals(tuple(.5*(a+a.T) for a in h), eri,
                                          nocc=nocc, frozen=frozen)
        return _density_ucc_from_integrals(ints, result, left, nmo=h[0].shape[0],
            nocc=nocc, frozen=frozen, config=config, dtypes=tuple(a.dtype for a in h),
            max_memory=max_memory, reserved_bytes=reserved_bytes)
    h = jnp.asarray(h1)
    # Preserve the raw API's Hermitian partial even for off-manifold inputs;
    # the outer right/left states still depend on the original raw Hamiltonian.
    reserved_bytes = preparation_reservation(lambda a, b: prepare_integrals(.5*(a+a.T), b,
        nocc=nocc, frozen=frozen), (h, eri), max_memory, reserved_bytes, consumer_owned=(result, left))
    ints = prepare_integrals(.5*(h+h.T), eri, nocc=nocc, frozen=frozen)
    density = _density_from_integrals(
        ints, result, left, nmo=h.shape[0], nocc=nocc, frozen=frozen, config=config,
        max_memory=max_memory, reserved_bytes=reserved_bytes
    )
    return density.astype(h.dtype)


def _density_from_integrals(ints, result, left, *, nmo, nocc, frozen=None, config=None,
                             max_memory=None, reserved_bytes=0):
    """Restricted MO 1-RDM from the local Hermitian one-body partial.

    ``nmo/nocc/frozen`` describe the full supplied orbital frame. Only active
    Fock entries, their diagonal energies and the full occupied reference trace
    depend on the zero one-body perturbation; ERI blocks stay supplied objects.
    The prepared Fock itself is not symmetrized. T and Lambda remain live outer
    dependencies, so nested derivatives include both states' implicit response.
    """
    cfg = CCConfig() if config is None else config
    frozen = frozen_indices(nmo, nocc, frozen)
    occupied = [i for i in range(nocc) if i not in frozen]
    virtual = [i for i in range(nocc, nmo) if i not in frozen]
    if (len(occupied), len(virtual)) != (ints.nocc, ints.nvir):
        raise ValueError("Full orbital metadata does not match prepared CC active space")
    if max_memory is not None:
        output = jax.ShapeDtypeStruct((8*nmo*nmo,), jnp.result_type(ints.fock, result.t1, left.adjoint))
        check_payload(ints, (result, left, output), max_memory, reserved_bytes,
                       ints.nocc, ints.nvir, cfg.method)
    active = jnp.asarray(occupied+virtual, dtype=jnp.int32)
    space = AmplitudeSpace(ints.nocc, ints.nvir, cfg.method)

    def lagrangian(raw_delta):
        delta = .5*(raw_delta+raw_delta.T)
        selected = delta[jnp.ix_(active, active)]
        perturbed = ints._replace(
            fock=ints.fock+selected,
            mo_energy=ints.mo_energy+jnp.diag(selected),
            reference_energy=ints.reference_energy+2*jnp.trace(delta[:nocc, :nocc]),
        )
        r = space.pack(*residual(result.t1, result.t2, perturbed, model=cfg.method))
        e = perturbed.reference_energy+correlation_energy(
            result.t1, result.t2, perturbed, model=cfg.method)
        return e + jnp.vdot(left.adjoint, r).real

    density = jax.grad(lagrangian)(jnp.zeros((nmo, nmo), dtype=ints.fock.dtype))
    return jnp.where(result.converged & left.converged, density, jnp.nan)


def _density_ucc_from_integrals(ints, result, left, *, nmo, nocc, frozen=None,
                                 config=None, dtypes=None, max_memory=None, reserved_bytes=0):
    """UHF local Hermitian one-body partial in the supplied full MO frame.

    The snapshot Fock is unchanged at the base point. Delta enters only its
    selected alpha/beta blocks and each full occupied reference trace; all ERIs
    remain the supplied objects. Right/left states remain live outer AD inputs.
    Private dtypes preserve the raw API's separate h-alpha/h-beta coordinates.
    """
    cfg = CCConfig() if config is None else config
    frozen = unrestricted_frozen_indices(nmo, nocc, frozen)
    occupied = tuple(tuple(i for i in range(no) if i not in fr)
                     for no, fr in zip(nocc, frozen))
    virtual = tuple(tuple(i for i in range(no, nmo) if i not in fr)
                    for no, fr in zip(nocc, frozen))
    sizes = tuple(tuple(map(len, group)) for group in (occupied, virtual))
    no, nv = map(sum, sizes)
    if ints.fock.shape != (no+nv, no+nv) or ints.oovv.shape != (no, no, nv, nv):
        raise ValueError("Full orbital metadata does not match prepared UCC active space")
    if max_memory is not None:
        dtype = jnp.result_type(ints.fock, left.adjoint, *jax.tree.leaves((result.t1, result.t2)))
        buffers = (jax.ShapeDtypeStruct((16*nmo*nmo,), dtype),
                   jax.ShapeDtypeStruct((no, nv), dtype),
                   jax.ShapeDtypeStruct((no, no, nv, nv), dtype))
        check_payload(ints, (result, left, buffers), max_memory, reserved_bytes,
                       no, nv, cfg.method, spin=True)
    space = SpinAmplitudeSpace(*sizes, cfg.method)
    dtypes = (ints.fock.dtype,)*2 if dtypes is None else dtypes
    t1, t2 = space.from_blocks(result.t1, result.t2, dtypes[0])
    active = tuple(jnp.asarray(o+v, dtype=jnp.int32) for o, v in zip(occupied, virtual))
    targets = tuple(jnp.asarray(tuple(range(sum(sizes[0][:s]), sum(sizes[0][:s+1])))
                    +tuple(range(no+sum(sizes[1][:s]), no+sum(sizes[1][:s+1]))),
                    dtype=jnp.int32) for s in range(2))

    def lagrangian(raw_delta):
        fock, reference = ints.fock, ints.reference_energy
        for delta, selection, target, count in zip(raw_delta, active, targets, nocc):
            delta = .5*(delta+delta.T)
            fock = fock.at[jnp.ix_(target, target)].add(delta[jnp.ix_(selection, selection)])
            reference = reference+jnp.trace(delta[:count, :count])
        perturbed = ints._replace(fock=fock, reference_energy=reference)
        r = space.pack(*_spin_equations.residual(t1, t2, perturbed))
        e = reference+_spin_equations.correlation_energy(t1, t2, perturbed)
        return e+jnp.vdot(left.adjoint, r).real

    densities = jax.grad(lagrangian)(tuple(jnp.zeros((nmo, nmo), dtype) for dtype in dtypes))
    return tuple(jnp.where(result.converged & left.converged, d, jnp.nan) for d in densities)


def make_rdm1(h1, eri, result, *, nocc, frozen=None, config=None,
               max_memory=None, reserved_bytes=0):
    """Real symmetric MO 1-RDM, including frozen occupied electrons.

    Restricted output is spin summed; unrestricted output is (alpha, beta).
    This includes T/Lambda response but not optimization of the reference
    orbitals. It is a CC-model density, not a CCSD(T)-corrected density.
    """
    cfg = CCConfig() if config is None else config
    left = solve_lambda(h1, eri, result, nocc=nocc, frozen=frozen, config=cfg,
                         max_memory=max_memory, reserved_bytes=reserved_bytes)
    return _density_from_lambda(
        h1, eri, result, left, nocc=nocc, frozen=frozen, config=cfg,
        max_memory=max_memory, reserved_bytes=reserved_bytes
    )


def _rdm2_from_lambda(h1, eri, result, left, *, nocc, frozen, config):
    h = jnp.asarray(h1[0] if isinstance(nocc, (tuple, list)) else h1)
    return _rdm2_from_states(result, left, nmo=h.shape[0], nocc=nocc,
                             frozen=frozen, dtype=h.dtype, config=config)


def _rdm2_from_states(result, left, *, nmo, nocc, dtype, frozen=None, config=None):
    """Dense 2-RDM from full-frame metadata and stationary right/left states.

    No integrals are required. ``dtype`` retains the raw h1 input chart; state
    amplitudes may promote it just as in the raw wrapper. The existing full
    spin-orbital four-index allocation remains an in-core reference algorithm.
    """
    cfg = CCConfig() if config is None else config
    if cfg.method not in {"ccsd", "ccd", "ccs"}:
        raise NotImplementedError("The CC 2-RDM currently supports CCS, CCD and CCSD")
    unrestricted = isinstance(nocc, (tuple, list))
    n = nmo
    if unrestricted:
        counts = tuple(nocc)
        frozen_sets = unrestricted_frozen_indices(n, counts, frozen)
    else:
        counts = (nocc, nocc)
        frozen_sets = (frozen_indices(n, nocc, frozen),)*2
    occupied = [[s*n+i for i in range(no) if i not in fr]
                for s, (no, fr) in enumerate(zip(counts, frozen_sets))]
    virtual = [[s*n+i for i in range(no, n) if i not in fr]
               for s, (no, fr) in enumerate(zip(counts, frozen_sets))]
    space = SpinAmplitudeSpace(tuple(map(len, occupied)), tuple(map(len, virtual)))

    def to_spin(a, b):
        if unrestricted:
            return space.from_blocks(a, b, dtype)
        same_spin = b-b.swapaxes(2, 3)
        return space.from_blocks((a, a), (same_spin, b, same_spin), dtype)

    t1, t2 = to_spin(result.t1, result.t2)
    l1, l2 = to_spin(left.l1, left.l2)
    active1, active2 = active_density_parts(t1, t2, l1, l2)
    active = jnp.asarray(occupied[0]+occupied[1]+virtual[0]+virtual[1], dtype=jnp.int32)
    dc = jnp.zeros((2*n, 2*n), t1.dtype).at[jnp.ix_(active, active)].set(active1)
    dm2 = jnp.zeros((2*n,)*4, t1.dtype).at[jnp.ix_(active, active, active, active)].set(active2)
    full_occ = jnp.asarray(list(range(counts[0]))+list(range(n, n+counts[1])), dtype=jnp.int32)
    p = jnp.zeros_like(dc).at[full_occ, full_occ].set(1.)
    # Restore reference and core-active terms in the full spin-orbital space.
    dm2 += (jnp.einsum("pq,rs->pqrs", p, dc)+jnp.einsum("pq,rs->pqrs", dc, p)
            -jnp.einsum("ps,rq->pqrs", p, dc)-jnp.einsum("ps,rq->pqrs", dc, p)
            +jnp.einsum("pq,rs->pqrs", p, p)-jnp.einsum("ps,rq->pqrs", p, p))
    dm2 = jnp.where(result.converged & left.converged, dm2, jnp.nan)
    aa, ab, bb = dm2[:n, :n, :n, :n], dm2[:n, :n, n:, n:], dm2[n:, n:, n:, n:]
    return (aa, ab, bb) if unrestricted else aa+ab+ab.transpose(2, 3, 0, 1)+bb


def make_rdm2(h1, eri, result, *, nocc, frozen=None, config=None):
    """Orbital-unrelaxed real MO 2-RDM, including Lambda and frozen electrons.

    dm2[p,q,r,s] = <a_p^+ a_r^+ a_s a_q>, Hermitian-symmetrized for real
    observables. Restricted: spin sum; unrestricted: (aa,ab,bb), whose energy
    contraction weights are (.5,1,.5). A dense spin-orbital intermediate is
    used: this is an in-core reference path, not a low-memory density algorithm.
    CCSD(T), CC2 and linearized-model 2-RDMs are not supplied by this function.
    """
    cfg = CCConfig() if config is None else config
    left = solve_lambda(h1, eri, result, nocc=nocc, frozen=frozen, config=cfg)
    return _rdm2_from_lambda(h1, eri, result, left, nocc=nocc, frozen=frozen, config=cfg)
