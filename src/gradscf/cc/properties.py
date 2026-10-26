"""CC one- and two-particle properties including the stationary left state."""

import jax
import jax.numpy as jnp
from functools import partial
from math import prod
from numbers import Integral
from .amplitudes import AmplitudeSpace
from .integrals import prepare_integrals
from .rccsd import residual, correlation_energy
from .lambda_equations import solve_lambda
from .types import CCConfig
from .uccsd import prepare_ucc_integrals
from .spin_amplitudes import SpinAmplitudeSpace
from . import _spin_equations
from gradscf.ao2mo import frozen_indices, unrestricted_frozen_indices
from ._spin_density import active_density_parts, indexed_density_parts, reference_density_terms
from ._memory import check_payload, preparation_reservation
from .._memory import (tree_storage_bytes, validate_reserved_bytes, workspace_bytes,
                       jaxpr_temporary_bytes)
from ..solvers.memory import _largest_fitting


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


def _rdm2_from_lambda(h1, eri, result, left, *, nocc, frozen, config,
                      indices=None, spin=None, max_memory=None, reserved_bytes=0):
    h = jnp.asarray(h1[0] if isinstance(nocc, (tuple, list)) else h1)
    reservation = validate_reserved_bytes(reserved_bytes)+tree_storage_bytes(h1,eri,exclude=(result,left))
    return _rdm2_from_states(result, left, nmo=h.shape[0], nocc=nocc,
        frozen=frozen, dtype=h.dtype, config=config, indices=indices, spin=spin,
        max_memory=max_memory, reserved_bytes=reservation)


def _rdm2_dense_from_states(result, left, *, nmo, nocc, dtype, frozen=None, config=None):
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
    dm2 += reference_density_terms(p,dc)
    dm2 = jnp.where(result.converged & left.converged, dm2, jnp.nan)
    aa, ab, bb = dm2[:n, :n, :n, :n], dm2[:n, :n, n:, n:], dm2[n:, n:, n:, n:]
    return (aa, ab, bb) if unrestricted else aa+ab+ab.transpose(2, 3, 0, 1)+bb


def _density_frame(nmo,nocc,frozen):
    unrestricted = isinstance(nocc,(tuple,list))
    counts = tuple(nocc) if unrestricted else (nocc,nocc)
    frozen_sets = (unrestricted_frozen_indices(nmo,counts,frozen) if unrestricted
                   else (frozen_indices(nmo,nocc,frozen),)*2)
    occupied = tuple(tuple(s*nmo+i for i in range(no) if i not in fr)
                     for s,(no,fr) in enumerate(zip(counts,frozen_sets)))
    virtual = tuple(tuple(s*nmo+i for i in range(no,nmo) if i not in fr)
                    for s,(no,fr) in enumerate(zip(counts,frozen_sets)))
    return unrestricted,counts,occupied,virtual


def _density_recipe(result,left,nmo,frame,dtype):
    from ._density_index import Tensor,einsum as indexed_einsum
    unrestricted,counts,occupied,virtual = frame
    space = SpinAmplitudeSpace(tuple(map(len,occupied)),tuple(map(len,virtual)))

    def convert(a,b):
        if unrestricted:
            return space.from_blocks(a,b,dtype)
        same = b-b.swapaxes(2,3)
        return space.from_blocks((a,a),(same,b,same),dtype)

    t1,t2 = convert(result.t1,result.t2)
    l1,l2 = convert(left.l1,left.l2)
    one,two = indexed_density_parts(t1,t2,l1,l2)
    active = occupied[0]+occupied[1]+virtual[0]+virtual[1]
    active_index = jnp.asarray(active,jnp.int32)
    dc = jnp.zeros((2*nmo,)*2,t1.dtype).at[jnp.ix_(active_index,active_index)].set(one)
    full = jnp.asarray(tuple(range(counts[0]))+tuple(range(nmo,nmo+counts[1])),jnp.int32)
    p = jnp.zeros_like(dc).at[full,full].set(1.)
    reference = reference_density_terms(Tensor(p),Tensor(dc),contract=indexed_einsum)
    lookup = {orbital:i for i,orbital in enumerate(active)}
    mapping = jnp.asarray(tuple(lookup.get(i,-1) for i in range(2*nmo)),jnp.int32)
    return two,reference,lookup,mapping


def _density_component(recipe,coordinates,spin,nmo):
    from ._density_index import Index
    two,reference,lookup,mapping = recipe
    if spin == 'ba':
        value = _density_component(recipe,coordinates[2:]+coordinates[:2],'ab',nmo)
        return value.transpose(2,3,0,1)
    spins = (0,0,0,0) if spin == 'aa' else ((1,1,1,1) if spin == 'bb' else (0,0,1,1))
    physical = tuple(tuple(i+s*nmo for i in axis) if isinstance(axis,tuple) else axis+s*nmo
                     for axis,s in zip(coordinates,spins))
    active = tuple(tuple(lookup.get(i,-1) for i in axis) if isinstance(axis,tuple) else mapping[axis]
                   for axis in physical)
    reference = reference.evaluate(tuple(Index(jnp.asarray(axis,jnp.int32)) for axis in physical),{})
    return two.take(active)+reference


def _density_layout(nmo,nocc,indices,spin):
    if spin not in (None,'aa','ab','ba','bb'):
        raise ValueError('spin must be aa, ab, ba, bb or None')
    if indices is None:
        indices = (tuple(range(nmo)),)*4
    if len(indices) != 4:
        raise ValueError('indices must contain four static orbital sequences')
    indices = tuple(tuple(axis) for axis in indices)
    if any(not isinstance(i,Integral) or isinstance(i,bool) or not 0 <= i < nmo
           for axis in indices for i in axis):
        raise ValueError('Density indices must be static integers in [0,nmo)')
    unrestricted = isinstance(nocc,(tuple,list))
    spins = ('aa','ab','bb') if spin is None and unrestricted else ((spin,) if spin else ('sum',))
    return indices,spins


def _density_preflight(result,*,nmo,nocc,dtype,indices=None,spin=None,
                        max_memory=None,reserved_bytes=0):
    """Reject an impossible required export before preparing T/Lambda buffers."""
    indices,spins = _density_layout(nmo,nocc,indices,spin)
    dtype = jnp.result_type(dtype,*(a.dtype for a in jax.tree.leaves((result.t1,result.t2))))
    resident = validate_reserved_bytes(reserved_bytes)+len(spins)*prod(map(len,indices))*dtype.itemsize
    if max_memory is not None:
        leaf = jax.tree.leaves(result.t1)[0]
        workspace_bytes(max_memory,resident_bytes=resident,device=getattr(leaf,'device',None))
    return indices,spins,dtype


def _render_density(recipe,coordinates,spins,nmo):
    values = []
    for channel in spins:
        if channel == 'sum':
            value = _density_component(recipe,coordinates,'aa',nmo)
            for s in ('ab','ba','bb'):
                value = value+_density_component(recipe,coordinates,s,nmo)
        else:
            value = _density_component(recipe,coordinates,channel,nmo)
        values.append(value)
    return tuple(values)


def _reduce_density(values,operators,coordinates,masks,spins):
    weights = (.5,1.,.5) if spins == ('aa','ab','bb') else (.5,)
    total = jnp.asarray(0.,jnp.result_type(*operators,*values))
    for weight,operator,value in zip(weights,operators,values):
        # Gather the box directly; sequential takes could produce B*N**3.
        integrals = operator[jnp.ix_(*coordinates)]
        for axis,mask in enumerate(masks):
            size = tuple(mask.size if k == axis else 1 for k in range(4))
            value = jnp.where(mask.reshape(size),value,0.)
        total = total+weight*jnp.sum(integrals*value)
    return total


@partial(jax.jit,static_argnames=('nmo','frame','dtype','indices','spins','widths','contract'))
def _density_execute(result,left,operators,*,nmo,frame,dtype,indices,spins,widths,contract):
    """One pure driver: all state and observable arrays remain dynamic inputs."""
    shape = tuple(map(len,indices))
    counts = tuple((n+w-1)//w for n,w in zip(shape,widths))
    axes = tuple(jnp.asarray(axis,jnp.int32) for axis in indices)
    recipe = _density_recipe(result,left,nmo,frame,dtype)
    output_dtype = jnp.result_type(dtype,*(a.dtype for a in jax.tree.leaves((result.t1,result.t2))))
    if prod(counts) == 1:
        # Static sector intersections avoid repeated/clipped O/V gathers for
        # a small affordable export. The planner still charges a dynamic box.
        values = _render_density(recipe,indices,spins,nmo)
        output = (_reduce_density(values,operators,axes,
                  tuple(jnp.ones(n,jnp.bool_) for n in shape),spins) if contract else values)
        return jax.tree.map(lambda value:jnp.where(result.converged & left.converged,value,jnp.nan),output)
    initial = (jnp.asarray(0.,jnp.result_type(output_dtype,*operators)) if contract else
               tuple(jnp.zeros(shape,output_dtype) for _ in spins))

    @jax.checkpoint
    def body(block,output):
        starts,coordinates,masks = [],[],[]
        for axis,(n,w,count) in enumerate(zip(shape,widths,counts)):
            cell = (block//prod(counts[axis+1:]))%count
            start = jnp.minimum(cell*w,n-w)
            positions = start+jnp.arange(w)
            starts.append(start)
            coordinates.append(axes[axis][positions])
            # Clamped tails overlap their predecessor. Exports overwrite the
            # same entries; scalar reductions count each entry exactly once.
            masks.append(positions >= cell*w)
        coordinates = tuple(coordinates)
        values = _render_density(recipe,coordinates,spins,nmo)
        if contract:
            return output+_reduce_density(values,operators,coordinates,masks,spins)
        return tuple(jax.lax.dynamic_update_slice(old,value,tuple(starts))
                     for old,value in zip(output,values))

    output = jax.lax.fori_loop(0,prod(counts),body,initial)
    return jax.tree.map(lambda value:jnp.where(result.converged & left.converged,value,jnp.nan),output)


def _density_driver(result,left,operators,*,nmo,nocc,dtype,frozen,config,
                     indices,spin,max_memory,reserved_bytes,contract):
    cfg = CCConfig() if config is None else config
    if cfg.method not in {'ccsd','ccd','ccs'}:
        raise NotImplementedError('The CC 2-RDM currently supports CCS, CCD and CCSD')
    indices,spins,output_dtype = _density_preflight(result,nmo=nmo,nocc=nocc,dtype=dtype,
        indices=indices,spin=spin,max_memory=None,reserved_bytes=reserved_bytes)
    frame = _density_frame(nmo,nocc,frozen)
    shape = tuple(map(len,indices))
    output_bytes = (6*jnp.result_type(output_dtype,*operators).itemsize if contract else
                    6*len(spins)*prod(shape)*output_dtype.itemsize)
    resident = (validate_reserved_bytes(reserved_bytes)+tree_storage_bytes(result,left,operators)
                +4*sum(shape)+output_bytes)
    leaf = jax.tree.leaves(result.t1)[0]
    available = None if max_memory is None else workspace_bytes(max_memory,
        resident_bytes=resident,device=getattr(leaf,'device',None))
    if not all(shape):
        return (jnp.asarray(0.,jnp.result_type(output_dtype,*operators)) if contract else
                tuple(jnp.zeros(shape,output_dtype) for _ in spins))

    side = max(shape)
    if available is not None:
        abstract = jax.eval_shape(lambda x:x,(result,left))
        working = {}

        def fits(size):
            if size not in working:
                widths = tuple(min(n,size) for n in shape)
                def block(state,adjoint,ops,coordinates,masks):
                    values = _render_density(_density_recipe(state,adjoint,nmo,frame,dtype),
                                             coordinates,spins,nmo)
                    return _reduce_density(values,ops,coordinates,masks,spins) if contract else values
                coordinate = tuple(jax.ShapeDtypeStruct((w,),jnp.int32) for w in widths)
                mask = tuple(jax.ShapeDtypeStruct((w,),jnp.bool_) for w in widths)
                ops = jax.eval_shape(lambda x:x,operators)
                graph = jax.make_jaxpr(block)(*abstract,ops,coordinate,mask)
                working[size] = (jaxpr_temporary_bytes(graph)+tree_storage_bytes(graph.consts)
                                 +16*sum(widths)+32)
            return working[size] <= available

        if not fits(side):
            if not fits(1):
                raise ValueError('Minimum CC density tile exceeds the working-memory budget')
            side = _largest_fitting(1,side,fits)
    return _density_execute(result,left,operators,nmo=nmo,frame=frame,dtype=jnp.dtype(dtype),
        indices=indices,spins=spins,widths=tuple(min(n,side) for n in shape),contract=contract)


def _rdm2_from_states(result,left,*,nmo,nocc,dtype,frozen=None,config=None,
                      indices=None,spin=None,max_memory=None,reserved_bytes=0):
    """Requested spatial CC density, without an active/full spin 2-RDM."""
    outputs = _density_driver(result,left,(),nmo=nmo,nocc=nocc,dtype=dtype,frozen=frozen,
        config=config,indices=indices,spin=spin,max_memory=max_memory,
        reserved_bytes=reserved_bytes,contract=False)
    return outputs if spin is None and isinstance(nocc,(tuple,list)) else outputs[0]


def _contract_rdm2_from_states(eri,result,left,*,nmo,nocc,dtype,frozen=None,
                               config=None,max_memory=None,reserved_bytes=0):
    operators = tuple(map(jnp.asarray,eri)) if isinstance(nocc,(tuple,list)) else (jnp.asarray(eri),)
    if len(operators) != (3 if isinstance(nocc,(tuple,list)) else 1) or any(
            value.shape != (nmo,)*4 for value in operators):
        raise ValueError('Two-body observable must have full spatial MO shape (nmo,)*4')
    return _density_driver(result,left,operators,nmo=nmo,nocc=nocc,dtype=dtype,frozen=frozen,
        config=config,indices=None,spin=None,max_memory=max_memory,
        reserved_bytes=reserved_bytes,contract=True)


def make_rdm2(h1, eri, result, *, nocc, frozen=None, config=None,
               indices=None,spin=None,max_memory=None,reserved_bytes=0):
    """Orbital-unrelaxed real MO 2-RDM, including Lambda and frozen electrons.

    dm2[p,q,r,s] = <a_p^+ a_r^+ a_s a_q>, Hermitian-symmetrized for real
    observables. Restricted: spin sum; unrestricted: (aa,ab,bb), whose energy
    contraction weights are (.5,1,.5). ``indices`` selects four spatial MO axes;
    ``spin`` selects aa/ab/ba/bb. Output boxes share the decimal-MB budget with
    states, integrals and the existing O**2 V**2 spin amplitude bridge.
    CCSD(T), CC2 and linearized-model 2-RDMs are not supplied by this function.
    """
    cfg = CCConfig() if config is None else config
    h = jnp.asarray(h1[0] if isinstance(nocc,(tuple,list)) else h1)
    _density_preflight(result,nmo=h.shape[0],nocc=nocc,dtype=h.dtype,indices=indices,
        spin=spin,max_memory=max_memory,reserved_bytes=validate_reserved_bytes(reserved_bytes)
        +tree_storage_bytes(h1,eri,result))
    left = solve_lambda(h1, eri, result, nocc=nocc, frozen=frozen, config=cfg,
                         max_memory=max_memory,reserved_bytes=reserved_bytes)
    return _rdm2_from_lambda(h1, eri, result, left, nocc=nocc, frozen=frozen, config=cfg,
        indices=indices,spin=spin,max_memory=max_memory,reserved_bytes=reserved_bytes)
