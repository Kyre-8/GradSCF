"""Known CC array owners and physical action scratch for shared solvers.

The counts describe the existing dense residual's largest full amplitude
batch. They do not bound compilation, allocator state or an external AD tape.
Numerical DIIS/GMRES capacities and the runtime reserve belong to solvers.memory.
"""
from dataclasses import replace
from functools import lru_cache

import jax
import jax.numpy as jnp

from gradscf._memory import (array_storage_bytes, jaxpr_live_bytes, resolve_memory_mb,
                             tree_storage_bytes, validate_reserved_bytes, workspace_bytes)
from gradscf.ao2mo.actions import VirtualERI, _native_action_workspace
from gradscf.solvers.memory import plan_linear, plan_nonlinear
from ._spin_virtual import _SpinVirtualERI


def _residual_dtype(ints, dtype=None):
    """Contraction precision shared by residual scratch and retained outputs."""
    operands = tuple(value.dtype for value in jax.tree.leaves(
        ints._replace(reference_energy=None)) if hasattr(value, 'dtype'))
    return jnp.result_type(*operands, *(() if dtype is None else (dtype,)))


def residual_workspace(ints, no, nv, method, *, spin=False, dtype=None):
    """Conservative live buffer envelope of the unchanged residual body.

    Count unpacked/old/new T/R/tau and doubles contraction buffers, four ovvv
    work arrays, occupied intermediates and singles matrices. Linear models
    retain the value and tangent of that same body. Bare operand permutations
    use actual stored leaves; a provider has no logical full-spin v**4 charge.
    """
    # Fock/initial precision cannot lower a contraction with higher-precision
    # integral operands. The reference scalar does not enter the residual.
    dtype = _residual_dtype(ints, dtype)
    item = dtype.itemsize
    singles = 8*(no*no+nv*nv)+12*no*nv
    if method == 'ccs':
        return item*(singles+no*no*nv*nv)
    work = item*(singles+20*no*no*nv*nv+4*no*nv**3+6*no**3*nv+4*no**4)
    if isinstance(ints.vvvv, VirtualERI):
        bare = 0
    elif isinstance(ints.vvvv, _SpinVirtualERI):
        copies = {id(value): jax.ShapeDtypeStruct(value.shape, jnp.result_type(value.dtype, dtype))
                  for value in jax.tree.leaves(ints.vvvv)}
        bare = 2*array_storage_bytes(*copies.values())
    else:
        cached = (3*nv**4 < no*no*(2*no*nv+no*no) if spin else
                  3*nv**4 < no**3*nv)
        bare = (3 if cached else 1)*array_storage_bytes(
            jax.ShapeDtypeStruct(ints.vvvv.shape, jnp.result_type(ints.vvvv.dtype, dtype)))
    return (work+bare)*(2 if method in ('lccd', 'lccsd') else 1)


def _native_scratch(provider):
    native, per_rhs = _native_action_workspace(provider.basis, provider.coefficients.shape[1])
    return native+per_rhs


def source_reservation(ints):
    """Retain opaque preparation reservations without assuming array aliases.

    A caller may replace individual prepared fields. Equal byte counts do not
    prove that those new arrays alias the original producer's reserved owners.
    """
    if not isinstance(ints.vvvv, VirtualERI):
        return 0
    return ints.vvvv.reserved_bytes


def _action_budget(ints, resident, working, native_min, no, nv, item, memory):
    if not isinstance(ints.vvvv, VirtualERI):
        return ints
    # The provider charges its own numeric children, RHS and three output
    # buffers. They already occur in the enclosing residual envelope.
    own = array_storage_bytes(ints.vvvv.coefficients,
                              *jax.tree.leaves(ints.vvvv.basis))+4*no*no*nv*nv*item
    reservation = max(0, resident+working-native_min-own)
    return ints._replace(vvvv=replace(ints.vvvv, max_memory=memory, reserved_bytes=reservation))


def ground_budget(ints, initial, owners, nonlinear, linear, max_memory,
                  reserved_bytes, no, nv, method, *, spin=False):
    reserved_bytes = validate_reserved_bytes(reserved_bytes)
    if max_memory is None:
        return ints, nonlinear, linear, {}
    memory = resolve_memory_mb(max_memory)
    external = validate_reserved_bytes(reserved_bytes)+source_reservation(ints)
    resident = external+tree_storage_bytes(ints, owners, exclude=(initial,))
    item = initial.dtype.itemsize
    scratch = residual_workspace(ints, no, nv, method, spin=spin, dtype=initial.dtype)
    native_min = _native_scratch(ints.vvvv) if isinstance(ints.vvvv, VirtualERI) else 0
    scratch += native_min
    device = getattr(initial, 'device', None)
    nonlinear, primal = plan_nonlinear(initial.size, initial.dtype, nonlinear, memory,
        resident_bytes=resident+array_storage_bytes(initial),
        operator_min_bytes=scratch, device=device)
    # Root attachment keeps its seed/initial and a response RHS. The two phases
    # share a physical action, so choose a tile fitting their maximum, not sum.
    linear, response = plan_linear(initial.size, initial.dtype, linear, memory,
        resident_bytes=resident+3*initial.size*item, operator_min_bytes=scratch, device=device)
    phase = max(initial.size*item+primal, 3*initial.size*item+response)
    planned = _action_budget(ints, resident, phase, native_min, no, nv, item, memory)
    return planned, nonlinear, linear, dict(max_memory=memory,
        reserved_bytes=resident, operator_min_bytes=scratch)


def preparation_reservation(prepare, inputs, max_memory, reserved_bytes, *, owners=(), consumer_owned=()):
    reserved_bytes = validate_reserved_bytes(reserved_bytes)
    if max_memory is None:
        return reserved_bytes
    planned = jax.eval_shape(prepare, *inputs)
    unrestricted = isinstance(planned, tuple) and len(planned) == 3 and hasattr(planned[0], 'fock')
    payload = planned[0] if unrestricted else planned
    # Raw UCC explicitly casts every complete h/G input before gathering its
    # selected spin blocks. Equal input objects can yield separate cast copies.
    copies = tuple(jax.ShapeDtypeStruct(value.shape, payload.fock.dtype)
        for value in jax.tree.leaves(inputs)
        if unrestricted and hasattr(value, 'dtype') and value.dtype != payload.fock.dtype)
    matrix = inputs[0][0] if unrestricted else inputs[0]
    matrix_bytes = ((12 if unrestricted else 6)*matrix.size*payload.fock.dtype.itemsize
                    if hasattr(matrix, 'size') else 0)
    source = tree_storage_bytes(inputs, owners, consumer_owned)
    workspace_bytes(max_memory, resident_bytes=reserved_bytes+source+
        tree_storage_bytes(planned, copies)+matrix_bytes)
    # Right/left/initial states survive preparation, but the consumer counts
    # them explicitly. Only its otherwise invisible preparation inputs carry on.
    return reserved_bytes+tree_storage_bytes(inputs, owners, exclude=consumer_owned)


def check_payload(ints, owners, max_memory, reserved_bytes, no, nv, method, *, spin=False):
    """Reject a physically impossible minimum before VJP/T3 tensor creation."""
    if max_memory is None:
        return
    resident = validate_reserved_bytes(reserved_bytes)+source_reservation(ints)+tree_storage_bytes(ints, owners)
    scratch = residual_workspace(ints, no, nv, method, spin=spin)
    if isinstance(ints.vvvv, VirtualERI):
        scratch += _native_scratch(ints.vvvv)
    if scratch > workspace_bytes(max_memory, resident_bytes=resident):
        raise ValueError('Minimum CC physical action exceeds the working-memory budget')


@lru_cache(maxsize=32)
def _streamed_response_workspace(shapes, dtypes, spin):
    """Whole caller lift/value/VJP/directional-second bytes, cached as an int.

    Original caller inputs are resident, so graph inputs are excluded. The
    real lift outputs are returned as profile auxiliaries to keep their eager
    caller lifetime through the kernel; this pins existing SSA variables rather
    than charging the bridge twice. Directions stay in the caller's space.
    """
    from .integrals import CCIntegrals
    from .spin_amplitudes import SpinAmplitudeSpace
    from ._streamed_triples import (
        _TriplesInputs, _restricted_inputs, _streamed_general_correction,
    )

    inputs = tuple(jax.ShapeDtypeStruct(shape, dtype) for shape, dtype in zip(shapes, dtypes))
    cotangent = jax.ShapeDtypeStruct((3,), jnp.result_type(*(jnp.dtype(t) for t in dtypes)))
    if spin:
        space = SpinAmplitudeSpace(tuple(s[0] for s in shapes[4:6]),
                                   tuple(s[1] for s in shapes[4:6]))

    def value(*xs):
        if spin:
            data = _TriplesInputs(*xs[:4])
            t1, t2 = space.from_blocks(xs[4:6], xs[6:], xs[0].dtype)
            held = (t1, t2)
        else:
            fock, ovvv, ovoo, ovov, t1, t2 = xs
            original = CCIntegrals(fock, None, None, ovoo, ovov, None, None, ovvv, None, None)
            data, t1, t2 = _restricted_inputs(original, t1, t2)
            held = (*data, t1, t2)
        result = _streamed_general_correction(data, t1, t2, denominator_tol=1e-10)
        return jnp.stack(result[:3]), held

    def reverse(xs, cot):
        primal, pullback, held = jax.vjp(value, *xs, has_aux=True)
        return primal, pullback(cot), held

    def second(xs, cot, direction, dcot):
        return jax.jvp(reverse, (xs, cot), (direction, dcot))

    forward = jaxpr_live_bytes(jax.make_jaxpr(value)(*inputs))
    first = jaxpr_live_bytes(jax.make_jaxpr(reverse)(inputs, cotangent))
    higher = jaxpr_live_bytes(jax.make_jaxpr(second)(
        inputs, cotangent, inputs, cotangent))
    seeds = array_storage_bytes(cotangent)
    return int(max(forward, first+seeds,
                   higher+tree_storage_bytes(inputs)+2*seeds))


def triples_budget(ints, result, max_memory, reserved_bytes, no, nv, method,
                   orbital_basis, algorithm, *, spin=False):
    reserved_bytes = validate_reserved_bytes(reserved_bytes)
    if max_memory is None:
        return ints
    memory = resolve_memory_mb(max_memory)
    external = validate_reserved_bytes(reserved_bytes)+source_reservation(ints)
    # These are the moment operands used by the spatial formula or its spin
    # bridge. A higher-precision integral promotes the panel even when the
    # supplied Fock matrix and amplitudes have lower precision.
    moments = (ints.ovvv, ints.ooov, ints.oovv) if spin else (ints.ovvv, ints.ovoo, ints.ovov)
    dtype = jnp.result_type(ints.fock, *moments,
                            *jax.tree.leaves((result.t1, result.t2)))
    item = dtype.itemsize
    resident = external+tree_storage_bytes(ints, result)
    native_min = _native_scratch(ints.vvvv) if isinstance(ints.vvvv, VirtualERI) else 0
    residual = residual_workspace(ints, no, nv, method, spin=spin, dtype=dtype)+native_min
    # Unrestricted kernels first expand physical amplitudes, while restricted
    # general/streamed kernels use only their existing small spin-block bridge.
    expanded = item*(no*nv+no*no*nv*nv) if spin else 0
    so, sv = (no, nv) if spin else (2*no, 2*nv)
    bridge = 0
    if not spin and (orbital_basis == 'semicanonical' or algorithm == 'streamed'):
        bridge = item*(16*(no*nv**3+no*no*nv*nv+no**3*nv)+4*(no+nv)**2)
    if algorithm == 'streamed':
        if min(so, sv) < 3:
            # Both public entries return before constructing a triples bridge.
            triples = 0
        else:
            inputs = (ints.fock, *moments, *jax.tree.leaves((result.t1, result.t2)))
            shapes = tuple(a.shape for a in inputs)
            dtypes = tuple(a.dtype.name for a in inputs)
            amplitude_dtype = jnp.result_type(ints.fock,
                *jax.tree.leaves((result.t1, result.t2)))
            checked_dtype = _residual_dtype(ints, dtype)
            spec = jax.ShapeDtypeStruct
            canonical_dtype = (ints.fock.dtype if spin else
                               jnp.result_type(ints.fock, ints.mo_energy))
            symmetry_dtype = amplitude_dtype if spin else result.t2.dtype
            held = [spec((), checked_dtype), spec((), canonical_dtype),
                    spec((), symmetry_dtype), spec((), jnp.bool_)]
            if spin:
                # UCC reduces inline: only eps and small diagnostics survive,
                # besides expanded T1/T2 already owned outside the phase max.
                held += [spec((no+nv,), ints.fock.dtype), spec((), amplitude_dtype)]
            else:
                from .amplitudes import AmplitudeSpace
                # Eager RCC retains named r1/r2/packed residual locals while
                # the correction runs. Their dtype includes live amplitudes.
                held += [spec((no,nv), checked_dtype), spec((no,no,nv,nv), checked_dtype),
                         spec((AmplitudeSpace(no, nv).size,), checked_dtype)]
            triples = (array_storage_bytes(*held)
                       +_streamed_response_workspace(shapes, dtypes, spin))
    elif orbital_basis == 'semicanonical':
        triples = 32*(so*sv)**3*item+bridge
    else:
        triples = (64*no**3+nv*(nv-1)*(nv-2)//2)*item
    # UCC expansion is inside the complete streamed correction graph. It also
    # exists during residual validation, but must not be paid twice in either
    # phase. Other algorithms retain their original separate expansion owner.
    working = (max(expanded+residual, triples) if algorithm == 'streamed' else
               expanded+max(residual, triples))
    if working > workspace_bytes(memory, resident_bytes=resident):
        raise ValueError('Minimum triples arrays exceed the working-memory budget')
    return _action_budget(ints, resident, working, native_min, no, nv, item, memory)
