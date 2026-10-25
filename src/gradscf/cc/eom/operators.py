"""Physical EOM actions built from a checked CCSD ground state."""

from dataclasses import dataclass, replace
from math import prod
import jax
import jax.numpy as jnp
from ..._memory import (array_storage_bytes, jaxpr_temporary_bytes,
                        resolve_memory_mb, validate_reserved_bytes, workspace_bytes)
from ...ao2mo import VirtualERI
from ...ao2mo.actions import _native_action_plan
from ...solvers import LinearOperator, NonHermitianSolverConfig, solve_nonhermitian
from ...solvers.memory import plan_nonhermitian
from ...solvers.diagnostics import require_converged_derivative
from ..types import CCConfig
from ..ground import METHODS
from ..integrals import prepare_integrals
from .._memory import source_reservation
from ..amplitudes import AmplitudeSpace
from ..rccsd import residual
from .amplitudes import EOMAmplitudeSpace
from .types import EOMConfig, EOMResult, EOMSubspaceResult
from ._charged import intermediates, ip_action, ea_action


@dataclass(frozen=True)
class _EOMMemory:
    resident_bytes: int
    closure_bytes: int
    operator_min_bytes: int
    operator_bytes_per_column: int
    config: NonHermitianSolverConfig
    solver_reserved_bytes: int
    action_columns: int
    max_memory: float


@dataclass(frozen=True)
class _EOMContext:
    operator: LinearOperator
    space: EOMAmplitudeSpace
    valid: object
    memory: _EOMMemory | None = None


def build_eom_operator(
    h1, eri, ground, *, nocc, frozen=None, config=None, cc_config=None,
    max_memory=None, reserved_bytes=0
):
    context = _operator_context(h1, eri, ground, nocc=nocc, frozen=frozen,
        config=config, cc_config=cc_config, max_memory=max_memory,
        reserved_bytes=reserved_bytes)
    return context.operator, context.space, context.valid


def _operator_context(h1, eri, ground, *, nocc, frozen=None, config=None,
                      cc_config=None, max_memory=None, reserved_bytes=0):
    cfg = EOMConfig() if config is None else config
    cc_cfg = CCConfig() if cc_config is None else cc_config
    if not isinstance(nocc, int) or isinstance(ground.t1, (tuple, list)):
        raise NotImplementedError(
            "Initial EOM-CCSD requires a real restricted reference"
        )
    space = _checked_space(ground, cfg, cc_cfg)
    if jnp.size(eri) > cfg.max_intermediate_elements:
        raise ValueError("EOM input integrals exceed max_intermediate_elements")
    if max_memory is not None:
        reserved_bytes = validate_reserved_bytes(reserved_bytes)
        h1, eri = jnp.asarray(h1), jnp.asarray(eri)
        reserved_bytes += array_storage_bytes(h1, eri)
        # Shape-only preparation charges the exact selected payload before
        # eager gathers allocate it, including frozen-core/virtual selections.
        selected = jax.eval_shape(lambda h, g: prepare_integrals(
            h, g, nocc=nocc, frozen=frozen), h1, eri)
        plan_nonhermitian(space.size, ground.t1.dtype, _solver_config(cfg), max_memory,
            resident_bytes=reserved_bytes+array_storage_bytes(
                *jax.tree.leaves(ground), *jax.tree.leaves(selected)),
            device=getattr(h1, "device", None))
    ints = prepare_integrals(h1, eri, nocc=nocc, frozen=frozen)
    return _planned_operator(ints, ground, space, cfg, cc_cfg, max_memory, reserved_bytes)


def build_eom_operator_from_integrals(ints, ground, *, config=None, cc_config=None,
                                      max_memory=None, reserved_bytes=0):
    """EE/IP/EA operator from the same prepared restricted integral snapshot.

    The capacity guard counts supplied storage leaves, without a full MO ERI
    estimate. Native VirtualERI supplies the same bare virtual action to
    EE/IP/EA; charged dressing is applied without a cached dense Wvvvv.
    max_memory plans known primal and isolated-pair response arrays in decimal
    MB, with the shared 20% runtime reserve and independent phase lifetimes.
    The array plans exclude compilation/allocator state and external AD storage;
    they are not a process memory limit.
    Native source reservations, declared tile arrays and selected-worker
    optimizer scratch share the same construction/primal/response budget.
    """
    context = _operator_context_from_integrals(ints, ground, config=config,
        cc_config=cc_config, max_memory=max_memory, reserved_bytes=reserved_bytes)
    return context.operator, context.space, context.valid


def _operator_context_from_integrals(ints, ground, *, config=None, cc_config=None,
                                     max_memory=None, reserved_bytes=0):
    cfg = EOMConfig() if config is None else config
    cc_cfg = CCConfig() if cc_config is None else cc_config
    if isinstance(ground.t1, (tuple, list)):
        raise NotImplementedError("Initial EOM-CCSD requires a real restricted reference")
    space = _checked_space(ground, cfg, cc_cfg)
    if not hasattr(ints.vvvv, "shape") and not isinstance(ints.vvvv, VirtualERI):
        raise NotImplementedError("Prepared EOM requires dense or native virtual ERIs")
    if sum(jnp.size(x) for x in jax.tree.leaves(ints)) > cfg.max_intermediate_elements:
        raise ValueError("Prepared EOM integral storage exceeds max_intermediate_elements")
    return _planned_operator(ints, ground, space, cfg, cc_cfg, max_memory, reserved_bytes)


def _array_bytes(variable):
    aval = getattr(variable, "aval", None)
    if not hasattr(aval, "shape") or not hasattr(aval, "dtype"):
        return 0
    return prod(aval.shape)*jnp.dtype(aval.dtype).itemsize


def _native_primitive_bytes(ints, count, dtype):
    """Declare selected native worker/staging scratch for this RHS shape."""
    if not isinstance(ints.vvvv, VirtualERI):
        return {}
    provider = ints.vvvv
    rhs = jax.ShapeDtypeStruct((count, ints.nvir, ints.nvir), dtype)
    _, _, native = _native_action_plan(provider.coefficients, provider.basis, rhs,
        provider.max_memory, provider.reserved_bytes, working_bytes=provider._working_bytes)
    names = ('gradscf_direct_jk', 'gradscf_direct_jk_coordinates',
             *(f'gradscf_direct_jk_geometry_order{order}_{kind}'
               for order in (1, 2) for kind in ('jvp', 'vjp')))
    return dict.fromkeys(names, native)


def _planned_operator(ints, ground, space, cfg, cc_cfg, max_memory, reserved_bytes):
    if max_memory is None:
        op, _, valid, _, _ = _build_operator(ints, ground, space, cfg, cc_cfg)
        return _EOMContext(op, space, valid)
    reserved_bytes = validate_reserved_bytes(reserved_bytes)
    max_memory = resolve_memory_mb(max_memory)
    reserved_bytes += validate_reserved_bytes(source_reservation(ints))
    device = getattr(ints.fock, "device", None)
    if isinstance(ints.vvvv, VirtualERI):
        ints = ints._replace(vvvv=replace(ints.vvvv, max_memory=max_memory,
            _working_bytes=workspace_bytes(max_memory, device=device)))
    base = jax.tree.leaves((ints, ground))
    resident = reserved_bytes+array_storage_bytes(*base)
    solver_cfg = _solver_config(cfg)
    no = ground.t1.shape[0]
    construction = _native_primitive_bytes(ints, no*no, jnp.result_type(ground.t1, ground.t2))
    if max(construction.values(), default=0) > workspace_bytes(
            max_memory, resident_bytes=resident, device=device):
        raise ValueError('Native EOM construction exceeds the working-memory budget')
    # This minimum check precedes linearize/intermediates, including their
    # abstract tracing. Every check retains the SAME original MB budget.
    plan_nonhermitian(space.size, ground.t1.dtype, solver_cfg, max_memory,
                       resident_bytes=resident, device=device)
    shaped = jax.make_jaxpr(lambda i, g: _build_operator(
        i, g, space, cfg, cc_cfg, storage_only=True))(ints, ground)
    if jaxpr_temporary_bytes(shaped, primitive_bytes=construction) > workspace_bytes(
            max_memory, resident_bytes=resident, device=device):
        raise ValueError("EOM operator construction exceeds the working-memory budget")
    # Outputs are residual, diagonal, validity and precisely the arrays held by
    # the linearized EE action or the charged-sector namespace. Input aliases
    # are already resident; no conceptual full-MO or provider nvir**4 is added.
    inputs = {id(v) for v in shaped.jaxpr.invars}
    held = {id(v): v for v in shaped.jaxpr.outvars[1:] if id(v) not in inputs}
    closure_estimate = sum(_array_bytes(v) for v in held.values())
    plan_nonhermitian(space.size, ground.t1.dtype, solver_cfg, max_memory,
                       resident_bytes=resident+closure_estimate, device=device)
    op, _, valid, held, action = _build_operator(ints, ground, space, cfg, cc_cfg)
    vector = jax.ShapeDtypeStruct((space.size,), op.dtype)
    right = jax.make_jaxpr(action)(vector)
    left = jax.make_jaxpr(lambda v: jax.linear_transpose(action, v)(v)[0])(vector)
    resident = reserved_bytes+array_storage_bytes(
        *base, *held, op.diagonal, valid, *right.consts, *left.consts)
    closure_bytes = resident-reserved_bytes-array_storage_bytes(*base)
    # Davidson complex Ritz vectors run through the same real operator. Charge
    # both real/imaginary work widths, also safely covering dense real actions.
    declared = tuple(jaxpr_temporary_bytes(graph) for graph in (right, left))
    per_column = 2*max(declared)
    # IP no longer calls bare virtual ERIs after construction. EE uses a
    # doubles RHS; EA uses its existing fused no-by-v-by-v RHS. The native
    # primitive batches root columns sequentially, so its opaque work is fixed
    # scratch; declared JAX arrays scale with the simultaneous column count.
    charges = ({} if cfg.sector == 'ip' else _native_primitive_bytes(
        ints, no*no if cfg.sector == 'ee' else no, op.dtype))
    native = 2*max(jaxpr_temporary_bytes(graph, primitive_bytes=charges)-arrays
                   for graph, arrays in zip((right, left), declared))
    planned, solver_bytes, columns = plan_nonhermitian(space.size, op.dtype,
        solver_cfg, max_memory, resident_bytes=resident,
        operator_min_bytes=native, operator_bytes_per_column=per_column, device=device)
    memory = _EOMMemory(resident, closure_bytes, native, per_column, planned,
                        solver_bytes, columns, max_memory)
    return _EOMContext(op, space, valid, memory)


def _checked_space(ground, cfg, cc_cfg):
    if cc_cfg.method != "ccsd":
        raise ValueError("EOM-CCSD requires the CCSD ground-state model")
    if jnp.ndim(ground.t1) != 2:
        raise ValueError("CC singles must have shape (nocc, nvir)")
    no, nv = ground.t1.shape
    space = EOMAmplitudeSpace(no, nv, cfg.sector)
    if cfg.solver == "dense" and space.size > cfg.max_dense:
        raise ValueError(
            "EOM reference exceeds max_dense before constructing intermediates"
        )
    if not space.size or cfg.nroots > space.size:
        raise ValueError("Invalid root count or empty EOM sector")
    return space


def _build_operator(ints, ground, space, cfg, cc_cfg, *, storage_only=False):
    no, nv = ground.t1.shape
    if (no, nv) != (ints.nocc, ints.nvir):
        raise ValueError("CC amplitudes and active EOM space differ")
    tspace = AmplitudeSpace(no, nv)
    t = tspace.pack(ground.t1, ground.t2)
    # Packing reads one triangle; validate the complete supplied tensor before
    # using the same independent coordinates in every sector.
    t1, t2 = tspace.unpack(t)
    symmetry_error = jnp.max(jnp.abs(ground.t2 - t2), initial=0.0)
    symmetry_tol = (
        64
        * jnp.finfo(t.dtype).eps
        * jnp.maximum(1.0, jnp.max(jnp.abs(ground.t2), initial=0.0))
    )

    def equations(x):
        return tspace.pack(*residual(*tspace.unpack(x), ints, model="ccsd"))

    current = equations(t)
    valid = (
        ground.converged
        & (ground.method_id == METHODS.index("ccsd"))
        & jnp.all(jnp.isfinite(t))
    )
    valid &= jnp.all(jnp.isfinite(ground.t2)) & (symmetry_error <= symmetry_tol)
    valid &= jnp.max(jnp.abs(current), initial=0.0) <= 10 * cc_cfg.residual_tol
    if cfg.sector == "ee":
        _, action = jax.linearize(equations, t)
        held = jax.tree.leaves(action)
    else:
        w = intermediates(t1, t2, ints, cfg.sector)
        held = jax.tree.leaves(vars(w))
        contraction = ip_action if cfg.sector == "ip" else ea_action

        def action(v):
            return space.pack(*contraction(*space.unpack(v), w))

    # Orbital-energy differences are a physical preconditioner approximation,
    # not a diagonal assembled by probing the EOM matrix.
    eps = jnp.diag(ints.fock)
    eo, ev = eps[:no], eps[no:]
    if cfg.sector == "ee":
        d1 = ev[None, :] - eo[:, None]
        d2 = d1[:, None, :, None] + d1[None, :, None, :]
    elif cfg.sector == "ip":
        d1 = -eo
        d2 = ev[None, None, :] - eo[:, None, None] - eo[None, :, None]
    else:
        d1 = ev
        d2 = ev[None, :, None] + ev[None, None, :] - eo[:, None, None]
    diagonal = space.pack(d1, d2) / space.pack(jnp.ones_like(d1), jnp.ones_like(d2))
    if storage_only:
        return current, diagonal, valid, *held

    def checked(v):
        return require_converged_derivative(action(v), valid)

    op = LinearOperator(
        (space.size, space.size),
        t.dtype,
        checked,
        diagonal=diagonal,
        matmat=lambda v: jax.vmap(checked, in_axes=1, out_axes=1)(v),
    )
    return op, space, valid, held, action


def run_eom(h1, eri, ground, *, nocc, frozen=None, config=None, cc_config=None,
            max_memory=None, reserved_bytes=0, probes=None):
    """EE/IP/EA energies relative to the same converged CCSD ground state.

    EA returns E(N+1)-E(N); the conventional electron affinity is its negative.
    Composition with run_cc inside an AD function includes T response. Right/
    left vectors are forward-only by default; vector_response opts into the
    shared isolated-real-eigenpair response needed by transition properties.
    EOMConfig(response=EigenResponseConfig(target="subspace")) instead returns
    invariant projection/energy sum for a complete isolated real cluster;
    optional probes request P@probes without assembling the full projector.
    """
    cfg = EOMConfig() if config is None else config
    reserved_bytes = validate_reserved_bytes(reserved_bytes)+array_storage_bytes(probes)
    context = _operator_context(
        h1, eri, ground, nocc=nocc, frozen=frozen, config=cfg, cc_config=cc_config,
        max_memory=max_memory, reserved_bytes=reserved_bytes
    )
    return _run_operator(context.operator, context.valid, cfg, context.memory, probes=probes)


def run_eom_from_integrals(ints, ground, *, config=None, cc_config=None,
                           max_memory=None, reserved_bytes=0, probes=None):
    """Prepared EE/IP/EA roots with the same solver and response contracts."""
    cfg = EOMConfig() if config is None else config
    reserved_bytes = validate_reserved_bytes(reserved_bytes)+array_storage_bytes(probes)
    context = _operator_context_from_integrals(
        ints, ground, config=cfg, cc_config=cc_config, max_memory=max_memory,
        reserved_bytes=reserved_bytes)
    return _run_operator(context.operator, context.valid, cfg, context.memory, probes=probes)


def _solver_config(cfg):
    return NonHermitianSolverConfig(
        method=cfg.solver, maxiter=cfg.max_cycle, max_space=cfg.max_space,
        guard_roots=cfg.guard_roots, seed=cfg.seed,
        preconditioner_floor=cfg.preconditioner_floor, nroots=cfg.nroots,
        atol=cfg.conv_tol, gap_atol=cfg.gap_tol, gap_rtol=cfg.gap_tol,
        imaginary_tol=cfg.imaginary_tol, max_condition=cfg.max_condition,
        max_dense=cfg.max_dense, block_size=cfg.block_size)


def _run_operator(op, valid, cfg, memory=None, *, probes=None):
    options = {} if memory is None else dict(_batch_size=memory.action_columns,
        _response_budget=dict(max_memory=memory.max_memory,
            reserved_bytes=memory.resident_bytes, operator_min_bytes=memory.operator_min_bytes,
            operator_bytes_per_column=memory.operator_bytes_per_column,
            action_columns=memory.action_columns))
    solved = solve_nonhermitian(
        op,
        vector_response=cfg.vector_response,
        response_config=cfg.response_config, response=cfg.response, probes=probes,
        config=_solver_config(cfg) if memory is None else memory.config,
        **options,
    )
    subspace = cfg.response is not None and cfg.response.target == 'subspace'
    invariant = solved if subspace else None
    if subspace:
        solved = solved.raw_result
    response = solved.response_valid & valid
    energy = require_converged_derivative(solved.values, response)
    result = EOMResult(
        energy,
        solved.right_vectors,
        solved.left_vectors,
        solved.residual_norms,
        solved.left_residual_norms,
        solved.converged & valid,
        response,
        solved.condition_numbers,
        solved.raw_eigenvalues,
        valid,
        solved.biorthogonality_error,
        solved.spectrum_complete,
        solved.iterations,
        solved.subspace_dimension,
        solved.spectral_gaps,
        solved.guard_residual_norms,
        solved.restarts,
        cfg.vector_response or (cfg.response is not None and cfg.response.target == 'eigenpairs'),
        cfg.sector,
    )
    if subspace:
        response = invariant.response_valid & valid
        projection = None if invariant.projection is None else require_converged_derivative(
            invariant.projection, response)
        return EOMSubspaceResult(projection,
            require_converged_derivative(invariant.eigenvalue_sum, response),
            response, invariant.boundary_gap, valid, jax.lax.stop_gradient(result))
    return result
