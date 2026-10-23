"""Left-state equations as a shared transposed residual Jacobian solve."""

import jax
import jax.numpy as jnp
from ..solvers import LinearOperator, solve_linear
from .types import CCConfig, LambdaResult
from .ground import METHODS, linear_config
from .integrals import prepare_integrals
from .amplitudes import AmplitudeSpace
from .rccsd import residual, correlation_energy
from .uccsd import prepare_ucc_integrals
from .spin_amplitudes import SpinAmplitudeSpace
from . import _spin_equations
from .._memory import jaxpr_temporary_bytes, tree_storage_bytes, validate_reserved_bytes
from ..ao2mo.actions import VirtualERI
from ..solvers.memory import plan_linear
from ._memory import (check_payload, preparation_reservation, residual_workspace,
                      source_reservation, _action_budget, _native_scratch)


def solve_lambda(h1, eri, result, *, nocc, frozen=None, config=None,
                 max_memory=None, reserved_bytes=0):
    """Solve J^T lambda = -dE/dt in independent norm-preserving coordinates.

    Convert the packed dual to the conventional restricted spin-adapted l1/l2:
    the full-tensor pairing is 2*l1*R1 + (2*l2-l2.swap(a,b))*R2.
    Lambda convergence is distinct from convergence of the right amplitudes.
    Unrestricted input instead returns spin blocks (la,lb)/(laa,lab,lbb), with
    full-spin pairing l1.R1 + .25*l2.R2. Its doubles dual conversion is fourfold.
    """
    cfg = CCConfig() if config is None else config
    unrestricted = isinstance(nocc, (tuple, list))
    if not unrestricted:
        reserved_bytes = preparation_reservation(lambda a, b: prepare_integrals(a, b,
            nocc=nocc, frozen=frozen), (h1, eri), max_memory, reserved_bytes, consumer_owned=(result,))
        ints = prepare_integrals(h1, eri, nocc=nocc, frozen=frozen)
        return solve_lambda_from_integrals(ints, result, config=cfg,
            max_memory=max_memory, reserved_bytes=reserved_bytes)
    if cfg.method not in {"ccsd", "ccd"}:
        raise NotImplementedError("Unrestricted Lambda supports CCSD and CCD")
    reserved_bytes = preparation_reservation(lambda a, b: prepare_ucc_integrals(a, b,
        nocc=nocc, frozen=frozen), (h1, eri), max_memory, reserved_bytes, consumer_owned=(result,))
    ints, occupied, virtual = prepare_ucc_integrals(h1, eri, nocc=nocc, frozen=frozen)
    return _solve_ucc_lambda(ints, result, occupied, virtual, cfg,
                             max_memory, reserved_bytes)


def _solve_ucc_lambda(ints, result, occupied, virtual, cfg, max_memory=None, reserved_bytes=0):
    space = SpinAmplitudeSpace(occupied, virtual, cfg.method)
    x = space.pack_blocks(result.t1, result.t2, ints.fock.dtype)
    return _solve_left(space, x, ints, result, cfg, unrestricted=True,
                       max_memory=max_memory, reserved_bytes=reserved_bytes)


def solve_lambda_from_integrals(ints, result, *, config=None, max_memory=None, reserved_bytes=0):
    """Restricted or UCC Lambda from prepared blocks and the shared VJP solve.

    The caller supplies the same active-space integral snapshot and model used
    for the right state. No integral preparation, symmetrization or copy is
    performed; derivatives follow the supplied integral chart.
    """
    cfg = CCConfig() if config is None else config
    if isinstance(result.t1, (tuple, list)):
        if cfg.method not in {"ccsd", "ccd"}:
            raise NotImplementedError("Unrestricted Lambda supports CCSD and CCD")
        occupied = tuple(t.shape[0] for t in result.t1)
        virtual = tuple(t.shape[1] for t in result.t1)
        return _solve_ucc_lambda(ints, result, occupied, virtual, cfg,
                                 max_memory, reserved_bytes)
    space = AmplitudeSpace(ints.nocc, ints.nvir, cfg.method)
    x = space.pack(result.t1, result.t2)
    return _solve_left(space, x, ints, result, cfg,
                       max_memory=max_memory, reserved_bytes=reserved_bytes)


def _solve_left(space, x, ints, result, cfg, *, unrestricted=False,
                max_memory=None, reserved_bytes=0):
    def functions(data):
        if unrestricted:
            return (lambda t: space.pack(*_spin_equations.residual(*space.unpack(t), data)),
                    lambda t: _spin_equations.packed_correlation_energy(t, space, data))
        return (lambda t: space.pack(*residual(*space.unpack(t), data, model=cfg.method)),
                lambda t: correlation_energy(*space.unpack(t), data, model=cfg.method))
    r, e = functions(ints)
    response = linear_config(cfg)
    external = validate_reserved_bytes(reserved_bytes)
    scratch = 0
    if max_memory is not None:
        if unrestricted:
            no = sum(t.shape[0] for t in result.t1)
            nv = sum(t.shape[1] for t in result.t1)
        else:
            no, nv = result.t1.shape
        check_payload(ints, (result, x), max_memory, external, no, nv, cfg.method,
                      spin=unrestricted)
        external += source_reservation(ints)
        scratch = residual_workspace(ints, no, nv, cfg.method, spin=unrestricted, dtype=x.dtype)
        def transpose_action(t, v):
            value, backward = jax.vjp(r, t)
            return value, backward(v)[0]
        # Include every declared forward/transpose intermediate, including
        # pack/unpack and layout copies, instead of treating the transpose as
        # one forward residual buffer. This is a static array sum, not a bound
        # on opaque compiled-library/allocator or external AD storage.
        scratch = max(scratch, jaxpr_temporary_bytes(
            jax.make_jaxpr(transpose_action)(x, x)))
        metadata, native_min = (), 0
        if isinstance(ints.vvvv, VirtualERI):
            # Metadata only: size the saved VJP owners before choosing the
            # native tile. The actual VJP below retains its original AD owner.
            metadata = jax.tree.leaves(jax.eval_shape(lambda t: jax.vjp(r, t), x))
            native_min = _native_scratch(ints.vvvv)
        scratch += native_min
        resident = external+tree_storage_bytes(ints, result, x, metadata)+x.size*x.dtype.itemsize
        response, working = plan_linear(x.size, x.dtype, response, max_memory,
            resident_bytes=resident, operator_min_bytes=scratch)
        if isinstance(ints.vvvv, VirtualERI):
            ints = _action_budget(ints, resident, working, native_min, no, nv,
                                   x.dtype.itemsize, max_memory)
            r, e = functions(ints)
    current, pullback = jax.vjp(r, x)
    valid = (
        result.converged
        & (result.method_id == METHODS.index(cfg.method))
        & (jnp.max(jnp.abs(current), initial=0.0) <= cfg.residual_tol * 10)
    )
    rhs = jnp.where(valid, -jax.grad(e)(x), jnp.nan)
    op = LinearOperator((x.size, x.size), x.dtype, lambda v: pullback(v)[0])
    budget = {}
    if max_memory is not None:
        budget = dict(max_memory=max_memory,
            reserved_bytes=external+tree_storage_bytes(ints, result, x, current,
                jax.tree.leaves(pullback), exclude=(rhs,)), operator_min_bytes=scratch)
    solved = solve_linear(op, rhs, config=response, **budget)
    if unrestricted:
        # x2=2*t2(independent), while <Lambda,R>=l1.R1 + .25*l2.R2.
        l1, dual2 = space.unpack_blocks(solved.solution)
        l2 = tuple(4*block for block in dual2)
    else:
        dual1, dual2 = space.unpack(solved.solution)
        l1 = dual1 / 2
        l2 = (2 * dual2 + dual2.swapaxes(2, 3)) / 3
    return LambdaResult(
        l1, l2, solved.solution, solved.residual_norm, solved.converged & valid
    )
