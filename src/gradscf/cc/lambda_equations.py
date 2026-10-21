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


def solve_lambda(h1, eri, result, *, nocc, frozen=None, config=None):
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
        ints = prepare_integrals(h1, eri, nocc=nocc, frozen=frozen)
        return solve_lambda_from_integrals(ints, result, config=cfg)
    if cfg.method not in {"ccsd", "ccd"}:
        raise NotImplementedError("Unrestricted Lambda supports CCSD and CCD")
    ints, occupied, virtual = prepare_ucc_integrals(h1, eri, nocc=nocc, frozen=frozen)
    space = SpinAmplitudeSpace(occupied, virtual, cfg.method)
    x = space.pack(*space.from_blocks(result.t1, result.t2, ints.fock.dtype))
    r = lambda t: space.pack(*_spin_equations.residual(*space.unpack(t), ints))
    e = lambda t: _spin_equations.correlation_energy(*space.unpack(t), ints)
    return _solve_left(space, x, r, e, result, cfg, unrestricted=True)


def solve_lambda_from_integrals(ints, result, *, config=None):
    """Restricted Lambda from prepared blocks, using the shared VJP solve.

    The caller supplies the same active-space integral snapshot and model used
    for the right state. No integral preparation, symmetrization or copy is
    performed; derivatives follow the supplied integral chart.
    """
    cfg = CCConfig() if config is None else config
    space = AmplitudeSpace(ints.nocc, ints.nvir, cfg.method)
    x = space.pack(result.t1, result.t2)
    r = lambda t: space.pack(*residual(*space.unpack(t), ints, model=cfg.method))
    e = lambda t: correlation_energy(*space.unpack(t), ints, model=cfg.method)
    return _solve_left(space, x, r, e, result, cfg)


def _solve_left(space, x, r, e, result, cfg, *, unrestricted=False):
    current, pullback = jax.vjp(r, x)
    valid = (
        result.converged
        & (result.method_id == METHODS.index(cfg.method))
        & (jnp.max(jnp.abs(current), initial=0.0) <= cfg.residual_tol * 10)
    )
    rhs = jnp.where(valid, -jax.grad(e)(x), jnp.nan)
    op = LinearOperator((x.size, x.size), x.dtype, lambda v: pullback(v)[0])
    solved = solve_linear(op, rhs, config=linear_config(cfg))
    dual1, dual2 = space.unpack(solved.solution)
    if unrestricted:
        # x2=2*t2(independent), while <Lambda,R>=l1.R1 + .25*l2.R2.
        l1, l2 = space.to_blocks(dual1, 4*dual2)
    else:
        l1 = dual1 / 2
        l2 = (2 * dual2 + dual2.swapaxes(2, 3)) / 3
    return LambdaResult(
        l1, l2, solved.solution, solved.residual_norm, solved.converged & valid
    )
