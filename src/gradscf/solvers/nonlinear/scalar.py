"""Batched independent real scalar roots, with optional AD frequency slopes.

The callback must be elementwise in x: its Jacobian is diagonal. For coupled
systems use the general nonlinear/root interfaces. Physics-specific branch
selection and implicit response remain the caller's responsibility.
"""
from dataclasses import dataclass
from math import isfinite
from numbers import Integral
from typing import NamedTuple

import jax
import jax.numpy as jnp


@dataclass(frozen=True)
class ScalarRootConfig:
    method: str = 'secant'
    ftol: float = 1e-6
    xtol: float = 1e-6
    maxiter: int = 100
    step_cap: float = .05
    slope_floor: float = 1e-10
    hybrid_every: int = 4
    max_backtrack: int = 4

    def __post_init__(self):
        if self.method not in {'secant', 'newton', 'hybrid'}:
            raise ValueError('Scalar root method must be secant, newton or hybrid')
        for name in ('ftol', 'xtol', 'step_cap', 'slope_floor'):
            if not isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f'{name} must be finite and positive')
        for name in ('maxiter', 'hybrid_every', 'max_backtrack'):
            v = getattr(self, name)
            if isinstance(v, bool) or not isinstance(v, Integral) or v < (0 if name == 'max_backtrack' else 1):
                raise ValueError(f'Invalid scalar root limit: {name}')


class ScalarRootResult(NamedTuple):
    roots: object
    residual: object
    converged: object
    iterations: object
    function_evaluations: object
    derivative_evaluations: object
    backtracks: object
    newton_steps: object


def bisect(function, lower, upper, *, tolerance=1e-12, maxiter=100):
    """Batched bracketed continuous scalar roots; returns (roots, converged).

    This is a primal solver. Use attach_root for derivatives of an isolated
    root; branch decisions of bisection do not define its physical derivative.
    """
    if tolerance <= 0 or not isfinite(tolerance) or maxiter < 1:
        raise ValueError('Bisection requires positive tolerance and maxiter')
    lo, hi = jnp.asarray(lower), jnp.asarray(upper)
    fl, fh = function(lo), function(hi)
    valid = (lo <= hi) & jnp.isfinite(fl) & jnp.isfinite(fh) & (jnp.sign(fl)*jnp.sign(fh) <= 0)
    def step(_, state):
        lo, hi, fl = state
        mid = lo+(hi-lo)/2
        fm = function(mid)
        left = jnp.sign(fl)*jnp.sign(fm) <= 0
        exact = fm == 0
        return jnp.where(exact,mid,jnp.where(left,lo,mid)), jnp.where(left,mid,hi), jnp.where(exact,0.,jnp.where(left,fl,fm))
    lo, hi, _ = jax.lax.fori_loop(0, maxiter, step, (lo, hi, fl))
    roots = lo+(hi-lo)/2
    residual = function(roots)
    valid &= jnp.isfinite(residual) & (jnp.abs(residual) <= tolerance)
    return roots, valid


class _State(NamedTuple):
    previous: object
    x: object
    fprevious: object
    fx: object
    slope: object
    best_x: object
    best_f: object
    done: object
    step: object
    iterations: object
    evaluations: object
    derivatives: object
    backtracks: object
    newton_steps: object
    stalled: object


def solve_scalar_roots(function, x0, *, x1=None, config=None, scan=False):
    """Solve independent roots; scan=True differentiates the finite trajectory.

    Newton uses JVP slopes, bounds steps, halves residual-increasing steps,
    then falls back to a secant step if necessary. Nonfinite hybrid secant
    trials also backtrack; an invalid fallback retains the last valid point.
    Hybrid refreshes AD slopes
    periodically or after poor progress. Secant preserves the original bounded
    two-point update. Both step and residual must converge. Failed lanes return
    the best finite-residual iterate and converged=False, never a false success.

    Evaluation counters count whole-batch callback evaluations (including JVP
    primals), not active lanes or FLOPs. Derivative evaluations are a subset.
    Full batches may be evaluated even when some lanes have already converged.
    In scan mode, rejected-trial values are stopped and accepted values are
    recomputed to keep domain errors out of the differentiated trajectory.
    No global root bracket, pole-crossing guarantee or implicit AD is supplied.
    """
    cfg = ScalarRootConfig() if config is None else config
    x0 = jnp.asarray(x0)
    if x0.ndim != 1 or not jnp.issubdtype(x0.dtype, jnp.floating):
        raise ValueError('Independent scalar roots require a real floating (n,) vector')
    x1 = x0 + 1e-4 if x1 is None else jnp.asarray(x1, dtype=x0.dtype)
    if x1.shape != x0.shape:
        raise ValueError('Initial root vectors must have equal shapes')

    def value_slope(x):
        return jax.jvp(function, (x,), (jnp.ones_like(x),))

    f0 = function(x0)
    f1, slope = value_slope(x1) if cfg.method == 'newton' else (function(x1), jnp.zeros_like(x1))
    if f0.shape != x0.shape or f1.shape != x0.shape or jnp.iscomplexobj(f1):
        raise ValueError('Residual must be a real vector with the same shape as x')
    better = jnp.isfinite(f1) & (~jnp.isfinite(f0) | (jnp.abs(f1) < jnp.abs(f0)))
    zeros = jnp.zeros(x0.shape, dtype=jnp.int32)
    state = _State(x0, x1, f0, f1, slope, jnp.where(better, x1, x0),
                   jnp.where(better, f1, f0), jnp.zeros(x0.shape, dtype=bool),
                   jnp.array(0), zeros, jnp.array(2), jnp.array(int(cfg.method == 'newton')),
                   jnp.array(0), zeros, zeros)

    def evaluate(x):
        if scan and cfg.method != 'secant':
            # Trial-domain errors belong only to line-search decisions. Their
            # discarded derivatives must not enter the accepted trajectory.
            return jax.lax.stop_gradient(function(x)), jnp.zeros_like(x)
        return value_slope(x) if cfg.method == 'newton' else (function(x), jnp.zeros_like(x))

    def body(s):
        active = ~s.done
        denom = s.fx - s.fprevious
        valid_secant = jnp.isfinite(denom) & (denom != 0.)
        secant = jnp.where(valid_secant, s.fx * (s.x-s.previous) /
                          jnp.where(valid_secant, denom, 1.), 0.)
        fevals, devals = s.evaluations, s.derivatives
        slope = s.slope
        use_ad = active & (cfg.method == 'newton')
        if cfg.method == 'hybrid':
            use_ad = active & ((s.step % cfg.hybrid_every == 0) | ~valid_secant | (s.stalled >= 2))
            def refresh(_):
                _, deriv = value_slope(s.x)
                return deriv, fevals+1, devals+1
            slope, fevals, devals = jax.lax.cond(jnp.any(use_ad), refresh,
                lambda _: (slope, fevals, devals), None)
        use_ad &= jnp.isfinite(slope) & (jnp.abs(slope) >= cfg.slope_floor)
        newton = s.fx / jnp.where(use_ad, slope, 1.)
        step = jnp.clip(jnp.where(use_ad, newton, secant), -cfg.step_cap, cfg.step_cap)
        candidate = jnp.where(active, s.x-step, s.x)
        fc, sc = evaluate(candidate)
        fevals += 1
        devals += int(cfg.method == 'newton' and not scan)
        backtracks = s.backtracks
        if cfg.method != 'secant':
            def bad(value):
                return active & (~jnp.isfinite(value) | (use_ad & (jnp.abs(value) > jnp.abs(s.fx))))
            def backtrack(_, carry):
                xc, fc, sc, step, ne, nd, nb = carry
                reject = bad(fc)
                def shorten(_):
                    smaller = jnp.where(reject, step*.5, step)
                    trial = jnp.where(reject, s.x-smaller, xc)
                    ft, st = evaluate(trial)
                    return trial, ft, st, smaller, ne+1, nd+int(cfg.method == 'newton' and not scan), nb+1
                return jax.lax.cond(jnp.any(reject), shorten, lambda _: carry, None)
            candidate, fc, sc, step, fevals, devals, backtracks = jax.lax.fori_loop(
                0, cfg.max_backtrack, backtrack, (candidate, fc, sc, step, fevals, devals, backtracks))
            reject = bad(fc)
            def fallback(_):
                trial = jnp.where(reject, s.x-jnp.clip(secant,-cfg.step_cap,cfg.step_cap), candidate)
                ft, st = evaluate(trial)
                return trial, ft, st, fevals+1, devals+int(cfg.method == 'newton' and not scan)
            candidate, fc, sc, fevals, devals = jax.lax.cond(jnp.any(reject), fallback,
                lambda _: (candidate, fc, sc, fevals, devals), None)
            use_ad &= ~reject
            # A secant fallback may itself leave the callback's domain.
            # Stay at the last valid point so a later AD refresh can recover.
            invalid = active & ~jnp.isfinite(fc)
            candidate = jnp.where(invalid, s.x, candidate)
            fc = jnp.where(invalid, s.fx, fc)
            sc = jnp.where(invalid, s.slope, sc)
            if scan:
                fc, sc = (value_slope(candidate) if cfg.method == 'newton'
                          else (function(candidate), jnp.zeros_like(candidate)))
                fevals += 1
                devals += int(cfg.method == 'newton')
        done = jnp.isfinite(fc) & (jnp.abs(candidate-s.x) < cfg.xtol) & (jnp.abs(fc) < cfg.ftol)
        better = active & jnp.isfinite(fc) & (~jnp.isfinite(s.best_f) | (jnp.abs(fc) < jnp.abs(s.best_f)))
        return _State(jnp.where(active,s.x,s.previous), jnp.where(active,candidate,s.x),
            jnp.where(active,s.fx,s.fprevious), jnp.where(active,fc,s.fx), sc,
            jnp.where(better,candidate,s.best_x), jnp.where(better,fc,s.best_f), s.done | done,
            s.step+1, s.iterations+active.astype(jnp.int32), fevals, devals, backtracks,
            s.newton_steps+use_ad.astype(jnp.int32),
            jnp.where(active & (jnp.abs(fc) >= .9*jnp.abs(s.fx)),s.stalled+1,0))

    if scan:
        def scan_body(s, _):
            return jax.lax.cond(jnp.all(s.done), lambda x:x, body, s), None
        state, _ = jax.lax.scan(scan_body, state, None, length=cfg.maxiter)
    else:
        state = jax.lax.while_loop(lambda s:(s.step < cfg.maxiter) & ~jnp.all(s.done), body, state)
    return ScalarRootResult(jnp.where(state.done,state.x,state.best_x),
        jnp.where(state.done,state.fx,state.best_f), state.done, state.iterations,
        state.evaluations,state.derivatives,state.backtracks,state.newton_steps)
