"""Ground-state CC/QCI orchestration: equations here, iteration/AD in solvers."""

import jax.numpy as jnp
from ..solvers import LinearSolverConfig
from ..solvers.nonlinear.iterate import NonlinearConfig, solve_nonlinear
from .types import CCConfig, CCResult
from .amplitudes import AmplitudeSpace
from .integrals import prepare_integrals, denominators
from .rccsd import residual, correlation_energy

METHODS = ("ccs", "ccd", "ccsd", "cc2", "lccd", "lccsd", "qcisd")


def linear_config(config):
    return LinearSolverConfig(
        rtol=config.adjoint_tol,
        maxiter=config.adjoint_max_cycle,
        restart=config.adjoint_restart,
    )


def run_cc(
    h1, eri, *, nocc, nuclear_repulsion=0.0, frozen=None, config=None, t1=None, t2=None
):
    """Solve a restricted model in static active space, with implicit response.

    Occupied orbitals precede virtuals. CCSD permits real noncanonical blocks;
    the conventional (T) correction separately requires a canonical reference.
    Zero/small denominators are diagnosed rather than silently regularized.
    """
    ints = prepare_integrals(
        h1, eri, nocc=nocc, nuclear_repulsion=nuclear_repulsion, frozen=frozen
    )
    return run_cc_from_integrals(ints, config=config, t1=t1, t2=t2)


def run_cc_from_integrals(ints, *, config=None, t1=None, t2=None):
    """Solve a restricted model from prepared blocks using the shared CC body.

    The caller owns the integral snapshot and all reference/frozen-space
    preparation. Iteration, convergence and implicit response are identical to
    ``run_cc``; this entry does not prepare or copy the integral tensors.
    """
    cfg = CCConfig() if config is None else config
    no, nv = ints.nocc, ints.nvir
    space = AmplitudeSpace(no, nv, cfg.method)
    d1, d2 = denominators(ints)
    # This is an iteration preconditioner, not a change to R(T) or to (T).
    d1, d2 = d1 - cfg.level_shift, d2 - 2 * cfg.level_shift
    if t1 is None:
        t1 = jnp.zeros((no, nv), dtype=ints.fock.dtype)
    if t2 is None:
        t2 = ints.ovov.transpose(0, 2, 1, 3) / jnp.where(
            jnp.abs(d2) > cfg.denominator_tol, d2, 1.0
        )
    t1, t2 = jnp.asarray(t1), jnp.asarray(t2)
    if jnp.iscomplexobj(t1) or jnp.iscomplexobj(t2):
        raise NotImplementedError(
            "Restricted CC currently requires real initial amplitudes"
        )
    if t1.shape != (no, nv) or t2.shape != (no, no, nv, nv):
        raise ValueError("Initial CC amplitudes have incorrect shapes")
    dtype = jnp.result_type(ints.fock.dtype, t1.dtype, t2.dtype)
    t1, t2 = t1.astype(dtype), t2.astype(dtype)
    initial = space.pack(t1, 0.5 * (t2 + t2.transpose(1, 0, 3, 2)))
    # pack is norm-preserving for amplitudes; diagonal action itself has no sqrt(2).
    ones = space.pack(jnp.ones_like(t1), jnp.ones_like(t2))
    diagonal = space.pack(d1, d2) / jnp.where(ones != 0, ones, 1.0)
    minimum = jnp.min(jnp.abs(diagonal), initial=jnp.inf)
    valid = (
        (minimum > cfg.denominator_tol)
        & jnp.all(jnp.isfinite(ints.fock))
        & jnp.isfinite(ints.reference_energy)
    )
    safe_diagonal = jnp.where(jnp.abs(diagonal) > cfg.denominator_tol, diagonal, 1.0)
    r = lambda vector: space.pack(
        *residual(*space.unpack(vector), ints, model=cfg.method)
    )
    e = lambda vector: correlation_energy(*space.unpack(vector), ints, model=cfg.method)
    solved = solve_nonlinear(
        r,
        initial,
        update=lambda x, rx: x + rx / safe_diagonal,
        observable=e,
        config=NonlinearConfig(
            maxiter=cfg.max_cycle,
            residual_tol=cfg.residual_tol,
            energy_tol=cfg.conv_tol,
            diis_space=cfg.diis_space,
            diis_start=cfg.diis_start_cycle,
            damping=cfg.damping,
        ),
        linear_config=linear_config(cfg),
        valid_inputs=valid,
    )
    t1, t2 = space.unpack(solved.solution)
    ecorr = e(solved.solution)
    return CCResult(
        ints.reference_energy + ecorr,
        ecorr,
        ints.reference_energy,
        t1,
        t2,
        solved.residual_norm,
        solved.energy_change,
        solved.iterations,
        solved.converged,
        minimum,
        jnp.asarray(METHODS.index(cfg.method), dtype=jnp.int32),
    )
