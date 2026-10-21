"""Physical EOM actions built from a checked CCSD ground state."""

import jax
import jax.numpy as jnp
from ...solvers import LinearOperator, NonHermitianSolverConfig, solve_nonhermitian
from ...solvers.diagnostics import require_converged_derivative
from ..types import CCConfig
from ..ground import METHODS
from ..integrals import prepare_integrals
from ..amplitudes import AmplitudeSpace
from ..rccsd import residual
from .amplitudes import EOMAmplitudeSpace
from .types import EOMConfig, EOMResult
from ._charged import intermediates, ip_action, ea_action


def build_eom_operator(
    h1, eri, ground, *, nocc, frozen=None, config=None, cc_config=None
):
    cfg = EOMConfig() if config is None else config
    cc_cfg = CCConfig() if cc_config is None else cc_config
    if not isinstance(nocc, int) or isinstance(ground.t1, (tuple, list)):
        raise NotImplementedError(
            "Initial EOM-CCSD requires a real restricted reference"
        )
    space = _checked_space(ground, cfg, cc_cfg)
    if jnp.size(eri) > cfg.max_intermediate_elements:
        raise ValueError("EOM input integrals exceed max_intermediate_elements")
    ints = prepare_integrals(h1, eri, nocc=nocc, frozen=frozen)
    return _build_operator(ints, ground, space, cfg, cc_cfg)


def build_eom_operator_from_integrals(ints, ground, *, config=None, cc_config=None):
    """EE/IP/EA operator from the same prepared restricted integral snapshot.

    The capacity guard counts supplied storage leaves, without a full MO ERI
    estimate. Matrix-free virtual ERIs are explicitly unsupported here: EA's
    current transformed Wvvvv remains dense and needs separate validation.
    """
    cfg = EOMConfig() if config is None else config
    cc_cfg = CCConfig() if cc_config is None else cc_config
    if isinstance(ground.t1, (tuple, list)):
        raise NotImplementedError("Initial EOM-CCSD requires a real restricted reference")
    space = _checked_space(ground, cfg, cc_cfg)
    if not hasattr(ints.vvvv, "shape"):
        raise NotImplementedError("Prepared EOM currently requires dense virtual ERIs")
    if sum(jnp.size(x) for x in jax.tree.leaves(ints)) > cfg.max_intermediate_elements:
        raise ValueError("Prepared EOM integral storage exceeds max_intermediate_elements")
    return _build_operator(ints, ground, space, cfg, cc_cfg)


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


def _build_operator(ints, ground, space, cfg, cc_cfg):
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
    else:
        w = intermediates(t1, t2, ints, cfg.sector)
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

    def checked(v):
        return require_converged_derivative(action(v), valid)

    op = LinearOperator(
        (space.size, space.size),
        t.dtype,
        checked,
        diagonal=diagonal,
        matmat=lambda v: jax.vmap(checked, in_axes=1, out_axes=1)(v),
    )
    return op, space, valid


def run_eom(h1, eri, ground, *, nocc, frozen=None, config=None, cc_config=None):
    """EE/IP/EA energies relative to the same converged CCSD ground state.

    EA returns E(N+1)-E(N); the conventional electron affinity is its negative.
    Composition with run_cc inside an AD function includes T response. Right/
    left vectors are forward-only by default; vector_response opts into the
    shared isolated-real-eigenpair response needed by transition properties.
    """
    cfg = EOMConfig() if config is None else config
    op, space, valid = build_eom_operator(
        h1, eri, ground, nocc=nocc, frozen=frozen, config=cfg, cc_config=cc_config
    )
    return _run_operator(op, valid, cfg)


def run_eom_from_integrals(ints, ground, *, config=None, cc_config=None):
    """Prepared EE/IP/EA roots with the same solver and response contracts."""
    cfg = EOMConfig() if config is None else config
    op, _, valid = build_eom_operator_from_integrals(
        ints, ground, config=cfg, cc_config=cc_config)
    return _run_operator(op, valid, cfg)


def _run_operator(op, valid, cfg):
    solved = solve_nonhermitian(
        op,
        vector_response=cfg.vector_response,
        response_config=cfg.response_config,
        config=NonHermitianSolverConfig(
            method=cfg.solver,
            maxiter=cfg.max_cycle,
            max_space=cfg.max_space,
            guard_roots=cfg.guard_roots,
            seed=cfg.seed,
            preconditioner_floor=cfg.preconditioner_floor,
            nroots=cfg.nroots,
            atol=cfg.conv_tol,
            gap_atol=cfg.gap_tol,
            gap_rtol=cfg.gap_tol,
            imaginary_tol=cfg.imaginary_tol,
            max_condition=cfg.max_condition,
            max_dense=cfg.max_dense,
            block_size=cfg.block_size,
        ),
    )
    response = solved.response_valid & valid
    energy = require_converged_derivative(solved.values, response)
    return EOMResult(
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
        cfg.vector_response,
        cfg.sector,
    )
