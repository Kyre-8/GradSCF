"""Static physical problem assembly through GradSCF's common eigensolver."""

import jax
import jax.numpy as jnp
from ..solvers import (
    EigenSolverConfig,
    EigenResponseConfig,
    LinearSolverConfig,
    solve_hermitian,
    solve_rpa,
)
from ..solvers.diagnostics import require_converged_derivative
from ..gw.screened import build_static_screening
from .space import make_bse_space, SpinBSESpace
from .types import BSEConfig, BSEResult
from .kernel import build_tda_operator, build_bse_operators


def run_bse(
    qp_energy,
    screening_energy,
    mo_factors,
    space,
    *,
    screening_space=None,
    qp_computed_mask=None,
    qp_converged_mask=None,
    config=None,
):
    """Real static BSE with explicit QP and screening spectra.

    Unspecified QP masks mean explicitly supplied energies, not inferred GW
    convergence. GW adapters must provide actual computed/converged masks.
    Requested isolated roots include an extra guard root for response checks.
    """
    spin = isinstance(space, SpinBSESpace)
    cfg = BSEConfig(singlet=None if spin else True) if config is None else config
    if (cfg.singlet is None) != spin:
        raise ValueError("Use singlet=None for unrestricted spin-conserving BSE only")
    qp, e, l = map(jnp.asarray, (qp_energy, screening_energy, mo_factors))
    expected_energy = (2, space.nmo) if spin else (space.nmo,)
    if (
        e.shape != expected_energy
        or qp.shape != e.shape
        or l.ndim != (4 if spin else 3)
        or (spin and l.shape[0] != 2)
        or l.shape[-2:] != (space.nmo,) * 2
    ):
        raise ValueError("BSE input shapes do not match the orbital space")
    if jnp.iscomplexobj(qp) or jnp.iscomplexobj(e) or jnp.iscomplexobj(l):
        raise NotImplementedError("Molecular BSE requires real collinear data")
    if l.size > cfg.max_factor_elements:
        raise ValueError("BSE factors exceed max_factor_elements")
    if not space.size or cfg.nroots > space.size:
        raise ValueError("Invalid BSE root count or empty excitation space")
    if cfg.solver == "dense" and space.size > cfg.max_dense:
        raise ValueError("BSE dense oracle exceeds max_dense")
    screen = (
        make_bse_space(space.nmo, space.nocc)
        if screening_space is None
        else screening_space
    )
    if (screen.nmo, screen.nocc) != (space.nmo, space.nocc):
        raise ValueError("Screening and excitation spaces must share a reference")
    # Canonicalize only contraction-sized roundoff, once, before both W and
    # the kernel use these factors. Larger asymmetry remains untouched and
    # is rejected by screening validation. Within the accepted neighborhood,
    # AD differentiates the same linear symmetric projection.
    axes = (-3, -2, -1)
    scale = jnp.maximum(1., jnp.max(jnp.abs(l), axis=axes, keepdims=True, initial=0.))
    tolerance = 64 * space.nmo * jnp.finfo(jnp.result_type(l, 1.)).eps * scale
    roundoff = jnp.max(jnp.abs(l - l.swapaxes(-1, -2)), axis=axes,
                      keepdims=True, initial=0.) <= tolerance
    l = jnp.where(roundoff, (l + l.swapaxes(-1, -2)) * .5, l)
    state = build_static_screening(
        e, l, occupied=screen.occupied, virtual=screen.virtual, max_aux=cfg.max_aux,
        config=cfg.screening_config
    )
    channels = space.channels if spin else (space,)
    spectra = qp if spin else (qp,)
    valid = state.valid
    for channel, energy in zip(channels, spectra):
        selected = jnp.asarray(channel.occupied + channel.virtual, dtype=int)
        valid &= jnp.all(jnp.isfinite(energy[selected]))
        valid &= jnp.all(energy[jnp.asarray(channel.virtual, dtype=int)][None, :]
                         - energy[jnp.asarray(channel.occupied, dtype=int)][:, None] > 0)
    for mask in (qp_computed_mask, qp_converged_mask):
        if mask is not None:
            mask = jnp.asarray(mask)
            if mask.shape != qp.shape or mask.dtype != jnp.bool_:
                raise ValueError(
                    "QP coverage/convergence masks must be boolean arrays of shape (nmo,)"
                )
            for channel, entries in zip(channels, mask if spin else (mask,)):
                selected = jnp.asarray(channel.occupied + channel.virtual, dtype=int)
                valid &= jnp.all(entries[selected])

    def amplitudes(vectors, response_valid):
        blocks, start = [], 0
        for channel in channels:
            shape = (cfg.nroots, len(channel.occupied), len(channel.virtual))
            block = vectors[start:start + channel.size].T.reshape(shape)
            blocks.append(require_converged_derivative(block, response_valid[:, None, None]))
            start += channel.size
        return tuple(blocks) if spin else blocks[0]
    if not cfg.tda:
        a, b = build_bse_operators(
            qp, l, space, state, singlet=cfg.singlet, block_size=cfg.block_size
        )
        solved = solve_rpa(
            a,
            b,
            config=EigenSolverConfig(
                method=cfg.solver,
                nroots=cfg.nroots,
                maxiter=cfg.max_cycle,
                max_subspace=None if cfg.solver == "dense" else cfg.max_space,
                atol=cfg.conv_tol,
                max_dense=cfg.max_dense,
                gradient_mode=cfg.gradient_mode,
                adjoint_tol=cfg.adjoint_tol,
                adjoint_maxiter=cfg.adjoint_max_cycle,
            ),
            gap_tol=cfg.gap_tol,
            seed=cfg.seed,
            response_config=cfg.response_config,
        )
        response_valid = solved.response_valid & valid
        energies = require_converged_derivative(solved.values, response_valid)
        x, y = [amplitudes(v, response_valid) for v in (solved.x, solved.y)]
        return BSEResult(
            energies,
            x,
            y,
            solved.residual_norms,
            solved.converged & valid,
            jnp.broadcast_to(solved.stable, energies.shape),
            response_valid,
            state.valid,
            state.min_gap,
            cfg.singlet,
            cfg.gradient_mode == "implicit_eigenvector",
            solved.stability_margins,
            solved.stability_certified,
            solved.stability_residual_norms,
        )
    op = build_tda_operator(
        qp, l, space, state, singlet=cfg.singlet, block_size=cfg.block_size
    )
    solved = solve_hermitian(
        op,
        config=EigenSolverConfig(
            method=cfg.solver,
            nroots=cfg.nroots,
            atol=cfg.conv_tol,
            maxiter=cfg.max_cycle,
            max_subspace=None if cfg.solver == "dense" else cfg.max_space,
            max_dense=cfg.max_dense,
            seed=cfg.seed,
        ),
        response=EigenResponseConfig(
            target=(
                "eigenpairs"
                if cfg.gradient_mode == "implicit_eigenvector"
                else "eigenvalues"
            ),
            gap_atol=cfg.gap_tol,
            gap_rtol=0.0,
            linear_config=cfg.response_config or LinearSolverConfig(
                rtol=cfg.adjoint_tol, maxiter=cfg.adjoint_max_cycle
            ),
        ),
    )
    energies = solved.values
    converged = solved.converged & valid
    stable = energies > cfg.gap_tol
    response_valid = solved.response_valid & valid & jnp.all(stable)
    energies = require_converged_derivative(energies, response_valid)
    x = amplitudes(solved.vectors[:, :cfg.nroots], response_valid)
    return BSEResult(
        energies,
        x,
        jax.tree.map(jnp.zeros_like, x),
        solved.residual_norms[: cfg.nroots],
        converged,
        stable,
        response_valid,
        state.valid,
        state.min_gap,
        cfg.singlet,
        cfg.gradient_mode == "implicit_eigenvector",
    )
