"""Eigenvalue-only self-consistent GW (evGW) -- Stage 3.

evGW re-injects the quasiparticle energies into both the Green's function
and the RPA screened interaction (orbitals and density frozen) until the
QP spectrum is self-consistent:

    e_{n+1} = G0W0[e_n]   (fixed-point iteration with optional damping)

The drivers re-evaluate the G0W0 machinery
(:func:`gradscf.gw.g0w0_cd_restricted` / ``_unrestricted`` /
:func:`gradscf.gw.pbc.krgw.g0w0_cd_gamma`) with the current QP spectrum
in G and W, retaining the original mean-field base in the Dyson equation.
The default outer loop is eager. ``differentiation`` opts into a compiled
primal and implicit outer root response; ``diff_mode`` alone controls only
the inner QP machinery and does not enable outer response.
Convergence certifies the on-shell residual of the chosen discrete CD
grid. Molecular static-W subtraction supplies the contour's own-pole limit;
frequency-grid accuracy still requires independent convergence checks.

References
----------
- M. S. Hybertsen and S. G. Louie, Phys. Rev. B 34, 5390 (1986).
  DOI:10.1103/PhysRevB.34.5390
- M. Shishkin and G. Kresse, Phys. Rev. B 75, 235102 (2007).
  DOI:10.1103/PhysRevB.75.235102 (evGW convergence behavior)
- X. Blase, C. Attaccalite and V. Olevano, Phys. Rev. B 83, 115103 (2011).
  DOI:10.1103/PhysRevB.83.115103
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import replace
from numbers import Integral

import jax
import jax.numpy as jnp
from jaxtyping import Array

from .g0w0 import (
    build_screened_w_restricted,
    g0w0_cd_restricted,
    g0w0_cd_unrestricted,
)
from .types import GWResult
from ..scf.autodiff import SCFDifferentiationConfig
from ..solvers.nonlinear import NonlinearConfig, attach_root, solve_nonlinear
from .outer_response import linear_config, require_occupation_gap, require_valid, require_real


def _converged_result(result, computed, tol, converged):
    """Keep fixed-G/W branch validity separate from the coupled outer root."""
    branch = jnp.ones_like(computed)
    if result.qp_branch_valid is not None:
        branch = branch & result.qp_branch_valid
    if result.qp_weight is not None:
        branch = branch & jnp.isfinite(result.qp_weight) & (result.qp_weight > 0)
    branch = ~computed | branch
    require_valid(jnp.all(branch), jnp.max(jnp.abs(result.qp_residual)),
                  "evGW quasiparticle slope")
    return replace(result, converged=converged,
                   converged_mask=(jnp.abs(result.qp_residual) < tol) & branch,
                   qp_computed_mask=computed)


def _evgw_loop(
    driver: Callable[..., GWResult],
    mo_energy_mf,
    *,
    max_iter: int = 20,
    tol: float = 1e-6,
    damping: float = 0.0,
    diis_space: int = 0,
    initial_guess=None,
    differentiation: SCFDifferentiationConfig | None = None,
    **kwargs,
) -> GWResult:
    """Shared fixed-point iteration around a G0W0 driver.

    ``mo_energy_mf`` stays the fixed mean-field base of the QP equation;
    only the pole spectrum (``mo_energy_poles`` of the drivers) is iterated.
    Iterate the diagonal self-consistent residual
    F(E) = E - e_mf - Re Sigma(E; G[E], W[E]) - delta_v.
    Each driver evaluates F with G/W rebuilt at E; the returned result
    and the unscaled residual share that same spectrum. Fixed-G/W partial
    frequency derivatives at a pole are not used to precondition this map.
    """
    if not 0.0 <= damping < 1.0:
        raise ValueError("damping must be in [0, 1).")
    if max_iter < 1 or tol <= 0.0:
        raise ValueError("max_iter and tol must be positive.")
    if not isinstance(diis_space, Integral) or isinstance(diis_space, bool) or diis_space < 0:
        raise ValueError("diis_space must be a nonnegative integer.")
    e_old = jnp.asarray(mo_energy_mf, dtype=jnp.float64)
    occupations = kwargs.get("nocc")
    selected = kwargs.get("orbs")
    selected = (jnp.arange(e_old.shape[-1]) if selected is None
                else jnp.asarray(tuple(selected), dtype=jnp.int32))
    computed = jnp.zeros_like(e_old, dtype=bool).at[..., selected].set(True)
    if initial_guess is not None:
        guess = jnp.asarray(initial_guess)
        if guess.shape != e_old.shape or jnp.iscomplexobj(guess):
            raise ValueError("initial_guess must be real and match the full mean-field spectrum.")
        require_valid(jnp.all(jnp.isfinite(guess)), jnp.max(jnp.abs(guess)), "evGW initial guess")
        # Numerical continuation cannot alter frozen levels or the MF base.
        e_old = jnp.where(computed, jax.lax.stop_gradient(guess), e_old)
    if occupations is not None:
        require_occupation_gap(e_old, occupations, "evGW")
    if differentiation is not None or diis_space:
        config = linear_config(differentiation or SCFDifferentiationConfig())

        def residual(energy):
            if occupations is not None:
                require_occupation_gap(energy, occupations, "evGW")
            result = driver(mo_energy_poles=energy, evaluate_only=True, **kwargs)
            # Unrequested levels follow the original MF spectrum. Their
            # equations must not be zero rows in the outer response Jacobian.
            return jnp.where(computed, result.qp_residual, energy - mo_energy_mf)

        def body(state):
            energy, iteration, _, _ = state
            error = residual(energy)
            norm = jnp.max(jnp.abs(error))
            done = jnp.isfinite(norm) & (norm < tol)
            energy = energy - jnp.where(done, 0., 1. - damping) * error
            return energy, iteration + 1, norm, done

        if diis_space:
            # Keep the same physical residual; only the numerical trajectory
            # changes. The shared solver owns DIIS and implicit JVP/VJP.
            shape = e_old.shape
            solved = solve_nonlinear(lambda e: residual(e.reshape(shape)).reshape(-1),
                e_old.reshape(-1), update=lambda e, error: e-error,
                # Sum of QP levels is an iteration monitor, not a total energy.
                observable=jnp.sum,
                config=NonlinearConfig(maxiter=max_iter, residual_tol=tol, energy_tol=tol,
                                       diis_space=int(diis_space), damping=damping),
                linear_config=config)
            energy, norm = solved.solution.reshape(shape), solved.residual_norm
            done = solved.converged & (norm < tol)
        else:
            def numerical_forward(seed):
                return jax.lax.while_loop(
                    lambda state: (state[1] < max_iter) & ~state[3], body,
                    (seed, jnp.array(0), jnp.array(jnp.inf), jnp.array(False)))
            # As in the shared nonlinear solver, stop captured physical
            # inputs before tracing the seed trajectory, including nested AD.
            forward, parameters = jax.closure_convert(numerical_forward, e_old)
            state = forward(jax.lax.stop_gradient(e_old),
                            *jax.tree.map(jax.lax.stop_gradient, parameters))
            energy, _, norm, done = jax.tree.map(jax.lax.stop_gradient, state)
        norm, done = jax.tree.map(jax.lax.stop_gradient, (norm, done))
        require_valid(done, norm, "evGW")
        if not diis_space:
            energy = attach_root(residual, jax.lax.stop_gradient(energy), config=config, converged=done)
        result = driver(mo_energy_poles=energy, evaluate_only=True, **kwargs)
        return _converged_result(result, computed, tol, done)
    for _ in range(int(max_iter)):
        if occupations is not None:
            require_occupation_gap(e_old, occupations, "evGW")
        result = driver(mo_energy_poles=e_old, evaluate_only=True, **kwargs)
        residual = result.qp_residual
        delta = jnp.max(jnp.abs(residual))
        if float(delta) < tol:
            return _converged_result(result, computed, tol, True)
        e_old = e_old - (1.0 - damping) * residual
    raise ArithmeticError(
        f"evGW did not converge in {max_iter} iterations (last max|Dyson residual| = "
        f"{float(delta):.3e} > {tol}). Increase max_iter or damping; "
        "no fallback result is returned."
    )


def evgw_cd_restricted(
    *,
    mo_energy: Array,
    mo_coeff: Array,
    nocc: int,
    df_factors: Array,
    fock_matrix: Array,
    hcore_matrix: Array,
    density_matrix: Array,
    nw: int = 100,
    eta: float = 1e-3,
    orbs: Sequence[int] | None = None,
    diff_mode: str = "implicit",
    max_iter: int = 20,
    tol: float = 1e-6,
    damping: float = 0.0,
    diis_space: int = 0,
    initial_guess: Array | None = None,
    update_w: bool = True,
    differentiation: SCFDifferentiationConfig | None = None,
    g_orbitals: Sequence[int] | None = None,
    screening_occupied: Sequence[int] | None = None,
    screening_virtual: Sequence[int] | None = None,
) -> GWResult:
    """Spin-restricted evGW with contour deformation.

    Parameters are those of :func:`gradscf.gw.g0w0_cd_restricted` plus the
    iteration controls ``max_iter`` / ``tol`` / ``damping``.  Note the evGW
    loop evaluates the requested orbitals; restricting ``orbs`` freezes the
    remaining energies at mean-field values. ``update_w=False`` implements
    evGW0 with the original spectrum in W throughout the iterations. W is
    built once per evGW0 call and reused across the outer iterations.
    Correlation windows are passed through unchanged to the shared CD driver.
    ``differentiation=SCFDifferentiationConfig(mode="implicit")`` enables
    JIT/JVP/VJP of the outer Dyson root, including G/W response. Cached W0
    retains its derivative with respect to the original inputs. The default
    eager path is unchanged. Unconverged roots raise; unresolved implicit
    linear solves return NaNs under the shared solver policy.
    ``diis_space>0`` selects the shared compiled DIIS iteration and its implicit
    outer response; zero preserves the original iteration. The existing
    max_iter/tol/damping controls still apply, with an additional tol gate on
    the change in the sum of QP levels (a convergence monitor, not total energy).
    ``initial_guess`` supplies numerical pole energies for local continuation;
    its shape is the full mean-field spectrum. Unrequested levels stay at MF.
    Its tangent is ignored, while the original MF/G/W inputs retain response.
    Neither an initial guess nor DIIS selects a global largest-weight branch.
    The fixed occupied/virtual labels require a finite positive occupation
    gap at every G/W evaluation; an inverted or closed gap is rejected.
    Final computed levels also require a finite positive fixed-G/W weight.
    This local frequency-slope check does not replace the coupled outer
    Jacobian or enumerate all quasiparticle branches.
    """
    require_real(mo_energy, mo_coeff, df_factors, fock_matrix, hcore_matrix, density_matrix)
    if type(update_w) is not bool:
        raise TypeError("update_w must be boolean")
    fixed_w = None
    if not update_w:
        fixed_w = build_screened_w_restricted(
            mo_energy=mo_energy, mo_coeff=mo_coeff, nocc=int(nocc),
            df_factors=df_factors, nw=int(nw), eta=float(eta),
            screening_occupied=screening_occupied,
            screening_virtual=screening_virtual,
        )
    result = _evgw_loop(
        g0w0_cd_restricted,
        mo_energy,
        mo_energy=mo_energy,
        mo_coeff=mo_coeff,
        nocc=int(nocc),
        df_factors=df_factors,
        fock_matrix=fock_matrix,
        hcore_matrix=hcore_matrix,
        density_matrix=density_matrix,
        nw=int(nw),
        eta=float(eta),
        orbs=orbs,
        diff_mode=diff_mode,
        max_iter=int(max_iter),
        tol=float(tol),
        damping=float(damping),
        diis_space=diis_space,
        initial_guess=initial_guess,
        differentiation=differentiation,
        screening_energy=None if update_w else mo_energy,
        wmn_fixed=None if fixed_w is None else fixed_w["wmn"],
        wmn_static_fixed=None if fixed_w is None else fixed_w["wmn_static"],
        resolvent_expansion=True,
        resolvent_data_fixed=None if fixed_w is None else fixed_w["resolvent_data"],
        g_orbitals=g_orbitals,
        screening_occupied=screening_occupied,
        screening_virtual=screening_virtual,
    )

    return replace(result, method="evgw" if update_w else "evgw0")


def evgw_cd_unrestricted(
    *,
    mo_energy: tuple[Array, Array],
    mo_coeff: tuple[Array, Array],
    nocc: tuple[int, int],
    df_factors: Array,
    fock_matrix: tuple[Array, Array] | None = None,
    hcore_matrix: Array | None = None,
    density_matrix: tuple[Array, Array] | None = None,
    nw: int = 100,
    eta: float = 1e-3,
    orbs: Sequence[int] | None = None,
    diff_mode: str = "implicit",
    max_iter: int = 20,
    tol: float = 1e-6,
    damping: float = 0.0,
    diis_space: int = 0,
    initial_guess=None,
    differentiation: SCFDifferentiationConfig | None = None,
    update_w: bool = True,
) -> GWResult:
    """Spin-unrestricted evGW; ``update_w=False`` keeps the initial W spectra.

    ``differentiation`` enables coupled alpha/beta outer implicit response,
    including the initial-spectrum dependence of W0. The unrestricted W0
    arrays are currently reevaluated per step, with the same fixed spectra.
    ``diis_space>0`` uses the same shared DIIS/implicit contract as the
    restricted driver, packing both spin spectra into one coupled residual.
    ``initial_guess`` is an optional real (2,nmo) full numerical pole spectrum;
    the original MF base and unrequested levels are unchanged.
    """
    require_real(mo_energy, mo_coeff, df_factors, fock_matrix, hcore_matrix, density_matrix)
    if type(update_w) is not bool:
        raise TypeError('update_w must be boolean')
    e0 = jnp.stack([jnp.asarray(mo_energy[0]), jnp.asarray(mo_energy[1])])

    def driver(mo_energy_poles, **kw):
        return g0w0_cd_unrestricted(mo_energy_poles=(mo_energy_poles[0], mo_energy_poles[1]), **kw)

    result = _evgw_loop(
        driver,
        e0,
        mo_energy=mo_energy,
        mo_coeff=mo_coeff,
        nocc=(int(nocc[0]), int(nocc[1])),
        df_factors=df_factors,
        fock_matrix=fock_matrix,
        hcore_matrix=hcore_matrix,
        density_matrix=density_matrix,
        nw=int(nw),
        eta=float(eta),
        orbs=orbs,
        diff_mode=diff_mode,
        max_iter=int(max_iter),
        tol=float(tol),
        damping=float(damping),
        diis_space=diis_space,
        initial_guess=initial_guess,
        differentiation=differentiation,
        screening_energy=None if update_w else mo_energy,
    )

    return replace(result, method="evgw" if update_w else "evgw0")


__all__ = ["evgw_cd_restricted", "evgw_cd_unrestricted"]
