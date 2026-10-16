"""Quasiparticle equation solver with implicit differentiation.

The quasiparticle energy e_p^QP is a nonlinear (Dyson) solution of

    f(w) = w - e_p^mf - [ Re Sigma_p(w) + v^x_pp - v^mf_pp ] = 0

solved by a shared independent-scalar iteration (default secant, as used by
PySCF ``gw_cd``).  All requested orbitals are solved **simultaneously** in
one vectorized ``lax.while_loop`` (one compiled XLA program instead of
hundreds of eager dispatches -- this matters for CPU-bound hosts).

Two differentiation modes are provided:

- ``implicit`` (default): the roots are differentiated through the
  implicit function theorem,

      d e_p^QP / d theta = -(df_p/dtheta) / (df_p/dw)

  where df_p/dw = 1 - d(Re Sigma_p)/dw is the inverse quasiparticle
  renormalization factor 1/Z. A custom JVP differentiates the residual at the
  converged roots; its transpose supplies the VJP. The forward trajectory is
  not differentiated.

- ``explicit``: differentiates through the chosen forward iterations via a
  static-length ``lax.scan`` (useful for cross-checking the implicit rule
  on small systems). ``unrolled`` remains a legacy alias.

The pole-selection mask in the self-energy is piecewise constant, so in
pole-dense regions the secant iteration can oscillate without meeting the
step tolerance; the solver tracks the best (min |f|) iterate and reports
per-orbital convergence flags explicitly -- never a silent fallback.

The implicit backward rule rejects unconverged roots and singular/nonfinite
df/dw (Z diverging near satellite structures), including under JIT. JAX
wraps runtime callback errors in its backend exception when compiled.

References
----------
- L. Hedin, Phys. Rev. 139, A796 (1965). DOI:10.1103/PhysRev.139.A796
- M. S. Hybertsen and S. G. Louie, "Electron correlation in semiconductors
  and insulators: Band gaps and quasiparticle energies", Phys. Rev. B 34,
  5390 (1986). DOI:10.1103/PhysRevB.34.5390 (Dyson equation for QP states)
- Implicit differentiation convention follows the project-wide
  ``SCFDifferentiationConfig`` pattern in ``gradscf.scf.autodiff``.
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from .self_energy import sigma_cd
from ..solvers.nonlinear import ScalarRootConfig, solve_scalar_roots

_DFDW_MIN = 1e-10
_TOL = 1e-6
_MAXITER = 100
_STEP_CAP = 0.05
# Memory threshold (bytes) above which the orbital batch is evaluated with
# a sequential lax.map instead of vmap (see gradscf.gw.self_energy).
_BATCH_MEM_LIMIT = 1 << 31


def qp_residual(omega: Array, e_mf: Array, delta_v: Array, ctx: dict) -> Array:
    """Dyson residual f(w) = w - e_mf - Re Sigma_p(w) - delta_v (real)."""
    sigma = sigma_cd(omega, ctx)
    return omega - e_mf - (jnp.real(sigma) + delta_v)


def _ctx_from_parts(shared: dict, wmn_p: Array, b_pm: Array, b_mp: Array, wmn_static=None) -> dict:
    ctx = {
        "mo_energy": shared["mo_energy"],
        "wmn_p": wmn_p,
        "b_pm": b_pm,
        "b_mp": b_mp,
        "channels": shared["channels"],
        "ef": shared["ef"],
        "eta": shared["eta"],
        "freqs": shared["freqs"],
        "wts": shared["wts"],
        "conjugate": shared.get("conjugate", False),
    }
    if wmn_static is not None:
        ctx["wmn_static"] = wmn_static
        if "occupation_sign" in shared:
            ctx["occupation_sign"] = shared["occupation_sign"]
    if "q0" in shared:
        ctx["q0"] = shared["q0"]
    if "g_indices" in shared:
        ctx["g_indices"] = shared["g_indices"]
    if "resolvent_data" in shared:
        ctx["resolvent_data"] = shared["resolvent_data"]
    return ctx


def sigma_cd_batch(omega: Array, shared: dict, stacked: dict) -> Array:
    """Vectorized self-energy Sigma_p(omega) for a batch of orbitals.

    Returns complex ``(norb,)``.  Same stacking conventions as
    :func:`qp_residual_batch`.
    """

    def fn(w, wp, bp, bm, ws, pidx, *extra):
        ctx = _ctx_from_parts(shared, wp, bp, bm, ws if "wmn_static" in stacked else None)
        ctx["p_index"] = pidx
        if extra:
            ctx["del_w"], ctx["p_index"] = extra
            if "q0" in shared:
                ctx["q0"] = {**shared["q0"], "p_index": extra[1]}
        return sigma_cd(w, ctx)

    args = (omega, stacked["wmn_p"], stacked["b_pm"], stacked["b_mp"],
            stacked.get("wmn_static", jnp.zeros_like(stacked["wmn_p"][:, 0, :])),
            stacked.get("p_index", jnp.arange(omega.shape[0])))
    if "del_w" in stacked:
        args = args + (stacked["del_w"], stacked["p_index"])
    naux = stacked["b_pm"].shape[1]
    est_bytes = omega.shape[0] * naux * naux * 16 * 2
    if est_bytes > _BATCH_MEM_LIMIT:
        return jax.lax.map(lambda a: fn(*a), args)
    return jax.vmap(fn)(*args)


def qp_residual_batch(
    omega: Array, e_mf: Array, delta_v: Array, shared: dict, stacked: dict
) -> Array:
    """Vectorized Dyson residual for a batch of orbitals.

    ``omega``, ``e_mf``, ``delta_v`` have shape ``(norb,)``; ``stacked``
    holds ``wmn_p`` ``(norb, nw, nmo)``, ``b_pm``/``b_mp``
    ``(norb, naux, nmo)``; ``shared`` holds the orbital-independent
    quantities (see :func:`gradscf.gw.self_energy.sigma_cd`).

    Falls back to a sequential ``lax.map`` when the batched residue
    intermediates would exceed ``_BATCH_MEM_LIMIT``.

    When ``stacked`` carries ``del_w`` ``(norb, nw)`` and ``p_index``
    ``(norb,)`` (periodic q -> 0 correction), they are mapped over the
    orbital batch as well.
    """
    has_q0 = "del_w" in stacked

    def fn(w, e, dv, wp, bp, bm, ws, pidx, *extra):
        ctx = _ctx_from_parts(shared, wp, bp, bm, ws if "wmn_static" in stacked else None)
        ctx["p_index"] = pidx
        if extra:
            ctx["del_w"], ctx["p_index"] = extra
            if "q0" in shared:
                ctx["q0"] = {**shared["q0"], "p_index": extra[1]}
        return qp_residual(w, e, dv, ctx)

    args = (omega, e_mf, delta_v, stacked["wmn_p"], stacked["b_pm"], stacked["b_mp"],
            stacked.get("wmn_static", jnp.zeros_like(stacked["wmn_p"][:, 0, :])),
            stacked.get("p_index", jnp.arange(omega.shape[0])))
    if has_q0:
        args = args + (stacked["del_w"], stacked["p_index"])
    naux = stacked["b_pm"].shape[1]
    est_bytes = omega.shape[0] * naux * naux * 16 * 2
    if est_bytes > _BATCH_MEM_LIMIT:
        return jax.lax.map(lambda a: fn(*a), args)
    return jax.vmap(fn)(*args)


def _df_dw_batch(root, e_mf, delta_v, shared, stacked):
    """Per-orbital df/dw at the roots (inverse Z factors), memory-bounded."""

    def fn(w, e, dv, wp, bp, bm, ws, pidx, *extra):
        ctx = _ctx_from_parts(shared, wp, bp, bm, ws if "wmn_static" in stacked else None)
        ctx["p_index"] = pidx
        if extra:
            ctx["del_w"], ctx["p_index"] = extra
            if "q0" in shared:
                ctx["q0"] = {**shared["q0"], "p_index": extra[1]}
        return jax.grad(lambda ww: qp_residual(ww, e, dv, ctx))(w)

    args = (root, e_mf, delta_v, stacked["wmn_p"], stacked["b_pm"], stacked["b_mp"],
            stacked.get("wmn_static", jnp.zeros_like(stacked["wmn_p"][:, 0, :])),
            stacked.get("p_index", jnp.arange(root.shape[0])))
    if "del_w" in stacked:
        args = args + (stacked["del_w"], stacked["p_index"])
    naux = stacked["b_pm"].shape[1]
    est_bytes = root.shape[0] * naux * naux * 16 * 2
    if est_bytes > _BATCH_MEM_LIMIT:
        return jax.lax.map(lambda a: fn(*a), args)
    return jax.vmap(fn)(*args)


def _qp_forward(f, x0, *, tol, maxiter, method, scan=False):
    cfg = ScalarRootConfig(method=method, ftol=tol, xtol=tol, maxiter=maxiter,
                           step_cap=_STEP_CAP, slope_floor=_DFDW_MIN)
    out = solve_scalar_roots(f, x0, config=cfg, scan=scan)
    return out.roots, out.converged


@partial(jax.custom_jvp, nondiff_argnums=(5, 6, 7, 8))
def _qp_solve_implicit(
    x0: Array, e_mf: Array, delta_v: Array, shared: dict, stacked: dict,
    conjugate: bool, tol: float, maxiter: int, method: str,
) -> tuple[Array, Array]:
    context = {**shared, "conjugate": conjugate}
    f = lambda w: qp_residual_batch(w, e_mf, delta_v, context, stacked)
    root, done = _qp_forward(f, x0, tol=tol, maxiter=maxiter, method=method)
    # Encode auxiliary status as a real output inside the custom rule. Boolean
    # custom outputs can acquire active float0 tangents on some JAX versions.
    return root, done.astype(root.dtype)


def _raise_invalid_implicit_gradient(done, df_dw):
    if not np.all(np.asarray(done)):
        raise ArithmeticError("QP implicit differentiation requires converged roots; QP not converged.")
    slopes = np.asarray(df_dw)
    if np.any(~np.isfinite(slopes) | (np.abs(slopes) < _DFDW_MIN)):
        raise ArithmeticError(
            "QP implicit differentiation is singular or nonfinite: "
            f"requires finite |df/dw| >= {_DFDW_MIN} (finite Z factors)."
        )


@_qp_solve_implicit.defjvp
def _qp_solve_implicit_jvp(conjugate, tol, maxiter, method, primals, tangents):
    x0, e_mf, delta_v, shared, stacked = primals
    _, de_mf, ddelta_v, dshared, dstacked = tangents
    root, status = _qp_solve_implicit(*primals, conjugate, tol, maxiter, method)
    done = status > .5
    context = {**shared, "conjugate": conjugate}
    # df_p/dw at the converged roots (inverse Z factors), one per orbital.
    df_dw = _df_dw_batch(root, e_mf, delta_v, context, stacked)
    if isinstance(df_dw, jax.core.Tracer) or isinstance(done, jax.core.Tracer):
        valid = jnp.all(done & jnp.isfinite(df_dw) & (jnp.abs(df_dw) >= _DFDW_MIN))
        jax.lax.cond(
            valid,
            lambda: None,
            lambda: jax.debug.callback(_raise_invalid_implicit_gradient, done, df_dw),
        )
    else:
        _raise_invalid_implicit_gradient(done, df_dw)
    # Hold the root fixed in this partial differential. Its live dependence
    # remains available when the rule is itself differentiated. The seed is
    # only a numerical choice; its tangent does not enter the physical root.
    _, partial_response = jax.jvp(
        lambda em, dv, sh, st: qp_residual_batch(
            root, em, dv, {**sh, "conjugate": conjugate}, st
        ),
        (e_mf, delta_v, shared, stacked),
        (de_mf, ddelta_v, dshared, dstacked),
    )
    return (root, status), (-partial_response/df_dw, jnp.zeros_like(status))


def solve_qp_batch(
    e_mf: Array,
    delta_v: Array,
    shared: dict,
    stacked: dict,
    *,
    occupied: Array,
    tol: float = _TOL,
    maxiter: int = _MAXITER,
    diff_mode: str = "implicit",
    method: str = "secant",
    initial_guess: Array | None = None,
) -> tuple[Array, Array]:
    """Solve the quasiparticle equation for a batch of orbitals.

    Parameters
    ----------
    e_mf, delta_v:
        ``(norb,)`` mean-field energies and ``v^x - v^mf`` corrections.
    shared, stacked:
        Self-energy contexts, see :func:`qp_residual_batch`.
    occupied:
        ``(norb,)`` boolean; selects the initial-guess offset sign
        (-1e-2 occupied, +1e-2 virtual, following PySCF ``gw_cd``).
    initial_guess:
        Optional real ``(norb,)`` QP seeds, overriding the mean-field offset.
        Useful for following a previously identified isolated root. Implicit
        response ignores the seed tangent; root selection itself is discrete.
    method:
        Static forward algorithm: "secant" (default), "newton" (AD JVP slopes),
        or "hybrid" (periodic/stalled AD refresh). All use the same implicit
        backward and step/residual convergence checks. Safeguards do not
        establish pole-free brackets or guarantee a particular satellite root.
    tol, maxiter:
        Absolute step and residual tolerance (Ha), and iteration cap.
        These and ``shared['conjugate']`` are static configuration under JIT.
    diff_mode:
        ``"implicit"`` (JVP/VJP via the implicit function theorem) or
        ``"explicit"`` (AD through a static scan of the chosen method).
        The historical ``"unrolled"`` spelling remains accepted.

    Returns
    -------
    (qp_energies, converged_mask), both shape ``(norb,)``. The actual mask
    is preserved under JAX transformations. Unconverged forward calls
    return the best-residual iterate; implicit backward calls reject it.
    """
    if diff_mode not in ("implicit", "explicit", "unrolled"):
        raise ValueError(f"diff_mode must be 'implicit' or 'explicit', got {diff_mode!r}")
    e_mf = jnp.asarray(e_mf, dtype=jnp.float64)
    delta_v = jnp.asarray(delta_v, dtype=jnp.float64)
    occupied = jnp.asarray(occupied, dtype=bool)
    if initial_guess is None:
        x0 = e_mf + jnp.where(occupied, -1e-2, 1e-2)
    else:
        x0 = jnp.asarray(initial_guess)
        if x0.shape != e_mf.shape or jnp.iscomplexobj(x0):
            raise ValueError("initial_guess must be a real vector matching the requested QP states")
        x0 = x0.astype(e_mf.dtype)
    f = lambda w: qp_residual_batch(w, e_mf, delta_v, shared, stacked)

    if diff_mode == "implicit":
        # Keep configuration out of the differentiable pytree. Always use
        # the same custom rule, including when only context leaves vary.
        dynamic_shared = {key: value for key, value in shared.items() if key != "conjugate"}
        root, status = _qp_solve_implicit(
            x0, e_mf, delta_v, dynamic_shared, stacked,
            shared.get("conjugate", False), float(tol), int(maxiter), method,
        )
        done = status > 0.5
    else:
        root, done = _qp_forward(f, x0, tol=float(tol), maxiter=int(maxiter), method=method, scan=True)

    # Status is an observable, not a differentiable output of the custom
    # root rule. In particular, boolean reductions must not enter its AD chain.
    done = jax.lax.stop_gradient(done)
    return root, done


def solve_qp_orbital(
    e_mf: float,
    delta_v: float,
    ctx: dict,
    *,
    occupied: bool,
    tol: float = 1e-6,
    maxiter: int = 100,
    diff_mode: str = "implicit",
    method: str = "secant",
) -> tuple[float, bool]:
    """Single-orbital convenience wrapper around :func:`solve_qp_batch`."""
    shared = {
        "mo_energy": ctx["mo_energy"],
        "channels": ctx["channels"],
        "ef": ctx["ef"],
        "eta": ctx["eta"],
        "freqs": ctx["freqs"],
        "wts": ctx["wts"],
        "conjugate": ctx.get("conjugate", False),
    }
    if "occupation_sign" in ctx:
        shared["occupation_sign"] = ctx["occupation_sign"]
    if "q0" in ctx:
        shared["q0"] = ctx["q0"]
    stacked = {
        "wmn_p": ctx["wmn_p"][None],
        "b_pm": ctx["b_pm"][None],
        "b_mp": ctx["b_mp"][None],
    }
    if "wmn_static" in ctx:
        stacked["wmn_static"] = ctx["wmn_static"][None]
    if "del_w" in ctx:
        stacked["del_w"] = ctx["del_w"][None]
        stacked["p_index"] = jnp.asarray([ctx["p_index"]])
    root, done = solve_qp_batch(
        jnp.asarray([e_mf]),
        jnp.asarray([delta_v]),
        shared,
        stacked,
        occupied=jnp.asarray([occupied]),
        tol=tol,
        maxiter=maxiter,
        diff_mode=diff_mode,
        method=method,
    )
    if isinstance(root, jax.core.Tracer):
        return root[0], done[0]
    return float(root[0]), bool(done[0])


__all__ = ["qp_residual", "qp_residual_batch", "sigma_cd_batch", "solve_qp_batch", "solve_qp_orbital"]
