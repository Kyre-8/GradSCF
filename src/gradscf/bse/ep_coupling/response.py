"""Low-population exciton Fan/DW and causal dipole response, energies in Ha."""

import jax
import jax.numpy as jnp

from ...gw.ep_coupling.phonon import bose_occupation
from ...gw.ep_coupling.self_energy import debye_waller as _debye_waller
from ...gw.ep_coupling.self_energy import _quadratic_average
from ...gw.ep_coupling.types import _require
from ...solvers import LinearSolverConfig, solve_complex
from ..properties import _absorption_from_polarizability
from .types import validate_model


def _config(config):
    if config is not None and not isinstance(config, LinearSolverConfig):
        raise TypeError("solver_config must be a LinearSolverConfig")
    return LinearSolverConfig(method="direct", rtol=1e-10) if config is None else config


def _inputs(hamiltonian, model, frequencies, beta, eta):
    h, w, beta, eta = map(jnp.asarray, (hamiltonian, frequencies, beta, eta))
    if (
        h.ndim not in (1, 2)
        or not h.shape[0]
        or (h.ndim == 2 and h.shape[0] != h.shape[1])
    ):
        raise ValueError(
            "hamiltonian must be a nonempty energy vector or square Hermitian matrix"
        )
    if h.ndim == 1 and jnp.iscomplexobj(h):
        raise ValueError("Excitation energies must be real")
    matrix = jnp.diag(h) if h.ndim == 1 else h
    _require(
        jnp.all(jnp.isfinite(matrix))
        & jnp.allclose(matrix, matrix.conj().T, rtol=1e-10, atol=1e-12),
        "Excitation Hamiltonian must be finite Hermitian",
    )
    # Cholesky is unique for a positive matrix, including internal degeneracy.
    _require(
        (
            jnp.all(h > 0)
            if h.ndim == 1
            else jnp.all(jnp.isfinite(jnp.linalg.cholesky(matrix)))
        ),
        "Bare excitation Hamiltonian must be strictly positive",
    )
    if w.ndim != 1 or not w.shape[0] or jnp.iscomplexobj(w):
        raise ValueError("frequencies must be a nonempty real vector")
    _require(jnp.all(jnp.isfinite(w)), "frequencies must be finite")
    if (
        beta.ndim != 0
        or eta.ndim != 0
        or jnp.iscomplexobj(beta)
        or jnp.iscomplexobj(eta)
    ):
        raise ValueError("beta and eta must be real scalars")
    _require(jnp.isfinite(eta) & (eta > 0), "eta must be finite and strictly positive")
    validate_model(model, nstates=h.shape[0])
    fields = [
        h,
        w,
        beta,
        eta,
        jnp.asarray(model.energies),
        jnp.asarray(model.couplings),
    ]
    if model.quadratic is not None:
        fields.append(jnp.asarray(model.quadratic))
    dtype = jnp.result_type(*fields)
    real_dtype = jnp.empty((), dtype=dtype).real.dtype
    h = h.astype(real_dtype if h.ndim == 1 else dtype)
    matrix = jnp.diag(h) if h.ndim == 1 else h
    w, beta, eta = [value.astype(real_dtype) for value in (w, beta, eta)]
    thermal = bose_occupation(jnp.asarray(model.energies, dtype=real_dtype), beta)
    return h, matrix, w + 1j * eta, thermal


def _fan(h, model, z, thermal, config):
    omega, g = jnp.asarray(model.energies), jnp.asarray(model.couplings)
    if h.ndim == 1:
        delta = z[:, None, None] - h[None, :, None]
        weight = (thermal + 1) / (delta - omega) + thermal / (delta + omega)
        return jnp.einsum("lia,wal,lja->wij", g, weight, g.conj())
    identity = jnp.eye(h.shape[0], dtype=jnp.result_type(h, z, g, omega, thermal))
    g = g.astype(identity.dtype)

    def frequency(value):
        def mode(index, state):
            sigma, valid = state
            plus = solve_complex(
                (value - omega[index]) * identity - h, g[index].conj().T, config=config
            )

            def absorption():
                result = solve_complex(
                    (value + omega[index]) * identity - h,
                    g[index].conj().T,
                    config=config,
                )
                return result.solution, result.converged

            minus, absorption_valid = jax.lax.cond(
                thermal[index] > 0,
                absorption,
                lambda: (jnp.zeros_like(identity), jnp.asarray(True)),
            )
            sigma += g[index] @ (
                (thermal[index] + 1) * plus.solution + thermal[index] * minus
            )
            return sigma, valid & plus.converged & absorption_valid

        return jax.lax.fori_loop(
            0, omega.size, mode, (jnp.zeros_like(identity), jnp.asarray(True))
        )

    sigma, valid = jax.vmap(frequency)(z)
    _require(jnp.all(valid), "Exciton Fan resolvent solve did not converge")
    return sigma


def fan_retarded(
    hamiltonian, model, frequencies, *, beta, eta=0.01, solver_config=None
):
    """One-phonon exciton self-energy with a fixed bath and no exciton population.

    Vector Hamiltonians use canonical pole sums. Matrix Hamiltonians use
    resolvents with the shared linear solver, without eigenvector AD. The
    Bose weights are n+1 for emission and n for absorption, with no Fermi
    factor or chemical potential. Internal propagators use the bare H_X.
    Frequencies/eta are in Ha relative to the neutral excitation ground state.
    """
    h, _, z, thermal = _inputs(hamiltonian, model, frequencies, beta, eta)
    return _fan(h, model, z, thermal, _config(solver_config))


def debye_waller(model, beta):
    """Shared static Lambda contraction, relative to the ground surface."""
    validate_model(model)
    if jnp.asarray(beta).ndim != 0 or jnp.iscomplexobj(beta):
        raise ValueError("beta must be a real scalar")
    return _debye_waller(model, beta)


def _green(matrix, sigma, z, shift, rhs, config):
    identity = jnp.eye(
        matrix.shape[0], dtype=jnp.result_type(matrix, z, sigma, shift, rhs)
    )

    def solve(value, self_energy):
        result = solve_complex(
            value * identity - (matrix + shift) - self_energy, rhs, config=config
        )
        return result.solution, result.converged

    value, valid = jax.vmap(solve)(z, sigma)
    _require(jnp.all(valid), "Exciton optical resolvent solve did not converge")
    return value


def _shift(model, thermal, include_dw):
    if type(include_dw) is not bool:
        raise TypeError("include_dw must be a static boolean")
    return (
        _quadratic_average(model, thermal)
        if include_dw
        else jnp.zeros_like(model.couplings[0])
    )


def spectral_function(
    hamiltonian,
    model,
    frequencies,
    *,
    beta,
    eta=0.01,
    include_dw=True,
    solver_config=None,
):
    """A_X=-(R-R†)/(2 pi i), a matrix per Ha; no linewidth sign repair."""
    h, matrix, z, thermal = _inputs(hamiltonian, model, frequencies, beta, eta)
    cfg = _config(solver_config)
    green = _green(
        matrix,
        _fan(h, model, z, thermal, cfg),
        z,
        _shift(model, thermal, include_dw),
        jnp.eye(matrix.shape[0], dtype=jnp.result_type(matrix, z)),
        cfg,
    )
    return -(green - green.swapaxes(-1, -2).conj()) / (2j * jnp.pi)


def polarizability(
    hamiltonian,
    model,
    dipoles,
    omega=0.0,
    *,
    beta,
    eta=0.01,
    include_dw=True,
    solver_config=None,
):
    """Causal tensor in a0³, including resonant and antiresonant responses.

    dipoles (nstate,3) are <state|r|ground>, including the BSE spin factor.
    Return (3,3) for scalar omega or (nfreq,3,3). Both quadrants use the
    same analytic self-energy. Do not feed EP-corrected bare energies and
    add their full EP correction again: no double-counting subtraction is
    inferred from the model reference label.
    """
    axis = jnp.asarray(omega)
    if axis.ndim > 1 or jnp.iscomplexobj(axis):
        raise ValueError("omega must be real scalar or vector")
    h, matrix, z, thermal = _inputs(hamiltonian, model, jnp.atleast_1d(axis), beta, eta)
    d = jnp.asarray(dipoles)
    if d.shape != (matrix.shape[0], 3):
        raise ValueError("dipoles must have shape (nstate,3)")
    _require(jnp.all(jnp.isfinite(d)), "dipoles must be finite")
    cfg = _config(solver_config)
    z = jnp.concatenate((z, -z.conj()))
    green_d = _green(
        matrix,
        _fan(h, model, z, thermal, cfg),
        z,
        _shift(model, thermal, include_dw),
        d,
        cfg,
    )
    moment = jnp.einsum("si,wsj->wij", d.conj(), green_d)
    positive, negative = jnp.split(moment, 2, axis=0)
    tensor = -positive - negative.conj()
    return tensor[0] if axis.ndim == 0 else tensor


def absorption_cross_section(
    hamiltonian,
    model,
    dipoles,
    omega,
    *,
    beta,
    eta=0.01,
    include_dw=True,
    solver_config=None,
    polarization=None,
    unit="au",
):
    """Passive finite-state absorption in a0² or Mb, with BSE unit conventions.

    The low-population approximation can resolve negative-frequency thermal
    satellites when phonons exceed excitation energies. Negative absorption
    beyond roundoff raises: the raw polarizability remains available, and
    no sign clipping or thermal detailed-balance correction is applied.
    """
    alpha = polarizability(
        hamiltonian,
        model,
        dipoles,
        omega,
        beta=beta,
        eta=eta,
        include_dw=include_dw,
        solver_config=solver_config,
    )
    value = _absorption_from_polarizability(
        alpha, omega, eta=eta, polarization=polarization, unit=unit
    )
    axis = jnp.asarray(omega)
    physical = jnp.isfinite(axis) & (axis >= 0)
    roundoff = 64 * jnp.finfo(value.dtype).eps * jnp.maximum(1.0, jnp.abs(value))
    _require(
        jnp.all(jnp.where(physical & jnp.isfinite(value), value >= -roundoff, True)),
        "Non-passive absorption in the low-population exciton model; inspect raw polarizability and thermal satellites",
    )
    return value
