"""TDA projection without constructing a transition-space phonon tensor."""

import jax.numpy as jnp

from ..space import BSESpace
from ...gw.ep_coupling.types import _require, validate_model as validate_mo_model
from ...solvers.diagnostics import require_converged_derivative
from .types import PhononModel, validate_model


def _amplitudes(values, space):
    if not isinstance(space, BSESpace):
        raise NotImplementedError(
            "Phonon projection currently requires a restricted TDA space"
        )
    x = jnp.asarray(values)
    shape = (len(space.occupied), len(space.virtual))
    if x.ndim != 3 or x.shape[1:] != shape or not x.shape[0] or not space.size:
        raise ValueError(
            "amplitudes must have shape (nstate,nocc_selected,nvir_selected)"
        )
    _require(jnp.all(jnp.isfinite(x)), "Excitation amplitudes must be finite")
    _require(
        jnp.allclose(
            jnp.einsum("sia,tia->st", x.conj(), x),
            jnp.eye(x.shape[0]),
            rtol=1e-9,
            atol=1e-10,
        ),
        "TDA amplitudes must be orthonormal",
    )
    return x


def _kernel_projection(kernel, x, nmode):
    flat = x.reshape(x.shape[0], -1).T
    if callable(kernel):
        projected = []
        for mode in range(nmode):
            action = jnp.asarray(kernel(mode, flat))
            if action.shape != flat.shape:
                raise ValueError("Kernel action must return (ntransition,nstate)")
            projected.append(flat.conj().T @ action)
        return jnp.stack(projected)
    kernel = jnp.asarray(kernel)
    if kernel.shape != (nmode, flat.shape[0], flat.shape[0]):
        raise ValueError(
            "Kernel derivatives must have shape (nmode,ntransition,ntransition)"
        )
    _require(
        jnp.all(jnp.isfinite(kernel))
        & jnp.allclose(kernel, kernel.swapaxes(-1, -2).conj(), rtol=1e-10, atol=1e-12),
        "Kernel derivatives must be finite Hermitian matrices",
    )
    return jnp.einsum("is,lij,jt->lst", flat.conj(), kernel, flat)


def project_couplings(amplitudes, space, couplings, *, kernel=None):
    """Project explicit fixed-basis amplitudes: X†(g_e-g_h+dK)X.

    No sqrt(2) spin factor enters this operator matrix element. Optional
    kernel is a small dense oracle or callback (mode, columns)->action.
    The callback acts only on (ntransition,nstate) columns; no full
    transition-space derivative matrix is required. Array inputs are
    explicit differentiable quantities, with no eigensolver implied.
    """
    x, g = _amplitudes(amplitudes, space), jnp.asarray(couplings)
    if g.ndim != 3 or g.shape[1:] != (space.nmo, space.nmo) or not g.shape[0]:
        raise ValueError("MO couplings must have shape (nmode,nmo,nmo)")
    _require(
        jnp.all(jnp.isfinite(g))
        & jnp.allclose(g, g.swapaxes(-1, -2).conj(), rtol=1e-10, atol=1e-12),
        "MO couplings must be finite Hermitian matrices",
    )
    oi, va = jnp.asarray(space.occupied), jnp.asarray(space.virtual)
    electron = g[:, va[:, None], va[None, :]]
    hole = g[:, oi[:, None], oi[None, :]]
    action = jnp.einsum("lab,sib->lsia", electron, x) - jnp.einsum(
        "lji,sja->lsia", hole, x
    )
    result = jnp.einsum("sia,ltia->lst", x.conj(), action)
    if kernel is not None:
        result = result + _kernel_projection(kernel, x, g.shape[0])
    _require(
        jnp.all(jnp.isfinite(result))
        & jnp.allclose(result, result.swapaxes(-1, -2).conj(), rtol=1e-9, atol=1e-11),
        "Projected phonon vertices must be finite Hermitian matrices",
    )
    return result


def project_phonons(
    result, space, phonons, *, kernel_derivative=None, kernel_quadratic=None
):
    """Convert a converged restricted TDA BSE result and MO phonon vertices.

    Missing kernel derivatives explicitly freeze the electronic BSE kernel.
    kernel_quadratic supplies diagonal-mode second operator derivatives in
    the same dense/callback format as kernel_derivative. Other mode-pair
    contributions may be supplied through phonons.quadratic.

    Full BSE X/Y amplitudes require a different metric and are rejected.
    Forward projection permits arbitrary bases within degenerate spaces;
    AD through this result adapter retains BSE's amplitude-response guards.
    Use project_couplings for explicit fixed-basis arrays instead.
    """
    if isinstance(phonons, PhononModel):
        raise TypeError(
            "project_phonons expects MO phonon data, not already projected excitations"
        )
    validate_mo_model(phonons, norb=space.nmo, real=False)
    x = _amplitudes(result.x_amplitudes, space)
    y = jnp.asarray(result.y_amplitudes)
    if y.shape != x.shape:
        raise ValueError("TDA X/Y shapes must match")
    _require(jnp.all(y == 0), "Phonon projection requires TDA amplitudes (Y=0)")
    _require(
        jnp.all(result.converged & result.stable) & result.screening_valid,
        "Phonon projection requires converged stable BSE roots and valid screening",
    )
    g = project_couplings(x, space, phonons.couplings, kernel=kernel_derivative)
    quadratic = None
    nmode = phonons.energies.shape[0]
    if phonons.quadratic is not None:
        q = jnp.asarray(phonons.quadratic)
        quadratic = project_couplings(x, space, q.reshape(-1, space.nmo, space.nmo))
        quadratic = quadratic.reshape(q.shape[:-2] + g.shape[-2:])
    if kernel_quadratic is not None:
        correction = _kernel_projection(kernel_quadratic, x, nmode)
        if quadratic is None:
            quadratic = correction
        elif quadratic.ndim == 3:
            quadratic = quadratic + correction
        else:
            indices = jnp.arange(nmode)
            quadratic = quadratic.at[indices, indices].add(correction)
    valid = jnp.all(result.response_valid) & result.amplitude_response
    g = require_converged_derivative(g, valid)
    if quadratic is not None:
        quadratic = require_converged_derivative(quadratic, valid)
    label = "TDA projection of " + phonons.reference
    model = PhononModel(phonons.energies, g, quadratic, label)
    validate_model(model, nstates=x.shape[0])
    return model
