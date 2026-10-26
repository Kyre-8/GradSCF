"""Unit-spatial-norm BSE transition moments and length-gauge strengths."""

import jax
import jax.numpy as jnp
from ..solvers.diagnostics import require_converged_derivative
from .space import SpinBSESpace


def transition_dipoles(result, dipole_mo, space):
    d = jnp.asarray(dipole_mo)
    spin = isinstance(space, SpinBSESpace)
    shape = (2, 3, space.nmo, space.nmo) if spin else (3, space.nmo, space.nmo)
    if d.shape != shape:
        raise ValueError(f"dipole_mo must have shape {shape}")
    if jnp.iscomplexobj(d):
        raise NotImplementedError(
            "Molecular BSE transition moments currently require real inputs"
        )
    channels = space.channels if spin else (space,)
    dipoles = d if spin else (d,)
    xs = result.x_amplitudes if spin else (result.x_amplitudes,)
    ys = result.y_amplitudes if spin else (result.y_amplitudes,)
    mu = jnp.zeros((result.excitation_energies.size, 3), dtype=d.dtype)
    for channel, dipole, x, y in zip(channels, dipoles, xs, ys):
        block = dipole[:, jnp.asarray(channel.occupied, dtype=int)[:, None],
                       jnp.asarray(channel.virtual, dtype=int)[None, :]]
        mu += jnp.einsum("xia,sia->sx", block, x + y)
    mu *= 1. if spin else jnp.sqrt(2.)
    if result.singlet is False:
        mu = jnp.zeros_like(mu)
    valid = result.converged & result.stable
    response = result.response_valid & result.amplitude_response
    if not result.amplitude_response:
        # Keep a live state dependency even though stopped X has no tangent.
        # Otherwise a grad through omega alone could masquerade as full response.
        sentinel = require_converged_derivative(result.excitation_energies, False)
        mu = mu + (sentinel - jax.lax.stop_gradient(sentinel))[:, None]
    mu = require_converged_derivative(mu, response[:, None])
    return jnp.where(valid[:, None], mu, jnp.nan)


def oscillator_strengths(result, dipole_mo, space):
    mu = transition_dipoles(result, dipole_mo, space)
    value = (2 / 3) * result.excitation_energies * jnp.sum(mu**2, axis=1)
    return require_converged_derivative(
        value, result.response_valid & result.amplitude_response
    )


def polarizability(result, dipole_mo, space, omega=0., *, eta=0.):
    """Retarded dipole polarizability tensor in a0^3; omega/eta in Hartree.

    Sum only over the computed roots; a truncated spectrum is not a complete
    polarizability. Both resonant and antiresonant poles are included even
    for TDA eigenstates. At omega=eta=0 this gives 2 sum_s mu_s mu_s.T/Omega_s.
    Real scalar/1D frequencies return (3,3)/(nfrequency,3,3), respectively.
    """
    omega, eta = jnp.asarray(omega), jnp.asarray(eta)
    if omega.ndim > 1 or eta.ndim != 0 or jnp.iscomplexobj(omega) or jnp.iscomplexobj(eta):
        raise ValueError("omega must be real scalar/1D and eta a real scalar")
    mu = transition_dipoles(result, dipole_mo, space)
    energies = result.excitation_energies
    z = omega[..., None] + 1j * eta
    poles = 1 / (energies - z) + 1 / (energies + z)
    tensor = jnp.einsum('...s,si,sj->...ij', poles, mu, mu)
    valid = (jnp.isfinite(omega) & jnp.isfinite(eta) & (eta >= 0))[..., None, None]
    return jnp.where(valid, require_converged_derivative(tensor, valid), jnp.nan)


def absorption_cross_section(result, dipole_mo, space, omega, *, eta=.01,
                             polarization=None, unit='au'):
    """Absorption area at nonnegative omega (Ha), eta>0 (Ha).

    Default: orientational average Tr(sigma)/3. A real nonzero length-three
    polarization vector selects its normalized direction. Output unit is a0^2
    ('au') or megabarn ('Mb', 1e-18 cm^2). This is a finite-root spectrum.
    """
    alpha = polarizability(result, dipole_mo, space, omega, eta=eta)
    return _absorption_from_polarizability(alpha,omega,eta=eta,polarization=polarization,unit=unit)


def _absorption_from_polarizability(alpha, omega, *, eta, polarization, unit):
    """Common static/dynamic length-gauge cross-section units and polarization."""
    if unit not in {'au', 'Mb'}:
        raise ValueError("Cross-section unit must be 'au' or 'Mb'")
    omega = jnp.asarray(omega)
    valid = (omega >= 0) & jnp.isfinite(omega) & (jnp.asarray(eta) > 0)
    if polarization is None:
        response = jnp.trace(alpha, axis1=-2, axis2=-1) / 3
    else:
        direction = jnp.asarray(polarization)
        if direction.shape != (3,) or jnp.iscomplexobj(direction):
            raise ValueError("polarization must be a real vector with shape (3,)")
        norm = jnp.linalg.norm(direction)
        valid &= jnp.isfinite(norm) & (norm > 0)
        direction = direction / jnp.where(norm > 0, norm, 1.)
        response = jnp.einsum('i,...ij,j->...', direction, alpha, direction)
    value = 4 * jnp.pi * omega * jnp.imag(response) / 137.035999084
    if unit == 'Mb':
        value = value * (0.529177210903**2 * 100)
    return jnp.where(valid, require_converged_derivative(value, valid), jnp.nan)
