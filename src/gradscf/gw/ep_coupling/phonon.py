"""Harmonic propagators for validated positive mode energies (Ha)."""

import jax.numpy as jnp

from .types import _require


def _validate_beta(beta):
    _require(jnp.isfinite(beta) & (jnp.asarray(beta) > 0),
             "beta must be finite and positive")


def bose_occupation(energies, beta):
    """Stable n_B for validated positive energies; beta is finite positive."""
    _validate_beta(beta)
    x = beta * jnp.asarray(energies)
    return jnp.exp(-x) / (-jnp.expm1(-x))


def phonon_propagator_iw(energies, frequencies):
    """D0(i nu) = -2 Omega/(nu^2 + Omega^2), shape (nnu, nmode)."""
    omega = jnp.asarray(energies)
    return -2 * omega / (jnp.asarray(frequencies)[:, None]**2 + omega**2)


def phonon_propagator_tau(energies, tau, beta):
    """D0(tau), shape (ntau, nmode), for 0 <= tau <= beta."""
    _validate_beta(beta)
    omega, tau = jnp.asarray(energies), jnp.asarray(tau)[:, None]
    return -(jnp.exp(-tau * omega) + jnp.exp(-(beta - tau) * omega)) / (-jnp.expm1(-beta * omega))
