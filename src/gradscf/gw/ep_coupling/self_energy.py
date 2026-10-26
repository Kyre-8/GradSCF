"""Fixed-D0 molecular Fan/DW and the low-level complex periodic Fan action."""

import jax
import jax.numpy as jnp

from ..matsubara import _reference
from .types import _require, _validate_q_data, validate_periodic_model
from .phonon import bose_occupation, phonon_propagator_tau


def debye_waller(model, beta):
    """Static DW = (1/2) sum_l Lambda_ll coth(beta Omega_l/2)."""
    return _quadratic_average(model,bose_occupation(model.energies,beta))


def _quadratic_average(model, occupation):
    """Common harmonic average in the caller's occupation compute precision."""
    if model.quadratic is None:
        return jnp.zeros_like(jnp.asarray(model.couplings)[0])
    thermal = 1 + 2 * occupation
    quadratic = jnp.asarray(model.quadratic)
    diagonal = quadratic if quadratic.ndim == 3 else jnp.einsum('llij->lij',quadratic)
    return 0.5 * jnp.einsum("lij,l->ij", diagonal, thermal)


def _fan_reference(fock, mu, model, grid):
    energy, coeff, _, _, _, green_tau = _reference(fock, mu, grid)
    coupling = jnp.einsum("pi,lpq,qj->lij", coeff, jnp.asarray(model.couplings), coeff)
    from .spectral import _fan_weights
    weight = _fan_weights(1j * grid.fermion, energy, model.energies, mu, grid.beta)
    sigma = jnp.einsum("lik,wkl,lkj->wij", coupling, weight, coupling)
    return jnp.einsum("pi,wij,qj->wpq", coeff, sigma, coeff), green_tau


def fan_self_energy(green_tau, fock, mu, model, grid):
    """Molecular Fan self-energy from full dressed G(tau) and fixed D0.

    Requires a validated real model and real symmetric fock, all expressed
    in the same fixed orthonormal frame. Subtracts the static fock Green's
    function in time and adds its analytic finite-temperature Fan poles.
    Thus soft modes do not acquire a finite Matsubara-cutoff error when
    G=G_ref. The remaining dressed correction uses midpoint quadrature and
    still requires grid convergence. G(tau) must have the canonical unit
    spectral moment; the caller should include its electronic plus phonon
    self-energy moment when reconstructing G(tau) from Matsubara samples.
    """
    if jnp.iscomplexobj(fock) or jnp.iscomplexobj(model.couplings):
        raise ValueError("molecular Fan requires a real orbital reference and real couplings")
    reference_sigma, reference_green = _fan_reference(jnp.asarray(fock), mu, model, grid)
    d_tau = phonon_propagator_tau(model.energies, grid.tau, grid.beta)
    g = jnp.asarray(model.couplings)
    difference_tau = -jnp.einsum("lik,tkm,lmj,tl->tij", g, green_tau - reference_green, g, d_tau)
    phase = jnp.exp(1j * grid.fermion[:, None] * grid.tau)
    sigma = reference_sigma + grid.beta / (2 * grid.nw) * jnp.einsum("wt,tij->wij", phase, difference_tau)
    moment = jnp.einsum("lik,lkj,l->ij", g, g, 1 + 2 * bose_occupation(model.energies, grid.beta))
    return dict(sigma_iw=sigma, sigma_moment=moment)


def periodic_fan_self_energy_tau(green_tau, couplings, phonon_tau, k_plus_q, q_weights):
    """Low-level complex k/q Fan contraction, without a periodic GW driver.

    green_tau: (ntau, nk, norb, norb).
    couplings: (nq, nk, nmode, norb, norb), with g[q,k,l] mapping
        internal orbitals at k+q to external orbitals at k. The reverse
        vertex is its Hermitian conjugate, not its transpose.
    phonon_tau: (ntau, nq, nmode), prescribed D0 with D=-<T X X>.
    k_plus_q: integer (nq, nk) indices into green_tau's k axis.
    q_weights: nonnegative normalized (nq,) quadrature weights.

    Returns Sigma[t,k] = -sum_ql w_q D[t,q,l] g[q,k,l]
    G[t,k+q] g[q,k,l]^dagger. Inputs provide the orbital/mode gauges
    and momentum mapping explicitly; this helper neither constructs
    vertices nor provides periodic Matsubara tails or self-consistency.
    """
    green, g, d = map(jnp.asarray, (green_tau, couplings, phonon_tau))
    if green.ndim != 4 or green.shape[-1] != green.shape[-2]:
        raise ValueError("green_tau must have shape (ntau,nk,norb,norb)")
    nt, nk, norb, _ = green.shape
    g, mapping, weights = _validate_q_data(g, k_plus_q, q_weights, nk=nk, norb=norb)
    if d.shape != (nt, g.shape[0], g.shape[2]):
        raise ValueError("phonon_tau shape is inconsistent")
    _require(jnp.all(jnp.isfinite(green)) & jnp.all(jnp.isfinite(d)),
             "periodic Fan inputs must be finite")

    @jax.checkpoint
    def accumulate(total, values):
        vertex, destination, weight, propagator = values
        internal = green[:, destination]
        value = -jnp.einsum('klia,tkab,kljb,tl->tkij', vertex, internal,
                           vertex.conj(), propagator)
        return total + weight * value, None

    initial = jnp.zeros_like(green, dtype=jnp.result_type(green, g, d, weights))
    return jax.lax.scan(accumulate, initial, (g, mapping, weights, d.swapaxes(0, 1)))[0]


def periodic_fan_self_energy(green_tau, fock, mu, model, grid):
    """Complex periodic Matsubara Fan with analytic reference and total tail.

    G(tau) is (ntau,nk,norb,norb), fock is (nk,norb,norb), in fixed local
    orbital frames. Uses one q slab at a time. This computes a physical map,
    not a periodic self-consistent solve or a real-axis analytic continuation.
    Fock derivatives currently require isolated band eigenvalues; g/Omega
    derivatives at a fixed reference remain independent of band labeling.
    """
    from .spectral import _fan_weights
    green, fock = jnp.asarray(green_tau), jnp.asarray(fock)
    if jnp.asarray(mu).ndim != 0 or jnp.iscomplexobj(mu):
        raise ValueError("mu must be a real scalar")
    _require(jnp.isfinite(mu), "mu must be finite")
    if fock.ndim != 3 or fock.shape[-1] != fock.shape[-2]:
        raise ValueError("fock must have shape (nk,norb,norb)")
    nk, norb, _ = fock.shape
    if green.shape != (2 * grid.nw, nk, norb, norb):
        raise ValueError("green_tau shape does not match fock/grid")
    validate_periodic_model(model, nk=nk, norb=norb)
    _require(jnp.all(jnp.isfinite(fock)) & jnp.allclose(fock, fock.swapaxes(-1, -2).conj(),
             rtol=1e-10, atol=1e-12), "fock must be finite Hermitian")
    energy, vectors = jnp.linalg.eigh(fock)
    xi = energy - mu
    thermal = -jnp.exp(-grid.tau[:, None, None] * xi - jnp.logaddexp(0., -grid.beta * xi))
    reference_green = jnp.einsum('kim,tkm,kjm->tkij', vectors, thermal, vectors.conj())
    omega = jnp.asarray(model.energies)
    d_tau = jax.vmap(lambda w: phonon_propagator_tau(w, grid.tau, grid.beta))(omega).swapaxes(0, 1)
    difference = periodic_fan_self_energy_tau(green - reference_green,
        model.couplings, d_tau, model.k_plus_q, model.q_weights)
    phase = jnp.exp(1j * grid.fermion[:, None] * grid.tau)
    correction = grid.beta / (2 * grid.nw) * jnp.einsum('wt,tkij->wkij', phase, difference)

    @jax.checkpoint
    def accumulate(state, values):
        vertex, modes, destination, weight = values
        transformed = jnp.einsum('klia,kam->klim', vertex, vectors[destination])
        poles = jax.vmap(lambda e: _fan_weights(1j * grid.fermion, e, modes, mu, grid.beta))(energy[destination])
        sigma = jnp.einsum('klim,kfml,kljm->fkij', transformed, poles, transformed.conj())
        moment = jnp.einsum('klia,klja,l->kij', vertex, vertex.conj(),
                            1 + 2 * bose_occupation(modes, grid.beta))
        return (state[0] + weight * sigma, state[1] + weight * moment), None

    sigma, moment = jax.lax.scan(accumulate, (jnp.zeros_like(correction),
        jnp.zeros(fock.shape, dtype=jnp.result_type(correction, model.couplings))),
        (jnp.asarray(model.couplings), omega, jnp.asarray(model.k_plus_q), jnp.asarray(model.q_weights)))[0]
    return dict(sigma_iw=sigma + correction, sigma_moment=moment)
