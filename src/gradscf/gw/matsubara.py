"""Real, spin-restricted matrix GW on finite-temperature imaginary axes.

All kernels use a fixed orthonormal orbital basis, Hartree atomic units,
and per-spin Green's functions/densities. beta has units Ha^-1. Static
Hartree-Fock reference subtraction supplies the Green's-function tail and
the independent-particle bubble exactly, including the thermal zero mode.
An analytic reference RPA interaction supplies the bosonic tail. None of
these reference terms replaces the dressed G in the GW diagrams.

The midpoint time grid has 2*nw points. Fermionic frequencies occur in
conjugate pairs; the bosonic grid includes both Nyquist endpoints with
half weights. This avoids an unpaired imaginary Nyquist contribution.

References: Yeh et al., Phys. Rev. B 106, 235104 (2022);
Caruso et al., Phys. Rev. B 88, 075105 (2013).
"""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from ..scf._pytree import pytree_dataclass
from ..solvers.eigen.spectral import matrix_function


@pytree_dataclass(static_fields=("beta", "nw"))
@dataclass(frozen=True)
class MatsubaraGrid:
    beta: float
    nw: int
    tau: jax.Array
    fermion: jax.Array
    boson: jax.Array
    boson_weights: jax.Array


def matsubara_grid(*, nw: int, beta: float) -> MatsubaraGrid:
    """nw positive fermionic frequencies, 2*nw times, 2*nw+1 bosonic nodes."""
    if int(nw) != nw or nw < 2 or not np.isfinite(beta) or beta <= 0:
        raise ValueError("nw must be an integer >= 2 and beta must be finite and positive.")
    nw, beta = int(nw), float(beta)
    tau = (jnp.arange(2 * nw, dtype=jnp.float64) + 0.5) * beta / (2 * nw)
    fermion = (2 * jnp.arange(-nw, nw) + 1) * jnp.pi / beta
    boson = 2 * jnp.arange(-nw, nw + 1) * jnp.pi / beta
    weights = jnp.ones(2 * nw + 1).at[0].set(0.5).at[-1].set(0.5)
    return MatsubaraGrid(beta, nw, tau, fermion, boson, weights)


def _reference_green(fock, mu, grid):
    """Static-reference matrices with smooth first response at degeneracy."""
    density = matrix_function(
        fock, mu, lambda energy, chemical: jax.nn.sigmoid(-grid.beta * (energy - chemical))
    )
    eye = jnp.eye(fock.shape[0], dtype=jnp.complex128)
    inverse = (1j * grid.fermion[:, None, None] + mu) * eye - fock
    giw = jnp.linalg.solve(inverse, jnp.broadcast_to(eye, inverse.shape))

    def time_matrix(tau):
        # The log form stays stable for deep occupied states and large beta.
        return matrix_function(fock, (mu, tau), lambda energy, params:
            -jnp.exp(-params[1] * (energy - params[0])
                     - jnp.logaddexp(0.0, -grid.beta * (energy - params[0]))))

    return density, giw, jax.vmap(time_matrix)(grid.tau)


def _reference(fock, mu, grid):
    energy, coeff = jnp.linalg.eigh(fock)
    occupation = jax.nn.sigmoid(-grid.beta * (energy - mu))
    density, giw, gtau = _reference_green(fock, mu, grid)
    return energy, coeff, occupation, density, giw, gtau


@jax.jit
def dyson_green_and_density(fock, sigma_iw, mu, grid):
    """Dyson G and per-spin density with analytic static-reference tails."""
    eye = jnp.eye(fock.shape[0], dtype=jnp.complex128)
    inverse = (1j * grid.fermion[:, None, None] + mu) * eye - fock - sigma_iw
    green = jnp.linalg.solve(inverse, jnp.broadcast_to(eye, inverse.shape))
    dref, gref, _ = _reference_green(fock, mu, grid)
    density = dref + jnp.sum(green - gref, axis=0).real / grid.beta
    return green, 0.5 * (density + density.T)


def _reference_polarizability(b, energy, occupation, grid):
    delta = energy[:, None] - energy[None, :]
    numerator = occupation[:, None] - occupation[None, :]
    denominator = 1j * grid.boson[:, None, None] + delta
    zero = (grid.boson[:, None, None] == 0.0) & (jnp.abs(delta) < 1e-12)
    chi = numerator / jnp.where(zero, 1.0, denominator)
    thermal = -grid.beta * occupation * (1 - occupation)
    chi = jnp.where(zero, thermal[None, :, None], chi)
    return (2 * jnp.einsum("Pij,wij,Qji->wPQ", b, chi, b)).real


def _screened(pi):
    eye = jnp.eye(pi.shape[-1], dtype=pi.dtype)
    return jnp.linalg.solve(eye - pi, pi)


def _boson_product_sum(a, b, beta):
    """(1/beta) sum_nu [(nu^2+a^2)(nu^2+b^2)]^-1, including a=b."""
    qa, qb = jnp.exp(-beta * a), jnp.exp(-beta * b)
    sa = (1 + qa) / (-jnp.expm1(-beta * a)) / (2 * a)
    sb = (1 + qb) / (-jnp.expm1(-beta * b)) / (2 * b)
    denominator = b**2 - a**2
    # A purely relative comparison is essential for soft modes: a fixed
    # energy floor would mistake, e.g., a=1e-6 and b=2e-6 for equal modes.
    close = jnp.abs(denominator) < 1e-6 * (a**2 + b**2)
    quotient = (sa - sb) / jnp.where(close, 1.0, denominator)
    middle = 0.5 * (a + b)
    q = jnp.exp(-beta * middle)
    thermal = -jnp.expm1(-beta * middle)
    limit = (1 + q) / (4 * middle**3 * thermal) + beta * q / (2 * middle**2 * thermal**2)
    return jnp.where(close, limit, quotient)


def _reference_screening(b, energy, occupation, pi_ref, grid):
    """Analytic finite-T reference RPA Wc, including any static zero mode."""
    wc_iv = _screened(pi_ref)
    left, right = jnp.triu_indices(energy.shape[0], k=1)
    if left.shape[0]:
        gap = energy[right] - energy[left]
        weight = jnp.maximum(gap * (occupation[left] - occupation[right]), 0.0)
        vertex = 2 * b[:, left, right] * jnp.sqrt(weight)[None, :]
        squared, modes = jnp.linalg.eigh(jnp.diag(gap**2) + vertex.T @ vertex)
        # A zero-energy mode has zero vertex; use a safe denominator there.
        omega = jnp.sqrt(jnp.where(squared > 0, squared, 1.0))
        amplitude = vertex @ modes
        safe_gap = jnp.where(gap > 0, gap, 1.0)
        pi_dynamic_zero = -jnp.einsum("Ps,Qs,s->PQ", vertex, vertex, 1 / safe_gap**2)
        wc_dynamic_zero = -jnp.einsum("Ps,Qs,s->PQ", amplitude, amplitude, 1 / omega**2)
        product_sum = _boson_product_sum(safe_gap[:, None], omega[None, :], grid.beta)
        ec_dynamic = -0.5 * jnp.sum((vertex.T @ amplitude)**2 * product_sum)
        thermal = -jnp.expm1(-grid.beta * omega)
        time_weight = (jnp.exp(-grid.tau[:, None] * omega)
                       + jnp.exp(-(grid.beta - grid.tau[:, None]) * omega)) / (2 * omega * thermal)
        wc_tau = -jnp.einsum("Ps,Qs,ts->tPQ", amplitude, amplitude, time_weight)
        zero_weight = (1 + jnp.exp(-grid.beta * omega)) / (2 * omega * thermal)
        wc_zero = -jnp.einsum("Ps,Qs,s->PQ", amplitude, amplitude, zero_weight)
    else:
        omega = jnp.zeros((0,), dtype=energy.dtype)
        amplitude = jnp.zeros((b.shape[0], 0), dtype=b.dtype)
        pi_dynamic_zero = jnp.zeros_like(wc_iv[0])
        wc_dynamic_zero = jnp.zeros_like(wc_iv[0])
        wc_tau = jnp.zeros((2 * grid.nw,) + wc_iv.shape[1:], dtype=wc_iv.dtype)
        wc_zero = jnp.zeros_like(wc_iv[0])
        ec_dynamic = jnp.asarray(0.0)
    # Intraband/degenerate thermal response contributes at nu=0 only.
    static = (wc_iv[grid.nw] - wc_dynamic_zero) / grid.beta
    ec_zero = -0.5 / grid.beta * (
        jnp.trace(pi_ref[grid.nw] @ wc_iv[grid.nw]) - jnp.trace(pi_dynamic_zero @ wc_dynamic_zero)
    )
    return dict(iw=wc_iv, tau=wc_tau + static, zero=wc_zero + static,
                omega=omega, amplitude=amplitude, static=static, energy=ec_dynamic + ec_zero)


def _reference_self_energy(b, b_ref, energy, coeff, occupation, mu, gref_iw, reference, grid):
    """Exact reference G0 W_RPA convolution, used only for tail subtraction."""
    omega = reference["omega"]
    coupling = jnp.einsum("Pij,Ps->sij", b_ref, reference["amplitude"])
    bose = jnp.exp(-grid.beta * omega) / (-jnp.expm1(-grid.beta * omega))
    z = 1j * grid.fermion[:, None, None] - (energy - mu)[None, :, None]
    weight = ((1 - occupation[None, :, None] + bose) / (z - omega)
              + (occupation[None, :, None] + bose) / (z + omega)) / (2 * omega)
    in_eigenbasis = jnp.einsum("sik,wks,skj->wij", coupling, weight, coupling)
    sigma = jnp.einsum("pi,wij,qj->wpq", coeff, in_eigenbasis, coeff)
    # The purely thermal/intraband bosonic zero mode is constant in tau.
    return sigma - jnp.einsum("Pik,wkl,Qlj,PQ->wij", b, gref_iw, b, reference["static"])


def _bubble(g_tau, b):
    # G(-tau) = -G(beta-tau). Factor two is the restricted spin sum.
    return -2 * jnp.einsum("Pij,tjk,Qkl,tli->tPQ", b, g_tau, b, g_tau[::-1])


@jax.jit
def gw_matsubara_step(green_iw, fock, mu, b, grid, sigma_moment=None):
    """Build Pi, Wc and Sigma_c from the supplied full matrix G.

    sigma_moment is the 1/(iw) moment of the self-energy used for G.
    Supplying it improves G(tau)'s third-order tail; the driver mixes this
    moment together with Sigma. Omit only when it is zero or unavailable.
    Thermal reference density/G have smooth first matrix response at repeated
    static levels. The exact B=0 branch supports first response; fourth-order
    interaction derivatives at that branch are not supplied. Interacting
    reference/RPA tails still require nondegenerate resolved spectra. This
    kernel is not an implicit derivative of the outer self-consistent state.
    """
    _, gref_iw, gref_tau = _reference_green(fock, mu, grid)
    residual_g = green_iw - gref_iw
    phase_f = jnp.exp(-1j * grid.tau[:, None] * grid.fermion)
    g_tau = gref_tau + jnp.einsum("tw,wij->tij", phase_f, residual_g).real / grid.beta
    if sigma_moment is not None:
        # A bounded rational reference with the same 1/(iw)^3 G tail.
        # R(z)=1/[z(z^2-a^2)] avoids the large beta^2 cancellations of a
        # bare third-order pole. Its equal-time limit is zero.
        scale = jnp.maximum(1.0, jnp.max(jnp.abs(jnp.linalg.eigvalsh(fock) - mu)))
        riw = -1 / (1j * grid.fermion * (grid.fermion**2 + scale**2))
        xi = jnp.array([-1.0, 1.0]) * scale
        scalar_g = -jnp.exp(-grid.tau[:, None] * xi - jnp.logaddexp(0.0, -grid.beta * xi))
        rtau = (0.5 + 0.5 * jnp.sum(scalar_g, axis=1)) / scale**2
        correction = rtau - (phase_f @ riw).real / grid.beta
        g_tau = g_tau + correction[:, None, None] * sigma_moment

    def zero_correlation(_):
        # Sigma_c and Pi vanish identically at B=0. Keep the Hartree/Fock
        # terms in the driver: they still carry the quadratic B dependence.
        # This exact branch supplies first response at the noninteracting
        # limit; it does not claim fourth-order interaction derivatives.
        zero_aux = jnp.zeros((2 * grid.nw + 1, b.shape[0], b.shape[0]), dtype=b.dtype)
        zero_time_aux = jnp.zeros((2 * grid.nw, b.shape[0], b.shape[0]), dtype=b.dtype)
        return dict(green_tau=g_tau, polarizability_inu=zero_aux, screened_inu=zero_aux,
                    screened_tau=zero_time_aux, sigma_iw=jnp.zeros_like(green_iw),
                    sigma_tau=jnp.zeros_like(g_tau), sigma_moment=jnp.zeros_like(fock),
                    correlation_energy=jnp.asarray(0.0, dtype=fock.dtype))

    return jax.lax.cond(jnp.any(b != 0), lambda _: _gw_correlation(
        g_tau, gref_iw, gref_tau, fock, mu, b, grid, phase_f
    ), zero_correlation, operand=None)


def _gw_correlation(g_tau, gref_iw, gref_tau, fock, mu, b, grid, phase_f):
    """Interacting reference/RPA tail path, separated from the exact B=0 limit."""
    energy, coeff = jnp.linalg.eigh(fock)
    occupation = jax.nn.sigmoid(-grid.beta * (energy - mu))
    b_ref = jnp.einsum("Ppq,pi,qj->Pij", b, coeff, coeff)
    pi_ref = _reference_polarizability(b_ref, energy, occupation, grid)
    bubble_delta = _bubble(g_tau, b) - _bubble(gref_tau, b)
    # Real restricted Pi(tau) is even about beta/2: use its cosine transform.
    cosine = jnp.cos(grid.boson[:, None] * grid.tau)
    pi = pi_ref + grid.beta / (2 * grid.nw) * jnp.einsum("wt,tPQ->wPQ", cosine, bubble_delta)
    pi = 0.5 * (pi + pi.swapaxes(-1, -2))
    wc_iv = _screened(pi)
    reference = _reference_screening(b_ref, energy, occupation, pi_ref, grid)
    difference = wc_iv - reference["iw"]
    wc_tau = reference["tau"] + jnp.einsum("wt,w,wPQ->tPQ", cosine, grid.boson_weights, difference) / grid.beta
    wc_zero = reference["zero"] + jnp.einsum("w,wPQ->PQ", grid.boson_weights, difference) / grid.beta

    sigma_tau = -jnp.einsum("Pik,tkl,Qlj,tPQ->tij", b, g_tau, b, wc_tau)
    moment = -jnp.einsum("Pik,Qkj,PQ->ij", b, b, wc_zero)
    reference_moment = -jnp.einsum("Pik,Qkj,PQ->ij", b, b, reference["zero"])
    reference_sigma_tau = -jnp.einsum("Pik,tkl,Qlj,tPQ->tij", b, gref_tau, b, reference["tau"])
    reference_sigma_iw = _reference_self_energy(
        b, b_ref, energy, coeff, occupation, mu, gref_iw, reference, grid
    )
    moment_delta = moment - reference_moment
    sigma_iw = (reference_sigma_iw + grid.beta / (2 * grid.nw)
                * jnp.einsum("tw,tij->wij", phase_f.conj(), sigma_tau - reference_sigma_tau + 0.5 * moment_delta)
                + moment_delta / (1j * grid.fermion[:, None, None]))
    # Ec_GM = -Tr(Pi Wc)/(2 beta), restricted spin convention. Reference
    # subtraction includes the infinite bosonic sum analytically and avoids
    # the O(dtau^2) cusp error of direct midpoint Sigma(tau) G(-tau).
    integrand_delta = (jnp.einsum("wPQ,wQP->w", pi, wc_iv)
                       - jnp.einsum("wPQ,wQP->w", pi_ref, reference["iw"]))
    ec = reference["energy"] - 0.5 / grid.beta * jnp.sum(grid.boson_weights * integrand_delta)
    return dict(
        green_tau=g_tau, polarizability_inu=pi, screened_inu=wc_iv,
        screened_tau=wc_tau, sigma_iw=sigma_iw, sigma_tau=sigma_tau,
        sigma_moment=moment, correlation_energy=ec,
    )


__all__ = ["MatsubaraGrid", "matsubara_grid", "dyson_green_and_density", "gw_matsubara_step"]
