"""Fixed-reference Fan poles, on-shell linewidths, and retarded spectra.

All energies, frequencies, vertices, and broadenings are in Ha, beta in
Ha^-1. Frequencies are measured relative to mu; electronic energies and
fock are absolute. These are direct real-axis fixed-reference kernels,
not analytic continuation of a dressed Matsubara scGW solution.
"""

import jax
import jax.numpy as jnp

from .phonon import bose_occupation
from .types import _require, validate_model


def _fan_occupations(energies, phonon_energies, mu, beta):
    """Absorption/emission weights (ninternal, nmode), without vertices."""
    occupation = jax.nn.sigmoid(-beta * (jnp.asarray(energies) - mu))
    bose = bose_occupation(phonon_energies, beta)
    return occupation[:, None] + bose, 1 - occupation[:, None] + bose


def _fan_weights(z, energies, phonon_energies, mu, beta):
    """Analytic Fan weights (nz, ninternal, nmode) for validated inputs.

    z is complex frequency relative to mu, while energies are absolute.
    This shared pole expression applies on both Matsubara and real axes.
    """
    absorption, emission = _fan_occupations(energies, phonon_energies, mu, beta)
    shifted = jnp.asarray(z)[:, None, None] - (jnp.asarray(energies) - mu)[None, :, None]
    omega = jnp.asarray(phonon_energies)
    return absorption / (shifted + omega) + emission / (shifted - omega)


def _real_vector(value, name):
    value = jnp.asarray(value)
    if value.ndim != 1 or not value.shape[0] or jnp.iscomplexobj(value):
        raise ValueError(f"{name} must be a nonempty real vector")
    _require(jnp.all(jnp.isfinite(value)), f"{name} must be finite")
    return value


def _validate_axis(mu, eta):
    for name, value in (("mu", mu), ("eta", eta)):
        value = jnp.asarray(value)
        if value.ndim != 0 or jnp.iscomplexobj(value):
            raise ValueError(f"{name} must be a real scalar")
        _require(jnp.isfinite(value), f"{name} must be finite")
    _require(jnp.asarray(eta) > 0, "eta must be strictly positive")


def fan_retarded(energies, model, frequencies, *, mu=0., beta, eta=.01):
    """Full retarded Fan matrix (nfreq, norb, norb) with fixed free G/D0.

    energies and model.couplings must be in the same canonical electronic
    eigenbasis. Complex Hermitian vertices are supported. Positive eta
    gives causal poles; no sign clipping or absolute-value repair is used.
    Quadratic vertices are not included: add debye_waller separately if
    the supplied external model contains a quadratic coupling.
    """
    energies = _real_vector(energies, "energies")
    frequencies = _real_vector(frequencies, "frequencies")
    _validate_axis(mu, eta)
    validate_model(model, norb=energies.shape[0], real=False)
    weight = _fan_weights(frequencies + 1j * eta, energies, model.energies, mu, beta)
    g = jnp.asarray(model.couplings)
    return jnp.einsum("lia,wal,lja->wij", g, weight, g.conj())


def fan_linewidth(energies, internal_energies, phonon_energies, couplings, *,
                  mu=0., beta, eta=.01, smearing="lorentzian"):
    """On-shell full linewidth Gamma (nexternal,) for a single k/q block.

    couplings has shape (nmode, nexternal, ninternal), mapping internal
    eigenstates to external eigenstates; rectangular complex vertices
    are allowed. For a periodic calculation sum q-weighted outputs.
    Gamma = -2 Im Sigma^R_ii(e_i-mu), in Ha (hbar=1); it is nonnegative
    for positive modes. Lorentzian broadening uses the analytic Fan poles.
    Gaussian broadening uses exp[-(x/eta)^2]/(sqrt(pi)*eta), an on-shell
    delta approximation only, not a full complex retarded self-energy.

    ElectronPhonon.jl a0ecee01b7413af5566134d94cad5b15431d6565,
    src/selfenergy_electron.jl, stores the positive halfwidth Gamma/2.
    Comparison requires the same energy units, occupations, and q weight.
    No acoustic threshold is applied: all supplied modes must be positive.
    """
    energies = _real_vector(energies, "energies")
    internal = _real_vector(internal_energies, "internal_energies")
    omega = _real_vector(phonon_energies, "phonon_energies")
    _require(jnp.all(omega > 0), "phonon energies must be strictly positive")
    _validate_axis(mu, eta)
    g = jnp.asarray(couplings)
    if g.shape != (omega.shape[0], energies.shape[0], internal.shape[0]):
        raise ValueError("couplings must have shape (nmode, nexternal, ninternal)")
    _require(jnp.all(jnp.isfinite(g)), "couplings must be finite")
    if smearing == "lorentzian":
        weight = -2 * _fan_weights(energies - mu + 1j * eta, internal, omega, mu, beta).imag
    elif smearing == "gaussian":
        absorption, emission = _fan_occupations(internal, omega, mu, beta)
        delta_e = energies[:, None, None] - internal[None, :, None]
        def delta(x):
            return jnp.exp(-(x / eta)**2) / (jnp.sqrt(jnp.pi) * eta)
        weight = 2 * jnp.pi * (absorption * delta(delta_e + omega)
                               + emission * delta(delta_e - omega))
    else:
        raise ValueError("smearing must be 'lorentzian' or 'gaussian'")
    return jnp.einsum("lia,ial->i", jnp.abs(g)**2, weight)


def electron_linewidth(energies, model, *, mu=0., beta, eta=.01, smearing="lorentzian"):
    """Molecular on-shell Gamma (norb,) in the canonical eigenbasis.

    See fan_linewidth for smearing and full-width versus halfwidth
    conventions; this wrapper validates the Hermitian PhononModel.
    """
    energies = _real_vector(energies, "energies")
    validate_model(model, norb=energies.shape[0], real=False)
    return fan_linewidth(energies, energies, model.energies, model.couplings,
                         mu=mu, beta=beta, eta=eta, smearing=smearing)


def spectral_function(fock, sigma, frequencies, *, mu=0., eta=.01):
    """Matrix A = -(G^R-G^R†)/(2 pi i), shape (nfreq, norb, norb).

    G^R = [(w+mu+i eta)I-fock-Sigma^R(w)]^-1. fock is Hermitian in
    an orthonormal basis; sigma must already be the desired retarded
    self-energy in that same basis. The caller owns its approximation,
    double-counting convention, and causality. No analytic continuation
    is performed. The matrix anti-Hermitian part is essential for complex
    orbitals; elementwise -Im(G)/pi gives an incorrect spectral matrix.
    The trace is the total spectral density per Ha. eta is the explicit
    bare resolvent broadening in addition to any width present in sigma.
    """
    frequencies = _real_vector(frequencies, "frequencies")
    _validate_axis(mu, eta)
    fock, sigma = jnp.asarray(fock), jnp.asarray(sigma)
    if fock.ndim != 2 or fock.shape[0] != fock.shape[1] or not fock.shape[0]:
        raise ValueError("fock must have shape (norb, norb)")
    if sigma.shape != (frequencies.shape[0], *fock.shape):
        raise ValueError("sigma must have shape (nfreq, norb, norb)")
    _require(jnp.all(jnp.isfinite(fock)) & jnp.all(jnp.isfinite(sigma)),
             "fock and sigma must be finite")
    _require(jnp.allclose(fock, fock.T.conj(), rtol=1e-10, atol=1e-12),
             "fock must be Hermitian")
    inverse = ((frequencies + mu + 1j * eta)[:, None, None]
               * jnp.eye(fock.shape[0]) - fock - sigma)
    green = jnp.linalg.inv(inverse)
    return -(green - green.swapaxes(-1, -2).conj()) / (2j * jnp.pi)
