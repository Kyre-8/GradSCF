"""Harmonic one-phonon loss/gain spectra with explicit input strengths.

These are kinematic spectral utilities, not a calibrated electron scattering
cross section or a multiple-scattering EELS simulation. Energies are Hartree,
temperatures are kelvin, coordinates are bohr, and momenta are inverse bohr.
All array kernels support JIT and differentiation, including value guards.
"""

from typing import NamedTuple

import jax.numpy as jnp
from scipy.constants import physical_constants

from ..gw.ep_coupling.phonon import bose_occupation
from ..gw.ep_coupling.types import _require


_KB_HARTREE_PER_K = (physical_constants["Boltzmann constant in eV/K"][0]
                     / physical_constants["Hartree energy in eV"][0])
_AMU_TO_ELECTRON_MASS = (physical_constants["atomic mass constant"][0]
                         / physical_constants["electron mass"][0])


class OnePhononSpectrum(NamedTuple):
    """Spectral densities on the supplied grid: positive loss, negative gain."""

    loss: object
    gain: object
    total: object


def _real_finite(value, name):
    value = jnp.asarray(value)
    if jnp.iscomplexobj(value):
        raise ValueError(f"{name} must be real")
    _require(jnp.all(jnp.isfinite(value)), f"{name} must be finite")
    return value


def one_phonon_spectrum(energies, strengths, energy_grid, *, temperature, fwhm):
    """Broaden harmonic lines at +energy (loss) and -energy (gain).

    ``energies`` is a nonempty positive vector in Ha; signed imaginary modes
    and zero-frequency translations must be treated explicitly by the caller.
    No absolute value, frequency floor, or stability correction is applied.
    ``strengths`` contains nonnegative zero-temperature integrated loss
    strengths, including any desired zero-point factor 1/(2*energy).
    ``temperature`` is a finite nonnegative scalar in kelvin. ``fwhm`` is the
    finite positive Gaussian instrumental full width at half maximum in Ha.
    ``energy_grid`` is a real one-dimensional grid in Ha, in any order.

    Each Gaussian has unit integral on the whole real axis, so the integrated
    line areas are strength*(n_B+1) and strength*n_B. For each positive-energy
    mode their gain/loss ratio is exp(-energy/(k_B*T)); at T=0 gain vanishes.
    This detailed-balance relation applies to line areas before broadening,
    not generally pointwise to overlapping broadened spectra. A finite grid
    can truncate Gaussian tails. Outputs have units of strength per Hartree.

    The same strength is used for the loss and gain partner, assuming reciprocal
    harmonic modes/vertices. Supply separate momentum spectra when that
    assumption does not apply. Traced invalid values raise a JAX callback error
    containing the validation message; eager invalid values raise ValueError.
    """
    energies = _real_finite(energies, "energies")
    strengths = _real_finite(strengths, "strengths")
    energy_grid = _real_finite(energy_grid, "energy_grid")
    temperature = _real_finite(temperature, "temperature")
    fwhm = _real_finite(fwhm, "fwhm")
    if energies.ndim != 1 or not energies.size or strengths.shape != energies.shape:
        raise ValueError("energies and strengths must have matching nonempty vector shape")
    if energy_grid.ndim != 1 or not energy_grid.size:
        raise ValueError("energy_grid must have nonempty vector shape")
    if temperature.ndim or fwhm.ndim:
        raise ValueError("temperature and fwhm must be scalar")
    _require(jnp.all(energies > 0), "energies must be strictly positive; do not clip imaginary modes")
    _require(jnp.all(strengths >= 0), "strengths must be nonnegative")
    _require(temperature >= 0, "temperature must be nonnegative")
    _require(fwhm > 0, "fwhm must be strictly positive")

    # Keep both evaluated branches finite at T=0, including their AD tangents.
    positive_temperature = jnp.where(temperature > 0, temperature, 1.)
    occupation = bose_occupation(energies, 1 / (_KB_HARTREE_PER_K * positive_temperature))
    occupation = jnp.where(temperature > 0, occupation, 0.)
    sigma = fwhm / jnp.sqrt(8 * jnp.log(2.))
    offset = energy_grid[:, None]
    normalization = sigma * jnp.sqrt(2 * jnp.pi)
    loss_profile = jnp.exp(-0.5 * ((offset - energies) / sigma)**2) / normalization
    gain_profile = jnp.exp(-0.5 * ((offset + energies) / sigma)**2) / normalization
    loss = loss_profile @ (strengths * (occupation + 1))
    gain = gain_profile @ (strengths * occupation)
    return OnePhononSpectrum(loss, gain, loss + gain)


def mode_scattering_weights(momentum_cartesian, coordinates_bohr, modes,
                            masses_amu, *, form_factors):
    r"""Return coherent relative kinematic weights, without 1/(2*energy).

    For a single actual momentum Q=G+q in inverse bohr, this evaluates

    W_nu = |sum_k f_k exp(-i Q.tau_k) (Q.e_knu) / sqrt(M_k)|**2.

    ``modes`` has shape (natom,3,nmode) or (3*natom,nmode), with dimensionless
    mass-weighted eigenvectors stored in columns. ``masses_amu`` is converted
    to electron masses. Eigenvector normalization is the caller's responsibility.
    ``form_factors`` is an explicit finite (natom,) vector, optionally complex;
    no atomic-number, screened-potential, or Debye-Waller model is assumed.

    The phase convention is displacement e_k(q)*exp(+i q.R) for lattice
    translations R, as in pbc.phonon and pbc.ifc. The atom-coordinate phase is
    supplied here exactly once through exp(-i Q.tau_k), corresponding to the
    density component rho_-Q. Eigenvectors defined with exp(i q.(R+tau_k))
    must first be multiplied by exp(i q.tau_k). Q is the full scattering
    momentum, not a fractional reciprocal coordinate or a wrapped q.

    Divide W_nu by 2*energy_nu (Ha) to obtain a zero-point-normalized relative
    strength for ``one_phonon_spectrum``. Arbitrary form factors yield relative
    strengths only: electron probe dynamics, detector geometry, screening and
    multiple scattering are absent. The individual weights in a degenerate
    subspace depend on the eigenvector basis; their sum is unitary invariant.
    """
    momentum = _real_finite(momentum_cartesian, "momentum_cartesian")
    coordinates = _real_finite(coordinates_bohr, "coordinates_bohr")
    masses = _real_finite(masses_amu, "masses_amu")
    modes, factors = jnp.asarray(modes), jnp.asarray(form_factors)
    if momentum.shape != (3,) or coordinates.ndim != 2 or coordinates.shape[1] != 3:
        raise ValueError("momentum and coordinates must have shape (3,) and (natom,3)")
    natom = coordinates.shape[0]
    if not natom or masses.shape != (natom,) or factors.shape != (natom,):
        raise ValueError("masses and form_factors must have shape (natom,) for nonempty atoms")
    if modes.ndim == 2 and modes.shape[0] == 3 * natom:
        modes = modes.reshape(natom, 3, modes.shape[-1])
    if modes.ndim != 3 or modes.shape[:2] != (natom, 3) or not modes.shape[-1]:
        raise ValueError("modes must have shape (natom,3,nmode) or (3*natom,nmode)")
    _require(jnp.all(masses > 0), "masses_amu must be strictly positive")
    _require(jnp.all(jnp.isfinite(modes)), "modes must be finite")
    _require(jnp.all(jnp.isfinite(factors)), "form_factors must be finite")
    atom_factors = factors * jnp.exp(-1j * (coordinates @ momentum)) / jnp.sqrt(masses * _AMU_TO_ELECTRON_MASS)
    amplitude = jnp.einsum("a,i,aim->m", atom_factors, momentum, modes)
    return jnp.real(amplitude * amplitude.conj())


__all__ = ["OnePhononSpectrum", "one_phonon_spectrum", "mode_scattering_weights"]
