"""Harmonic spectra, mode charges and the polar long-wavelength limit.

These are kinematic spectral utilities, not a calibrated electron scattering
cross section or a multiple-scattering EELS simulation. Energies are Hartree,
temperatures are kelvin, coordinates are bohr, and momenta are inverse bohr.
The infrared kernels consume supplied Born charges and electronic dielectric
tensors; they do not calculate electronic response or finite-q polar IFCs.
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


def _born_charges(born_charges):
    born = _real_finite(born_charges, "born_charges")
    if born.ndim != 3 or born.shape[1:] != (3, 3) or not born.shape[0]:
        raise ValueError("born_charges must have shape (natom,3,3)")
    return born


def _volume(volume):
    volume = _real_finite(volume, "volume")
    if volume.ndim:
        raise ValueError("volume must be scalar in bohr**3")
    _require(volume > 0, "volume must be strictly positive")
    return volume


def nonanalytic_force_constants(direction, born_charges, dielectric, volume):
    r"""Unweighted Gamma directional correction in Eh/bohr**2.

    ``born_charges[a,i,j] = volume * dP_i/dR_aj`` is in units of e;
    ``dielectric`` is the real symmetric positive electronic epsilon_infinity.
    ``direction`` is a nonzero Cartesian reciprocal-space direction, not a
    fractional q point. Its magnitude and sign do not change the result.
    ``volume`` is the primitive-cell volume in bohr**3.

    Phi_NA[aj,bk] = 4*pi/V * (q_i Z_aij)(q_l Z_blk)/(q.epsilon.q).

    Add this matrix to the short-circuit Gamma force constants, then use the
    existing mass weighting/mode solver. No masses enter this unweighted
    correction. Raw Born charge-sum defects remain: no neutrality, ASR or
    symmetry projection is applied. Validate these separately before making
    a material-level LO-TO claim. This q->0 limit does not implement the
    dipolar subtraction/restoration needed for finite-q IFC interpolation.

    Convention and formula: Gonze and Lee, Phys. Rev. B 55, 10355 (1997),
    https://doi.org/10.1103/PhysRevB.55.10355; also
    https://phonopy.github.io/phonopy/formulation.html#non-analytical-term-correction.
    """
    born = _born_charges(born_charges)
    q = _real_finite(direction, "direction")
    epsilon = _real_finite(dielectric, "dielectric")
    volume = _volume(volume)
    if q.shape != (3,) or epsilon.shape != (3, 3):
        raise ValueError("direction and dielectric must have shape (3,) and (3,3)")
    _require(jnp.linalg.norm(q) > 0, "direction must be nonzero")
    _require(jnp.allclose(epsilon, epsilon.T, rtol=1e-10, atol=1e-12),
             "dielectric must be symmetric")
    _require(jnp.all(jnp.linalg.eigvalsh(epsilon) > 0),
             "dielectric must be positive definite")
    q = q / jnp.linalg.norm(q)
    charge = jnp.einsum("i,aij->aj", q, born).reshape(-1)
    return (4 * jnp.pi / volume) * jnp.outer(charge, charge) / (q @ epsilon @ q)


def mode_effective_charges(born_charges, modes, masses_amu):
    r"""Gamma mode charges Z_mode[i,nu] in e/sqrt(electron_mass).

    Contract Z_mode[i,nu] = sum_aj Z[a,i,j] e[a,j,nu]/sqrt(M_a).
    ``modes`` are orthonormal mass-weighted eigenvectors in columns, with
    shape (3*natom,nmode) or (natom,3,nmode). ``masses_amu`` contains one
    positive mass per atom and is converted to electron masses internally.
    A normalized subset of modes is allowed. Frequencies do not enter this
    contraction; callers must select stable optical modes for dielectric
    spectra. Nonneutral charge tensors retain nonzero translation strengths.

    Complex mode gauges are allowed. Individual strengths within a degenerate
    group depend on the basis; the sum of Z_mode Z_mode^dagger over a complete
    group is invariant. No new eigenvector derivative rule is introduced.
    """
    born = _born_charges(born_charges)
    masses = _real_finite(masses_amu, "masses_amu")
    vectors = jnp.asarray(modes)
    natom = born.shape[0]
    if masses.shape != (natom,):
        raise ValueError("masses_amu must have shape (natom,)")
    if vectors.ndim == 3 and vectors.shape[:2] == (natom, 3):
        vectors = vectors.reshape(3 * natom, -1)
    if (vectors.ndim != 2 or vectors.shape[0] != 3 * natom
            or not 0 < vectors.shape[1] <= 3 * natom):
        raise ValueError("modes must have shape (3*natom,nmode) or (natom,3,nmode)")
    _require(jnp.all(masses > 0), "masses_amu must be strictly positive")
    _require(jnp.all(jnp.isfinite(vectors)), "modes must be finite")
    _require(jnp.allclose(vectors.conj().T @ vectors, jnp.eye(vectors.shape[1]),
                         rtol=1e-8, atol=1e-10), "modes must be orthonormal")
    displacement = vectors.reshape(natom, 3, -1) / jnp.sqrt(masses[:, None, None] * _AMU_TO_ELECTRON_MASS)
    return jnp.einsum("aij,ajm->im", born, displacement)


def ionic_dielectric(energies, mode_charges, energy_grid, *, volume, broadening):
    r"""Retarded ionic contribution Delta epsilon(omega), dimensionless.

    Delta epsilon_ij(w) = 4*pi/V sum_n Z_in Z_jn^*/[w_n**2-(w+i*eta)**2].

    ``energies`` contains positive transverse optical hbar*omega in Eh;
    ``mode_charges`` has shape (3,nmode), in e/sqrt(electron_mass), as returned
    by ``mode_effective_charges``. ``energy_grid`` is a real vector in Eh.
    ``broadening`` is a nonnegative scalar eta in Eh; zero is allowed only
    away from poles (including the exact static limit). No imaginary-mode
    clipping, acoustic-mode filtering, weight renormalization or electronic
    background is applied. Add epsilon_infinity to obtain total epsilon.

    This is the harmonic displacement response to a macroscopic transverse
    field, not a photon absorption cross section or a computed lifetime.
    Positive eta gives nonnegative diagonal loss at positive frequency.
    At finite eta the zero-frequency value includes that resolution shift;
    use eta=0 for the undamped static dielectric tensor and LST identities.
    """
    omega = _real_finite(energies, "energies")
    charges = jnp.asarray(mode_charges)
    grid = _real_finite(energy_grid, "energy_grid")
    volume = _volume(volume)
    eta = _real_finite(broadening, "broadening")
    if omega.ndim != 1 or not omega.size or charges.shape != (3, omega.size):
        raise ValueError("energies and mode_charges must have shape (nmode,) and (3,nmode)")
    if grid.ndim != 1 or not grid.size or eta.ndim:
        raise ValueError("energy_grid must be a nonempty vector and broadening a scalar")
    _require(jnp.all(omega > 0), "energies must be strictly positive")
    _require(jnp.all(jnp.isfinite(charges)), "mode_charges must be finite")
    _require(eta >= 0, "broadening must be nonnegative")
    denominator = omega[None, :]**2 - (grid[:, None] + 1j * eta)**2
    _require(jnp.all(jnp.abs(denominator) > 0), "zero broadening is undefined at a mode pole")
    return 4 * jnp.pi / volume * jnp.einsum("im,jm,wm->wij", charges, charges.conj(), 1 / denominator)


__all__ = ["OnePhononSpectrum", "one_phonon_spectrum", "mode_scattering_weights",
           "nonanalytic_force_constants", "mode_effective_charges", "ionic_dielectric"]
