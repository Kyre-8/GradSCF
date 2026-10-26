"""Small fixed-model vibronic KHD scattering workflows, without a new solver.

The initial state is the ground oscillator vacuum; final states are its single
quanta. One excited electronic surface has an orthogonal Duschinsky map in the
same mass-weighted coordinate space and a linear transition dipole. Enumeration
is host-side, not a full-spectrum JAX differentiation API.

We use the scattering S-matrix convention G(E)=(E-H+i*Gamma)^-1 for BOTH time
orderings. In positive excitation-energy denominators this gives -i*Gamma in
both terms. This is not the retarded induced-polarization response convention:
see Mukamel, PRA 76, 021803 (2007), Eqs. 5/11/12 versus 13/14, and Chen et al.,
arXiv:1809.04498, Eq. 2. Gamma is a positive phenomenological energy HWHM.
Amplitudes have units (e bohr)^2/Eh; no photon prefactor or absolute cross-section
conversion is included. Complex nonsymmetric Cartesian tensors are retained.
"""
import numpy as np

from .franck_condon import ground_state_quadratic_coordinate_overlaps


def _dipole_inputs(initial, dipole, derivatives):
    if np.iscomplexobj(initial):
        raise ValueError("Ground frequencies must be real.")
    omega = np.asarray(initial, dtype=float)
    mu, derivative = np.asarray(dipole), np.asarray(derivatives)
    if omega.ndim != 1 or not omega.size or np.any(omega <= 0) or not np.isfinite(omega).all():
        raise ValueError("Ground frequencies must be a positive finite vector.")
    if mu.shape != (3,) or derivative.shape != (3, len(omega)):
        raise ValueError("Dipole and derivatives must have shapes (3,) and (3,nmode).")
    if not np.isfinite(mu).all() or not np.isfinite(derivative).all():
        raise ValueError("Dipole inputs must be finite.")
    return omega, mu, derivative


def linear_dipole_vertices(initial, final, rotation, shift, dipole, derivatives, *, max_quanta=12):
    """Return occupations, a[n,j]=<n_e|mu_j|0_g>, b[n,k,j]=<n_e|mu_j|1_g,k>.

    All signs/phases are retained. The Hermitian emission vertex is b.conj();
    b itself uses the same electronic transition orientation as a. The identity
    |1_g,k>=sqrt(2*omega_g,k)*Q_g,k|0_g> requires the second coordinate moment
    even though the electronic dipole surface is only linear.
    """
    omega, mu, derivative = _dipole_inputs(initial, dipole, derivatives)
    states, overlap, coordinate, quadratic = ground_state_quadratic_coordinate_overlaps(
        initial, final, rotation, shift, max_quanta=max_quanta)
    absorption = overlap[:, None]*mu+coordinate@derivative.T
    emission = coordinate[:, :, None]*mu+quadratic@derivative.T
    emission *= np.sqrt(2*omega)[None, :, None]
    return states, absorption, emission


def vertex_closure(initial, dipole, derivatives):
    """Exact infinite-state Gram matrices a^H a and b_k^H b_k.

    These unweighted dipole closures bound omitted amplitudes by Cauchy-Schwarz;
    they are not oscillator-strength sums and do not renormalize finite sums.
    """
    omega, mu, derivative = _dipole_inputs(initial, dipole, derivatives)
    vacuum = np.outer(mu.conj(), mu)+(derivative.conj()/(2*omega))@derivative.T
    excited = vacuum[None, :, :]+np.einsum("ik,jk->kij", derivative.conj(), derivative)/omega[:, None, None]
    return vacuum, excited


def khd_tensor(energies, absorption, emission, final_energies, incident_energies, gamma,
               *, state_chunk=65536, photon_chunk=8):
    """Return resonant and antiresonant arrays (nlaser,nfinal,3,3).

    energies[n]=E_n-E_initial includes the model 0-0 and intermediate quanta.
    final_energies[k]=E_final,k-E_initial; outgoing=incident-final must be >0.
    A_ij=sum_n b_ki* a_j/(Delta_n-incident-i Gamma)
                 + b_kj* a_i/(Delta_n+outgoing-i Gamma).
    The sign of A is minus the bare two-photon T-matrix convention; the two
    terms' relative sign is unchanged. No tensor symmetrization is performed.
    """
    if any(np.iscomplexobj(x) for x in (energies, final_energies, incident_energies)):
        raise ValueError("Intermediate, final and photon energies must be real.")
    if np.ndim(gamma) != 0 or np.iscomplexobj(gamma):
        raise ValueError("Gamma must be a real scalar.")
    gamma = float(gamma)
    delta, final, laser = (np.asarray(x, dtype=float) for x in (energies, final_energies, incident_energies))
    a, b = np.asarray(absorption), np.asarray(emission)
    if delta.ndim != 1 or final.ndim != 1 or laser.ndim != 1 or not min(delta.size, final.size, laser.size):
        raise ValueError("Energies must be nonempty vectors.")
    if a.shape != (len(delta), 3) or b.shape != (len(delta), len(final), 3):
        raise ValueError("Vertex arrays must have shapes (nstate,3), (nstate,nfinal,3).")
    if not all(np.isfinite(x).all() for x in (delta, final, laser, a, b)):
        raise ValueError("Energies and vertices must be finite.")
    if not np.isfinite(gamma) or gamma <= 0 or np.any(delta <= 0) or np.any(final < 0):
        raise ValueError("Intermediate energies and gamma must be positive; final gaps nonnegative.")
    if np.any(laser[:, None]-final[None, :] <= 0):
        raise ValueError("All outgoing Stokes photon energies must be positive.")
    if any(type(x) is not int or x <= 0 for x in (state_chunk, photon_chunk)):
        raise ValueError("Chunk sizes must be positive static integers.")
    resonant = np.zeros((len(laser), len(final), 3, 3), dtype=complex)
    anti = np.zeros_like(resonant)
    for start in range(0, len(delta), state_chunk):
        sl = slice(start, start+state_chunk)
        for first in range(0, len(laser), photon_chunk):
            lp = slice(first, first+photon_chunk)
            frequency = laser[lp]
            denominator = 1/(delta[sl, None]-frequency[None, :]-1j*gamma)
            incoming = (denominator[:, :, None]*a[sl, None, :]).reshape(len(delta[sl]), -1)
            for mode, gap in enumerate(final):
                bra = b[sl, mode, :].conj().T
                resonant[lp, mode] += (bra@incoming).reshape(3, len(frequency), 3).transpose(1, 0, 2)
                denominator = 1/(delta[sl, None]+frequency[None, :]-gap-1j*gamma)
                outgoing = (denominator[:, :, None]*a[sl, None, :]).reshape(len(delta[sl]), -1)
                anti[lp, mode] += (bra@outgoing).reshape(3, len(frequency), 3).transpose(1, 2, 0)
    return resonant, anti


def isotropic_strengths(tensor):
    """Return parallel/perpendicular SO(3) averages for general complex A.

    The analyzer vectors are parallel or perpendicular real unit vectors; A's
    first Cartesian index is scattered and the second incident polarization.
    Return shape (2,...) for input (...,3,3). Units are amplitude squared.
    """
    a = np.asarray(tensor)
    if a.shape[-2:] != (3, 3) or not np.isfinite(a).all():
        raise ValueError("A finite Cartesian (...,3,3) tensor is required.")
    trace = abs(np.trace(a, axis1=-2, axis2=-1))**2
    norm = np.sum(abs(a)**2, axis=(-2, -1))
    transpose = np.sum(a*a.swapaxes(-1, -2).conj(), axis=(-2, -1)).real
    return np.stack(((trace+norm+transpose)/15, (4*norm-trace-transpose)/30))


__all__ = ["linear_dipole_vertices", "vertex_closure", "khd_tensor", "isotropic_strengths"]
