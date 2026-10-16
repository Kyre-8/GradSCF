"""Cell-normalized length vertices; no macroscopic absorption approximation."""
from numbers import Integral
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np
from .kernel import _eager, _bands


class OpticalTransitions(NamedTuple):
    energies: object
    dipoles: object
    strengths: object


def ks_dipoles(mf, *, valence=None, conduction=None):
    """Length vertices for a current restricted LDA/GGA KS reference.

    Return (nk,nv,nc,3) in Bohr, with the same band windows as the BSE kernel.
    The shared AO velocity includes the GTH nonlocal-potential commutator.
    HF/hybrid and meta-GGA velocity completions are not implemented here.
    """
    from ...pbc.scf import validate_reference
    from ...pbc.optics import velocity_matrix
    from ...dft.libxc_jax.jax_libxc import hybrid_coeff, xc_type
    validate_reference(mf, low_dimensional=True)
    if mf.cell.dimension < 3 and mf.exxdiv is not None:
        raise ValueError('Reduced-dimensional optical references require exxdiv=None.')
    if mf.unrestricted or hybrid_coeff(mf.xc) != 0 or xc_type(mf.xc) not in ('LDA', 'GGA'):
        raise NotImplementedError('KS optical vertices currently require restricted LDA/GGA references.')
    c, energy = mf.result.mo_coeff_spin[0], mf.result.mo_energy_spin[0]
    no = mf.cell.nelec[0]
    occupied, virtual = _bands(valence, 0, no), _bands(conduction, no, energy.shape[1])
    velocity = jnp.stack([jnp.einsum('pi,dpq,qa->iad', ck[:, occupied].conj(),
        velocity_matrix(mf.cell, kpt=k), ck[:, virtual]) for k, ck in zip(mf.kpts, c)])
    gaps = energy[:, virtual][:, None, :]-energy[:, occupied][:, :, None]
    return ks_transition_dipoles(velocity, gaps)


def independent_particles(qp_energies, nocc, dipoles, *, valence=None, conduction=None,
                          polarization=None):
    """QP independent-particle lines using the same fixed KS length vertices.

    Windows and (nk,nv,nc,3) dipoles must agree with build_tda_matrix/ks_dipoles.
    Unselected QP entries may be NaN. Unit transition vectors are implicit:
    no identity matrix is allocated. The singlet/k weight is sqrt(2/Nk).
    """
    _eager(qp_energies, dipoles, polarization)
    energy, d = jnp.asarray(qp_energies), jnp.asarray(dipoles)
    if (energy.ndim != 2 or not len(energy) or isinstance(nocc, bool)
            or not isinstance(nocc, Integral) or not 0 < nocc < energy.shape[1]):
        raise ValueError('Use QP energies (nk,nmo) and a nonempty occupied/virtual space.')
    occupied = _bands(valence, 0, nocc)
    virtual = _bands(conduction, nocc, energy.shape[1])
    if d.shape != (len(energy), len(occupied), len(virtual), 3):
        raise ValueError('Dipole shape must match the selected QP band windows.')
    gaps = (energy[:, virtual][:, None, :]-energy[:, occupied][:, :, None]).reshape(-1)
    moments = jnp.sqrt(2./len(energy))*d.reshape(-1, 3)
    return OpticalTransitions(gaps, moments, oscillator_strengths(gaps, moments, polarization=polarization))


def ks_transition_dipoles(velocity, gaps):
    """d_vc=i*v_vc/Delta_KS, shape (nk,nv,nc,3), in Bohr.

    Supply the complete KS velocity, including nonlocal-pseudopotential terms.
    The existing periodic p+GTH velocity helper covers local LDA/GGA KS
    references. No HF/hybrid, meta-GGA, QP or scissor velocity completion is
    inferred here. At fixed orbitals, changing QP gaps leaves this vertex
    unchanged; its velocity representation must then scale by Delta_QP/Delta_KS.
    """
    _eager(velocity, gaps)
    v, gap = jnp.asarray(velocity), jnp.asarray(gaps)
    if v.ndim != 4 or v.shape[-1] != 3 or gap.shape != v.shape[:-1]:
        raise ValueError('Use velocities (nk,nv,nc,3) and matching KS gaps.')
    if jnp.iscomplexobj(gap) or not np.isfinite(gap).all() or np.any(np.asarray(gap) <= 0) or not np.isfinite(v).all():
        raise ValueError('Optical vertices require finite velocities and positive real KS gaps.')
    return 1j*v/gap[..., None]


def transition_dipoles(dipoles, vectors):
    """D_S=sqrt(2/Nk)*sum_kvc d_vc(k)*x_S(kvc), in Bohr.

    Input spatial amplitudes are columns, ordered (k,v,c). The spin-singlet
    sqrt(2) and full equal-weight k normalization are included exactly once.
    Orbital-phase covariance requires this contraction without conjugating x.
    """
    _eager(dipoles, vectors)
    d, x = jnp.asarray(dipoles), jnp.asarray(vectors)
    if d.ndim != 4 or d.shape[-1] != 3 or not all(d.shape[:-1]) or x.ndim != 2 or x.shape[0] != np.prod(d.shape[:-1]):
        raise ValueError('Use vertices (nk,nv,nc,3) and amplitudes (nk*nv*nc,nstates).')
    norm = jnp.sum(jnp.abs(x)**2, axis=0)
    if not np.isfinite(d).all() or not np.isfinite(x).all() or np.any(np.asarray(norm) <= 0):
        raise ValueError('Transition amplitudes must be finite with positive norms.')
    return jnp.sqrt(2./d.shape[0])*((d.reshape(-1, 3).T@(x/jnp.sqrt(norm)[None, :])).T)


def oscillator_strengths(energies, dipoles, *, polarization=None):
    """Dimensionless per-cell strengths from length dipoles and Hartree energies.

    Default is the orientational average 2*Omega*sum_alpha|D_alpha|²/3.
    A real polarization vector selects 2*Omega*|e.D|². A broadened collection
    of these strengths is not, by itself, a macroscopic absorption coefficient.
    """
    _eager(energies, dipoles, polarization)
    energy, d = jnp.asarray(energies), jnp.asarray(dipoles)
    if energy.ndim != 1 or d.shape != (len(energy), 3):
        raise ValueError('Use excitation energies (nstates,) and dipoles (nstates,3).')
    if jnp.iscomplexobj(energy) or not np.isfinite(energy).all() or np.any(np.asarray(energy) <= 0) or not np.isfinite(d).all():
        raise ValueError('Optical strengths require positive real energies and finite dipoles.')
    if polarization is None:
        strength = jnp.sum(jnp.abs(d)**2, axis=1)/3
    else:
        p = jnp.asarray(polarization)
        if p.shape != (3,) or jnp.iscomplexobj(p) or not np.isfinite(p).all() or float(jnp.linalg.norm(p)) == 0:
            raise ValueError('Use a real nonzero Cartesian polarization vector.')
        strength = jnp.abs(d@(p/jnp.linalg.norm(p)))**2
    return 2*energy*strength


def polarizability(energies, dipoles, omega=0., *, eta=0., lattice, dimension=3):
    """Complex retarded length response per periodic measure, in atomic units.

    D_S=<0|r|S> includes the singlet and k weights. Both spectral legs are
    retained, with their own conjugation for complex vertices. Energies,
    omega and eta are Hartree. Divide by |a1|, |a1 x a2| or V for dimension
    1, 2 or 3, respectively; the resulting units are Bohr**(3-dimension).
    This is a line/sheet/bulk polarizability, not an absorption coefficient.
    The scalar-frequency result is (3,3); a frequency array gives (nw,3,3).
    """
    _eager(energies, dipoles, omega, eta, lattice)
    energy, moments, frequency, width, cell = map(jnp.asarray,
        (energies, dipoles, omega, eta, lattice))
    if energy.ndim != 1 or moments.shape != (len(energy), 3):
        raise ValueError('Use excitation energies (nstates,) and dipoles (nstates,3).')
    if (jnp.iscomplexobj(energy) or not np.isfinite(energy).all()
            or np.any(np.asarray(energy) <= 0) or not np.isfinite(moments).all()):
        raise ValueError('Polarizability requires positive real energies and finite dipoles.')
    if (frequency.ndim > 1 or width.ndim or jnp.iscomplexobj(frequency)
            or jnp.iscomplexobj(width) or not np.isfinite(frequency).all()
            or not np.isfinite(width) or float(width) < 0):
        raise ValueError('Use real finite frequencies and a nonnegative scalar broadening.')
    if (isinstance(dimension, bool) or not isinstance(dimension, Integral)
            or dimension not in (1, 2, 3) or cell.shape != (3, 3)
            or jnp.iscomplexobj(cell) or not np.isfinite(cell).all()
            or float(jnp.linalg.det(cell)) <= 0):
        raise ValueError('Use dimension 1, 2 or 3 and a right-handed lattice in Bohr.')
    measure = (jnp.linalg.norm(cell[0]) if dimension == 1 else
               jnp.linalg.norm(jnp.cross(cell[0], cell[1])) if dimension == 2 else jnp.linalg.det(cell))
    z = jnp.atleast_1d(frequency)+1j*width
    minus, plus = energy[None, :]-z[:, None], energy[None, :]+z[:, None]
    if np.any(np.asarray(minus) == 0) or np.any(np.asarray(plus) == 0):
        raise ValueError('An undamped pole requires nonzero broadening.')
    response = (jnp.einsum('ws,si,sj->wij', 1/minus, moments, moments.conj())
                +jnp.einsum('ws,si,sj->wij', 1/plus, moments.conj(), moments))/measure
    return response[0] if frequency.ndim == 0 else response
