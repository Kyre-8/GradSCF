"""Polar IFC interpolation with an explicit reciprocal dipole decomposition.

All force constants are unweighted, in Eh/bohr**2. The Gaussian reciprocal
model uses the lattice-translation phase convention of pbc.ifc. Its fixed
finite G sum is differentiable away from reciprocal-integer Gamma points;
G-cutoff, smearing and q-mesh convergence are separate requirements.
"""
from dataclasses import dataclass
from numbers import Integral
import itertools

import jax.numpy as jnp
import numpy as np

from .ifc import ForceConstants
from .vibrational import _born_charges, _real_finite, nonanalytic_force_constants
from ..gw.ep_coupling.types import _require


@dataclass(frozen=True)
class DipoleCorrection:
    r"""Gaussian reciprocal dipole model, without an onsite ASR convention.

    Lattice rows and positions are in bohr. Born[a,i,j] has polarization index
    i and displacement index j, in e. Dielectric is epsilon_infinity; smearing
    Lambda is in bohr^-1, using angular reciprocal vectors including 2*pi.
    ``cutoff`` is the integer half-width of a fixed reciprocal cube.

    For Q=(G+q).B and b[a,j]=exp(iQ.tau_a)*sum_i Q_i Z[a,i,j],
    L(q)=4*pi/V sum_G b b^dagger exp[-Q.epsilon.Q/(4*Lambda**2)]/(Q.epsilon.Q).

    A fixed finite G set is only approximately periodic in q. Neither q nor
    its derivatives are silently wrapped; verify cutoff convergence at the
    representatives used in the calculation. Raw Born charge-sum defects are
    retained. See Gonze/Lee, Phys. Rev. B 55,10355 (1997), and Togo et al.,
    J. Phys.: Condens. Matter 35,353001 (2023), Eqs.51--54.
    """
    lattice: object
    positions: object
    born_charges: object
    dielectric: object
    smearing: object = .5
    cutoff: int = 4

    def __post_init__(self):
        lattice = _real_finite(self.lattice, 'lattice')
        positions = _real_finite(self.positions, 'positions')
        born = _born_charges(self.born_charges)
        dielectric = _real_finite(self.dielectric, 'dielectric')
        smearing = _real_finite(self.smearing, 'smearing')
        if lattice.shape != (3, 3) or dielectric.shape != (3, 3):
            raise ValueError('lattice and dielectric must have shape (3,3)')
        if positions.shape != (born.shape[0], 3) or smearing.ndim:
            raise ValueError('positions must have shape (natom,3) and smearing must be scalar')
        if not isinstance(self.cutoff, Integral) or self.cutoff < 1:
            raise ValueError('cutoff must be a positive static integer')
        _require(jnp.linalg.det(lattice) > 0, 'lattice must have positive volume')
        _require(smearing > 0, 'smearing must be strictly positive')
        _require(jnp.allclose(dielectric, dielectric.T, atol=1e-12, rtol=1e-10),
                 'dielectric must be symmetric')
        _require(jnp.all(jnp.linalg.eigvalsh(dielectric) > 0), 'dielectric must be positive definite')
        for name, value in [('lattice', lattice), ('positions', positions),
                            ('born_charges', born), ('dielectric', dielectric), ('smearing', smearing)]:
            object.__setattr__(self, name, value)

    def _sum(self, q):
        integers = np.array(list(itertools.product(range(-self.cutoff, self.cutoff+1), repeat=3)))
        reciprocal = 2*jnp.pi*jnp.linalg.inv(self.lattice).T
        vectors = (integers+q)@reciprocal
        denominator = jnp.einsum('gi,ij,gj->g', vectors, self.dielectric, vectors)
        nonzero = jnp.sum(vectors*vectors, axis=1) > 0
        safe = jnp.where(nonzero, denominator, 1.)
        weights = jnp.where(nonzero, jnp.exp(-denominator/(4*self.smearing**2))/safe, 0.)
        phase = jnp.exp(1j*(vectors@self.positions.T))
        amplitude = (phase[:, :, None]*jnp.einsum('gi,aij->gaj', vectors, self.born_charges))
        amplitude = amplitude.reshape(len(integers), -1)
        return 4*jnp.pi/jnp.linalg.det(self.lattice) * (amplitude.T@(weights[:, None]*amplitude.conj()))

    def matrix(self, q):
        """Finite-G model at one non-Gamma fractional q; JIT/JVP/VJP supported.

        The Gamma point has no unique q derivative. Use gamma() for its
        short-circuit value or gamma(direction) for a specified Cartesian
        directional limit; parameters of either explicit limit may respond.
        """
        q = _real_finite(q, 'q')
        if q.shape != (3,):
            raise ValueError('q must have shape (3,) in fractional reciprocal coordinates')
        _require(jnp.linalg.norm(q-jnp.round(q)) > 1e-12,
                 'Use gamma() at reciprocal-integer Gamma; its q derivative is undefined')
        return self._sum(q)

    def gamma(self, direction=None):
        """G=0-excluded Gamma value, optionally with a Cartesian NA direction."""
        value = self._sum(jnp.zeros(3, dtype=self.lattice.dtype))
        if direction is not None:
            value = value+nonanalytic_force_constants(direction, self.born_charges,
                self.dielectric, jnp.linalg.det(self.lattice))
        return value


@dataclass(frozen=True)
class PolarForceConstants:
    """Total polar interpolation built from raw sampled unweighted matrices.

    Construction is eager, like ForceConstants.from_qmesh; it is not an AD
    route through the source SCF or sampled force constants. matrix(q) supports
    the finite-sum q response away from Gamma. No new mode solver is supplied.
    """
    short_range: ForceConstants
    correction: DipoleCorrection
    raw_asr_residual: object
    sample_reconstruction_error: float

    @classmethod
    def from_qmesh(cls, cell, qpoints, matrices, *, masses, correction, tolerance=1e-8):
        """Subtract on the mesh, interpolate the remainder, restore at target.

        All raw total matrices are validated before subtraction. No ASR is
        applied to either part. The separately stored total raw_asr_residual
        must not be confused with the residual short-range object's value.
        A q-independent atom-block-diagonal onsite convention cancels in this
        construction; omitting it keeps the reciprocal model Hermitian even
        for general nonsymmetric Born tensors.
        """
        if not isinstance(correction, DipoleCorrection):
            raise TypeError('correction must be a DipoleCorrection')
        if (not np.allclose(cell.lattice, correction.lattice, atol=1e-12, rtol=0)
                or not np.allclose(cell.coords, correction.positions, atol=1e-12, rtol=0)):
            raise ValueError('The dipole model must use the same lattice and atomic representatives')
        raw = ForceConstants.from_qmesh(cell, qpoints, matrices, masses=masses,
                                       asr=False, tolerance=tolerance)
        qpoints = np.asarray(qpoints, dtype=float)
        at_gamma = np.linalg.norm(qpoints-np.rint(qpoints), axis=1) < 1e-12
        long_range = jnp.stack([correction.gamma() if gamma else correction.matrix(q)
                               for q, gamma in zip(qpoints, at_gamma)])
        residual = jnp.asarray(matrices)-long_range
        short = ForceConstants.from_qmesh(cell, qpoints, residual, masses=masses,
                                         asr=False, tolerance=tolerance)
        error = float(jnp.max(jnp.abs(short.matrix(qpoints)+long_range-jnp.asarray(matrices))))
        if error > tolerance*max(1., float(jnp.max(jnp.abs(matrices)))):
            raise ValueError('The polar decomposition failed sampled-matrix reconstruction')
        return cls(short, correction, raw.raw_asr_residual, error)

    def matrix(self, q):
        """Restore the reciprocal dipole model at one non-Gamma fractional q."""
        return self.short_range.matrix(q)+self.correction.matrix(q)

    def gamma(self, direction=None):
        """Analytic sampled Gamma, or its specified Cartesian directional limit."""
        return self.short_range.matrix(jnp.zeros(3))+self.correction.gamma(direction)


__all__ = ['DipoleCorrection', 'PolarForceConstants']
