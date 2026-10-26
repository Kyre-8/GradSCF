"""Reciprocal coordinates use radians per Bohr; lattice vectors are rows."""
import numpy as np
import jax.numpy as jnp
from typing import NamedTuple


def reciprocal_vectors(lattice):
    return 2*jnp.pi*jnp.linalg.inv(jnp.asarray(lattice)).T


def uniform_kpoints(lattice, mesh, *, shift=(0.,0.,0.)):
    mesh=np.asarray(mesh)
    if mesh.shape!=(3,) or np.any(mesh<1) or np.any(mesh!=mesh.astype(int)):
        raise ValueError('k mesh must contain three positive integers.')
    shift=np.asarray(shift,dtype=float)
    if shift.shape!=(3,) or not np.isfinite(shift).all():
        raise ValueError('shift must contain three finite grid offsets')
    scaled=np.stack(np.meshgrid(*[(np.arange(int(n))+s)/n for n,s in zip(mesh,shift)],indexing='ij'),axis=-1).reshape(-1,3)
    scaled=scaled-np.floor(scaled+.5)
    return jnp.asarray(scaled)@reciprocal_vectors(lattice)


def kmesh_shape(lattice,kpoints):
    scaled=np.asarray(kpoints)@np.linalg.inv((2*np.pi*np.linalg.inv(np.asarray(lattice)).T))
    scaled=np.mod(np.round(scaled,10),1.)
    axes=[np.unique(scaled[:,i]) for i in range(3)]
    shape=tuple(len(v) for v in axes)
    if np.prod(shape)!=len(scaled) or len(np.unique(scaled,axis=0))!=len(scaled):
        raise ValueError('Use a complete uniform Cartesian-product k mesh.')
    for values in axes:
        if not np.allclose(np.diff(np.r_[values,values[0]+1]),1/len(values),atol=1e-8):
            raise ValueError('k points must be uniformly spaced.')
    return shape


def weights_for(inputs):
    """Normalized integration weights, retaining legacy equal-weight inputs."""
    n=len(inputs.kpoints)
    weights=getattr(inputs,'kweights',None)
    return jnp.full(n,1/n) if weights is None else jnp.asarray(weights)


class KPointReduction(NamedTuple):
    representatives: object
    full_to_ir: object
    weights: object


class TimeReversalPlan(NamedTuple):
    """Representative k points plus conjugation maps for scalar collinear AOs."""
    representatives: object
    full_to_ir: object
    conjugate: object
    partners: object

    def expand(self, matrices):
        values=jnp.take(matrices,self.full_to_ir,axis=-3)
        mask=jnp.asarray(self.conjugate).reshape((1,)*(values.ndim-3)+(-1,1,1))
        return jnp.where(mask,values.conj(),values)

    def reduce_weights(self, weights):
        return jnp.zeros(len(self.representatives),dtype=weights.dtype).at[self.full_to_ir].add(weights)

    def weights_consistent(self, weights):
        return jnp.all(jnp.abs(weights-weights[self.partners])<1e-12)

    def consistent(self, matrices, tolerance=1e-8):
        # Self-paired Gamma/TRIM points must also equal their own conjugate.
        restored=jnp.take(matrices,self.partners,axis=-3).conj()
        return jnp.max(jnp.abs(matrices-restored))<=tolerance


def time_reversal_plan(kpoints, *, tolerance=1e-8):
    """Pair fractional k and -k modulo reciprocal vectors; no spatial rotations.

    AO matrices use the lattice-translation Bloch convention, for which paired
    matrices are complex conjugates. The SCF additionally verifies this on
    overlap and Fock matrices: finite FFT cutoffs need not preserve it exactly.
    """
    from .momentum import wrap_to_kmesh
    k=wrap_to_kmesh(kpoints)
    reduced=symmetry_reduce(k,time_reversal=True,tolerance=tolerance)
    partners=[]
    for point in k:
        found=np.flatnonzero(np.max(np.abs(wrap_to_kmesh(k+point)),axis=1)<tolerance)
        if len(found)!=1:raise ValueError('Time reversal requires a unique partner for every k point')
        partners.append(found[0])
    conjugate=np.arange(len(k))!=reduced.representatives[reduced.full_to_ir]
    return TimeReversalPlan(reduced.representatives,reduced.full_to_ir,conjugate,np.asarray(partners))


def symmetry_reduce(kpoints, *, rotations=None, time_reversal=True, tolerance=1e-8):
    """Orbit metadata for fractional k points under supplied reciprocal rotations.

    This reduces scalar Brillouin-zone sums. It does not rotate Bloch orbitals
    or AO density matrices, and is not an irreducible-SCF evaluation backend.
    Callers must supply symmetries of their physical Hamiltonian; time reversal
    is not a default physical assumption for magnetic/spinor calculations.
    """
    from .momentum import wrap_to_kmesh
    k=wrap_to_kmesh(kpoints)
    if k.ndim!=2 or k.shape[1]!=3 or not len(k) or not np.isfinite(k).all():
        raise ValueError('Expected finite fractional (nk,3) points')
    operations=np.eye(3)[None] if rotations is None else np.asarray(rotations)
    if operations.ndim!=3 or operations.shape[1:]!=(3,3):
        raise ValueError('Reciprocal rotations require shape (nrotation,3,3)')
    if time_reversal:operations=np.concatenate([operations,-operations])
    parent=np.arange(len(k))
    def root(i):
        while parent[i]!=i:i=parent[i]
        return i
    for op in operations:
        for i,target in enumerate(k@op.T):
            matches=np.flatnonzero(np.max(np.abs(wrap_to_kmesh(k-target)),axis=1)<tolerance)
            if len(matches)!=1:raise ValueError('Symmetry does not map the full mesh uniquely onto itself')
            a,b=root(i),root(matches[0]);parent[max(a,b)]=min(a,b)
    representatives,mapping=np.unique([root(i) for i in range(len(k))],return_inverse=True)
    return KPointReduction(representatives,mapping,np.bincount(mapping)/len(k))
