"""Real-space force constants and shortest-image Fourier interpolation.

Phi_ab(q) = sum_R C_ab(R) exp(2*pi*i*q.R), with fractional reciprocal q.
Atomic positions select the shortest images but do not enter this phase.
This is the lattice-translation convention used by the native DFPT driver.
Construction is eager; the matrix Fourier sum supports JAX derivatives in q.
Long-range polar corrections must be handled separately.
"""
from dataclasses import dataclass
from typing import NamedTuple
import itertools
import jax
import jax.numpy as jnp
import numpy as np
from .phonon import _modes,_project_translations


class PhononBands(NamedTuple):
    qpoints: object
    frequencies: object
    modes: object


def _shortest_images(offset,super_lattice):
    """All closest lattice images, with a singular-value-certified search box."""
    center=-np.rint(offset@np.linalg.inv(super_lattice)).astype(int)
    residual=offset+center@super_lattice
    singular=np.linalg.svd(super_lattice,compute_uv=False)[-1]
    radius=int(np.ceil(2*np.linalg.norm(residual)/singular))
    if (2*radius+1)**3>1_000_000:
        raise ValueError('Shortest-image search is too large; use a better conditioned lattice basis')
    offsets=np.asarray(list(itertools.product(range(-radius,radius+1),repeat=3)),dtype=int)
    images=center+offsets
    distances=np.sum((residual+offsets@super_lattice)**2,axis=1)
    tolerance=1e-10*max(1.,float(np.min(distances)))
    return images[distances<=np.min(distances)+tolerance]


@dataclass(frozen=True)
class ForceConstants:
    """Sparse pair/image blocks in Eh/Bohr**2, with explicit masses in amu.

    from_qmesh accepts raw unweighted Phi(q), not the dynamical matrix.
    Signed frequencies are hbar*omega in Eh; negative values denote imaginary
    modes. bands() uses the existing numerical mode solver, without a new
    eigenvector derivative promise at degeneracy.
    """
    translations: object
    pairs: object
    values: object
    masses: object
    mesh: tuple
    raw_asr_residual: object
    asr_correction_norm: object

    @classmethod
    def from_qmesh(cls,cell,qpoints,matrices,*,masses,asr=False,tolerance=1e-8):
        """Inverse-transform a complete Gamma-centered rectangular q mesh.

        Input order and reciprocal representatives are arbitrary. Equal-length
        shortest images of each atom pair share the aliased coefficient. ASR
        is opt-in: project the Gamma matrix before Fourier transformation,
        leaving other sampled matrices unchanged and retaining the raw defect.
        """
        if not np.isfinite(tolerance) or tolerance<=0:raise ValueError('tolerance must be positive and finite')
        lattice=np.asarray(cell.lattice,dtype=float);coords=np.asarray(cell.coords,dtype=float)
        if lattice.shape!=(3,3) or not np.isfinite(lattice).all() or np.linalg.det(lattice)<=0:
            raise ValueError('Use a finite right-handed nonsingular lattice')
        if coords.ndim!=2 or coords.shape[1]!=3 or not len(coords) or not np.isfinite(coords).all():
            raise ValueError('Atomic coordinates must be a finite (natom,3) array in Bohr')
        masses=np.asarray(masses,dtype=float);nat=len(coords);dimension=3*nat
        if masses.shape!=(nat,) or not np.isfinite(masses).all() or np.any(masses<=0):
            raise ValueError('Supply one finite positive mass in amu per atom')
        if np.iscomplexobj(qpoints):raise ValueError('q mesh must be real')
        q=np.asarray(qpoints,dtype=float)
        if q.ndim!=2 or q.shape[1]!=3 or not len(q) or not np.isfinite(q).all():
            raise ValueError('q mesh must have finite shape (nq,3)')
        wrapped=np.mod(np.round(np.mod(q,1.),10),1.)
        mesh=tuple(len(np.unique(wrapped[:,i])) for i in range(3))
        scaled=q*np.asarray(mesh);indices=np.rint(scaled).astype(int)%np.asarray(mesh)
        if (np.prod(mesh)!=len(q) or not np.allclose(scaled,np.rint(scaled),atol=1e-8,rtol=0)
                or len(np.unique(indices,axis=0))!=len(q)):
            raise ValueError('Use a complete Gamma-centered rectangular q grid')
        matrices=jnp.asarray(matrices)
        if not jnp.issubdtype(matrices.dtype,jnp.inexact):matrices=matrices.astype(jnp.asarray(0.).dtype)
        if matrices.shape!=(len(q),dimension,dimension) or not np.isfinite(matrices).all():
            raise ValueError('Force constants must be finite (nq,3*natom,3*natom) matrices')
        scale=max(1.,float(jnp.max(jnp.abs(matrices))))
        if float(jnp.max(jnp.abs(matrices-matrices.conj().swapaxes(-1,-2))))>tolerance*scale:
            raise ValueError('Force-constant matrices must be Hermitian')
        order=np.argsort(np.ravel_multi_index(indices.T,mesh))
        ordered=matrices[order]
        negative=np.ravel_multi_index(((-indices[order])%np.asarray(mesh)).T,mesh)
        if float(jnp.max(jnp.abs(ordered[negative]-ordered.conj())))>tolerance*scale:
            raise ValueError('Raw q/-q conjugacy must hold before any ASR projection')
        translations=jnp.tile(jnp.eye(3),(nat,1))
        raw_asr=jnp.linalg.norm(ordered[0]@translations)
        correction=jnp.array(0.,dtype=matrices.real.dtype)
        if asr:
            gamma=_project_translations(ordered[0])
            correction=jnp.linalg.norm(gamma-ordered[0])
            ordered=ordered.at[0].set(gamma)
        realspace=jnp.fft.fftn(ordered.reshape(mesh+(dimension,dimension)),axes=(0,1,2))/len(q)
        if float(jnp.max(jnp.abs(realspace.imag)))>tolerance*scale:
            raise ValueError('q/-q conjugacy is required for real-space force constants')
        coefficients=realspace.real.reshape(len(q),nat,3,nat,3)
        super_lattice=np.asarray(mesh)[:,None]*lattice
        images=[];pairs=[];source=[];weights=[]
        for index,r in enumerate(itertools.product(*[range(n) for n in mesh])):
            r=np.asarray(r)
            for a in range(nat):
                for b in range(nat):
                    closest=_shortest_images(r@lattice+coords[b]-coords[a],super_lattice)
                    for shift in closest:
                        images.append(r+shift*np.asarray(mesh));pairs.append((a,b))
                        source.append((index,a,b));weights.append(1/len(closest))
        source=np.asarray(source)
        values=coefficients[source[:,0],source[:,1],:,source[:,2],:]*jnp.asarray(weights)[:,None,None]
        return cls(jnp.asarray(np.asarray(images)),jnp.asarray(np.asarray(pairs)),values,
                   jnp.asarray(masses),mesh,raw_asr,correction)

    def matrix(self,q):
        """Interpolate unweighted Phi(q); q has shape (...,3) and is real."""
        q=jnp.asarray(q)
        if q.ndim<1 or q.shape[-1]!=3 or jnp.iscomplexobj(q):
            raise ValueError('q must have real shape (...,3)')
        if not isinstance(q,jax.core.Tracer) and not np.isfinite(q).all():raise ValueError('q must be finite')
        q=q.astype(jnp.result_type(q,self.values.real))
        phases=jnp.exp(2j*jnp.pi*jnp.einsum('...i,ti->...t',q,self.translations))
        dimension=3*len(self.masses)
        rows=3*self.pairs[:,0,None]+jnp.arange(3)
        columns=3*self.pairs[:,1,None]+jnp.arange(3)
        def assemble(phase):
            result=jnp.zeros((dimension,dimension),dtype=jnp.result_type(phases,self.values))
            return result.at[rows[:,:,None],columns[:,None,:]].add(phase[:,None,None]*self.values)
        matrices=jax.vmap(assemble)(phases.reshape(-1,len(self.translations)))
        return matrices.reshape(q.shape[:-1]+(dimension,dimension))

    def bands(self,qpoints):
        """Numerical modes at supplied fractional reciprocal points."""
        q=jnp.asarray(qpoints)
        if q.ndim!=2 or q.shape[1]!=3 or not len(q):raise ValueError('Use a nonempty (nq,3) path')
        interpolate=jax.jit(lambda point:self.matrix(point))
        frequencies=[];modes=[]
        for point in q:
            matrix=interpolate(point)
            if float(jnp.max(jnp.abs(matrix-matrix.conj().T)))>1e-8*max(1.,float(jnp.max(jnp.abs(matrix)))):
                raise RuntimeError('Interpolated force constants fail Hermiticity')
            _,values,vectors=_modes(matrix,self.masses,False)
            frequencies.append(values);modes.append(vectors)
        return PhononBands(q,jnp.stack(frequencies),jnp.stack(modes))

    def dos(self,qpoints,energies,*,width,weights=None):
        """Unit-area Gaussian DOS in states/Eh/cell; width is sigma in Eh.

        Integrates to 3*natom on a sufficiently wide/resolved energy grid.
        Imaginary frequencies retain their negative signed-energy positions.
        """
        if not np.isfinite(width) or width<=0:raise ValueError('width must be positive and finite')
        energy=np.asarray(energies,dtype=float)
        if energy.ndim!=1 or len(energy)<2 or not np.isfinite(energy).all() or np.any(np.diff(energy)<=0):
            raise ValueError('Use a finite increasing one-dimensional energy grid')
        frequencies=self.bands(qpoints).frequencies;nq=len(frequencies)
        weights=np.full(nq,1/nq) if weights is None else np.asarray(weights,dtype=float)
        if weights.shape!=(nq,) or not np.isfinite(weights).all() or np.any(weights<0) or not np.isclose(weights.sum(),1.):
            raise ValueError('DOS q weights must be nonnegative and normalized')
        levels=frequencies.ravel();strengths=jnp.repeat(jnp.asarray(weights),frequencies.shape[1])
        spectrum=jnp.zeros(len(energy),dtype=frequencies.dtype)
        for start in range(0,len(levels),256):
            offsets=(jnp.asarray(energy)[:,None]-levels[None,start:start+256])/width
            profiles=jnp.exp(-.5*offsets**2)/(width*jnp.sqrt(2*jnp.pi))
            spectrum+=profiles@strengths[start:start+256]
        return spectrum
