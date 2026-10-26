"""Static AO-aware stars for exact finite-discretization spatial reduction.

The constructor consumes absolute Cartesian k points (radians/Bohr), whereas
Operation consumes fractional reciprocal coordinates. Source k points and AOs
remain full: this plan reconstructs target matrices and sums target weights.
Frames are static; explicit derivatives require symmetry-preserving changes.
An SCF using this plan must certify the full residual and differentiate that
full residual for unrestricted implicit response.
"""
from dataclasses import dataclass, field
import numpy as np
import jax.numpy as jnp
from .symmetry import Operation, space_group, _readonly


def _adjoint(matrix):
    return jnp.swapaxes(matrix.conj(),-1,-2)


def _conjugate(values,mask):
    mask=jnp.asarray(mask).reshape((1,)*(values.ndim-3)+(-1,1,1))
    return jnp.where(mask,values.conj(),values)


@dataclass(frozen=True,eq=False)
class SpatialReduction:
    """Immutable star metadata with contravariant density/covariant Fock maps.

    All matrix methods accept (...,nk,nao,nao). Consistency checks include every
    retained operation at every k, including stabilizers of representatives.
    Explicit operations incompatible with the sampling raise ValueError;
    automatic discovery retains only compatible operations. Identity is always
    present, so discovery can legitimately produce no reduction.
    """
    representatives: object
    full_to_ir: object
    operations: tuple
    conjugate: object
    _frames: object=field(repr=False)
    _inverse_frames: object=field(repr=False)
    _permutations: object=field(repr=False)
    _edge_frames: object=field(repr=False)
    _edge_inverses: object=field(repr=False)
    _antiunitary: object=field(repr=False)
    _fractional_k: object=field(repr=False)
    _integer_g: object=field(repr=False)
    _mesh: tuple=field(repr=False)
    tolerance: float=1e-8
    integrates: bool=field(default=True,init=False)

    @classmethod
    def from_cell(cls,cell,kpoints,operations=None,time_reversal=False,kweights=None,tolerance=1e-8):
        if np.ndim(tolerance)!=0 or not np.isfinite(tolerance) or tolerance<=0:
            raise ValueError('tolerance must be finite and positive.')
        k=np.asarray(kpoints)
        if k.ndim!=2 or k.shape[1]!=3 or not len(k) or np.iscomplexobj(k) or not np.isfinite(k).all():
            raise ValueError('Expected finite real absolute Cartesian (nk,3) k points.')
        mesh=np.asarray(cell.mesh)
        if mesh.shape!=(3,) or np.any(mesh<3) or np.any(mesh!=mesh.astype(int)) or np.any(mesh%2==0):
            raise ValueError('Exact reduction requires three odd FFT mesh sizes >=3.')
        nk=len(k);fractional=k@np.asarray(cell.lattice).T/(2*np.pi)
        distance=np.max(np.abs(fractional[:,None]-fractional[None,:]),axis=-1)
        if np.any(distance[np.triu_indices(nk,1)]<=tolerance):
            raise ValueError('The full k mesh must contain unique unwrapped points.')
        weights=np.full(nk,1/nk) if kweights is None else np.asarray(kweights)
        if (weights.shape!=(nk,) or np.iscomplexobj(weights) or not np.isfinite(weights).all()
                or np.any(weights<0) or not np.isclose(weights.sum(),1.,atol=tolerance,rtol=0)):
            raise ValueError('k weights must be finite, nonnegative and normalized.')
        automatic=operations is None
        supplied=space_group(cell,tolerance=tolerance) if automatic else tuple(operations)
        for op in supplied:
            if not isinstance(op,Operation):raise TypeError('operations must contain Operation objects.')
            op.validate(cell)
        identity=Operation.from_cell(cell,np.eye(3),tolerance=tolerance)
        retained=[];permutations=[];edges=[];antiunitary=[];seen=set()
        for op in (identity,)+tuple(supplied):
            key=(op.rotation.tobytes(),op.translation.tobytes())
            if key in seen:continue
            seen.add(key)
            inverse=np.rint(np.linalg.inv(op.rotation)).astype(int)
            half=mesh//2
            real_grid=op.rotation.T*mesh[None,:]/mesh[:,None]
            compatible=(np.all(half@np.abs(inverse)<=half)
                        and np.allclose(real_grid,np.rint(real_grid),atol=tolerance,rtol=0)
                        and np.allclose(op.translation*mesh,np.rint(op.translation*mesh),atol=tolerance,rtol=0))
            if not compatible:
                if automatic:continue
                raise ValueError('Operation does not preserve the finite reciprocal box or real FFT grid.')
            added=False
            for anti in ((False,True) if time_reversal else (False,)):
                target=(-fractional if anti else fractional)@inverse
                matches=np.max(np.abs(target[:,None]-fractional[None,:]),axis=-1)<=tolerance
                if not (np.all(matches.sum(axis=1)==1) and np.all(matches.sum(axis=0)==1)):
                    if automatic:continue
                    raise ValueError('Operation does not preserve the unwrapped finite k mesh.')
                permutation=matches.argmax(axis=1)
                if not np.allclose(weights,weights[permutation],atol=tolerance,rtol=0):
                    if automatic:continue
                    raise ValueError('k weights break the requested symmetry of the discretization.')
                matrices=np.stack([np.asarray(op.matrix(-point if anti else point)) for point in fractional])
                permutations.append(permutation);edges.append(matrices);antiunitary.append(anti);added=True
            if added:retained.append(op)
        permutations=np.asarray(permutations);edges=np.asarray(edges);antiunitary=np.asarray(antiunitary)
        nao=cell.topology.nao;mapping=np.full(nk,-1,dtype=int)
        frames=np.zeros((nk,nao,nao),dtype=complex);conjugate=np.zeros(nk,dtype=bool);representatives=[]
        for representative in range(nk):
            if mapping[representative]>=0:continue
            ir=len(representatives);representatives.append(representative)
            mapping[representative]=ir;frames[representative]=np.eye(nao);queue=[representative]
            for source in queue:
                for permutation,edge,anti in zip(permutations,edges,antiunitary):
                    target=int(permutation[source])
                    if mapping[target]>=0:continue
                    mapping[target]=ir
                    frames[target]=edge[source]@(frames[source].conj() if anti else frames[source])
                    conjugate[target]=bool(conjugate[source])^bool(anti)
                    queue.append(target)
        integer_g=np.stack(np.meshgrid(*[np.fft.fftfreq(int(n))*n for n in mesh],indexing='ij'),axis=-1).reshape(-1,3)
        return cls(_readonly(representatives,int),_readonly(mapping,int),tuple(retained),_readonly(conjugate,bool),
                   _readonly(frames),_readonly(np.linalg.inv(frames)),_readonly(permutations,int),
                   _readonly(edges),_readonly(np.linalg.inv(edges)),_readonly(antiunitary,bool),
                   _readonly(fractional),_readonly(integer_g),tuple(map(int,mesh)),float(tolerance))

    def _matrices(self,matrices,nk):
        values=jnp.asarray(matrices);nao=self._frames.shape[-1]
        if values.ndim<3 or values.shape[-3:]!=(nk,nao,nao):
            raise ValueError(f'Matrices require (...,{nk},{nao},{nao}).')
        return values

    def expand(self,matrices):
        """Reconstruct full contravariant densities from representative blocks."""
        values=self._matrices(matrices,len(self.representatives))
        values=_conjugate(jnp.take(values,self.full_to_ir,axis=-3),self.conjugate)
        frames=jnp.asarray(self._frames)
        return frames@values@_adjoint(frames)

    def expand_operator(self,matrices):
        """Reconstruct covariant operators, including nonunitary Cartesian AOs."""
        values=self._matrices(matrices,len(self.representatives))
        values=_conjugate(jnp.take(values,self.full_to_ir,axis=-3),self.conjugate)
        inverse=jnp.asarray(self._inverse_frames)
        return _adjoint(inverse)@values@inverse

    def reduce_weights(self,weights):
        weights=jnp.asarray(weights)
        if weights.shape!=(len(self.full_to_ir),):raise ValueError('Expected full k weights.')
        return jnp.zeros(len(self.representatives),dtype=weights.dtype).at[self.full_to_ir].add(weights)

    def weights_consistent(self,weights):
        weights=jnp.asarray(weights)
        if weights.shape!=(len(self.full_to_ir),) or jnp.iscomplexobj(weights):return jnp.asarray(False)
        return (jnp.all(jnp.isfinite(weights)) & jnp.all(weights>=0)
                & (jnp.abs(jnp.sum(weights)-1)<=self.tolerance)
                & jnp.all(jnp.abs(weights[self._permutations]-weights)<=self.tolerance))

    def _consistent(self,matrices,tolerance,operator):
        values=self._matrices(matrices,len(self.full_to_ir));valid=jnp.asarray(True)
        for permutation,frame,inverse,anti in zip(self._permutations,self._edge_frames,self._edge_inverses,self._antiunitary):
            source=values.conj() if anti else values
            frame=jnp.asarray(frame);inverse=jnp.asarray(inverse)
            moved=_adjoint(inverse)@source@inverse if operator else frame@source@_adjoint(frame)
            valid=valid & (jnp.max(jnp.abs(jnp.take(values,permutation,axis=-3)-moved))<=tolerance)
        return valid

    def consistent(self,matrices,tolerance=1e-8):
        """Check covariant operator symmetry, including every self-stabilizer."""
        return self._consistent(matrices,tolerance,True)

    def consistent_density(self,matrices,tolerance=1e-8):
        """Check contravariant density symmetry, including every self-stabilizer."""
        return self._consistent(matrices,tolerance,False)

    def sampling_consistent(self,inputs):
        """Check fractional sampling without freezing the physical lattice.

        This guard supports symmetry-preserving strain/geometry derivatives;
        it is not a replacement for full overlap/Fock/residual certification.
        """
        lattice=getattr(inputs,'lattice',None)
        if lattice is None:return jnp.asarray(False)
        lattice=jnp.asarray(lattice);k=jnp.asarray(inputs.kpoints);gv=jnp.asarray(inputs.gvectors)
        if lattice.shape!=(3,3) or k.shape!=self._fractional_k.shape or gv.shape!=self._integer_g.shape:
            return jnp.asarray(False)
        weights=getattr(inputs,'kweights',None)
        if weights is None:weights=jnp.full(len(k),1/len(k))
        return (jnp.all(jnp.abs(k@lattice.T/(2*jnp.pi)-self._fractional_k)<=self.tolerance)
                & jnp.all(jnp.abs(gv@lattice.T/(2*jnp.pi)-self._integer_g)<=self.tolerance)
                & self.weights_consistent(weights))
